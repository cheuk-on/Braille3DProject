import io
import math
import cv2
import numpy as np
import streamlit as st

# Try optional imports
try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import louis  # python-louis binding (if available)
except Exception:
    louis = None

st.set_page_config(page_title="3D Braille & Tactile Converter", layout="centered")

def text_to_braille(text: str) -> str:
    """
    Try to use liblouis (python-louis) if available, otherwise fall back to
    a simple Unicode Braille mapping for ASCII letters.
    """
    if not text:
        return ""
    if louis:
        try:
            return louis.translateString(None, text)
        except Exception:
            pass

    mapping = {
        "a": "\u2801", "b": "\u2803", "c": "\u2809", "d": "\u2819", "e": "\u2811",
        "f": "\u280b", "g": "\u281b", "h": "\u2813", "i": "\u280a", "j": "\u281a",
        "k": "\u2805", "l": "\u2807", "m": "\u280d", "n": "\u281d", "o": "\u2815",
        "p": "\u280f", "q": "\u281f", "r": "\u2817", "s": "\u280e", "t": "\u281e",
        "u": "\u2825", "v": "\u2827", "w": "\u283a", "x": "\u282d", "y": "\u283d",
        "z": "\u2835", " ": "\u2800"
    }
    out = []
    for ch in text:
        lower = ch.lower()
        out.append(mapping.get(lower, "\u283f" if ch.isdigit() else mapping.get(lower, "\u2800")))
    return "".join(out)


def braille_char_to_dots(ch: str):
    """Return a list of 6 booleans for dots 1..6 for a unicode braille char."""
    if not ch:
        return [False]*6
    code = ord(ch) - 0x2800
    dots = [(code >> i) & 1 == 1 for i in range(6)]
    return dots


def render_braille_image(braille_text: str, cell_size=40, padding=12):
    """
    Render braille text to a grayscale image (white background, black dots).
    """
    if not braille_text:
        return None
    cols = len(braille_text)
    cell_w = cell_size
    cell_h = int(cell_size * 1.6)
    width = cols * (cell_w + padding) + padding
    height = cell_h + 2 * padding
    img = np.ones((height, width), dtype=np.uint8) * 255

    radius = int(cell_size * 0.12)
    spacing_y = int(cell_h / 4)
    for i, ch in enumerate(braille_text):
        dots = braille_char_to_dots(ch)
        cx = padding + i * (cell_w + padding)
        x_left = cx + int(cell_w * 0.25)
        x_right = cx + int(cell_w * 0.70)
        y_top = padding + spacing_y
        y_mid = padding + spacing_y * 2
        y_bot = padding + spacing_y * 3
        positions = [(x_left, y_top), (x_left, y_mid), (x_left, y_bot),
                     (x_right, y_top), (x_right, y_mid), (x_right, y_bot)]
        for present, (x, y) in zip(dots, positions):
            if present:
                cv2.circle(img, (x, y), radius, (0,), -1)
    return img


def image_to_text(img_bytes: bytes) -> str:
    """Run OCR on uploaded image using pytesseract (if available)."""
    if not pytesseract:
        return ""
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        text = pytesseract.image_to_string(th)
    except Exception:
        text = ""
    return text.strip()


def heightmap_edges(img_bytes: bytes, scale=1.0):
    """Produce a simple heightmap (0..1) from edges of the image."""
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    max_dim = 128
    h, w = img.shape[:2]
    factor = max(1, max(h, w) // max_dim)
    if factor > 1:
        img = cv2.resize(img, (w//factor, h//factor), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(img, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    return (edges.astype(np.float32) / 255.0)


def heightmap_to_ascii_stl(heightmap: np.ndarray, voxel_size_mm=0.5, dot_height_mm=1.6, base_thickness_mm=1.0):
    """
    Convert a binary-ish heightmap (values 0..1) to a simple ASCII STL by placing small boxes.
    """
    if heightmap is None:
        return b""
    h, w = heightmap.shape
    lines = []
    lines.append("solid braille_tactile")
    def add_box(x, y, sx, sy, hgt):
        z0 = 0.0
        z1 = hgt
        v = [
            (x, y, z0),
            (x + sx, y, z0),
            (x + sx, y + sy, z0),
            (x, y + sy, z0),
            (x, y, z1),
            (x + sx, y, z1),
            (x + sx, y + sy, z1),
            (x, y + sy, z1),
        ]
        faces = [
            (0,1,2),(0,2,3),
            (4,7,6),(4,6,5),
            (0,4,5),(0,5,1),
            (1,5,6),(1,6,2),
            (2,6,7),(2,7,3),
            (3,7,4),(3,4,0),
        ]
        for a,b,c in faces:
            va = v[a]; vb = v[b]; vc = v[c]
            ux = vb[0]-va[0]; uy = vb[1]-va[1]; uz = vb[2]-va[2]
            vx = vc[0]-va[0]; vy = vc[1]-va[1]; vz = vc[2]-va[2]
            nx = uy*vz - uz*vy
            ny = uz*vx - ux*vz
            nz = ux*vy - uy*vx
            norm = math.sqrt(nx*nx + ny*ny + nz*nz) or 1.0
            nx, ny, nz = nx/norm, ny/norm, nz/norm
            lines.append(f"  facet normal {nx:.6f} {ny:.6f} {nz:.6f}")
            lines.append("    outer loop")
            lines.append(f"      vertex {va[0]:.6f} {va[1]:.6f} {va[2]:.6f}")
            lines.append(f"      vertex {vb[0]:.6f} {vb[1]:.6f} {vb[2]:.6f}")
            lines.append(f"      vertex {vc[0]:.6f} {vc[1]:.6f} {vc[2]:.6f}")
            lines.append("    endloop")
            lines.append("  endfacet")

    base_w = w * voxel_size_mm
    base_h = h * voxel_size_mm
    add_box(0.0, 0.0, base_w, base_h, base_thickness_mm)

    threshold = 0.1
    for r in range(h):
        for c in range(w):
            val = float(heightmap[r, c])
            if val > threshold:
                x = c * voxel_size_mm
                y = (h - 1 - r) * voxel_size_mm
                sx = voxel_size_mm
                sy = voxel_size_mm
                hgt = dot_height_mm * val
                add_box(x, y, sx, sy, base_thickness_mm + hgt)
    lines.append("endsolid braille_tactile")
    return ("\n".join(lines)).encode("utf-8")


# Streamlit UI
st.title("3D Braille & Tactile Converter")

st.sidebar.header("Input")
input_mode = st.sidebar.radio("Mode", ["Text", "Image"])

if input_mode == "Text":
    text = st.text_area("Enter text to convert", value="", height=120)
    if st.button("Translate to Braille"):
        braille = text_to_braille(text)
        st.subheader("Braille (Unicode)")
        st.write(braille)
        img = render_braille_image(braille)
        if img is not None:
            st.image(img, caption="Braille dots (preview)", width="content")
            _, png = cv2.imencode(".png", img)
            st.download_button("Download Braille PNG", data=png.tobytes(), file_name="braille.png", mime="image/png")
            hm = (255 - img).astype(np.float32) / 255.0
            stl_bytes = heightmap_to_ascii_stl(hm, voxel_size_mm=0.8, dot_height_mm=1.8, base_thickness_mm=1.2)
            st.download_button("Download Tactile STL", data=stl_bytes, file_name="braille_tactile.stl", mime="application/sla")
elif input_mode == "Image":
    uploaded = st.file_uploader("Upload image (photo)", type=["png", "jpg", "jpeg"])
    if uploaded:
        img_bytes = uploaded.read()
        st.image(img_bytes, caption="Uploaded image", width="stretch")
        if st.button("Run OCR and Convert"):
            ocr_text = image_to_text(img_bytes) if pytesseract else ""
            st.subheader("OCR Text")
            st.write(ocr_text or "No OCR (pytesseract not installed or no text found).")
            braille = text_to_braille(ocr_text)
            st.subheader("Braille")
            st.write(braille)
            img = render_braille_image(braille)
            if img is not None:
                st.image(img, caption="Braille dots (preview)")
                _, png = cv2.imencode(".png", img)
                st.download_button("Download Braille PNG", data=png.tobytes(), file_name="braille_from_image.png", mime="image/png")
            hm = heightmap_edges(img_bytes)
            if hm is not None:
                st.write("Preview of edge heightmap (small)")
                vis = (hm * 255).astype(np.uint8)
                st.image(vis, caption="Edge heightmap", width="content")
                stl_bytes = heightmap_to_ascii_stl(hm, voxel_size_mm=0.6, dot_height_mm=2.0, base_thickness_mm=1.5)
                st.download_button("Download Photo Tactile STL", data=stl_bytes, file_name="photo_tactile.stl", mime="application/sla")

st.info("Notes: This app attempts to use python-louis and pytesseract if installed. If not present, it falls back to simple mappings and basic processing.")
