# 3D Braille and Tactile Model Converter

A Streamlit web app for converting text and photos into 3D-printable Braille and tactile models, enhancing accessibility for visually impaired users. Supports English, Chinese Traditional, Chinese Simplified, and more.

## Features
- **Text-to-Braille**: Convert text (e.g., "hello", "你好") to Braille using `liblouis`.
- **Photo-to-Braille**: Extract text from images (OCR with OpenCV, pytesseract).
- **Photo-to-Tactile**: Convert building photos into tactile 3D models (e.g., walls, windows).
- **Multi-Language**: Prioritizes English, Chinese Traditional, Chinese Simplified.
- **Mobile-Friendly**: Works on mobile browsers with camera input.
- **Awards**: Subject Award, Mingpao Sustainability Award.

## Installation
```bash
pip install streamlit opencv-python pytesseract python-louis
