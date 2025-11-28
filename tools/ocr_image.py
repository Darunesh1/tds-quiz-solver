import os

import pytesseract
from langchain_core.tools import tool
from PIL import Image


@tool
def ocr_image(image_path: str) -> str:
    """Extract all text from image using Tesseract OCR. Fast & lightweight."""
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text.strip()
    except Exception as e:
        return f"OCR failed: {str(e)}"
