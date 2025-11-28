import logging

import pytesseract
from langchain_core.tools import tool
from PIL import Image

logger = logging.getLogger(__name__)


@tool
def ocr_image(image_path: str) -> str:
    """
    Extracts all visible text from an image using Tesseract OCR.

    Args:
        image_path (str): Path to image file (relative to current working directory or absolute)

    Returns:
        str: Extracted text (whitespace-stripped)
             On error: "OCR error: <details>"

    Supported Formats: PNG, JPEG, JPG, TIFF, BMP, GIF

    Usage:
    - For downloaded files: ocr_image("LLMFiles/screenshot.png")
    - Works best with high-contrast text
    - Returns empty string if no text detected

    Example:
        ocr_image("LLMFiles/quiz_question.png")
    """

    logger.info(f"👁️ OCR PROCESSING: {image_path}")
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        clean_text = text.strip()

        log_preview = (
            clean_text[:50].replace("\n", " ") + "..."
            if len(clean_text) > 50
            else clean_text
        )
        logger.info(f"✅ OCR Success: {log_preview}")

        return clean_text
    except Exception as e:
        logger.error(f"💥 OCR Failed: {e}")
        return f"OCR error: {str(e)}"
