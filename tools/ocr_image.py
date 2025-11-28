import pytesseract
from langchain_core.tools import tool
from PIL import Image


@tool
def ocr_image(image_path: str) -> str:
    """
    Extract all visible text from an image file using OCR.

    Use this for:
    - Reading text from screenshots
    - Extracting codes/secrets from images
    - Processing visual quiz content

    Args:
        image_path: Full path to image file (e.g., 'LLMFiles/quiz.png')

    Returns:
        Extracted text as string

    Example: ocr_image('LLMFiles/screenshot.png')
    """
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text.strip()
    except Exception as e:
        return f"OCR error: {str(e)}"
