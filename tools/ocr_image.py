"""Image text extraction tool using Tesseract OCR."""

import logging
import os

import pytesseract
from langchain_core.tools import tool
from PIL import Image

logger = logging.getLogger(__name__)


@tool
def ocr_image(image_path: str) -> dict:
    """
    Extracts text from images using Tesseract OCR engine.

    **When to Use:**
    - Page has images that might contain text (questions, clues, instructions)
    - get_rendered_html() shows images but minimal text content
    - Challenge seems incomplete without examining images
    - Screenshots, diagrams, or scanned documents are present

    **How CTF Challenges Hide Information in Images:**
    1. **Screenshot Questions**: Entire question as an image (avoid text scraping)
    2. **Partial Clues**: Key numbers/words in images while context in text
    3. **Steganography Hints**: Instructions for decoding hidden data
    4. **Visual Puzzles**: Math equations, codes, or patterns as images

    **Workflow:**
    ```
    # Step 1: Identify image URLs
    page = get_rendered_html("https://quiz.com/level")
    image_url = page['assets']['images'][0]

    # Step 2: Download image locally
    file_info = download_file(image_url)

    # Step 3: Extract text
    result = ocr_image(file_info['filepath'])
    print(result['text'])
    ```

    **OCR Quality Factors:**
    - **Best**: High-contrast text, clean backgrounds, standard fonts
    - **Good**: Screenshots with clear text
    - **Poor**: Handwriting, artistic fonts, low resolution
    - **Fails**: Pure graphics, photos without text, encrypted images

    **Supported Formats:**
    PNG, JPEG, JPG, TIFF, BMP, GIF

    Args:
        image_path: Path to image file (use path from download_file)
                   Can be relative ("LLMFiles/image.png") or just filename

    Returns:
        dict: {
            'text': str,  # Extracted text (empty if none found)
            'char_count': int,  # Number of characters extracted
            'confidence': str,  # Quality estimate (high/medium/low)
            'filepath': str  # Original file path
        }
        On error: dict with 'error' field and 'suggestion'

    **Interpreting Results:**
    - Empty text + success: Image has no text (purely visual)
    - Short text (<10 chars): May be noise, verify manually
    - Garbled text: Try preprocessing (enhance contrast, resize)
    - Complete sentences: High confidence in accuracy

    **Pro Tips:**
    - If OCR fails, image might need preprocessing
    - Try multiple images if page has several
    - Some CTFs use QR codes (requires different tool)
    - Check for text orientation (Tesseract handles rotations)
    """
    logger.info(f"👁️ OCR PROCESSING: {image_path}")

    try:
        # Handle both full paths and bare filenames
        if not os.path.exists(image_path) and not image_path.startswith("LLMFiles"):
            alt_path = os.path.join("LLMFiles", image_path)
            if os.path.exists(alt_path):
                image_path = alt_path

        # Verify file exists
        if not os.path.exists(image_path):
            error_msg = f"Image file not found: {image_path}"
            logger.error(f"💥 {error_msg}")

            # List available files for debugging
            llm_files = os.listdir("LLMFiles") if os.path.exists("LLMFiles") else []
            return {
                "error": error_msg,
                "suggestion": f"Available files in LLMFiles/: {llm_files}",
                "filepath": image_path,
            }

        # Open and process image
        img = Image.open(image_path)
        logger.info(f"   Image size: {img.size}x{img.size} pixels")[3]

        # Extract text
        text = pytesseract.image_to_string(img)
        clean_text = text.strip()

        # Estimate confidence based on output characteristics
        char_count = len(clean_text)
        if char_count > 50 and clean_text.count(" ") > 5:
            confidence = "high"
        elif char_count > 10:
            confidence = "medium"
        else:
            confidence = "low"

        # Logging with preview
        if clean_text:
            preview = clean_text[:100].replace("\n", " ")
            logger.info(f"✅ OCR Success ({char_count} chars): {preview}...")
        else:
            logger.info("✅ OCR Complete: No text detected in image")

        return {
            "text": clean_text,
            "char_count": char_count,
            "confidence": confidence,
            "filepath": image_path,
            "success": True,
        }

    except FileNotFoundError as e:
        error_msg = f"Image file not found: {image_path}"
        logger.error(f"💥 {error_msg}")
        return {
            "error": error_msg,
            "suggestion": "Use download_file() first to get the image locally",
            "filepath": image_path,
        }

    except Exception as e:
        logger.error(f"💥 OCR Failed: {e}")
        return {
            "error": str(e),
            "suggestion": "Verify image format is supported (PNG, JPG, etc.)",
            "filepath": image_path,
        }
