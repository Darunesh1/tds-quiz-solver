"""File download tool with smart filename handling and validation."""

import logging
import os
import re
from urllib.parse import unquote, urlparse

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def download_file(url: str) -> dict:
    """
    Downloads a file from a URL to the local 'LLMFiles/' directory.

    **When to Use:**
    - After get_rendered_html() identifies interesting files (images, audio, data)
    - When you need to process file contents locally (OCR, transcription, data analysis)
    - Before using ocr_image(), transcribe_audio(), or run_code() on external files

    **File Naming:**
    - Automatically extracts filename from URL
    - Handles URL-encoded filenames (e.g., %20 → space)
    - Falls back to descriptive names based on content type if no filename found
    - Preserves file extensions for tool compatibility

    **Supported File Types:**
    - Images: PNG, JPG, GIF, SVG, etc. (for ocr_image)
    - Audio: MP3, WAV, OGG, M4A (for transcribe_audio)
    - Data: CSV, JSON, XML, TXT (for run_code analysis)
    - Documents: PDF, DOCX (may require additional processing)
    - Any other binary format

    **Common Patterns:**
    ```
    # Pattern 1: Image with hidden text
    result = download_file("https://example.com/clue.png")
    text = ocr_image(result['filepath'])

    # Pattern 2: Audio clue
    result = download_file("https://example.com/hint.mp3")
    text = transcribe_audio(result['filepath'])

    # Pattern 3: Data analysis
    result = download_file("https://example.com/data.csv")
    run_code("import pandas as pd; df = pd.read_csv('data.csv'); print(df.head())")
    ```

    Args:
        url: Direct download URL (must be accessible via HTTP GET)

    Returns:
        dict: {
            'filepath': str,  # Relative path (e.g., "LLMFiles/image.png")
            'filename': str,  # Just the filename (e.g., "image.png")
            'size_bytes': int,  # File size
            'content_type': str  # MIME type from server
        }
        On error: dict with 'error' field and 'suggestion'

    Important Notes:
    - Files are saved to LLMFiles/ directory (auto-created if missing)
    - When using downloaded files with run_code(), reference by filename only
        ✓ Correct: pd.read_csv('data.csv')
        ✗ Wrong: pd.read_csv('LLMFiles/data.csv')
    - Timeout: 30 seconds for download
    - Existing files with same name will be overwritten
    """
    try:
        # Ensure download directory exists
        os.makedirs("LLMFiles", exist_ok=True)

        # Extract filename from URL with better handling
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        filename = unquote(filename)  # Decode URL encoding

        # Fallback if no filename in URL
        if not filename or filename == "":
            filename = "downloaded_file.dat"

        # Remove query parameters from filename if present
        filename = filename.split("?")[0]

        filepath = os.path.join("LLMFiles", filename)

        logger.info(f"📥 DOWNLOADING: {url}")
        logger.info(f"   └─ Saving to: {filepath}")

        # Download with streaming for large files
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Write file in chunks
        total_size = 0
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                total_size += len(chunk)

        content_type = response.headers.get("Content-Type", "unknown")

        logger.info(f"✅ Download Complete: {total_size} bytes ({content_type})")

        return {
            "filepath": filepath,
            "filename": filename,
            "size_bytes": total_size,
            "content_type": content_type,
            "success": True,
        }

    except requests.exceptions.Timeout:
        error_msg = f"Download timeout (30s limit) for {url}"
        logger.error(f"💥 {error_msg}")
        return {
            "error": error_msg,
            "url": url,
            "suggestion": "File may be too large or server is slow. Try a different URL.",
        }

    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP Error {e.response.status_code}: {url}"
        logger.error(f"💥 {error_msg}")
        return {
            "error": error_msg,
            "url": url,
            "suggestion": "Check if URL is correct and accessible. May require authentication.",
        }

    except Exception as e:
        logger.error(f"💥 Download Failed: {e}")
        return {
            "error": str(e),
            "url": url,
            "suggestion": "Verify URL format and network connectivity",
        }

