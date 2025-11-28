import os

import requests
from langchain_core.tools import tool


@tool
def download_file(url: str, filename: str = None) -> str:
    """
    Download any file (image, audio, data) from a URL.

    Use this for:
    - Downloading images for OCR processing
    - Saving audio files for transcription
    - Fetching data files for analysis

    Args:
        url: Full URL to download from
        filename: Optional filename (auto-generates if not provided)

    Returns:
        Path to saved file

    Example: download_file('https://example.com/image.png', 'LLMFiles/image.png')
    """
    try:
        if not filename:
            filename = f"LLMFiles/{url.split('/')[-1]}"

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        with open(filename, "wb") as f:
            f.write(response.content)

        return f"Downloaded to: {filename}"
    except Exception as e:
        return f"Download failed: {str(e)}"

