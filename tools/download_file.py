import logging
import os

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def download_file(url: str) -> str:
    """
    Downloads a file from a URL to the local 'LLMFiles/' directory.

    Args:
        url (str): Direct download URL (supports http/https)

    Returns:
        str: Success: "File saved at: LLMFiles/<filename>"
             Failure: Error message with details

    File Naming:
    - Extracts filename from URL (last segment before query params)
    - Falls back to 'downloaded_file.dat' if no filename found

    Timeout: 30 seconds

    Supported File Types: All (images, PDFs, CSVs, audio, video, etc.)

    Example:
        download_file("https://example.com/data.csv")
        # Returns: "File saved at: LLMFiles/data.csv"
    """

    try:
        os.makedirs("LLMFiles", exist_ok=True)
        filename = url.split("/")[-1].split("?")[0]
        if not filename:
            filename = "downloaded_file.dat"

        filepath = os.path.join("LLMFiles", filename)
        logger.info(f"📥 DOWNLOADING: {url} -> {filepath}")

        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("✅ Download Complete")
        return f"File saved at: {filepath}"

    except Exception as e:
        logger.error(f"💥 Download Failed: {e}")
        return f"Download failed: {str(e)}"

