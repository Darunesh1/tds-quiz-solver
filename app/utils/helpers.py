"""
Common utility functions.
"""

import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def safe_filename(url: str) -> str:
    """
    Generate a safe filename from URL.

    Args:
        url: URL to convert

    Returns:
        Safe filename string
    """
    # Use hash to create unique but safe filename
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    return f"file_{url_hash}"


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, create if not.

    Args:
        path: Directory path

    Returns:
        The path object
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_json(data: Dict[str, Any], indent: int = 2) -> str:
    """
    Format dictionary as pretty JSON string.

    Args:
        data: Dictionary to format
        indent: Indentation spaces

    Returns:
        Formatted JSON string
    """
    return json.dumps(data, indent=indent, ensure_ascii=False)


def encode_image_base64(image_path: Path) -> str:
    """
    Encode image file to base64 string.

    Args:
        image_path: Path to image file

    Returns:
        Base64 encoded string with data URI prefix
    """
    with open(image_path, "rb") as f:
        image_data = f.read()

    b64_data = base64.b64encode(image_data).decode("utf-8")

    # Determine MIME type
    suffix = image_path.suffix.lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_types.get(suffix, "image/png")

    return f"data:{mime_type};base64,{b64_data}"


def get_file_size_mb(path: Path) -> float:
    """
    Get file size in megabytes.

    Args:
        path: File path

    Returns:
        Size in MB
    """
    return path.stat().st_size / (1024 * 1024)
