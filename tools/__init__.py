"""
Tools package for CTF Quiz Solver Agent.

All tools follow agentic design principles:
- Rich docstrings with strategic guidance
- Structured output formats
- Observable error handling
- Context-aware suggestions
"""

from .add_dependencies import add_dependencies
from .download_file import download_file
from .ocr_image import ocr_image
from .run_code import run_code
from .send_request import post_request
from .transcribe_audio import transcribe_audio
from .web_scraper import get_rendered_html

__all__ = [
    "run_code",
    "download_file",
    "get_rendered_html",
    "post_request",
    "add_dependencies",
    "transcribe_audio",
    "ocr_image",
]

