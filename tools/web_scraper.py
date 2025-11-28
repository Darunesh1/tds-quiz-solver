"""Web scraping tool with semantic content categorization for agent efficiency."""

import logging
import re
import time
from urllib.parse import urljoin

import html2text
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from playwright.sync_api import sync_playwright

logger = logging.getLogger(__name__)


@tool
def get_rendered_html(url: str) -> dict:
    """
    Scrapes a webpage and returns structured, categorized content for efficient analysis.

    This tool is designed for agents solving CTF-style challenges where information
    may be hidden in various formats (text, images, audio, data files).

    **When to Use:**
    - First step for any new webpage/quiz question
    - To discover what types of content are available on the page
    - When you need to understand the page structure

    **What It Returns:**
    A dictionary with categorized content:
    - `text_content`: Plain text from the page (no HTML markup)
    - `markdown`: Full markdown version with links preserved
    - `assets`: Categorized URLs by type:
        - `images`: .png, .jpg, .jpeg, .gif, .svg files
        - `audio`: .mp3, .wav, .ogg files
        - `data`: .csv, .json, .xml, .txt files
        - `other`: Everything else (PDFs, videos, etc.)
    - `forms`: List of form actions (for submission endpoints)
    - `metadata`: Page title and base URL

    **Common Patterns in CTF Challenges:**
    - Empty/minimal text + images → Question is IN the image (use OCR)
    - Audio files present → Clue is spoken (use transcribe)
    - CSV/JSON files → Data analysis required (download + run_code)
    - Multiple similar URLs → May need to process all of them

    **Example Workflow:**
    ```
    1. result = get_rendered_html("https://quiz.com/level1")
    2. Check result['text_content'] for instructions
    3. If text seems incomplete, check result['assets']['images']
    4. Use download_file() + ocr_image() on suspicious images
    ```

    Args:
        url: Full URL of the webpage to scrape

    Returns:
        dict: Structured content with text, assets, forms, and metadata
        On error: dict with 'error' field describing the failure

    Rate Limits:
    - Uses browser automation (slower but more reliable)
    - Includes 1.5s of scrolling to trigger lazy-loaded content
    - Timeout: 60 seconds for page load
    """
    logger.info(f"🌍 SCRAPING: {url}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            page = context.new_page()

            # Navigate and wait for network to stabilize
            page.goto(url, wait_until="networkidle", timeout=60000)

            # Scroll to trigger lazy loading (modern sites often load content on scroll)
            for _ in range(3):
                page.mouse.wheel(0, 3000)
                time.sleep(0.5)

            content = page.content()
            browser.close()

        soup = BeautifulSoup(content, "html.parser")

        # === LINK EXTRACTION WITH CATEGORIZATION ===
        all_urls = set()

        # Extract all URLs from href, src, and action attributes
        for tag in soup.find_all(True):
            for attr in ["href", "src", "action"]:
                if tag.get(attr):
                    absolute_url = urljoin(url, tag.get(attr))
                    if absolute_url.startswith("http"):
                        all_urls.add(absolute_url)

        # Categorize by file extension and purpose
        assets = {"images": [], "audio": [], "data": [], "other": []}

        for link in all_urls:
            link_lower = link.lower()
            if re.search(r"\.(png|jpe?g|gif|svg|webp|bmp)(\?|$)", link_lower):
                assets["images"].append(link)
            elif re.search(r"\.(mp3|wav|ogg|flac|m4a)(\?|$)", link_lower):
                assets["audio"].append(link)
            elif re.search(r"\.(csv|json|xml|txt|tsv)(\?|$)", link_lower):
                assets["data"].append(link)
            else:
                assets["other"].append(link)

        # === FORM EXTRACTION (for submission endpoints) ===
        forms = []
        for form in soup.find_all("form"):
            forms.append(
                {
                    "action": urljoin(url, form.get("action", "")),
                    "method": form.get("method", "GET").upper(),
                }
            )

        # === TEXT EXTRACTION ===
        # Plain text version (easier for analysis)
        text_content = soup.get_text(separator="\n", strip=True)

        # Markdown version (preserves structure)
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.body_width = 0
        markdown = h.handle(content)

        # === METADATA ===
        metadata = {
            "title": soup.title.string if soup.title else "No Title",
            "url": url,
            "has_forms": len(forms) > 0,
        }

        # Logging for transparency
        total_assets = sum(len(v) for v in assets.values())
        logger.info(
            f"✅ Scraped: {len(text_content)} chars text, "
            f"{total_assets} assets ({len(assets['images'])} images, "
            f"{len(assets['audio'])} audio, {len(assets['data'])} data files)"
        )

        return {
            "text_content": text_content,
            "markdown": markdown,
            "assets": assets,
            "forms": forms,
            "metadata": metadata,
        }

    except Exception as e:
        logger.error(f"💥 Scrape Error: {e}")
        return {
            "error": f"Failed to scrape {url}: {str(e)}",
            "url": url,
            "suggestion": "Check if URL is accessible and valid",
        }

