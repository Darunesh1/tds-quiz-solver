import logging
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
    Scrapes the webpage and returns the raw Markdown content and a list of ALL links.
    It does NOT categorize assets. The AI must interpret the links based on context.

    Returns:
    - markdown: The textual content of the page.
    - links: A list of every URL found in 'href', 'src', or 'action' attributes.
    """
    logger.info(f"🌍 SCRAPING: {url}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            # Use a standard user agent to look like a real browser
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
            page = context.new_page()

            # Navigate and wait for network to settle
            page.goto(url, wait_until="networkidle", timeout=60000)

            # Scroll to trigger lazy loading (crucial for modern sites)
            for _ in range(3):
                page.mouse.wheel(0, 3000)
                time.sleep(0.5)

            content = page.content()
            browser.close()

        soup = BeautifulSoup(content, "html.parser")

        # --- 1. GENERAL LINK EXTRACTION (Capture Everything) ---
        # We look for ANY tag with href, src, or action (forms).
        # We do NOT filter by extension.
        all_urls = set()

        # Tags to check: a, link, img, script, audio, video, source, iframe, form, etc.
        # Instead of listing tags, we look for attributes.
        for tag in soup.find_all(True):  # find_all(True) gets every tag
            href = tag.get("href")
            src = tag.get("src")
            action = tag.get("action")

            # If a link exists, resolve it to absolute URL and add to list
            if href:
                all_urls.add(urljoin(url, href))
            if src:
                all_urls.add(urljoin(url, src))
            if action:
                all_urls.add(urljoin(url, action))

        # Filter out empty strings or javascript: calls
        clean_links = [u for u in list(all_urls) if u.startswith("http")]

        # --- 2. TEXT EXTRACTION ---
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.body_width = 0
        markdown = h.handle(content)

        logger.info(
            f"✅ Scraped {len(markdown)} chars. Found {len(clean_links)} raw links."
        )

        return {
            "markdown": markdown,
            "links": clean_links,  # <--- The AI receives this raw list
            "url": url,
        }

    except Exception as e:
        logger.error(f"💥 Scrape Error: {e}")
        return {"error": f"Scrape failed: {str(e)}", "url": url}

