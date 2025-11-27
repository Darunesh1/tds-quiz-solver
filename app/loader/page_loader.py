from __future__ import annotations

from typing import Any, Dict, List

from app.config import settings
from app.logger import setup_logger
from app.primitives.browser import browser_manager
from app.utils.exceptions import BrowserError

logger = setup_logger(__name__)


async def load_raw_page(url: str) -> Dict[str, Any]:
    """
    Load a web page (with JS) and return raw content.

    Returns:
        {
            "url": final_url_after_redirects,
            "base_url": same_as_final_url,
            "html": full_html,
            "text": visible_text,
            "links": [absolute_link_1, ...],
        }

    Raises:
        BrowserError on failure.
    """
    logger.info(f"🌐 Loading page: {url}")

    try:
        # Use existing browser_manager from primitives
        result = await browser_manager.load_page(
            url,
            timeout=settings.playwright_timeout,
        )
        page = result["page"]
        html = result["html"]
        text = result["text"]
        final_url = page.url

        # Extract all links
        try:
            links: List[str] = await browser_manager.extract_links(page)
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract links: {e}")
            links = []

        await page.close()

        return {
            "url": final_url,
            "base_url": final_url,
            "html": html,
            "text": text,
            "links": links,
        }

    except Exception as e:
        logger.error(f"❌ Failed to load page {url}: {e}")
        raise BrowserError(f"Failed to load page {url}: {e}")
