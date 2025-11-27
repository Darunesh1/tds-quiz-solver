from __future__ import annotations

from typing import Any, Dict, Optional

from app.logger import setup_logger
from app.utils.exceptions import QuizSolverError

logger = setup_logger(__name__)


async def scrape(
    url: str,
    selector: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Scrape a web page using tools/web_scraper.py.

    Args:
        url: Target URL.
        selector: Optional CSS selector to focus on part of the page.

    Returns:
        {
          "html": full_or_partial_html,
          "text": visible_text,
          "links": [absolute_links],
        }

    Raises:
        QuizSolverError if scraping fails.
    """
    try:
        from tools import web_scraper as web_scraper_mod  # type: ignore
    except Exception as e:
        raise QuizSolverError(f"web_scraper tool not available: {e}")

    logger.info(f"🕸️ Scraping URL: {url} (selector={selector})")

    try:
        # Adjust this call to match the real API of tools/web_scraper.py.
        # Example: async def scrape(url: str, selector: str | None) -> dict
        result: Dict[str, Any] = await web_scraper_mod.scrape(
            url=url, selector=selector
        )  # type: ignore[attr-defined]
        return {
            "html": result.get("html", ""),
            "text": result.get("text", ""),
            "links": result.get("links", []),
        }
    except AttributeError:
        # Fallback for synchronous implementation
        try:
            result: Dict[str, Any] = web_scraper_mod.scrape(url=url, selector=selector)  # type: ignore[attr-defined]
            return {
                "html": result.get("html", ""),
                "text": result.get("text", ""),
                "links": result.get("links", []),
            }
        except Exception as e:
            logger.error(f"❌ web_scraper failed: {e}")
            raise QuizSolverError(f"web_scraper failed: {e}")
    except Exception as e:
        logger.error(f"❌ web_scraper failed: {e}")
        raise QuizSolverError(f"web_scraper failed: {e}")
