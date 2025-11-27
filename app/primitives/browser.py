"""
Browser automation using Playwright.

Handles JavaScript-rendered quiz pages.
"""

import asyncio
from typing import Dict, List, Optional

from playwright.async_api import Browser, Page, async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeout

from app.config import settings
from app.logger import setup_logger
from app.utils.exceptions import BrowserError

logger = setup_logger(__name__)


class BrowserManager:
    """
    Manages Playwright browser lifecycle.
    Supports pre-warming for faster page loads.
    """

    def __init__(self) -> None:
        self.playwright = None
        self.browser: Optional[Browser] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Playwright and browser."""
        if self._initialized:
            return
        try:
            logger.info("🌐 Initializing Playwright browser...")
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=settings.headless,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                ],
            )
            self._initialized = True
            logger.info("✅ Browser initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize browser: {e}")
            raise BrowserError(f"Browser initialization failed: {e}")

    async def close(self) -> None:
        """Close browser and cleanup."""
        if self.browser:
            await self.browser.close()
            logger.info("🔒 Browser closed")
        if self.playwright:
            await self.playwright.stop()
            logger.info("🔒 Playwright stopped")
        self._initialized = False

    async def load_page(
        self, url: str, timeout: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Load a page and extract key elements.

        Args:
            url: URL to load
            timeout: Page load timeout in ms (default from config)

        Returns:
            {
              "page": Playwright Page object,
              "html": page HTML content,
              "text": visible text content,
              "url": final URL,
            }

        Raises:
            BrowserError if page loading fails.
        """
        if not self._initialized:
            await self.initialize()

        timeout = timeout or settings.playwright_timeout
        page: Optional[Page] = None

        try:
            logger.info(f"📄 Loading page: {url}")
            page = await self.browser.new_page()  # type: ignore[union-attr]

            # Set viewport for consistent rendering
            await page.set_viewport_size({"width": 1280, "height": 720})

            # Navigate with timeout
            response = await page.goto(url, wait_until="networkidle", timeout=timeout)
            if not response or response.status != 200:
                status = response.status if response else "No response"
                raise BrowserError(f"Page load failed with status: {status}")

            # Wait a bit for dynamic content
            await asyncio.sleep(1)

            # Extract content
            html = await page.content()
            text = await page.inner_text("body")

            logger.info(f"✅ Page loaded successfully ({len(html)} chars)")

            return {"page": page, "html": html, "text": text, "url": page.url}
        except PlaywrightTimeout:
            logger.error(f"⏱️ Timeout loading page: {url}")
            if page:
                await page.close()
            raise BrowserError(f"Page load timeout after {timeout}ms")
        except Exception as e:
            logger.error(f"❌ Error loading page: {e}")
            if page:
                await page.close()
            raise BrowserError(f"Failed to load page: {e}")

    async def extract_links(self, page: Page, selector: str = "a") -> List[str]:
        """
        Extract all links from page.

        Args:
            page: Playwright page object
            selector: CSS selector for links (default: 'a')

        Returns:
            List of URLs
        """
        try:
            links: List[str] = await page.locator(selector).evaluate_all(
                "elements => elements.map(el => el.href).filter(href => href)"
            )
            logger.debug(f"Found {len(links)} links")
            return links
        except Exception as e:
            logger.warning(f"Failed to extract links: {e}")
            return []

    async def find_submit_url(self, page: Page) -> Optional[str]:
        """
        Find the submit URL on quiz page.

        Looks for:
        1. Form action attribute
        2. Submit-like URLs in attributes
        3. 'submit to:' phrases in text

        Returns:
            Submit URL if found, None otherwise.
        """
        try:
            # Strategy 1: form action
            form_action = await page.locator("form").get_attribute("action")
            if form_action:
                logger.info(f"✅ Found submit URL in form action: {form_action}")
                return form_action

            # Strategy 2: patterns in HTML
            html = await page.content()
            import re

            patterns = [
                r'submit["\']?\s*:\s*["\']([^"\']+)',  # submit: "url"
                r'action\s*=\s*["\']([^"\']*submit[^"\']*)',  # action="...submit..."
                r'data-submit-url\s*=\s*["\']([^"\']+)',  # data-submit-url="url"
            ]

            for pattern in patterns:
                match = re.search(pattern, html, re.IGNORECASE)
                if match:
                    url = match.group(1)
                    logger.info(f"✅ Found submit URL via pattern: {url}")
                    return url

            # Strategy 3: in visible text
            text = await page.inner_text("body")
            match = re.search(r"submit\s+to:?\s+(https?://[^\s]+)", text, re.IGNORECASE)
            if match:
                url = match.group(1)
                logger.info(f"✅ Found submit URL in text: {url}")
                return url

            logger.warning("⚠️ Could not find submit URL on page")
            return None
        except Exception as e:
            logger.error(f"❌ Error finding submit URL: {e}")
            return None

    async def extract_instructions(self, page: Page) -> str:
        """
        Extract quiz instructions/problem statement.

        Returns:
            Instruction text.
        """
        try:
            selectors = [
                ".instructions",
                "#instructions",
                ".problem",
                "#problem",
                "main",
                "article",
            ]

            for selector in selectors:
                try:
                    text = await page.locator(selector).first.inner_text()
                    if text and len(text) > 20:
                        logger.debug(f"Found instructions in {selector}")
                        return text.strip()
                except Exception:
                    continue

            text = await page.inner_text("body")
            logger.debug("Using full body text as instructions")
            return text.strip()
        except Exception as e:
            logger.warning(f"Failed to extract instructions: {e}")
            return ""


# Global browser manager instance
browser_manager = BrowserManager()
