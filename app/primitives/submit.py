"""
Quiz answer submission with retry logic and error handling.
"""

import asyncio
from typing import Any, Dict, Optional

import httpx

from app.logger import setup_logger
from app.utils.exceptions import SubmissionError

logger = setup_logger(__name__)


class SubmissionHandler:
    """
    Handles quiz answer submission with retries and validation.
    """

    def __init__(self, max_retries: int = 3, timeout: int = 30):
        """
        Initialize submission handler.

        Args:
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        self.max_retries = max_retries
        self.timeout = timeout

    async def submit_answer(
        self,
        submit_url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Submit answer to quiz endpoint.

        Args:
            submit_url: Submission URL
            payload: Answer payload (must be <1MB)
            headers: Optional HTTP headers

        Returns:
            Response JSON with 'correct', 'url', 'reason'

        Raises:
            SubmissionError: If submission fails after retries
        """
        import json

        # Validate payload size
        payload_str = json.dumps(payload)
        payload_size_mb = len(payload_str.encode("utf-8")) / (1024 * 1024)

        if payload_size_mb > 1.0:
            raise SubmissionError(
                f"Payload too large: {payload_size_mb:.2f}MB (max 1MB)"
            )

        logger.info(f"📤 Submitting answer to: {submit_url}")
        logger.info(f"   Payload size: {payload_size_mb:.3f}MB")

        default_headers = {
            "Content-Type": "application/json",
            "User-Agent": "TDS-Quiz-Solver/1.0",
        }
        if headers:
            default_headers.update(headers)

        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        submit_url, json=payload, headers=default_headers
                    )
                    response.raise_for_status()

                    result = response.json()

                    is_correct = result.get("correct", False)
                    next_url = result.get("url")
                    reason = result.get("reason")

                    if is_correct:
                        logger.info(f"✅ Answer CORRECT!")
                    else:
                        logger.warning(f"❌ Answer INCORRECT")
                        if reason:
                            logger.warning(f"   Reason: {reason}")

                    if next_url:
                        logger.info(f"➡️  Next URL: {next_url}")
                    else:
                        logger.info("🏁 Quiz complete (no next URL)")

                    return result

            except httpx.HTTPStatusError as e:
                logger.warning(
                    f"⚠️ HTTP {e.response.status_code} on attempt {attempt}/{self.max_retries}"
                )

                if e.response.status_code == 400:
                    # Bad request - don't retry
                    raise SubmissionError(f"Bad request: {e.response.text}")

                if attempt == self.max_retries:
                    raise SubmissionError(
                        f"Submission failed: HTTP {e.response.status_code}"
                    )

                await asyncio.sleep(2**attempt)  # Exponential backoff

            except httpx.TimeoutException:
                logger.warning(f"⏱️ Timeout on attempt {attempt}/{self.max_retries}")

                if attempt == self.max_retries:
                    raise SubmissionError("Submission timeout")

                await asyncio.sleep(2**attempt)

            except Exception as e:
                logger.error(f"❌ Submission error on attempt {attempt}: {e}")

                if attempt == self.max_retries:
                    raise SubmissionError(f"Submission failed: {e}")

                await asyncio.sleep(2**attempt)

        raise SubmissionError("Submission failed after all retries")

    def validate_payload(self, payload: Dict[str, Any]) -> bool:
        """
        Validate payload before submission.

        Args:
            payload: Answer payload

        Returns:
            True if valid
        """
        import json

        try:
            # Check JSON serializable
            json.dumps(payload)

            # Check size
            payload_str = json.dumps(payload)
            size_mb = len(payload_str.encode("utf-8")) / (1024 * 1024)

            if size_mb > 1.0:
                logger.error(f"❌ Payload too large: {size_mb:.2f}MB")
                return False

            logger.info(f"✅ Payload valid ({size_mb:.3f}MB)")
            return True

        except Exception as e:
            logger.error(f"❌ Payload validation failed: {e}")
            return False


# Global submission handler instance
submission_handler = SubmissionHandler()
