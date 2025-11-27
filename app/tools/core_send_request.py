from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from app.logger import setup_logger
from app.utils.exceptions import QuizSolverError

logger = setup_logger(__name__)


async def send_request(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout: int = 20,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Send an HTTP request with basic retry logic.

    Args:
        method: HTTP method (GET, POST, etc.).
        url: Target URL.
        headers: Optional headers.
        json_body: Optional JSON body.
        timeout: Per-request timeout in seconds.
        max_retries: Max retry attempts for transient errors.

    Returns:
        {
          "status_code": int,
          "headers": dict,
          "json": parsed_json_or_None,
          "text": response_text,
        }

    Raises:
        QuizSolverError if all retries fail.
    """
    method = method.upper()
    headers = headers or {}

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.request(
                    method, url, headers=headers, json=json_body
                )
                text = resp.text
                data: Any = None
                try:
                    data = resp.json()
                except Exception:
                    data = None

                if resp.is_error:
                    # 4xx/5xx: decide retry based on status
                    if resp.status_code in (429,) or 500 <= resp.status_code < 600:
                        raise httpx.HTTPStatusError(
                            f"HTTP error {resp.status_code}",
                            request=resp.request,
                            response=resp,
                        )

                return {
                    "status_code": resp.status_code,
                    "headers": dict(resp.headers),
                    "json": data,
                    "text": text,
                }

        except (
            httpx.TimeoutException,
            httpx.TransportError,
            httpx.HTTPStatusError,
        ) as e:
            last_error = e
            logger.warning(
                f"⚠️ send_request error on {url} (attempt {attempt}/{max_retries}): {e}"
            )
            if attempt == max_retries:
                break
        except Exception as e:
            last_error = e
            logger.error(f"❌ send_request unexpected error: {e}")
            break

    raise QuizSolverError(f"send_request failed for {url}: {last_error}")
