import json
import logging
from typing import Any, Dict, Optional

import requests
from langchain_core.tools import tool  # ✅ FIXED: langchain_core.tools
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger("tools.sendrequest")

# Configure retry logic
wait_ = wait_exponential(multiplier=1, min=2, max=20)  # 2s, 4s, 8s... up to 20s
stop = stop_after_attempt(5)


@retry(
    wait=wait_,
    stop=stop,
    retry=retry_if_exception_type(
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.HTTPError,
        )
    ),
)
@tool
def postrequest(
    url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None
) -> Any:
    """
    Send an HTTP POST request to the given URL with the provided payload.

    Args:
        url: The endpoint to send the POST request to.
        payload: The JSON-serializable request body.
        headers: Optional HTTP headers.

    Returns:
        Response body (parsed JSON dict or raw text).
    """
    headers = headers or {"Content-Type": "application/json"}

    logger.info(f"📤 Sending Answer to: {url}")

    # Log payload safely (truncate if too long)
    payload_str = json.dumps(payload, indent=2)
    if len(payload_str) > 1000:
        logger.debug(f"Payload truncated: {payload_str[:1000]}...")
    else:
        logger.debug(f"Payload: {payload_str}")

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        # Parse response
        try:
            data = response.json()
        except ValueError:
            data = response.text

        # Extract key fields for logging
        delay = data.get("delay", 0) if isinstance(data, dict) else 0
        correct = data.get("correct") if isinstance(data, dict) else None
        next_url = data.get("url") if isinstance(data, dict) else None

        # Enhanced logging for quiz progress tracking
        logger.info(
            f"✅ RESULT: correct={correct}, delay={delay}s, next_url={next_url or 'NONE'}"
        )

        return data

    except requests.HTTPError as e:
        err_resp = e.response
        try:
            err_data = err_resp.json()
        except ValueError:
            err_data = err_resp.text
        logger.error(f"HTTP {err_resp.status_code} Error: {err_data}")
        return err_data
    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        return str(e)


post_request = postrequest  # Export for __init__.py compatibility
