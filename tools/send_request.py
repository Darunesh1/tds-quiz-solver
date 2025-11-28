import json
import logging
import time

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)
URL_STATS = {}


@tool
def post_request(url: str, payload: dict) -> dict:
    """
    Sends a POST request with JSON payload to a specified endpoint.

    Args:
        url (str): Target endpoint URL
        payload (dict): JSON-serializable dictionary

    Returns:
        dict: Server response (parsed JSON) with added 'status_code' field
              On error: {'error': <message>}

    Rate Limiting:
    - Enforces 2-second delay between requests to same URL
    - Auto-sleeps 2s on 4xx/5xx errors to preserve quota

    Timeout: 10 seconds

    Headers: Automatically sets 'Content-Type: application/json'

    Example Payload:
        {
            "answer": "42",
            "email": "user@example.com",
            "secret": "SECRET_KEY"
        }

    Example:
        post_request("https://quiz.com/submit", {"answer": "Paris", "email": EMAIL, "secret": SECRET})
    """

    try:
        # Track retry stats
        if url not in URL_STATS:
            URL_STATS[url] = {"attempts": 0, "last_time": 0}
        stats = URL_STATS[url]
        stats["attempts"] += 1

        # Rate Limit Buffer
        elapsed = time.time() - stats["last_time"]
        if elapsed < 2:
            time.sleep(2)
        stats["last_time"] = time.time()

        logger.info(f"🚀 POST {url} (Attempt {stats['attempts']})")
        logger.info(f"📦 Payload: {json.dumps(payload)[:200]}...")

        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers, timeout=10)

        try:
            data = response.json()
        except:
            data = {"text": response.text}

        data["status_code"] = response.status_code
        if response.status_code >= 400:
            logger.warning(
                f"⚠️ Request Failed ({response.status_code}). Sleeping for 2.0s to save Quota..."
            )
            time.sleep(2.0)

        if response.status_code == 200:
            logger.info(f"✅ Response: {json.dumps(data)[:200]}...")
        else:
            logger.warning(f"⚠️ Response ({response.status_code}): {json.dumps(data)}")

        return data

    except Exception as e:
        logger.error(f"💥 POST Failed: {e}")
        time.sleep(2.0)
        return {"error": str(e)}

