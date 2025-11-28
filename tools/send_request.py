"""HTTP POST request tool with intelligent retry hints."""

import json
import logging
import time

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def post_request(url: str, payload: dict) -> dict:
    """
    Sends POST request with JSON payload to submit answers or interact with APIs.

    **When to Use:**
    - Submitting quiz answers to the server
    - Interacting with API endpoints discovered on the page
    - Sending form data to progress to next level

    **Common Usage Pattern for CTF:**
    ```python
    # After solving the challenge
    result = post_request(
        url="https://quiz.com/submit",
        payload={
            "answer": "calculated_answer_here",
            "email": "your@email.com",
            "secret": "your_secret_key"
        }
    )

    # Check result
    if result['correct']:
        next_url = result['url']
        # Proceed to next challenge
    else:
        # Re-analyze the problem
    ```

    **Response Interpretation:**
    The server typically responds with:
    - `correct`: boolean (True if answer is right)
    - `url`: string (next challenge URL, or None if quiz complete)
    - `message`: string (optional feedback)
    - `status_code`: int (HTTP status, added by this tool)

    **Status Codes:**
    - 200: Success (check 'correct' field for answer validity)
    - 400: Bad Request (check payload format)
    - 429: Rate Limited (wait before retry)
    - 500: Server Error (safe to retry)

    **Auto-Retry Guidance:**
    This tool provides 'retry_suggestion' in responses to help decide next steps:
    - "Answer incorrect. Review constraints" → Re-analyze problem
    - "Rate limited. Wait 5 seconds" → Pause before next attempt
    - "Server error. Safe to retry" → Immediate retry OK

    Args:
        url: Target endpoint URL (usually from form action or documentation)
        payload: Dictionary to send as JSON body

    Returns:
        dict: Server response (parsed JSON) with added fields:
            - Original server fields (correct, url, message, etc.)
            - status_code: HTTP status code
            - retry_suggestion: Guidance on whether/how to retry
        On error: dict with 'error' and 'suggestion' fields

    **Rate Limiting:**
    - Automatically enforces 2-second delay between requests
    - Additional 2-second pause on 4xx/5xx errors
    - Prevents API quota exhaustion

    **Security:**
    - Automatically sets Content-Type: application/json
    - Timeout: 10 seconds
    - Does not follow redirects by default

    **Tips:**
    - Always include required fields (email, secret) from environment
    - Double-check answer format (string, int, list, etc.)
    - If "correct: false", review problem constraints before retry
    - Some challenges have attempt limits - use SKIP if stuck
    """
    logger.info(f"🚀 POST REQUEST: {url}")
    logger.info(f"   Payload preview: {json.dumps(payload, indent=2)[:200]}...")

    try:
        # Enforce rate limiting (2 second minimum between requests)
        time.sleep(2)

        headers = {"Content-Type": "application/json"}

        response = requests.post(
            url, json=payload, headers=headers, timeout=10, allow_redirects=False
        )

        # Parse response
        try:
            data = response.json()
        except json.JSONDecodeError:
            # If response isn't JSON, return text content
            data = {"text": response.text}

        # Add metadata
        data["status_code"] = response.status_code

        # Generate intelligent retry suggestions
        retry_suggestion = None

        if response.status_code == 429:
            retry_suggestion = "Rate limited. Wait 5-10 seconds before retrying."
            logger.warning("⚠️ Rate Limited (429)")
            time.sleep(5)

        elif response.status_code >= 500:
            retry_suggestion = "Server error. Safe to retry immediately."
            logger.warning(f"⚠️ Server Error ({response.status_code})")

        elif response.status_code == 400:
            retry_suggestion = "Bad Request. Check payload format and required fields."
            logger.warning("⚠️ Bad Request (400) - Check payload structure")

        elif response.status_code == 200:
            # Check if answer was correct
            if "correct" in data:
                if data["correct"]:
                    logger.info("✅ Answer CORRECT!")
                    retry_suggestion = "Success! Process next URL if provided."
                else:
                    logger.warning("❌ Answer INCORRECT")
                    retry_suggestion = (
                        "Answer incorrect. Review problem constraints and calculations."
                    )
            else:
                logger.info("✅ Request Successful")

        else:
            retry_suggestion = (
                f"Unexpected status {response.status_code}. Check server documentation."
            )

        data["retry_suggestion"] = retry_suggestion

        # Log response summary
        response_preview = json.dumps(data, indent=2)[:300]
        logger.info(f"   Response: {response_preview}...")

        return data

    except requests.exceptions.Timeout:
        error_msg = "Request timeout (10s limit)"
        logger.error(f"💥 {error_msg}")
        return {
            "error": error_msg,
            "suggestion": "Server may be slow or unresponsive. Retry or check URL.",
            "url": url,
        }

    except requests.exceptions.ConnectionError:
        error_msg = "Connection failed"
        logger.error(f"💥 {error_msg}")
        return {
            "error": error_msg,
            "suggestion": "Check network connectivity and verify URL is correct.",
            "url": url,
        }

    except Exception as e:
        logger.error(f"💥 POST Failed: {e}")
        return {
            "error": str(e),
            "suggestion": "Verify URL and payload format",
            "url": url,
        }

