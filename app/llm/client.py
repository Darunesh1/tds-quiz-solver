from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings
from app.logger import setup_logger
from app.timer import QuestionTimer
from app.utils.exceptions import QuizSolverError

logger = setup_logger(__name__)


@dataclass
class LLMProviderStatus:
    available: bool = False
    last_error: Optional[str] = None


class LLMClient:
    """
    LLM client with retry and optional provider fallback.

    Supports:
    - Primary provider: GEMINI or AIPIPE (configured in settings.llm_provider)
    - Optional fallback to the other provider if llm_fallback_enabled is True
    - Network / rate limit retries with exponential backoff
    - Awareness of per-question timer to avoid exceeding 3-minute limit
    """

    def __init__(self) -> None:
        self.primary = settings.llm_provider.upper()
        self.fallback_enabled = settings.llm_fallback_enabled

        self._gemini_status = LLMProviderStatus()
        self._aipipe_status = LLMProviderStatus()

        # Basic HTTP client; can be reused
        self._client = httpx.AsyncClient(timeout=30)

    # ------------------------------------------------------------------
    # Public properties for /health
    # ------------------------------------------------------------------
    @property
    def gemini_available(self) -> bool:
        return self._gemini_status.available

    @property
    def aipipe_available(self) -> bool:
        return self._aipipe_status.available

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        timer: Optional[QuestionTimer] = None,
        model: Optional[str] = None,
        max_output_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> str:
        """
        Generate a completion from the LLM with retry + fallback.

        Args:
            prompt: User prompt content.
            system: Optional system message.
            timer: Optional QuestionTimer to respect per-question limit.
            model: Optional model name override.
            max_output_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Returns:
            Generated text.

        Raises:
            QuizSolverError if all providers / retries fail.
        """
        messages = self._build_messages(prompt, system)

        providers_order = self._build_provider_order()

        last_error: Optional[Exception] = None

        for provider in providers_order:
            try:
                if provider == "GEMINI":
                    result = await self._call_with_retries(
                        self._call_gemini,
                        messages=messages,
                        model=model or settings.gemini_model,
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                        timer=timer,
                    )
                    self._gemini_status.available = True
                    self._gemini_status.last_error = None
                    return result

                if provider == "AIPIPE":
                    result = await self._call_with_retries(
                        self._call_aipipe,
                        messages=messages,
                        model=model or settings.aipipe_model,
                        max_output_tokens=max_output_tokens,
                        temperature=temperature,
                        timer=timer,
                    )
                    self._aipipe_status.available = True
                    self._aipipe_status.last_error = None
                    return result

            except Exception as e:
                last_error = e
                if provider == "GEMINI":
                    self._gemini_status.available = False
                    self._gemini_status.last_error = str(e)
                    logger.warning(f"⚠️ Gemini call failed: {e}")
                elif provider == "AIPIPE":
                    self._aipipe_status.available = False
                    self._aipipe_status.last_error = str(e)
                    logger.warning(f"⚠️ AIpipe call failed: {e}")

                # If this was the last provider in the order, we'll raise below

        raise QuizSolverError(
            f"LLM generation failed after retries and fallback: {last_error}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_messages(
        self, prompt: str, system: Optional[str]
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_provider_order(self) -> List[str]:
        """
        Determine which providers to try and in what order.
        """
        primary = self.primary
        others = []

        if self.fallback_enabled:
            if primary == "GEMINI":
                others.append("AIPIPE")
            elif primary == "AIPIPE":
                others.append("GEMINI")

        return [primary] + others

    async def _call_with_retries(
        self,
        func,
        messages: List[Dict[str, str]],
        model: str,
        max_output_tokens: int,
        temperature: float,
        timer: Optional[QuestionTimer],
        max_attempts: int = 4,
    ) -> str:
        """
        Generic retry wrapper for LLM calls.

        Retries on network / transient errors with exponential backoff,
        while respecting the per-question timer.
        """
        backoffs = [0.0, 0.5, 1.0, 2.0]  # seconds

        last_error: Optional[Exception] = None

        for attempt in range(max_attempts):
            # Check timer before each attempt
            if timer is not None:
                remaining = timer.time_remaining()
                # If almost no time is left, don't risk another long request
                if remaining < 5.0:
                    raise QuizSolverError(
                        f"LLM call aborted: only {remaining:.1f}s remaining for this question."
                    )

            try:
                if backoffs[attempt] > 0:
                    await asyncio.sleep(backoffs[attempt])

                return await func(
                    messages=messages,
                    model=model,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                )

            except Exception as e:
                last_error = e
                # Decide whether this seems transient or permanent
                if not self._is_retriable_error(e) or attempt == max_attempts - 1:
                    break
                logger.warning(
                    f"⚠️ LLM call error (attempt {attempt + 1}/{max_attempts}): {e}"
                )

        # If we reach here, all attempts failed or error was non-retriable
        raise QuizSolverError(f"LLM call failed: {last_error}")

    def _is_retriable_error(self, e: Exception) -> bool:
        """
        Determine if an error is transient and worth retrying.

        For now, treat HTTP 429/5xx and network/timeouts as retriable.
        """
        if isinstance(e, httpx.HTTPStatusError):
            status = e.response.status_code
            return status == 429 or 500 <= status < 600

        if isinstance(e, (httpx.TimeoutException, httpx.TransportError)):
            return True

        # Generic catch-all: by default, non-HTTP/transport errors are not retried
        return False

    # ------------------------------------------------------------------
    # Provider-specific calls
    # ------------------------------------------------------------------
    async def _call_gemini(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_output_tokens: int,
        temperature: float,
    ) -> str:
        """
        Call Gemini directly using its REST API.

        Assumes:
        - settings.gemini_api_key is set.
        - Model name in `model`.
        """
        if not settings.gemini_api_key:
            raise QuizSolverError("Gemini API key not configured")

        # Simple text-only request to Gemini's REST endpoint.
        # Adjust URL if you use a different endpoint.
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": settings.gemini_api_key,
        }

        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            # Map roles to parts; for simplicity we concatenate into a single user turn
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_output_tokens,
                "temperature": temperature,
            },
        }

        resp = await self._client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()

        try:
            candidates = data.get("candidates") or []
            first = candidates[0]
            parts = first["content"]["parts"]
            # Concatenate all text parts
            text = "".join(p.get("text", "") for p in parts)
            return text.strip()
        except Exception as e:
            raise QuizSolverError(f"Failed to parse Gemini response: {e}")

    async def _call_aipipe(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_output_tokens: int,
        temperature: float,
    ) -> str:
        """
        Call Gemini via AIpipe (or similar proxy) using its REST API.

        Assumes:
        - settings.aipipe_api_key is set.
        - settings.aipipe_model is configured.
        """
        if not settings.aipipe_api_key:
            raise QuizSolverError("AIpipe API key not configured")

        # This URL depends on your AIpipe deployment; using a generic placeholder path.
        # Adjust to the actual AIpipe endpoint.
        url = "https://api.aipipe.xyz/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.aipipe_api_key}",
        }

        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
        }

        resp = await self._client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()

        try:
            choices = data.get("choices") or []
            first = choices[0]
            text = first["message"]["content"]
            return text.strip()
        except Exception as e:
            raise QuizSolverError(f"Failed to parse AIpipe response: {e}")


# Global client instance
llm_client = LLMClient()
