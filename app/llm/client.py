import logging
import os
from typing import Literal

import httpx

from app.config import settings

# Import Google's SDK
try:
    import google.generativeai as genai
except ImportError:
    genai = None

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM client supporting multiple providers with fallback.

    Providers:
    - GEMINI: Direct Google Gemini API (free tier for testing)
    - AIPIPE: AIpipe.org proxy service (production)
    """

    def __init__(self):
        self.primary_provider = settings.llm_provider
        self.fallback_enabled = settings.llm_fallback_enabled

        # Initialize providers
        self._init_gemini()
        self._init_aipipe()

        logger.info(
            f"LLM initialized: primary={self.primary_provider}, fallback={self.fallback_enabled}"
        )

    def _init_gemini(self):
        """Initialize Google Gemini Direct API"""
        if not settings.gemini_api_key:
            logger.warning("GEMINI_API_KEY not set - Gemini provider unavailable")
            self.gemini_available = False
            return

        if genai is None:
            logger.error("google-generativeai package not installed")
            self.gemini_available = False
            return

        genai.configure(api_key=settings.gemini_api_key)
        self.gemini_model = genai.GenerativeModel(settings.gemini_model)
        self.gemini_available = True
        logger.info(f"✅ Gemini initialized: {settings.gemini_model}")

    def _init_aipipe(self):
        """Initialize AIpipe proxy"""
        if not settings.aipipe_api_key:
            logger.warning("AIPIPE_API_KEY not set - AIpipe provider unavailable")
            self.aipipe_available = False
            return

        self.aipipe_base_url = "https://aipipe.org/geminiv1beta/models"
        self.aipipe_model = settings.aipipe_model
        self.aipipe_available = True
        logger.info(f"✅ AIpipe initialized: {settings.aipipe_model}")

    async def generate(
        self, prompt: str, system: str = "", max_retries: int = 2
    ) -> str:
        """
        Generate text with automatic fallback.

        Args:
            prompt: User prompt
            system: System instruction (optional)
            max_retries: Retry attempts per provider

        Returns:
            Generated text

        Raises:
            Exception: If all providers fail
        """
        providers = self._get_provider_order()

        for provider in providers:
            try:
                if provider == "GEMINI" and self.gemini_available:
                    return await self._generate_gemini(prompt, system, max_retries)
                elif provider == "AIPIPE" and self.aipipe_available:
                    return await self._generate_aipipe(prompt, system, max_retries)
                else:
                    logger.warning(f"Provider {provider} not available, skipping")
            except Exception as e:
                logger.error(f"Provider {provider} failed: {e}")
                if not self.fallback_enabled or provider == providers[-1]:
                    raise
                logger.info(f"Falling back to next provider...")

        raise Exception("All LLM providers failed")

    def _get_provider_order(self) -> list[Literal["GEMINI", "AIPIPE"]]:
        """Get provider priority order"""
        if self.primary_provider == "GEMINI":
            return ["GEMINI", "AIPIPE"] if self.fallback_enabled else ["GEMINI"]
        else:
            return ["AIPIPE", "GEMINI"] if self.fallback_enabled else ["AIPIPE"]

    async def _generate_gemini(self, prompt: str, system: str, max_retries: int) -> str:
        """Generate using Google Gemini Direct API"""
        logger.debug(f"Calling Gemini API (retries={max_retries})")

        # Combine system instruction with prompt if provided
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        for attempt in range(max_retries):
            try:
                response = self.gemini_model.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": 0.1,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 2048,
                    },
                )

                # Extract text from response
                if not response.parts:
                    raise ValueError("Empty response from Gemini")

                result = response.text
                logger.info(f"✅ Gemini success (attempt {attempt + 1}/{max_retries})")
                return result

            except Exception as e:
                logger.warning(
                    f"Gemini attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                if attempt == max_retries - 1:
                    raise

        raise Exception("Gemini max retries exceeded")

    async def _generate_aipipe(self, prompt: str, system: str, max_retries: int) -> str:
        """Generate using AIpipe proxy"""
        logger.debug(f"Calling AIpipe API (retries={max_retries})")

        url = f"{self.aipipe_base_url}/{self.aipipe_model}:generateContent"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 2048,
            },
        }

        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        headers = {
            "Authorization": f"Bearer {settings.aipipe_api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            for attempt in range(max_retries):
                try:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()

                    result = data["candidates"][0]["content"]["parts"][0]["text"]
                    logger.info(
                        f"✅ AIpipe success (attempt {attempt + 1}/{max_retries})"
                    )
                    return result

                except httpx.HTTPStatusError as e:
                    logger.warning(
                        f"AIpipe HTTP {e.response.status_code} (attempt {attempt + 1}/{max_retries})"
                    )
                    if attempt == max_retries - 1:
                        raise
                except Exception as e:
                    logger.warning(
                        f"AIpipe attempt {attempt + 1}/{max_retries} failed: {e}"
                    )
                    if attempt == max_retries - 1:
                        raise

        raise Exception("AIpipe max retries exceeded")


# Global LLM client instance
llm_client = LLMClient()
