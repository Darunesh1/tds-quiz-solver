from __future__ import annotations

from app.llm import client as llm_client_module
from app.logger import setup_logger

logger = setup_logger(__name__)


async def summarize_text(text: str, max_words: int = 300) -> str:
    """
    Summarize a long piece of text using the LLM.

    Args:
        text: Original text.
        max_words: Approximate max length of summary.

    Returns:
        Shorter summary string.
    """
    llm_client = llm_client_module.llm_client

    prompt = f"""Summarize the following text in at most {max_words} words.
Focus only on information that might be relevant for solving a data-science quiz question.

TEXT:
{text}
"""

    logger.info(f"📝 Summarizing text (len={len(text)}) to ~{max_words} words")

    summary = await llm_client.generate(
        prompt=prompt, system="You summarize text concisely."
    )
    return summary.strip()
