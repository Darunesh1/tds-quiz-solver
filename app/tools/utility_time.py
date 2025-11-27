from __future__ import annotations

from app.timer import QuestionTimer


def get_time_remaining(timer: QuestionTimer) -> float:
    """
    Return seconds remaining before force-submit threshold.

    Args:
        timer: QuestionTimer instance.

    Returns:
        Seconds remaining (float, never negative).
    """
    return timer.time_remaining()
