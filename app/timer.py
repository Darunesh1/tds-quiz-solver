"""
Per-question timer with automatic force-submit at 170 seconds.
"""

import time
from typing import Optional

from app.logger import setup_logger

logger = setup_logger(__name__)


class QuestionTimer:
    """
    Timer for individual quiz questions.

    Enforces the 3-minute (180s) deadline with 10s buffer.
    Force-submits at 170s to ensure submission completes before timeout.
    """

    def __init__(self, timeout: int = 170):
        """
        Initialize timer.

        Args:
            timeout: Seconds before force-submit (default 170s = 2m50s)
        """
        self.timeout = timeout
        self.start_time: Optional[float] = None
        self._question_count = 0

    def start(self) -> None:
        """Start or restart timer for new question."""
        self.start_time = time.time()
        self._question_count += 1
        logger.info(
            f"⏱️  Timer started for question #{self._question_count} (timeout: {self.timeout}s)"
        )

    def elapsed(self) -> float:
        """
        Get elapsed time in seconds.

        Returns:
            Seconds since timer started (0 if not started)
        """
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def remaining(self) -> float:
        """
        Get remaining time before force-submit.

        Returns:
            Seconds remaining (0 if expired)
        """
        return max(0.0, self.timeout - self.elapsed())

    def should_force_submit(self) -> bool:
        """
        Check if force-submit threshold reached.

        Returns:
            True if elapsed >= timeout
        """
        is_expired = self.elapsed() >= self.timeout
        if is_expired:
            logger.warning(
                f"⚠️  Force-submit triggered at {self.elapsed():.1f}s "
                f"(threshold: {self.timeout}s)"
            )
        return is_expired

    def reset(self) -> None:
        """Reset timer for next question (alias for start)."""
        self.start()

    def get_status(self) -> dict:
        """
        Get current timer status.

        Returns:
            Dictionary with elapsed, remaining, and should_submit status
        """
        return {
            "elapsed": round(self.elapsed(), 2),
            "remaining": round(self.remaining(), 2),
            "timeout": self.timeout,
            "should_force_submit": self.should_force_submit(),
            "question_number": self._question_count,
        }
