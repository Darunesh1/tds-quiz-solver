"""
Question timer with force-submit logic.
"""

import time

from app.logger import setup_logger

logger = setup_logger(__name__)


class QuestionTimer:
    """
    Tracks time per question and enforces a hard deadline.
    """

    def __init__(self, timeout: int = 175) -> None:
        """
        Args:
            timeout: Force-submit threshold in seconds (default 175 = 2:55).
        """
        self.timeout = timeout
        self.start_time: float | None = None

    def start(self) -> None:
        """Start the timer for this question."""
        self.start_time = time.time()
        logger.info(f"⏱️  Timer started (timeout: {self.timeout}s)")

    def elapsed(self) -> float:
        """Return elapsed seconds since start."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def time_remaining(self) -> float:
        """Return seconds remaining before force-submit threshold."""
        if self.start_time is None:
            return float(self.timeout)
        remaining = self.timeout - self.elapsed()
        return max(0.0, remaining)

    def should_force_submit(self) -> bool:
        """Check if we should force-submit now."""
        return self.time_remaining() <= 0.0
