"""Custom exceptions for the quiz solver."""


class QuizSolverError(Exception):
    """Base exception for quiz solver errors."""

    pass


class BrowserError(QuizSolverError):
    """Browser/Playwright related errors."""

    pass


class DownloadError(QuizSolverError):
    """File download errors."""

    pass


class SubmissionError(QuizSolverError):
    """Answer submission errors."""

    pass


class TimeoutError(QuizSolverError):
    """Timeout errors."""

    pass


class InvalidSecretError(QuizSolverError):
    """Invalid secret provided."""

    pass


class LLMError(QuizSolverError):
    """LLM call errors."""

    pass
