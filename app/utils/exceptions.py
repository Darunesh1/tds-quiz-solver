"""
Custom exceptions for the quiz solver application.
"""


class QuizSolverError(Exception):
    """Base exception for all quiz solver errors."""

    pass


class TimeoutError(QuizSolverError):
    """Raised when operation exceeds time limit."""

    pass


class BrowserError(QuizSolverError):
    """Raised when browser automation fails."""

    pass


class DownloadError(QuizSolverError):
    """Raised when file download fails."""

    pass


class ParseError(QuizSolverError):
    """Raised when file parsing fails."""

    pass


class AnalysisError(QuizSolverError):
    """Raised when data analysis fails."""

    pass


class SubmissionError(QuizSolverError):
    """Raised when answer submission fails."""

    pass


class LLMError(QuizSolverError):
    """Raised when LLM call fails."""

    pass


class InvalidSecretError(QuizSolverError):
    """Raised when secret validation fails."""

    pass
