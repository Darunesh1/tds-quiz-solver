"""
Pydantic models for API requests and responses.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, EmailStr, Field, HttpUrl


class SolveRequest(BaseModel):
    """
    Request model for /solve endpoint.

    Attributes:
        email: Student email address
        secret: Authentication secret
        url: Quiz page URL to solve
    """

    email: EmailStr = Field(..., description="Student email address")
    secret: str = Field(..., min_length=1, description="Authentication secret")
    url: HttpUrl = Field(..., description="Quiz page URL")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "email": "student@example.com",
                    "secret": "my-secret-key",
                    "url": "https://example.com/quiz-123",
                }
            ]
        }
    }


class SolveResponse(BaseModel):
    """
    Response model for /solve endpoint.

    Attributes:
        accepted: Whether request was accepted
        job_id: Unique job identifier
        message: Additional information
    """

    accepted: bool = Field(..., description="Request accepted")
    job_id: str = Field(..., description="Unique job identifier")
    message: str = Field(default="Job started", description="Status message")


class QuizSubmission(BaseModel):
    """
    Quiz answer submission payload.

    Can contain any combination of answer types based on question requirements.
    """

    answer: Optional[Any] = Field(
        None, description="Answer value (string, number, boolean, etc.)"
    )

    # Allow any additional fields for flexible answer structures
    model_config = {
        "extra": "allow",
        "json_schema_extra": {
            "examples": [
                {"answer": 42},
                {"answer": "Paris", "confidence": 0.95},
                {"result": 123.45, "chart": "data:image/png;base64,..."},
                {
                    "analysis": {"mean": 50, "median": 48},
                    "visualization": "data:image/png;base64,...",
                },
            ]
        },
    }


class QuizResponse(BaseModel):
    """
    Response from quiz submission endpoint.

    Attributes:
        correct: Whether answer was correct
        url: Next quiz URL (if exists)
        reason: Explanation if wrong (optional)
    """

    correct: bool = Field(..., description="Answer correctness")
    url: Optional[HttpUrl] = Field(None, description="Next quiz URL")
    reason: Optional[str] = Field(None, description="Explanation if incorrect")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy", description="Service status")
    llm_provider: str = Field(..., description="Active LLM provider")
    gemini_available: bool = Field(..., description="Gemini availability")
    aipipe_available: bool = Field(..., description="AIpipe availability")
