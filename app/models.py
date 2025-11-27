from __future__ import annotations

from pydantic import BaseModel


class SolveRequest(BaseModel):
    """Request body for POST /solve."""

    email: str
    secret: str
    url: str


class SolveResponse(BaseModel):
    """Response body for POST /solve."""

    accepted: bool
    job_id: str
    message: str


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
    llm_provider: str
    gemini_available: bool
    aipipe_available: bool
