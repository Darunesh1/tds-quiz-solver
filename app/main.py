import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.agent.solver import run_solver_job  # NEW IMPORT
from app.config import settings
from app.llm.client import llm_client
from app.logger import setup_logger
from app.models import HealthResponse, SolveRequest, SolveResponse
from app.primitives.browser import browser_manager
from app.utils.exceptions import QuizSolverError

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: startup and shutdown."""
    logger.info("🚀 Starting Quiz Solver Service")
    logger.info(f"   Config: LLM={settings.llm_provider}, Headless={settings.headless}")
    # Startup (browser manager init is lazy)
    yield
    # Shutdown
    logger.info("🛑 Shutting down service")
    await browser_manager.close()


app = FastAPI(title="TDS Quiz Solver", version="0.2.0", lifespan=lifespan)


@app.post("/solve", response_model=SolveResponse)
async def solve_quiz(request: SolveRequest, background_tasks: BackgroundTasks):
    """
    Start a quiz solving job.

    Args:
        request: SolveRequest containing email, secret, url
        background_tasks: FastAPI background tasks handler

    Returns:
        SolveResponse with job_id
    """
    # 1. Validate Secret
    if request.secret != settings.quiz_secret:
        logger.warning(f"⚠️ Invalid secret provided: {request.secret[:3]}...")
        raise HTTPException(status_code=403, detail="Invalid secret")

    # 2. Generate Job ID
    import uuid

    job_id = str(uuid.uuid4())
    logger.info(f"📥 Received job {job_id} for {request.email}")

    # 3. Start Background Job
    background_tasks.add_task(run_solver_job, job_id, request)

    return SolveResponse(
        accepted=True, job_id=job_id, message="Job started successfully"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        llm_provider=settings.llm_provider,
        gemini_available=llm_client.gemini_available,
        aipipe_available=llm_client.aipipe_available,
    )


@app.exception_handler(QuizSolverError)
async def quiz_exception_handler(request: Request, exc: QuizSolverError):
    """Handle custom application exceptions."""
    logger.error(f"🔥 Application Error: {exc}")
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"🔥 Unexpected Error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
