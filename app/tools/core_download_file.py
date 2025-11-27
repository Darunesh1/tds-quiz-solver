from __future__ import annotations

from pathlib import Path

from app.logger import setup_logger
from app.primitives.download import downloader
from app.utils.exceptions import QuizSolverError

logger = setup_logger(__name__)


async def download_file(url: str, job_id: str) -> str:
    """
    Download files into the job-specific directory.

    Args:
        url: File URL.
        job_id: Job identifier.

    Returns:
        Local file path as string.

    Raises:
        QuizSolverError if download fails.
    """
    logger.info(f"⬇️ Downloading file for job {job_id}: {url}")
    try:
        path: Path = await downloader.download_file(url, job_id)
        return str(path)
    except Exception as e:
        logger.error(f"❌ download_file failed: {e}")
        raise QuizSolverError(f"download_file failed: {e}")
