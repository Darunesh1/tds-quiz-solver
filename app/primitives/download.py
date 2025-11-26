"""
File download utilities with streaming and size limits.
"""

import asyncio
from pathlib import Path
from typing import List, Optional

import httpx

from app.config import settings
from app.logger import setup_logger
from app.utils.exceptions import DownloadError
from app.utils.helpers import ensure_dir, safe_filename

logger = setup_logger(__name__)


class FileDownloader:
    """
    Downloads files with streaming, size limits, and retries.
    """

    def __init__(
        self, max_size_mb: float = 50.0, timeout: int = 30, max_retries: int = 3
    ):
        """
        Initialize downloader.

        Args:
            max_size_mb: Maximum file size in MB
            timeout: Download timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.timeout = timeout
        self.max_retries = max_retries
        self.download_dir = Path("/tmp/quiz-jobs")
        ensure_dir(self.download_dir)

    async def download_file(
        self, url: str, job_id: str, custom_filename: Optional[str] = None
    ) -> Path:
        """
        Download a single file.

        Args:
            url: File URL
            job_id: Job identifier for organizing downloads
            custom_filename: Optional custom filename

        Returns:
            Path to downloaded file

        Raises:
            DownloadError: If download fails
        """
        # Create job-specific directory
        job_dir = self.download_dir / job_id
        ensure_dir(job_dir)

        # Determine filename
        if custom_filename:
            filename = custom_filename
        else:
            # Extract from URL or generate safe name
            url_parts = url.rstrip("/").split("/")
            filename = url_parts[-1] if url_parts[-1] else safe_filename(url)

        file_path = job_dir / filename

        logger.info(f"⬇️  Downloading: {url}")

        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream("GET", url) as response:
                        response.raise_for_status()

                        # Check content length if available
                        content_length = response.headers.get("content-length")
                        if content_length and int(content_length) > self.max_size_bytes:
                            raise DownloadError(
                                f"File too large: {int(content_length) / 1024 / 1024:.1f}MB "
                                f"(max: {self.max_size_bytes / 1024 / 1024:.1f}MB)"
                            )

                        # Stream to file with size check
                        downloaded = 0
                        with open(file_path, "wb") as f:
                            async for chunk in response.aiter_bytes(chunk_size=8192):
                                downloaded += len(chunk)

                                # Check size limit
                                if downloaded > self.max_size_bytes:
                                    file_path.unlink(missing_ok=True)
                                    raise DownloadError(
                                        f"File exceeded size limit during download "
                                        f"({downloaded / 1024 / 1024:.1f}MB)"
                                    )

                                f.write(chunk)

                        logger.info(
                            f"✅ Downloaded: {filename} ({downloaded / 1024:.1f}KB)"
                        )
                        return file_path

            except httpx.HTTPStatusError as e:
                logger.warning(
                    f"⚠️ HTTP {e.response.status_code} on attempt {attempt}/{self.max_retries}"
                )
                if attempt == self.max_retries:
                    raise DownloadError(
                        f"Download failed: HTTP {e.response.status_code}"
                    )
                await asyncio.sleep(2**attempt)  # Exponential backoff

            except httpx.TimeoutException:
                logger.warning(f"⏱️ Timeout on attempt {attempt}/{self.max_retries}")
                if attempt == self.max_retries:
                    raise DownloadError(f"Download timeout after {self.timeout}s")
                await asyncio.sleep(2**attempt)

            except Exception as e:
                logger.error(f"❌ Download error on attempt {attempt}: {e}")
                if attempt == self.max_retries:
                    raise DownloadError(f"Download failed: {e}")
                await asyncio.sleep(2**attempt)

        raise DownloadError("Download failed after all retries")

    async def download_multiple(self, urls: List[str], job_id: str) -> List[Path]:
        """
        Download multiple files in parallel.

        Args:
            urls: List of file URLs
            job_id: Job identifier

        Returns:
            List of downloaded file paths
        """
        logger.info(f"⬇️  Downloading {len(urls)} files in parallel...")

        tasks = [self.download_file(url, job_id) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate successful downloads from errors
        downloaded_files = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to download {urls[i]}: {result}")
            else:
                downloaded_files.append(result)

        logger.info(
            f"✅ Downloaded {len(downloaded_files)}/{len(urls)} files successfully"
        )
        return downloaded_files

    def cleanup_job(self, job_id: str) -> None:
        """
        Delete all files for a job.

        Args:
            job_id: Job identifier
        """
        job_dir = self.download_dir / job_id
        if job_dir.exists():
            import shutil

            shutil.rmtree(job_dir)
            logger.info(f"🗑️  Cleaned up job directory: {job_id}")


# Global downloader instance
downloader = FileDownloader()
