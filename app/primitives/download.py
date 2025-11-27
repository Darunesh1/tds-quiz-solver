"""
File download utilities with streaming, size limits and retries.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Optional

import httpx

from app.config import settings
from app.logger import setup_logger
from app.utils.exceptions import DownloadError

logger = setup_logger(__name__)


class FileDownloader:
    """
    Downloads files with streaming, size limits, and retries.

    Files are stored under /tmp/quiz-jobs/{job_id}/ by default.
    """

    def __init__(
        self,
        max_size_mb: float = 50.0,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        """
        Args:
            max_size_mb: Maximum allowed file size in MB.
            timeout: Per-request timeout in seconds.
            max_retries: Maximum retry attempts.
        """
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.timeout = timeout
        self.max_retries = max_retries
        self.download_dir = Path("/tmp/quiz-jobs")

    async def download_file(
        self,
        url: str,
        job_id: str,
        custom_filename: Optional[str] = None,
    ) -> Path:
        """
        Download a single file.

        Args:
            url: File URL.
            job_id: Job identifier for organizing downloads.
            custom_filename: Optional custom filename.

        Returns:
            Path to downloaded file.

        Raises:
            DownloadError if download fails.
        """
        job_dir = self.download_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        if custom_filename:
            filename = custom_filename
        else:
            # Derive filename from URL
            url_parts = url.rstrip("/").split("/")
            filename = url_parts[-1] or "downloaded_file"
        filepath = job_dir / filename

        logger.info(f"⬇️ Downloading {url} -> {filepath.name}")

        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream("GET", url) as response:
                        response.raise_for_status()

                        content_length = response.headers.get("content-length")
                        if content_length and int(content_length) > self.max_size_bytes:
                            raise DownloadError(
                                f"File too large: {int(content_length) / (1024 * 1024):.1f}MB "
                                f"(max {self.max_size_bytes / (1024 * 1024):.1f}MB)"
                            )

                        downloaded = 0
                        with open(filepath, "wb") as f:
                            async for chunk in response.aiter_bytes(chunk_size=8192):
                                downloaded += len(chunk)
                                if downloaded > self.max_size_bytes:
                                    filepath.unlink(missing_ok=True)
                                    raise DownloadError(
                                        f"File exceeded size limit during download: "
                                        f"{downloaded / (1024 * 1024):.1f}MB "
                                        f"(max {self.max_size_bytes / (1024 * 1024):.1f}MB)"
                                    )
                                f.write(chunk)

                logger.info(
                    f"✅ Downloaded {filepath.name} ({downloaded / 1024:.1f}KB)"
                )
                return filepath

            except httpx.HTTPStatusError as e:
                logger.warning(
                    f"⚠️ HTTP {e.response.status_code} on attempt "
                    f"{attempt}/{self.max_retries} for {url}"
                )
                if attempt == self.max_retries:
                    raise DownloadError(
                        f"Download failed: HTTP {e.response.status_code}"
                    ) from e
                await asyncio.sleep(2 * attempt)

            except httpx.TimeoutException as e:
                logger.warning(
                    f"⚠️ Timeout on attempt {attempt}/{self.max_retries} for {url}"
                )
                if attempt == self.max_retries:
                    raise DownloadError(
                        f"Download timeout after {self.timeout}s"
                    ) from e
                await asyncio.sleep(2 * attempt)

            except Exception as e:
                logger.error(
                    f"❌ Download error on attempt {attempt}/{self.max_retries} for {url}: {e}"
                )
                if attempt == self.max_retries:
                    raise DownloadError(f"Download failed: {e}") from e
                await asyncio.sleep(2 * attempt)

        raise DownloadError("Download failed after all retries")

    async def download_multiple(self, urls: List[str], job_id: str) -> List[Path]:
        """
        Download multiple files in parallel.

        Args:
            urls: List of file URLs.
            job_id: Job identifier.

        Returns:
            List of downloaded file paths (successful only).
        """
        logger.info(f"⬇️ Downloading {len(urls)} files in parallel...")
        tasks = [self.download_file(url, job_id) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        downloaded_files: List[Path] = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                logger.error(f"❌ Failed to download {url}: {result}")
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
            job_id: Job identifier.
        """
        job_dir = self.download_dir / job_id
        if job_dir.exists():
            import shutil

            shutil.rmtree(job_dir)
            logger.info(f"🧹 Cleaned up job directory for {job_id}")
        else:
            logger.debug(f"No download directory to clean for {job_id}")


# Global downloader instance used by core_download_file.py
downloader = FileDownloader()
