from __future__ import annotations

from typing import Any, Callable, Dict

from app.logger import setup_logger
from app.timer import QuestionTimer
from app.tools.core_add_dependencies import add_dependencies
from app.tools.core_download_file import download_file
from app.tools.core_list_installed import list_installed_packages
from app.tools.core_run_code import run_code
from app.tools.core_send_request import send_request
from app.tools.core_web_scraper import scrape
from app.tools.utility_summarize import summarize_text
from app.tools.utility_time import get_time_remaining

logger = setup_logger(__name__)


class ToolRegistry:
    """
    Registry exposing all tools the agent can call.

    Each tool is referenced by a simple name and mapped to a Python callable.
    """

    def __init__(self, job_id: str, timer: QuestionTimer) -> None:
        self.job_id = job_id
        self.timer = timer

        self._tools: Dict[str, Callable[..., Any]] = {
            # Core tools
            "add_dependencies": self._add_dependencies,
            "download_file": self._download_file,
            "send_request": self._send_request,
            "run_code": self._run_code,
            "web_scraper": self._web_scraper,
            "list_installed_packages": self._list_installed_packages,
            # Utility tools
            "get_time_remaining": self._get_time_remaining,
            "summarize_text": self._summarize_text,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found")
        return self._tools[name]

    def list_for_prompt(self) -> str:
        """
        Return a human-readable description of tools for the LLM prompt.
        """
        descriptions = [
            "- add_dependencies(packages: List[str]): Install Python packages needed for analysis.",
            "- download_file(url: str): Download a file (CSV, JSON, PDF, Excel, image, etc.).",
            "- send_request(method: str, url: str, headers: dict, json_body: dict): Call any HTTP API.",
            "- run_code(code: str): Execute Python code that can read downloaded files and perform analysis/visualization.",
            "- web_scraper(url: str, selector: Optional[str]): Scrape a web page (JS-supported) and extract text/HTML/links.",
            "- list_installed_packages(): List installed Python packages so you avoid reinstalling.",
            "- get_time_remaining(): Check how many seconds are left before the 3-minute limit.",
            "- summarize_text(text: str, max_words: int): Summarize long text/HTML before reasoning about it.",
        ]
        return "\n".join(descriptions)

    # ------------------------------------------------------------------
    # Internal wrappers that inject job_id / timer
    # ------------------------------------------------------------------
    async def _add_dependencies(self, packages: list[str]) -> Any:
        return await add_dependencies(packages)

    async def _download_file(self, url: str) -> Any:
        return await download_file(url=url, job_id=self.job_id)

    async def _send_request(
        self,
        method: str,
        url: str,
        headers: dict | None = None,
        json_body: dict | None = None,
    ) -> Any:
        return await send_request(
            method=method, url=url, headers=headers, json_body=json_body
        )

    async def _run_code(self, code: str, workdir: str | None = None) -> Any:
        # Default workdir: job-specific workspace
        from pathlib import Path

        if workdir is None:
            workdir = str(Path("/app/data/quiz-jobs") / self.job_id)
        return await run_code(code=code, workdir=workdir)

    async def _web_scraper(self, url: str, selector: str | None = None) -> Any:
        return await scrape(url=url, selector=selector)

    async def _list_installed_packages(self, workdir: str | None = None) -> Any:
        from pathlib import Path

        if workdir is None:
            workdir = str(Path("/app/data/quiz-jobs") / self.job_id)
        return await list_installed_packages(workdir=workdir)

    async def _get_time_remaining(self) -> float:
        return get_time_remaining(self.timer)

    async def _summarize_text(self, text: str, max_words: int = 300) -> str:
        return await summarize_text(text=text, max_words=max_words)
