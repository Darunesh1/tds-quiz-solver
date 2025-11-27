from __future__ import annotations

from typing import Any, Dict, List

from app.logger import setup_logger
from app.tools.core_run_code import run_code
from app.utils.exceptions import QuizSolverError

logger = setup_logger(__name__)


async def list_installed_packages(workdir: str) -> List[str]:
    """
    List installed Python packages in the run_code environment.

    Args:
        workdir: Working directory (same as for run_code).

    Returns:
        List of package names (lowercased).
    """
    logger.info("📦 Listing installed packages via run_code")

    code = """
import pkg_resources
pkgs = [d.project_name.lower() for d in pkg_resources.working_set]
result = {"packages": pkgs}
"""

    result: Dict[str, Any] = await run_code(code=code, workdir=workdir)
    packages = result.get("result", {}).get("packages") or result.get("packages")

    if packages is None:
        # Try parsing stdout as a fallback
        stdout = result.get("stdout", "")
        logger.warning(
            "⚠️ Could not find 'packages' in run_code result; returning empty list"
        )
        return []

    return list(packages)
