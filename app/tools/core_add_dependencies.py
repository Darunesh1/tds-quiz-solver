from __future__ import annotations

from typing import Any, Dict, List

from app.logger import setup_logger
from app.utils.exceptions import QuizSolverError

logger = setup_logger(__name__)


async def add_dependencies(packages: List[str]) -> Dict[str, Any]:
    """
    Call tools/add_dependencies.py to install Python packages.

    Args:
        packages: List of package names to ensure installed.

    Returns:
        {
          "requested": [...],
          "installed": [...],   # actually newly installed
          "skipped": [...],     # already present
        }

    Raises:
        QuizSolverError if the helper script fails.
    """
    # Import lazily to avoid import-time side effects
    try:
        from tools import add_dependencies as add_deps_mod  # type: ignore
    except Exception as e:
        raise QuizSolverError(f"add_dependencies tool not available: {e}")

    if not packages:
        return {"requested": [], "installed": [], "skipped": []}

    logger.info(f"🧩 Installing dependencies: {packages}")

    try:
        # Assuming tools/add_dependencies.py exposes a main-like function.
        # Adjust this to match the actual API of that script.
        result = await add_deps_mod.ensure_dependencies(packages)  # type: ignore[attr-defined]
        # Expected result format is repo-dependent; normalize if needed.
        return {
            "requested": packages,
            "installed": result.get("installed", []),
            "skipped": result.get("skipped", []),
        }
    except AttributeError:
        # Fallback: script may only expose a synchronous function
        try:
            installed = add_deps_mod.ensure_dependencies(packages)  # type: ignore[attr-defined]
            return {
                "requested": packages,
                "installed": installed,
                "skipped": [],
            }
        except Exception as e:
            logger.error(f"❌ add_dependencies failed: {e}")
            raise QuizSolverError(f"add_dependencies failed: {e}")
    except Exception as e:
        logger.error(f"❌ add_dependencies failed: {e}")
        raise QuizSolverError(f"add_dependencies failed: {e}")
