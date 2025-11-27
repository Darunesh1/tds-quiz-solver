from __future__ import annotations

from typing import Any, Dict

from app.logger import setup_logger
from app.utils.exceptions import QuizSolverError

logger = setup_logger(__name__)


async def run_code(code: str, workdir: str) -> Dict[str, Any]:
    """
    Execute arbitrary Python code via tools/run_code.py.

    Args:
        code: Python code to execute.
        workdir: Working directory where downloaded files live.

    Returns:
        {
          "stdout": str,
          "stderr": str,
          "result": Any  # If the run_code tool supports returning a result
        }

    Raises:
        QuizSolverError if execution fails.
    """
    try:
        from tools import run_code as run_code_mod  # type: ignore
    except Exception as e:
        raise QuizSolverError(f"run_code tool not available: {e}")

    logger.info("🧮 Running user code via run_code tool")

    try:
        # Adjust this call to match the actual interface of tools/run_code.py.
        # Example signature: async def execute(code: str, workdir: str) -> dict
        result: Dict[str, Any] = await run_code_mod.execute(code=code, workdir=workdir)  # type: ignore[attr-defined]
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        logger.info(
            f"🧮 run_code completed (stdout len={len(stdout)}, stderr len={len(stderr)})"
        )
        return result
    except AttributeError:
        # Fallback for synchronous implementation
        try:
            result: Dict[str, Any] = run_code_mod.execute(code=code, workdir=workdir)  # type: ignore[attr-defined]
            return result
        except Exception as e:
            logger.error(f"❌ run_code failed: {e}")
            raise QuizSolverError(f"run_code failed: {e}")
    except Exception as e:
        logger.error(f"❌ run_code failed: {e}")
        raise QuizSolverError(f"run_code failed: {e}")
