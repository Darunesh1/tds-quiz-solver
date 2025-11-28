import logging
import subprocess

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def add_dependencies(dependencies: str) -> str:
    """
    Installs Python packages using the 'uv' package manager.

    Args:
        dependencies (str): Space-separated package names (e.g., "pandas numpy requests")

    Returns:
        str: Success message ("Installed: <packages>") or error details

    Timeout: 60 seconds

    Usage Notes:
    - Install multiple packages in one call: "pandas numpy matplotlib"
    - Package names must be valid PyPI identifiers
    - On error, returns detailed stderr from uv

    Examples:
        add_dependencies("pandas")
        add_dependencies("beautifulsoup4 lxml html2text")
    """

    logger.info(f"📦 INSTALLING: {dependencies}")
    try:
        result = subprocess.run(
            ["uv", "add"] + dependencies.split(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            logger.info("✅ Install Success")
            return f"Installed: {dependencies}"
        else:
            logger.error(f"❌ Install Failed: {result.stderr}")
            return f"Error installing {dependencies}: {result.stderr}"
    except Exception as e:
        return f"Exception: {str(e)}"
