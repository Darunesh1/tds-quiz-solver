import subprocess

from langchain_core.tools import tool


@tool
def list_dependencies() -> str:
    """
    List all currently installed Python packages in the environment.
    Use this to check if a library (like pandas, scipy, sklearn) is already
    installed before attempting to add it.

    Returns:
        str: A formatted string list of installed packages and versions.
    """
    try:
        # Using uv pip list is faster and consistent with the environment
        result = subprocess.run(
            ["uv", "pip", "list"], capture_output=True, text=True, check=True
        )
        return f"Installed Dependencies:\n{result.stdout}"
    except Exception as e:
        return f"Error listing dependencies: {e}"
