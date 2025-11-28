"""Python package installation tool using uv package manager."""

import logging
import subprocess

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def add_dependencies(packages: str) -> dict:
    """
    Installs Python packages using the ultra-fast 'uv' package manager.

    **When to Use:**
    - Before run_code() when you need libraries not in standard library
    - When you see "ModuleNotFoundError" in run_code() stderr
    - For data science: pandas, numpy, matplotlib
    - For web: requests, beautifulsoup4
    - For crypto: cryptography, hashlib (hashlib is built-in)

    **Common Packages for CTF Challenges:**
    ```
    Data Analysis:  add_dependencies("pandas numpy")
    Web Scraping:   add_dependencies("beautifulsoup4 lxml")
    Images:         add_dependencies("pillow")
    Crypto:         add_dependencies("cryptography pycryptodome")
    Math:           add_dependencies("scipy sympy")
    Encoding:       (base64, json are built-in, no install needed)
    ```

    **How It Works:**
    - Uses 'uv' package manager (faster than pip)
    - Installs to project environment (not global)
    - Packages available immediately in subsequent run_code() calls
    - Multiple packages can be installed in one call

    **Package Names:**
    - Use PyPI package names (check pypi.org if unsure)
    - Separate multiple packages with spaces
    - No version specifiers needed (installs latest)

    Args:
        packages: Space-separated package names
                 Examples: "pandas", "numpy matplotlib", "requests beautifulsoup4"

    Returns:
        dict: {
            'installed': list,  # Successfully installed packages
            'success': bool,
            'message': str
        }
        On error: dict with 'error' and 'suggestion' fields

    **Important Notes:**
    - Installation takes 5-30 seconds depending on package size
    - Timeout: 60 seconds
    - If installation fails, check package name spelling
    - Some packages have system dependencies (rare, but may fail)

    **Example Workflow:**
    ```
    # Scenario: Need to analyze CSV data

    # Try to run code without pandas
    result = run_code("import pandas as pd; df = pd.read_csv('data.csv')")
    # → Error: ModuleNotFoundError: No module named 'pandas'

    # Install pandas
    add_dependencies("pandas")
    # → Success

    # Run code again
    result = run_code("import pandas as pd; df = pd.read_csv('data.csv'); print(df.head())")
    # → Success
    ```

    **Tips:**
    - Install all needed packages in one call (faster)
    - Check run_code() stderr for exact package name
    - Some packages have different import names (e.g., PIL vs pillow)
    """
    logger.info(f"📦 INSTALLING PACKAGES: {packages}")

    try:
        # Parse and validate package names
        package_list = packages.strip().split()

        if not package_list:
            return {
                "error": "No packages specified",
                "suggestion": 'Provide space-separated package names (e.g., "pandas numpy")',
            }

        logger.info(f"   Packages to install: {package_list}")

        # Run uv add command
        result = subprocess.run(
            ["uv", "add"] + package_list,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            logger.info(f"✅ Successfully installed: {', '.join(package_list)}")
            return {
                "installed": package_list,
                "success": True,
                "message": f"Successfully installed {len(package_list)} package(s): {', '.join(package_list)}",
            }
        else:
            # Installation failed
            error_output = result.stderr.strip()
            logger.error(f"❌ Installation Failed: {error_output[:200]}")

            # Provide helpful suggestions based on error
            suggestion = "Check package name spelling on pypi.org"
            if "not found" in error_output.lower():
                suggestion = "Package not found on PyPI. Verify the exact package name."
            elif "conflict" in error_output.lower():
                suggestion = "Version conflict. Try installing packages separately."

            return {
                "error": f"Installation failed for: {', '.join(package_list)}",
                "stderr": error_output,
                "suggestion": suggestion,
                "success": False,
            }

    except subprocess.TimeoutExpired:
        error_msg = "Installation timeout (60s limit)"
        logger.error(f"💥 {error_msg}")
        return {
            "error": error_msg,
            "suggestion": "Package may be very large. Try installing smaller packages first.",
            "success": False,
        }

    except FileNotFoundError:
        error_msg = "'uv' command not found"
        logger.error(f"💥 {error_msg}")
        return {
            "error": error_msg,
            "suggestion": "Install uv package manager: curl -LsSf https://astral.sh/uv/install.sh | sh",
            "success": False,
        }

    except Exception as e:
        logger.error(f"💥 Installation Error: {e}")
        return {
            "error": str(e),
            "suggestion": "Verify uv is properly installed and accessible",
            "success": False,
        }
