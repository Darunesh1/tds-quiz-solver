import os
import subprocess
import sys

from langchain_core.tools import tool


@tool
def run_code(code: str) -> str:
    """
    Execute Python code dynamically. USE THIS EXTENSIVELY.

    Use this for:
    - ALL calculations and data processing
    - Creating missing functionality (OCR, screenshots, parsing)
    - Installing libraries when needed
    - ANY task requiring computation or file processing

    Args:
        code: Complete Python code to execute

    Returns:
        Output (stdout) from code execution

    CRITICAL: This is your most powerful tool. When stuck, write code.

    Examples:
    - OCR: run_code("import pytesseract; from PIL import Image; print(pytesseract.image_to_string(Image.open('file.png')))")
    - Math: run_code("numbers = [1,2,3]; print(sum(numbers))")
    - Install: run_code("import subprocess; subprocess.run(['pip', 'install', 'requests'])")
    """
    try:
        runner_path = os.path.join("LLMFiles", "runner.py")
        os.makedirs("LLMFiles", exist_ok=True)

        with open(runner_path, "w") as f:
            f.write(code)

        result = subprocess.run(
            [sys.executable, runner_path], capture_output=True, text=True, timeout=30
        )

        output = result.stdout if result.stdout else result.stderr
        return output or "Code executed successfully (no output)"
    except Exception as e:
        return f"Execution error: {str(e)}"

