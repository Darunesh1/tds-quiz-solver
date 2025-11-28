import logging
import os
import subprocess

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def strip_code_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1]
    if code.endswith("```"):
        code = code.rsplit("\n", 1)[0]
    return code.strip()


@tool
def run_code(code: str) -> dict:
    """
    Executes arbitrary Python code in an isolated environment.

    Execution Details:
    - Code runs INSIDE the 'LLMFiles/' directory
    - Access files by name only (e.g., 'data.csv'), NOT 'LLMFiles/data.csv'
    - 30-second timeout limit
    - Code fences (```

    Args:
        code (str): Python code to execute. Can include markdown code fences.

    Returns:
        dict: {
            'stdout': str,  # Program output (truncated at 5000 chars)
            'stderr': str,  # Error messages with debugging context
            'return_code': int  # 0 = success, non-zero = error
        }

    Error Context: On failure, stderr includes:
    - Working directory path
    - List of available files in cwd
    - File access hint

    Example:
        run_code("import pandas as pd\\ndf = pd.read_csv('data.csv')\\nprint(df.head())")
    """
    try:
        clean_code = strip_code_fences(code)
        logger.info("💻 EXECUTING Python Code...")

        # Ensure directory exists
        work_dir = "LLMFiles"
        os.makedirs(work_dir, exist_ok=True)

        filename = os.path.join(work_dir, "runner.py")
        with open(filename, "w") as f:
            f.write(clean_code)

        # Run the code
        proc = subprocess.Popen(
            ["uv", "run", "runner.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=work_dir,  # Execution happens HERE
        )

        stdout, stderr = proc.communicate(timeout=30)

        # --- GENERALIZED SELF-CORRECTION LOGIC ---
        # If there is an error, give the AI context about the environment
        if proc.returncode != 0 or stderr:
            # 1. Get list of files currently in the directory
            files_in_dir = os.listdir(work_dir)

            # 2. Append this context to the error message
            debug_info = f"\n\n--- DEBUGGING CONTEXT ---\n"
            debug_info += f"Working Directory: {work_dir}\n"
            debug_info += f"Files available in cwd: {files_in_dir}\n"
            debug_info += f"Note: Your code runs *inside* {work_dir}. Access files directly (e.g., 'file.csv'), not '{work_dir}/file.csv'."

            stderr += debug_info

            logger.error(f"❌ Code Error: {stderr[:300]}...")
        else:
            logger.info(f"✅ Code Result: {stdout[:100]}...")

        # Truncate for LLM context
        if len(stdout) > 5000:
            stdout = stdout[:5000] + "...[TRUNCATED]"
        if len(stderr) > 5000:
            stderr = stderr[:5000] + "...[TRUNCATED]"

        return {"stdout": stdout, "stderr": stderr, "return_code": proc.returncode}

    except subprocess.TimeoutExpired:
        proc.kill()
        return {"error": "Execution timed out (30s limit)."}
    except Exception as e:
        return {"error": str(e)}

