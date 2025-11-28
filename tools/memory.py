import logging

from langchain_core.tools import tool


@tool
def reset_memory(next_task_instruction: str) -> str:
    """
    CRITICAL: Call this tool IMMEDIATELY after you successfully submit an answer and receive a response stating the answer is correct and you decide to move to the next URL.

    Purpose: It deletes your conversation history to free up space for the next quiz level.

    Args:
        next_task_instruction: A clear instruction for yourself for the next step.
                               MUST include the NEW URL you just received.
                               Example: "Level 1 complete. Go to https://.../quiz/2 and solve it."
    """
    return f"__RESET__:{next_task_instruction}"
