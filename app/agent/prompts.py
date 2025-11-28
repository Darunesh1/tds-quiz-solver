from __future__ import annotations

from textwrap import dedent


def system_prompt() -> str:
    return dedent(
        """
        You are an autonomous data-science quiz agent.

        Rules:
        - Each question is independent. Do NOT rely on previous questions.
        - You have access only to the tools described to you. Do not fabricate external data.
        - Per question, you have a hard time limit of about 175 seconds (just under 3 minutes).
        - Always construct and maintain a submission JSON template:
            {
              "email": "...",
              "secret": "...",
              "url": "...",
              "answer": ...
            }
          Fill in as much as you can as you progress.
        - If time is almost over, finalize the best possible answer from available information and submit it.
        - Prefer simple, robust plans over complex ones.
        - Use tools to:
            - Scrape web pages (including JS-generated content),
            - Download and parse data files,
            - Call APIs,
            - Run Python code for data transformation, ML, statistics, visualization.

        Output:
        - When asked for a plan, always respond in valid JSON.
        - When asked to finalize an answer, respond with JSON containing at least the "answer" field.
        """
    ).strip()


def planning_prompt(
    page_text: str,
    page_html: str,
    tools_description: str,
    current_url: str,
    time_remaining: float,
) -> str:
    return dedent(
        f"""
        You are about to solve a single quiz question.

        Current page URL:
        {current_url}

        Time remaining (approximate):
        {time_remaining:.1f} seconds

        Page TEXT (truncated if very long):
        ----
        {page_text[:2000]}
        ----

        Page HTML SNIPPET (may be truncated):
        ----
        {page_html[:2000]}
        ----

        Available tools:
        {tools_description}

        TASKS:

        1. Understand what this question is asking.
        2. Identify the submit URL (either a full URL or a path like /submit).
           - Look for phrases like "POST this JSON to /submit".
           - If you only see a path (e.g., /submit), assume it is relative to the current page's origin.
        3. Infer the expected answer format (number, string, JSON structure, base64 image, etc).
        4. Decide which tools you will need (web_scraper, download_file, run_code, send_request, etc).
        5. Produce an initial submission template.

       
        Respond in STRICT JSON with this structure:

        {{
          "understanding": "brief description of what the question asks",
          "submit_url": "path or full URL where the answer JSON should be POSTed",
          "answer_format": "description of expected answer format",
          "submission_template": {{
            "email": "to be filled by system",
            "secret": "to be filled by you when known",
            "url": "to be filled by system with current question URL",
            "answer": null
          }},
          "tools_needed": ["tool_name1", "tool_name2", ...],
          "plan": [
            "Step 1 ...",
            "Step 2 ...",
            "Step 3 ..."
          ]
        }}

        """
    ).strip()


def react_step_prompt(
    current_context: str,
    tools_description: str,
    submission_template: dict,
    time_remaining: float,
) -> str:
    """
    Prompt used on each ReAct iteration.

    The backend will append tool results / observations into `current_context`.
    """
    return dedent(
        f"""
        You are in the middle of solving a data-science quiz question.

        Time remaining for this question (approximate): {time_remaining:.1f} seconds

        Current submission template (JSON):
        {submission_template}

        So far, you have this context (thoughts, tool calls, observations):
        ----
        {current_context[:4000]}
        ----

        Available tools:
        {tools_description}

        You can continue to think and, if needed, call ONE tool at a time.

        Use the following decision pattern:
        - THINK about what to do next.
        - Optionally CALL a tool if you need external information or computation.
        - If you have enough information to produce the final answer, STOP using tools and output a final answer update for the submission template.

        Respond in one of these TWO JSON formats:

        1) To CALL a tool:

        {
            "action": "tool",
          "tool_name": "one of: add_dependencies, download_file, send_request, run_code, web_scraper, list_installed_packages, get_time_remaining, summarize_text",
          "tool_args": {...},
          "reason": "why you chose this tool"
        }

        2) To PRODUCE FINAL ANSWER (no more tool calls):

        {
            "action": "final",
          "updated_submission": {
                "email": "...",
            "secret": "...",
            "url": "...",
            "answer": ...
          },
          "explanation": "brief explanation of how you arrived at this answer"
        }

        Always use valid JSON. Do not include any comments or extra keys.
        """
    ).strip()


def finalize_prompt(
    current_context: str,
    submission_template: dict,
    time_remaining: float,
) -> str:
    """
    Prompt used when time is almost over and we must force-submit.
    """
    return dedent(
        f"""
        You are OUT OF TIME for this question (only about {
            time_remaining:.1f} seconds remain).

        You must now produce the BEST POSSIBLE ANSWER using only the information you have.

        Current submission template:
        {submission_template}

        Context of what has happened so far:
        ----
        {current_context[:4000]}
        ----

        TASK:
        - Do NOT call any tools.
        - Update the submission template's "answer" field (and "secret" if required), using your best guess from the context.
        - Keep the structure of the submission template exactly the same.

        Respond in JSON:

        {
            "updated_submission": {
                "email": "...",
            "secret": "...",
            "url": "...",
            "answer": ...
          },
          "explanation": "very brief explanation"
        }
        """
    ).strip()
