from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from app.agent import prompts
from app.config import settings
from app.llm.client import llm_client
from app.loader.page_loader import load_raw_page
from app.logger import setup_logger
from app.models import SolveRequest
from app.timer import QuestionTimer
from app.tools.registry import ToolRegistry
from app.utils.exceptions import QuizSolverError

logger = setup_logger(__name__)


# ----------------------------------------------------------------------
# Top-level chain runner (called from app.main)
# ----------------------------------------------------------------------
async def run_solver_job(job_id: str, request: SolveRequest) -> None:
    """
    Run the full quiz chain starting from the initial URL.

    Loops over questions until the quiz endpoint returns url = null.
    """
    email = request.email
    secret = request.secret
    current_url: Optional[str] = request.url
    question_index = 0

    logger.info(f"🚀 Starting solver job {job_id}")
    logger.info(f"   Email: {email}")
    logger.info(f"   Initial URL: {current_url}")

    try:
        while current_url:
            question_index += 1
            logger.info("============================================================")
            logger.info(f"QUESTION {question_index}")
            logger.info("============================================================")

            timer = QuestionTimer(timeout=settings.forces_submit_time)

            agent = QuestionAgent(
                job_id=job_id,
                email=email,
                secret=secret,
                llm=llm_client,
                timer=timer,
            )

            result = await agent.solve_one_question(current_url)

            next_url = result.get("url")
            if next_url:
                logger.info(f"➡️  Moving to next question: {next_url}")
                current_url = next_url
            else:
                logger.info("✅ Quiz chain complete (no next URL).")
                current_url = None

        logger.info(f"🏁 Job {job_id} finished after {question_index} question(s).")

    except Exception as e:
        logger.error(f"🔥 Job {job_id} failed: {e}", exc_info=True)
        raise


# ----------------------------------------------------------------------
# Per-question agent
# ----------------------------------------------------------------------
class QuestionAgent:
    """
    Handles solving a single quiz question end-to-end.
    """

    def __init__(
        self,
        job_id: str,
        email: str,
        secret: str,
        llm,
        timer: QuestionTimer,
    ) -> None:
        self.job_id = job_id
        self.email = email
        self.secret_value = secret
        self.llm = llm
        self.timer = timer

        # Job workspace
        self.workspace_dir = Path("/app/data/quiz-jobs") / job_id
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Tool registry (injects job_id + timer)
        self.tool_registry = ToolRegistry(job_id=job_id, timer=timer)

        # Per-question state
        self.submission_template: Dict[str, Any] = {}
        self.context_log: str = ""  # text log of thoughts, tools, observations

    async def solve_one_question(self, url: str) -> Dict[str, Any]:
        """
        Solve a single question at the given URL.

        Returns:
            The parsed response from /submit, including the next URL (if any).
        """
        logger.info(f"🎯 Starting question: {url}")

        self.timer.start()

        # STEP 1: Load page (with basic retry handled by primitives/browser)
        page = await load_raw_page(url)
        page_text = page["text"]
        page_html = page["html"]
        base_url = page["base_url"]

        # STEP 2: Planning phase (find submit URL, format, initial template)
        plan = await self._planning_phase(
            current_url=url,
            page_text=page_text,
            page_html=page_html,
        )

        submit_url = self._make_absolute_submit_url(base_url, plan.get("submit_url"))
        self.submission_template = plan.get("submission_template") or {}
        # Fill in system-level fields
        self.submission_template["email"] = self.email
        self.submission_template["url"] = url
        # secret will be filled by the agent once known

        logger.info(f"📬 Planned submit URL: {submit_url}")
        logger.info(f"📝 Answer format: {plan.get('answer_format')}")
        logger.info(f"🛠️ Tools needed: {plan.get('tools_needed')}")

        # Log initial understanding
        self._append_context(f"PLAN: {json.dumps(plan, ensure_ascii=False)[:1000]}")

        # STEP 3: ReAct loop (tools + reasoning)
        await self._react_loop(submit_url)

        # STEP 4: Submit final or best-effort answer
        result = await self._submit_answer(submit_url)

        return result

    # ------------------------------------------------------------------
    # Planning Phase
    # ------------------------------------------------------------------
    async def _planning_phase(
        self,
        current_url: str,
        page_text: str,
        page_html: str,
    ) -> Dict[str, Any]:
        time_remaining = self.timer.time_remaining()
        tools_description = self.tool_registry.list_for_prompt()

        prompt_text = prompts.planning_prompt(
            page_text=page_text,
            page_html=page_html,
            tools_description=tools_description,
            current_url=current_url,
            time_remaining=time_remaining,
        )

        response = await self.llm.generate(
            prompt=prompt_text,
            system=prompts.system_prompt(),
            timer=self.timer,
        )

        try:
            plan = json.loads(response)
        except Exception:
            # Try to extract JSON object from text
            try:
                start = response.index("{")
                end = response.rindex("}") + 1
                plan = json.loads(response[start:end])
            except Exception as e:
                raise QuizSolverError(f"Failed to parse planning JSON: {e}")

        return plan

    def _make_absolute_submit_url(
        self, base_url: str, submit_url: Optional[str]
    ) -> str:
        from urllib.parse import urljoin, urlparse

        if not submit_url:
            # Fallback to /submit on same origin
            parsed = urlparse(base_url)
            return f"{parsed.scheme}://{parsed.netloc}/submit"

        submit_url = submit_url.strip()
        if submit_url.startswith("http://") or submit_url.startswith("https://"):
            return submit_url

        # treat as relative path
        return urljoin(base_url, submit_url)

    # ------------------------------------------------------------------
    # ReAct Loop
    # ------------------------------------------------------------------
    async def _react_loop(self, submit_url: str) -> None:
        """
        Repeatedly think → optionally call a tool → observe, until:
        - LLM decides final answer, or
        - Timer forces us to stop.
        """
        tools_description = self.tool_registry.list_for_prompt()

        while not self.timer.should_force_submit():
            time_remaining = self.timer.time_remaining()

            prompt_text = prompts.react_step_prompt(
                current_context=self.context_log,
                tools_description=tools_description,
                submission_template=self.submission_template,
                time_remaining=time_remaining,
            )

            response = await self.llm.generate(
                prompt=prompt_text,
                system=prompts.system_prompt(),
                timer=self.timer,
            )

            # Try to parse JSON
            try:
                step = json.loads(response)
            except Exception:
                # Try to extract JSON object
                try:
                    start = response.index("{")
                    end = response.rindex("}") + 1
                    step = json.loads(response[start:end])
                except Exception as e:
                    # Log and break; we'll finalize with best-effort
                    logger.warning(f"⚠️ Failed to parse ReAct step JSON: {e}")
                    break

            action = step.get("action")
            if action == "tool":
                tool_name = step.get("tool_name")
                tool_args = step.get("tool_args") or {}
                reason = step.get("reason", "")
                self._append_context(
                    f"THOUGHT: Using tool {tool_name} because {reason}"
                )

                await self._execute_tool_step(tool_name, tool_args)

            elif action == "final":
                updated = step.get("updated_submission") or {}
                # Merge into current template
                self.submission_template.update(updated)
                explanation = step.get("explanation", "")
                self._append_context(f"FINAL: {updated}, explanation={explanation}")
                break
            else:
                logger.warning(f"⚠️ Unknown action in ReAct step: {action}")
                break

        # If timer expired, we will handle in finalize phase

    async def _execute_tool_step(
        self, tool_name: str, tool_args: Dict[str, Any]
    ) -> None:
        try:
            tool_fn = self.tool_registry.get(tool_name)
        except KeyError as e:
            self._append_context(f"ERROR: Tool '{tool_name}' not found ({e})")
            return

        try:
            result = await tool_fn(**tool_args)
            # Store observation as JSON string (truncated)
            obs_str = json.dumps(result, ensure_ascii=False)
            self._append_context(f"OBSERVATION from {tool_name}: {obs_str[:1500]}")
        except Exception as e:
            self._append_context(f"ERROR from {tool_name}: {e}")

    # ------------------------------------------------------------------
    # Finalization and Submission
    # ------------------------------------------------------------------
    async def _submit_answer(self, submit_url: str) -> Dict[str, Any]:
        """
        Ensure we have a best-effort submission, then POST to submit_url
        using the send_request tool.
        """
        # If timer already forcing submission or we never got a "final" action:
        if self.timer.should_force_submit():
            await self._force_finalize()

        # Ensure required fields present
        self.submission_template.setdefault("email", self.email)
        self.submission_template.setdefault(
            "url", self.submission_template.get("url", "")
        )
        # secret: if the question uses the TDS secret, the LLM should have filled it
        # but as a fallback, we can default to self.secret_value if nothing else.
        self.submission_template.setdefault("secret", self.secret_value)
        self.submission_template.setdefault("answer", None)

        payload = dict(self.submission_template)

        logger.info(f"📨 Submitting payload to {submit_url}: {str(payload)[:500]}")

        send_request_tool = self.tool_registry.get("send_request")

        # One quick retry on transient network error
        for attempt in range(2):
            try:
                resp = await send_request_tool(
                    method="POST",
                    url=submit_url,
                    headers={"Content-Type": "application/json"},
                    json_body=payload,
                )
                status = resp["status_code"]
                body_json = resp.get("json") or {}
                logger.info(f"📨 Submit response status={status}, body={body_json}")
                return body_json
            except Exception as e:
                logger.warning(f"⚠️ Submit attempt {attempt + 1} failed: {e}")
                if attempt == 1:
                    raise QuizSolverError(f"Submit failed for {submit_url}: {e}")

        # Should not reach here
        raise QuizSolverError("Submit failed unexpectedly")

    async def _force_finalize(self) -> None:
        """
        When time is almost over, ask LLM once to finalize the best possible answer
        using current context and template. No more tool calls.
        """
        time_remaining = self.timer.time_remaining()
        prompt_text = prompts.finalize_prompt(
            current_context=self.context_log,
            submission_template=self.submission_template,
            time_remaining=time_remaining,
        )

        response = await self.llm.generate(
            prompt=prompt_text,
            system=prompts.system_prompt(),
            timer=self.timer,
        )

        try:
            data = json.loads(response)
        except Exception:
            try:
                start = response.index("{")
                end = response.rindex("}") + 1
                data = json.loads(response[start:end])
            except Exception as e:
                logger.warning(f"⚠️ Failed to parse finalize JSON: {e}")
                return

        updated = data.get("updated_submission") or {}
        self.submission_template.update(updated)
        explanation = data.get("explanation", "")
        self._append_context(f"FORCED FINAL: {updated}, explanation={explanation}")

    # ------------------------------------------------------------------
    # Context logging
    # ------------------------------------------------------------------
    def _append_context(self, text: str) -> None:
        """
        Append a line to the internal context log, trimming if it becomes too large.
        """
        self.context_log += text + "\n"

        # Simple size control to avoid unbounded growth
        if len(self.context_log) > 12000:
            # Keep only last ~8000 chars
            self.context_log = self.context_log[-8000:]
