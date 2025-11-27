"""
Main solver orchestration logic.
Connects all primitives to solve quiz questions end-to-end.
"""

import asyncio
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

from app.config import settings
from app.llm import llm_client
from app.logger import setup_logger
from app.models import QuizSubmission, SolveRequest
from app.primitives.analyze import data_analyzer
from app.primitives.browser import browser_manager
from app.primitives.download import downloader
from app.primitives.parse import file_parser
from app.primitives.submit import submission_handler
from app.primitives.visualize import visualizer
from app.timer import QuestionTimer
from app.utils.exceptions import (
    BrowserError,
    QuizSolverError,
    SubmissionError,
    TimeoutError,
)

logger = setup_logger(__name__)


class QuizSolver:
    """
    Main orchestrator for solving quiz questions.
    Handles the complete workflow from page load to submission.
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.timer = QuestionTimer(timeout=settings.force_submit_time)
        self.workspace = Path(f"/app/data/quiz-jobs/{job_id}")
        self.workspace.mkdir(parents=True, exist_ok=True)

    async def solve_question(self, url: str) -> Dict[str, Any]:
        """
        Solve a single quiz question.

        Args:
            url: Quiz page URL

        Returns:
            Submission response (correct, url, reason)
        """
        logger.info(f"🎯 Starting question: {url}")
        self.timer.start()  # Reset timer for this question

        try:
            # Step 1: Load page and extract information
            page_data = await self._load_and_extract_page(url)

            # Check timer
            if self.timer.should_force_submit():
                return await self._force_submit_partial(page_data["submit_url"])

            # Step 2: Download files (if any)
            files = await self._download_files(page_data.get("file_urls", []), url)

            # Check timer
            if self.timer.should_force_submit():
                return await self._force_submit_partial(page_data["submit_url"])

            # Step 3: Parse files
            parsed_data = self._parse_files(files)

            # Check timer
            if self.timer.should_force_submit():
                return await self._force_submit_partial(page_data["submit_url"])

            # Step 4: Analyze with LLM guidance
            answer = await self._analyze_and_generate_answer(
                instructions=page_data["instructions"],
                parsed_data=parsed_data,
                page_html=page_data.get("html", ""),
            )

            # Check timer
            if self.timer.should_force_submit():
                return await self._submit_answer(page_data["submit_url"], answer)

            # Step 5: Submit answer
            result = await self._submit_answer(page_data["submit_url"], answer)

            logger.info(f"⏱️  Question completed in {self.timer.elapsed():.1f}s")
            return result

        except Exception as e:
            logger.error(f"❌ Question failed: {e}")

            # Try to submit partial answer if we have submit URL
            if "submit_url" in locals():
                try:
                    return await self._force_submit_partial(page_data["submit_url"])
                except:
                    pass

            raise QuizSolverError(f"Question solving failed: {e}")

    async def _load_and_extract_page(self, url: str) -> Dict[str, Any]:
        """
        Load quiz page and extract all relevant information.

        Returns:
            Dict with instructions, submit_url, file_urls, html
        """
        logger.info("🌐 Loading quiz page...")

        try:
            # Load page with Playwright
            page_result = await browser_manager.load_page(
                url, timeout=settings.playwright_timeout
            )
            page = page_result["page"]
            html = page_result["html"]
            text = page_result["text"]

            # Extract submit URL (CRITICAL!)
            submit_url = await browser_manager.find_submit_url(page)
            if not submit_url:
                raise BrowserError("Could not find submit URL on page")

            # Make submit URL absolute
            submit_url = urljoin(url, submit_url)

            # Extract instructions
            instructions = await browser_manager.extract_instructions(page)

            # Find downloadable files (CSV, PDF, JSON, XLSX, images)
            file_urls = await self._extract_file_urls(page, url)

            # Extract embedded data (JSON in script tags, etc.)
            embedded_data = self._extract_embedded_data(html)

            # Close page
            await page.close()

            logger.info(f"✅ Page extracted:")
            logger.info(f"   Submit URL: {submit_url}")
            logger.info(f"   Instructions: {len(instructions)} chars")
            logger.info(f"   File URLs: {len(file_urls)}")
            logger.info(f"   Embedded data: {len(embedded_data)} keys")

            return {
                "submit_url": submit_url,
                "instructions": instructions,
                "file_urls": file_urls,
                "embedded_data": embedded_data,
                "html": html,
                "text": text,
            }

        except Exception as e:
            logger.error(f"❌ Page extraction failed: {e}")
            raise BrowserError(f"Failed to extract page data: {e}")

    async def _extract_file_urls(self, page, base_url: str) -> List[str]:
        """Extract URLs of downloadable files."""
        try:
            # Get all links
            all_links = await browser_manager.extract_links(page)

            # Filter for data files
            file_extensions = [
                ".csv",
                ".json",
                ".pdf",
                ".xlsx",
                ".xls",
                ".png",
                ".jpg",
                ".jpeg",
            ]
            file_urls = []

            for link in all_links:
                parsed = urlparse(link)
                path = parsed.path.lower()

                # Check if link points to a data file
                if any(path.endswith(ext) for ext in file_extensions):
                    # Make absolute URL
                    absolute_url = urljoin(base_url, link)
                    file_urls.append(absolute_url)

            # Also check for data-url attributes and download buttons
            additional_urls = await page.locator(
                '[data-url], [data-file], [href*="download"]'
            ).evaluate_all(
                """elements => elements.map(el => 
                    el.getAttribute('data-url') || 
                    el.getAttribute('data-file') || 
                    el.getAttribute('href')
                ).filter(url => url)"""
            )

            for url in additional_urls:
                absolute_url = urljoin(base_url, url)
                if absolute_url not in file_urls:
                    file_urls.append(absolute_url)

            return file_urls

        except Exception as e:
            logger.warning(f"⚠️ File URL extraction error: {e}")
            return []

    def _extract_embedded_data(self, html: str) -> Dict[str, Any]:
        """Extract JSON data embedded in script tags or data attributes."""
        import json

        embedded = {}

        try:
            # Look for JSON in script tags
            script_pattern = r"<script[^>]*>(.*?)</script>"
            scripts = re.findall(script_pattern, html, re.DOTALL | re.IGNORECASE)

            for script in scripts:
                # Try to find JSON objects
                json_pattern = r"\{[^{}]*\}"
                potential_jsons = re.findall(json_pattern, script)

                for pj in potential_jsons:
                    try:
                        data = json.loads(pj)
                        if isinstance(data, dict) and len(data) > 0:
                            embedded.update(data)
                    except:
                        continue

            logger.debug(f"Extracted {len(embedded)} embedded data keys")

        except Exception as e:
            logger.warning(f"⚠️ Embedded data extraction error: {e}")

        return embedded

    async def _download_files(self, file_urls: List[str], base_url: str) -> List[Path]:
        """Download all files in parallel."""
        if not file_urls:
            logger.info("ℹ️  No files to download")
            return []

        logger.info(f"⬇️  Downloading {len(file_urls)} files...")

        try:
            files = await downloader.download_multiple(file_urls, self.job_id)
            logger.info(f"✅ Downloaded {len(files)} files successfully")
            return files
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return []

    def _parse_files(self, files: List[Path]) -> Dict[str, Any]:
        """Parse downloaded files into usable data structures."""
        parsed = {}

        for file_path in files:
            try:
                logger.info(f"📄 Parsing {file_path.name}...")

                suffix = file_path.suffix.lower()

                if suffix == ".csv":
                    df = file_parser.parse_csv(file_path)
                    parsed[file_path.stem] = {
                        "type": "dataframe",
                        "data": df,
                        "shape": df.shape,
                        "columns": list(df.columns),
                    }

                elif suffix == ".json":
                    data = file_parser.parse_json(file_path)
                    parsed[file_path.stem] = {"type": "json", "data": data}

                elif suffix in [".xlsx", ".xls"]:
                    df = file_parser.parse_excel(file_path)
                    parsed[file_path.stem] = {
                        "type": "dataframe",
                        "data": df,
                        "shape": df.shape,
                        "columns": list(df.columns),
                    }

                elif suffix == ".pdf":
                    text = file_parser.parse_pdf_text(file_path)
                    parsed[file_path.stem] = {
                        "type": "text",
                        "data": text,
                        "length": len(text),
                    }

                    # Try extracting tables too
                    try:
                        tables = file_parser.parse_pdf_tables(file_path)
                        if tables:
                            parsed[f"{file_path.stem}_tables"] = {
                                "type": "tables",
                                "data": tables,
                            }
                    except:
                        pass

                else:
                    logger.warning(f"⚠️ Unsupported file type: {suffix}")

            except Exception as e:
                logger.error(f"❌ Failed to parse {file_path.name}: {e}")

        logger.info(f"✅ Parsed {len(parsed)} files")
        return parsed

    async def _analyze_and_generate_answer(
        self, instructions: str, parsed_data: Dict[str, Any], page_html: str
    ) -> Dict[str, Any]:
        """
        Use LLM to understand instructions and generate answer.
        This is the AI-powered core of the solver.
        """
        logger.info("🧠 Analyzing with LLM...")

        # Build context for LLM
        context = self._build_llm_context(instructions, parsed_data, page_html)

        # Step 1: Understand what needs to be done
        task_prompt = f"""You are analyzing a data science quiz question.

INSTRUCTIONS:
{instructions[:1000]}

AVAILABLE DATA:
{context["data_summary"]}

Your task:
1. Identify what analysis is required (e.g., calculate mean, create visualization, filter data, etc.)
2. Identify which dataset(s) to use
3. Determine the expected output format (number, string, chart, JSON object, etc.)

Respond in JSON format:
{{
    "task_type": "statistical_analysis | visualization | data_transformation | text_processing",
    "required_operations": ["operation1", "operation2"],
    "datasets_to_use": ["dataset_name"],
    "output_format": "number | string | chart | json_object"
}}"""

        try:
            task_analysis = await llm_client.generate(
                task_prompt, system="You are a data science expert."
            )
            logger.info(f"✅ Task understood: {task_analysis[:200]}...")

            # Parse LLM response
            import json

            task_info = self._extract_json_from_text(task_analysis)

            # Step 2: Execute the analysis based on task type
            result = await self._execute_analysis(task_info, parsed_data, instructions)

            return result

        except Exception as e:
            logger.error(f"❌ LLM analysis failed: {e}")

            # Fallback: try deterministic analysis
            return await self._fallback_analysis(parsed_data, instructions)

    def _build_llm_context(
        self, instructions: str, parsed_data: Dict[str, Any], page_html: str
    ) -> Dict[str, Any]:
        """Build concise context for LLM."""
        data_summary = []

        for name, info in parsed_data.items():
            if info["type"] == "dataframe":
                df = info["data"]
                summary = f"- {name}: DataFrame with {df.shape[0]} rows, {df.shape[1]} columns\n"
                summary += f"  Columns: {', '.join(df.columns[:10])}"
                if len(df.columns) > 10:
                    summary += f" ... ({len(df.columns)} total)"
                summary += f"\n  Sample data:\n{df.head(3).to_string()}"
                data_summary.append(summary)

            elif info["type"] == "json":
                data_summary.append(
                    f"- {name}: JSON with keys: {list(info['data'].keys())[:10]}"
                )

            elif info["type"] == "text":
                data_summary.append(f"- {name}: Text document ({info['length']} chars)")

        return {
            "instructions": instructions[:500],
            "data_summary": "\n".join(data_summary),
        }

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response text."""
        import json

        # Try direct JSON parse
        try:
            return json.loads(text)
        except:
            pass

        # Try finding JSON in markdown code blocks
        json_pattern = r"``````"
        matches = re.findall(json_pattern, text, re.DOTALL)
        if matches:
            try:
                return json.loads(matches[0])
            except:
                pass

        # Try finding any JSON object
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except:
                continue

        # Default
        return {}

    async def _execute_analysis(
        self, task_info: Dict[str, Any], parsed_data: Dict[str, Any], instructions: str
    ) -> Dict[str, Any]:
        """Execute the required analysis based on task info."""
        logger.info(f"🔬 Executing analysis: {task_info.get('task_type', 'unknown')}")

        result = {}

        # Get the main dataset
        datasets_to_use = task_info.get("datasets_to_use", [])
        if not datasets_to_use and parsed_data:
            datasets_to_use = [list(parsed_data.keys())[0]]

        for dataset_name in datasets_to_use:
            if dataset_name not in parsed_data:
                continue

            data_info = parsed_data[dataset_name]

            if data_info["type"] == "dataframe":
                df = data_info["data"]

                # Clean the data
                df = data_analyzer.clean_dataframe(df)

                # Perform analysis based on instructions
                analysis_result = await self._analyze_dataframe(
                    df, instructions, task_info
                )
                result.update(analysis_result)

        return result

    async def _analyze_dataframe(
        self, df, instructions: str, task_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze DataFrame based on instructions."""
        result = {}

        # Get basic statistics
        stats = data_analyzer.basic_statistics(df)

        # Use LLM to determine specific calculation
        analysis_prompt = f"""Given this data and instructions, provide the exact answer.

INSTRUCTIONS: {instructions[:500]}

DATA SUMMARY:
- Shape: {df.shape}
- Columns: {list(df.columns)}
- First few rows:
{df.head().to_string()}

Provide the answer in this JSON format:
{{
    "answer": <your calculated value>,
    "explanation": "brief explanation"
}}

If visualization is needed, indicate:
{{
    "answer": "chart_required",
    "chart_type": "line|bar|scatter|histogram",
    "x_column": "column_name",
    "y_column": "column_name"
}}"""

        try:
            response = await llm_client.generate(analysis_prompt)
            answer_data = self._extract_json_from_text(response)

            # Check if chart is required
            if answer_data.get("answer") == "chart_required":
                chart_type = answer_data.get("chart_type", "bar")
                x_col = answer_data.get("x_column", df.columns[0])
                y_col = answer_data.get(
                    "y_column", df.columns[1] if len(df.columns) > 1 else df.columns[0]
                )

                # Generate chart
                if chart_type == "line":
                    chart = visualizer.line_chart(
                        df, x_col, y_col, title=f"{y_col} vs {x_col}"
                    )
                elif chart_type == "bar":
                    chart = visualizer.bar_chart(
                        df.head(20), x_col, y_col, title=f"{y_col} by {x_col}"
                    )
                elif chart_type == "scatter":
                    chart = visualizer.scatter_plot(
                        df, x_col, y_col, title=f"{y_col} vs {x_col}"
                    )
                else:
                    chart = visualizer.histogram(
                        df, y_col, title=f"Distribution of {y_col}"
                    )

                result["answer"] = chart
                result["type"] = "visualization"
            else:
                result["answer"] = answer_data.get("answer")
                result["explanation"] = answer_data.get("explanation", "")

        except Exception as e:
            logger.error(f"❌ DataFrame analysis failed: {e}")

            # Fallback: return basic stats
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                result["answer"] = float(df[numeric_cols[0]].sum())

        return result

    async def _fallback_analysis(
        self, parsed_data: Dict[str, Any], instructions: str
    ) -> Dict[str, Any]:
        """Deterministic fallback when LLM fails."""
        logger.warning("⚠️ Using fallback analysis")

        # Find first DataFrame
        for name, info in parsed_data.items():
            if info["type"] == "dataframe":
                df = info["data"]
                numeric_cols = df.select_dtypes(include=["number"]).columns

                if len(numeric_cols) > 0:
                    # Return sum of first numeric column
                    return {"answer": float(df[numeric_cols[0]].sum())}

        # Default fallback
        return {"answer": 0}

    async def _submit_answer(
        self, submit_url: str, answer_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Submit answer to quiz endpoint."""
        logger.info("📤 Submitting answer...")

        # Format payload
        payload = {"answer": answer_data.get("answer")}

        # Add any additional fields
        if "explanation" in answer_data:
            payload["explanation"] = answer_data["explanation"]

        # Validate size
        if not submission_handler.validate_payload(payload):
            logger.warning("⚠️ Payload too large, simplifying...")
            payload = {"answer": str(answer_data.get("answer"))[:1000]}

        # Submit
        result = await submission_handler.submit_answer(submit_url, payload)
        return result

    async def _force_submit_partial(self, submit_url: str) -> Dict[str, Any]:
        """Force submit partial/empty answer when timer expires."""
        logger.warning("⚠️ FORCE SUBMITTING due to timeout")

        payload = {"answer": 0, "note": "Partial submission due to timeout"}

        try:
            return await submission_handler.submit_answer(submit_url, payload)
        except:
            # Return fake response to continue chain
            return {"correct": False, "url": None}


async def run_solver_job(job_id: str, request: SolveRequest):
    """
    Main entry point for solving a quiz job.
    Handles chaining of multiple questions.

    Args:
        job_id: Unique job identifier
        request: SolveRequest with email, secret, url
    """
    logger.info(f"🚀 Starting solver job {job_id}")
    logger.info(f"   Email: {request.email}")
    logger.info(f"   Initial URL: {request.url}")

    solver = QuizSolver(job_id)
    current_url = str(request.url)
    question_count = 0
    max_questions = 20  # Safety limit

    try:
        while current_url and question_count < max_questions:
            question_count += 1
            logger.info(f"\n{'=' * 60}")
            logger.info(f"QUESTION {question_count}")
            logger.info(f"{'=' * 60}\n")

            # Solve this question
            result = await solver.solve_question(current_url)

            # Check if there's a next question
            next_url = result.get("url")
            if next_url:
                logger.info(f"➡️  Moving to next question: {next_url}")
                current_url = next_url
                # Timer will reset automatically in solve_question()
            else:
                logger.info("🏁 Quiz chain complete!")
                current_url = None

        if question_count >= max_questions:
            logger.warning(f"⚠️ Stopped after {max_questions} questions (safety limit)")

        logger.info(f"\n{'=' * 60}")
        logger.info(f"✅ Job {job_id} completed: {question_count} questions solved")
        logger.info(f"{'=' * 60}\n")

    except Exception as e:
        logger.error(f"❌ Job {job_id} failed: {e}")
        raise

    finally:
        # Cleanup
        downloader.cleanup_job(job_id)
        logger.info(f"🗑️  Cleaned up job {job_id}")
