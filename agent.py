import json
import logging
import os
import sys
import time
from typing import Annotated, Any, List, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.pregel.main import GraphRecursionError

# Import your tools (Make sure these files are updated!)
from tools import (
    add_dependencies,
    download_file,
    get_rendered_html,
    ocr_image,
    post_request,
    run_code,
    transcribe_audio,
    # reset_memory is technically not needed if we Auto-Reset, but good to keep.
    # If you deleted it, remove it here.
)

load_dotenv()
EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")
RECURSION_LIMIT = 100  # Lower this to prevent infinite loops

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("QuizAgent")


# STATE
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    task_start_time: float
    attempts: int  # <--- NEW: Track failed attempts


TOOLS = [
    run_code,
    get_rendered_html,
    download_file,
    post_request,
    add_dependencies,
    transcribe_audio,
    ocr_image,
]

# LLM SETUP
rate_limiter = InMemoryRateLimiter(
    requests_per_second=9 / 60, check_every_n_seconds=1, max_bucket_size=9
)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google_genai")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")

llm = init_chat_model(
    model_provider=LLM_PROVIDER,
    model=LLM_MODEL,
    rate_limiter=rate_limiter,
).bind_tools(TOOLS)

# -------------------------------------------------
# SYSTEM PROMPT (Enhanced for Forensic "Deep Dive")
# -------------------------------------------------
SYSTEM_PROMPT = f"""
<role>
You are an expert Autonomous Capture-The-Flag (CTF) Solver. 
**CORE PHILOSOPHY**: The answer (or the clue to find it) is **ALWAYS** present on the webpage. It is NEVER missing. If you cannot see it in the text, it is hidden in an image, an audio file, or a specific HTML attribute.
Your job is to be a forensic investigator: Scrape -> Detect -> Decode -> Submit.
</role>

<credentials>
EMAIL: "{EMAIL}"
SECRET: "{SECRET}"
</credentials>

<tools>
1. **`get_rendered_html(url)`**: 
   - *Usage*: CALL THIS FIRST. Scrapes text AND lists all assets (images, audio, links).
2. **`download_file(url)`**: 
   - *Usage*: Downloads files to `LLMFiles/`. Returns the filename.
3. **`run_code(script)`**: 
   - *Usage*: Executes Python. Runs INSIDE `LLMFiles/`.
   - *Constraint*: Access files directly by name (e.g., `pd.read_csv("data.csv")`). DO NOT use `LLMFiles/` prefix in code.
4. **`ocr_image(image_path)`**:
   - *Usage*: Extracts text from images. **CRITICAL**: Use this if the page has no obvious text instructions but contains images.
5. **`transcribe_audio(audio_path)`**:
   - *Usage*: Converts audio to text.
6. **`add_dependencies(deps)`**: 
   - *Usage*: Install libs (e.g., `uv add pandas`).
7. **`post_request(url, payload)`**: 
   - *Usage*: Submits answer. Payload: {{"answer": "VAL", "email": "{EMAIL}", "secret": "{SECRET}"}}
</tools>

<protocol>
### PHASE 1: DEEP ANALYSIS
1. Call `get_rendered_html`.
2. **THE VISUAL CHECK (Crucial)**:
   - Does the text prompt look incomplete? 
   - **YES**: The missing data is in an IMAGE or AUDIO file.
   - **ACTION**: Check the `links`/`images` list from the scraper.

### PHASE 2: FORENSIC EXECUTION
- **Scenario A: The "Invisible" Text**: 
  - If the page looks empty but has an image: `download_file(img_url)` -> `ocr_image(local_path)`.
  - Treat images as screenshots containing the question.
- **Scenario B: Audio**:
  - If there is an `.mp3/.wav`: `download_file` -> `transcribe_audio`.
- **Scenario C: Data Mining**:
  - If there is a CSV/JSON: `download_file` -> `run_code`.

### PHASE 3: SOLVE & VERIFY
- **NEVER GUESS**. Use code to calculate.
- **DEBUGGING**: If a file path fails in `run_code`, read the `stderr` to see the actual file list.
- **RETRY LOGIC**: If the answer is wrong, assume you missed a constraint (e.g., "Sort by X", "Filter by Y"). Re-read the scraped data.

### PHASE 4: SUBMISSION
1. Submit via `post_request`.
2. If successful (`correct: true`), call `reset_memory`.

### EMERGENCY SKIP
If you fail **3 times** OR time > **175s**, you **MUST** submit:
`{{ "answer": "SKIP", "email": "{EMAIL}", "secret": "{SECRET}" }}
</protocol>

<constraints>
1. **Persistence**: Do not give up because text is missing. Dig into the assets.
2. **Code Safety**: Do not assume variables persist between `run_code` calls.
3. **Termination**: If no new URL is returned after success, output: `END`.
</constraints>

<current_state>
Session started. The answer is waiting to be found.
</current_state>
"""

prompt = ChatPromptTemplate.from_messages(
    [
        # Wrap the string in SystemMessage so LangChain doesn't parse it
        SystemMessage(content=SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
llm_with_prompt = prompt | llm


def agent_node(state: AgentState):
    messages = state["messages"]
    current_time = time.time()
    task_start_time = state.get("task_start_time", current_time)
    current_attempts = state.get("attempts", 0)

    # 1. ANALYZE LAST TOOL OUTPUT
    if (
        messages
        and isinstance(messages[-1], ToolMessage)
        and messages[-1].name == "post_request"
    ):
        # --- IMPROVED: ROBUSTLY FIND WHAT WE SUBMITTED ---
        last_tool_call_id = messages[-1].tool_call_id
        last_submitted_answer = None

        # Look at the tool call that triggered this result
        for msg in reversed(messages[:-1]):
            if hasattr(msg, "tool_calls"):
                for tc in msg.tool_calls:
                    if tc["id"] == last_tool_call_id:
                        last_submitted_answer = tc["args"].get("answer")
                        break
            if last_submitted_answer:
                break

        try:
            data = json.loads(messages[-1].content)
            is_correct = data.get("correct")
            next_url = data.get("url")

            # ---------------------------------------------------------
            # SCENARIO 1: ANSWER IS CORRECT
            # ---------------------------------------------------------
            if is_correct:
                if next_url:
                    logger.info(f"🚀 Success! Moving to next: {next_url}")
                    return {
                        "messages": [
                            HumanMessage(content=f"Level Complete. New URL: {next_url}")
                        ],
                        "task_start_time": current_time,
                        "attempts": 0,
                    }
                else:
                    logger.info("🏁 Success, but URL is NULL (End of Quiz). Stopping.")
                    return {"messages": [HumanMessage(content="END")], "attempts": 0}

            # ---------------------------------------------------------
            # SCENARIO 2: ANSWER IS WRONG
            # ---------------------------------------------------------
            else:
                # A. CHECK IF THIS WAS A "SKIP" ATTEMPT
                if last_submitted_answer == "SKIP":
                    # Case 1: Skip worked and gave us a new URL -> MOVE
                    if next_url and next_url != "UNKNOWN_URL":
                        logger.info(f"⏭️ Skip Accepted. Moving to next: {next_url}")
                        return {
                            "messages": [
                                HumanMessage(
                                    content=f"Skipped Level. New URL: {next_url}"
                                )
                            ],
                            "task_start_time": current_time,
                            "attempts": 0,
                        }

                    # Case 2: Skip returned NULL URL -> END QUIZ (Your specific request)
                    elif next_url is None:
                        logger.info("🏁 Skip returned NULL URL. Game Over. Stopping.")
                        return {
                            "messages": [HumanMessage(content="END")],
                            "attempts": 0,
                        }

                    # Case 3: Skip failed (correct: false) -> END (Prevent Infinite Loop)
                    else:
                        logger.error(
                            "🛑 Skip Rejected by Server. Terminating to prevent infinite loop."
                        )
                        return {
                            "messages": [HumanMessage(content="END")],
                            "attempts": 0,
                        }

                # B. CHECK TIMEOUT (Safety Valve for the "Death Spiral")
                if (current_time - task_start_time) > 180:
                    logger.error("🛑 Timeout exceeded during failure. Terminating.")
                    return {"messages": [HumanMessage(content="END")], "attempts": 0}

                # C. NORMAL RETRY (Attempts 1 & 2)
                current_attempts += 1
                logger.warning(
                    f"❌ Answer Incorrect (Attempt #{current_attempts}). Sleeping 2s..."
                )
                time.sleep(2)  # Prevent rapid API hammering

        except Exception as e:
            current_attempts += 1
            logger.error(f"⚠️ Error parsing response: {e}")
            time.sleep(2)  # Safety sleep

    # 2. DYNAMIC URL HUNTING
    task_url = "UNKNOWN_URL"
    for msg in reversed(messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call["name"] == "get_rendered_html":
                    task_url = tool_call["args"].get("url")
                    break
        if task_url != "UNKNOWN_URL":
            break

    # 3. FORCE SKIP LOGIC
    force_skip_message = None

    # Check for Loop
    if (
        messages
        and isinstance(messages[-1], HumanMessage)
        and "SYSTEM: You have failed" in str(messages[-1].content)
    ):
        logger.error("🛑 STUCK IN SKIP LOOP. Terminating.")
        return {"messages": [HumanMessage(content="END")], "attempts": 0}

    if current_attempts >= 3:
        logger.warning("🛑 3 Failed Attempts. Forcing SKIP.")
        force_skip_message = (
            f"SYSTEM: You have failed {current_attempts} times. STOP TRYING.\n"
            f"Submit the SKIP payload immediately.\n"
            f"CRITICAL: The 'url' field in JSON must be '{task_url}'\n"
            f"PAYLOAD: {{ 'answer': 'SKIP', 'email': '{EMAIL}', 'secret': '{SECRET}', 'url': '{task_url}' }}"
        )

    elif (current_time - task_start_time) > 175:
        logger.warning("⏰ TIMEOUT. Forcing SKIP.")
        force_skip_message = (
            f"SYSTEM: Timeout Reached. SKIP to next.\n"
            f"CRITICAL: The 'url' field in JSON must be '{task_url}'\n"
            f"PAYLOAD: {{ 'answer': 'SKIP', 'email': '{EMAIL}', 'secret': '{SECRET}', 'url': '{task_url}' }}"
        )

    # 4. EXECUTION
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break
    valid_history = (
        messages[last_human_idx:] if last_human_idx != -1 else messages[-10:]
    )

    if force_skip_message:
        valid_history.append(HumanMessage(content=force_skip_message))
        # Reset attempts effectively to avoid double-triggering,
        # but ensure we track the skip status in the next loop.
        current_attempts = 0

    logger.info(f"🧠 Thinking... (Attempts: {current_attempts})")
    result = llm_with_prompt.invoke({"messages": valid_history})
    time.sleep(1.5)

    return {"messages": [result], "attempts": current_attempts}


def route(state):
    """
    Determines the next step in the graph.
    1. If the Agent/System said "END", stop the graph.
    2. If the Agent called a tool, go to 'tools'.
    3. Otherwise, go back to 'agent' (loop).
    """
    messages = state["messages"]
    last_message = messages[-1]

    # 1. CHECK FOR STOP SIGNAL (From LLM or System)
    # We strip whitespace to ensure "END " or " END" triggers it.
    if isinstance(last_message.content, str) and last_message.content.strip() == "END":
        logger.info("🛑 Stop Signal Detected. Terminating Graph.")
        return END

    # 2. CHECK FOR TOOL CALLS
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # 3. DEFAULT LOOP
    return "agent"


graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_conditional_edges("agent", route)
app = graph.compile()


def run_agent(url: str):
    try:
        app.invoke(
            {
                "messages": [HumanMessage(content=f"Solve this: {url}")],
                "task_start_time": time.time(),
            },
            config={"recursion_limit": RECURSION_LIMIT},
        )
    except GraphRecursionError:
        logger.error("🛑 CRITICAL: Recursion Limit Reached. Agent stuck in a loop.")
        # Optional: You could insert logic here to force-post a skip via raw requests if needed
    except Exception as e:
        logger.error(f"💥 Application Error: {e}")
