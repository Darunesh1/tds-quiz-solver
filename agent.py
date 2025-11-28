import logging
import os
import sys
import time
from typing import Annotated, Any, List, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, RemoveMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_ollama import ChatOllama  # ← CRITICAL IMPORT
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import (
    add_dependencies,
    download_file,
    get_rendered_html,
    ocr_image,
    post_request,
    run_code,
    transcribe_audio,
)

load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")
RECURSION_LIMIT = 5000


# CONFIGURE GLOBAL LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("QuizAgent")


# -------------------------------------------------
# STATE
# -------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


TOOLS = [
    run_code,
    get_rendered_html,
    download_file,
    post_request,
    add_dependencies,
    transcribe_audio,
    ocr_image,
]


# -------------------------------------------------
# GEMINI LLM
# -------------------------------------------------
rate_limiter = InMemoryRateLimiter(
    requests_per_second=9 / 60, check_every_n_seconds=1, max_bucket_size=9
)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google_genai")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")  # Default model name
AI_PIPE_API_KEY = os.getenv("AI_PIPE_API_KEY")

if LLM_PROVIDER == "ollama":
    # Ollama setup (Run locally)
    # Ensure 'ollama serve' is running and model is pulled (e.g., 'ollama pull qwen2.5-coder')
    llm = init_chat_model(
        model_provider="ollama",
        model=LLM_MODEL,  # e.g., "llama3.2", "qwen2.5-coder"
        base_url="http://localhost:11434",  # Default Ollama port
    ).bind_tools(TOOLS)

elif LLM_PROVIDER == "aipipe":
    # AI Pipe setup
    llm = init_chat_model(
        model_provider="aipipe",
        api_key=AI_PIPE_API_KEY,
        model=LLM_MODEL,
        rate_limiter=rate_limiter,
    ).bind_tools(TOOLS)

else:
    # Google GenAI setup (Default)
    llm = init_chat_model(
        model_provider="google_genai",
        model=LLM_MODEL,
        rate_limiter=rate_limiter,
    ).bind_tools(TOOLS)


SYSTEM_PROMPT = f"""
<role>
You are an elite Autonomous Quiz-Solving Agent. Your goal is to solve multi-stage CTF (Capture The Flag) style challenges programmatically. You operate in a loop: Scrape -> Analyze -> Execute -> Submit -> Transition.
</role>

<credentials>
EMAIL: "{EMAIL}"
SECRET: "{SECRET}"
</credentials>

<tools>
You have access to the following python-defined tools. Do not hallucinate new tools.
1. `get_rendered_html(url)`: Fetches the DOM (handles dynamic JS). Returns HTML string.
2. `download_file(url, save_path)`: Downloads assets (images, audio, CSVs).
3. `ocr_image(image_path)`: Returns text extracted from an image.
4. `transcribe_audio(audio_path)`: Returns text from an audio file.
5. `run_code(script)`: Executes Python code in a sandboxed environment. Use this for math, dataframes, and logic.
   - *CRITICAL*: Print the final result to `stdout` so you can read it.
   - *DEPENDENCIES*: If a library is missing, install it via `subprocess`.
6. `post_request(url, payload)`: Sends a JSON POST request.
   - Payload format: {{"answer": "YOUR_ANSWER", "email": "{EMAIL}", "secret": "{SECRET}"}}
7. `reset_memory(instruction)`: Clears context. Call ONLY after specific success criteria.
</tools>

<protocol>
### PHASE 1: INGESTION
1. Call `get_rendered_html` on the current URL.
2. Parse the HTML to find:
   - The question/task.
   - The data source (image, audio, text, or file URL).
   - The submission endpoint (usually `/submit` or similar).

### PHASE 2: ANALYSIS & STRATEGY
1. Determine the domain: **Math**, **Vision**, **Audio**, or **Data Mining**.
2. **Chain of Thought**: Before calling tools, explicitly state your plan in `<thought>` tags.
3. *NEVER GUESS*. If you need to count pixels, sum a CSV column, or decode base64, YOU MUST WRITE CODE using `run_code`.

### PHASE 3: EXECUTION
- **Vision**: Download -> `ocr_image`.
- **Audio**: Download -> `transcribe_audio`.
- **Logic/Math**: Write a Python script to calculate the exact answer.
- **Data**: Load into pandas, process, and print the result.

### PHASE 4: SUBMISSION
1. Construct the JSON payload with the credentials provided above.
2. Call `post_request` to the endpoint found in Phase 1.

### PHASE 5: EVALUATION & TRANSITION
- **IF RESPONSE IS `correct: true`**:
  - Extract the `next_url` from the response.
  - Call `reset_memory` with the argument: "Previous level solved. Starting new level at {{next_url}}".
- **IF RESPONSE IS `correct: false`**:
  - **DO NOT** reset memory.
  - Enter **DEBUG MODE**:
    1. Re-read instructions (did you miss a sorting order or formatting rule?).
    2. Print intermediate variables in `run_code`.
    3. Try a radically different approach.
    4. If 5 consecutive failures occur, submit answer: "SKIP_TIMEOUT".
</protocol>

<constraints>
1. **Output Format**: For simple answers, just output the answer. For complex reasoning, use `<thought>` tags first.
2. **Code Safety**: Do not rely on hardcoded values. Extract values dynamically from the HTML/Files.
3. **Termination**: If the server returns no new URL after a correct submission, output exactly: `END`.
</constraints>

<current_state>
You are starting a new session.
</current_state>
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), MessagesPlaceholder(variable_name="messages")]
)

llm_with_prompt = prompt | llm


# -------------------------------------------------
# AGENT NODE
# -------------------------------------------------


def agent_node(state: AgentState):
    messages = state["messages"]

    # ------------------------------------------------------------------
    # 1. HARD RESET LOGIC: Physically delete old messages
    # ------------------------------------------------------------------
    if (
        messages
        and isinstance(messages[-1], ToolMessage)
        and "__RESET__:" in messages[-1].content
    ):
        # Extract instruction
        next_task_instruction = messages[-1].content.split("__RESET__:", 1)[1]

        logger.info(f"🧹 MEMORY WIPE TRIGGERED. Deleting {len(messages)} messages...")

        # Create a list of RemoveMessage operations for every message in history
        # Note: LangGraph assigns IDs to messages automatically during execution.
        delete_operations = [RemoveMessage(id=m.id) for m in messages if m.id]

        # Create the new starting message
        new_task_message = HumanMessage(content=f"New Task: {next_task_instruction}")

        # Return the deletions AND the new message
        # This clears the state and sets the new prompt in one go.
        return {"messages": delete_operations + [new_task_message]}

    # ------------------------------------------------------------------
    # 2. HISTORY FILTERING (Backup Safety)
    # ------------------------------------------------------------------
    # Even with Hard Reset, we keep this to ensure the current run is clean
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break

    if last_human_idx != -1:
        valid_history = messages[last_human_idx:]
    else:
        valid_history = messages[-10:]

    # ------------------------------------------------------------------
    # 3. EXECUTION
    # ------------------------------------------------------------------
    # time.sleep(5)  # Uncomment if rate limits are tight

    # Call LLM with the CLEAN history
    result = llm_with_prompt.invoke({"messages": valid_history})

    return {"messages": [result]}


# -------------------------------------------------
# GRAPH
# -------------------------------------------------
def route(state):
    last = state["messages"][-1]
    # support both objects (with attributes) and plain dicts
    tool_calls = None
    if hasattr(last, "tool_calls"):
        tool_calls = getattr(last, "tool_calls", None)
    elif isinstance(last, dict):
        tool_calls = last.get("tool_calls")

    if tool_calls:
        return "tools"
    # get content robustly
    content = None
    if hasattr(last, "content"):
        content = getattr(last, "content", None)
    elif isinstance(last, dict):
        content = last.get("content")

    if isinstance(content, str) and content.strip() == "END":
        return END
    if isinstance(content, list) and content[0].get("text").strip() == "END":
        return END
    return "agent"


graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))


graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_conditional_edges("agent", route)

app = graph.compile()


# -------------------------------------------------
# TEST
# -------------------------------------------------
def run_agent(url: str) -> str:
    app.invoke(
        {"messages": [{"role": "user", "content": url}]},
        config={"recursion_limit": RECURSION_LIMIT},
    )
    print("Tasks completed succesfully")
