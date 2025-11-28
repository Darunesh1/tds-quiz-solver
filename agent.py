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


SYSTEM_PROMPT = f"""You are an autonomous CTF quiz solver. 

Your goal: Extract questions from webpages, solve them, and submit answers.

Available tools:
- get_rendered_html(url): Scrapes page content and lists all asset URLs
- download_file(url): Downloads files to LLMFiles/ directory
- run_code(code): Executes Python (runs inside LLMFiles/)
- ocr_image(path): Extracts text from images  
- transcribe_audio(path): Converts audio to text
- add_dependencies(packages): Installs Python packages
- post_request(url, payload): Submits JSON payloads

Credentials: EMAIL="{EMAIL}", SECRET="{SECRET}"

Strategy:
1. Analyze the page structure to identify where the question/data is located
2. Use appropriate tools to gather and process information
3. Compute answers programmatically (never guess)
4. Submit via post_request with: {{"answer": "...", "email": "{EMAIL}", "secret": "{SECRET}"}}

If a task seems unsolvable after multiple attempts max try 3 times, submit {{"answer": "SKIP", ...}} and move to solve the next question

The server responds with {{"correct": bool, "url": str}}. On success, proceed to the next URL.
only when the url is null output exactly 'END'
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
    """Simplified agent that lets LLM handle strategy."""
    messages = state["messages"]

    # Only inject system context, don't force decisions
    enriched_messages = messages.copy()

    # Provide metadata as context, not commands
    elapsed = time.time() - state.get("task_start_time", time.time())
    if elapsed > 150:
        enriched_messages.append(
            HumanMessage(content=f"Note: {elapsed:.0f}s elapsed on this task.")
        )

    result = llm_with_prompt.invoke({"messages": enriched_messages})
    return {"messages": [result]}


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
