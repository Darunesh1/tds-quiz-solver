import os
from typing import Annotated, Any, List, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.rate_limiters import InMemoryRateLimiter
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
AI_PIPE_API_KEY = os.getenv("AI_PIPE_API_KEY")

if LLM_PROVIDER == "aipipe":
    llm = init_chat_model(
        model_provider="aipipe",
        api_key=AI_PIPE_API_KEY,
        model="gemini-2.5-flash",
        rate_limiter=rate_limiter,
    ).bind_tools(TOOLS)
else:
    llm = init_chat_model(
        model_provider="google_genai",
        model="gemini-2.5-flash",
        rate_limiter=rate_limiter,
    ).bind_tools(TOOLS)


# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------

SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent.

Your job is to:
1. Load the quiz page from the given URL.
2. Extract ALL instructions, required parameters, submission rules, and the submit endpoint.
3. Solve the task exactly as required.
4. Submit the answer ONLY to the endpoint specified on the current page (never make up URLs).
5. Read the server response and:
   - If it contains a new quiz URL → fetch it immediately and continue.
   - If no new URL is present → return "END".

STRICT RULES — FOLLOW EXACTLY:

GENERAL RULES:
- NEVER stop early. Continue solving tasks until no new URL is provided.
- NEVER hallucinate URLs, endpoints, fields, values, or JSON structure.
- NEVER shorten or modify URLs. Always submit the full URL.
- NEVER re-submit unless the server explicitly allows or it's within the 3-minute limit.
- ALWAYS inspect the server response before deciding what to do next.
- ALWAYS use the tools provided to fetch, scrape, download, render HTML, or send requests.

CODE-FIRST PROCESSING RULES (CRITICAL):
- You MUST ONLY interpret TEXT directly. NEVER interpret images, audio, or visual content directly.
- For ANY calculations, data analysis, transformations, or complex processing → ALWAYS write Python code using run_code.
- For images/visuals: download → write code (OCR libraries like easyocr, pytesseract) → extract text → analyze.
- For audio: download → write code (speechrecognition, pydub) → transcribe → analyze.
- For data processing (CSV, JSON, math, statistics): ALWAYS write code, never compute manually.
- If you need a library that's not installed, use add_dependencies first, then run_code.
- Only use existing tools (get_rendered_html, download_file, post_request, transcribe_audio) when they directly fit the task.
- When in doubt or for non-trivial tasks → write and execute code.

EXAMPLE WORKFLOWS:
- Image with text → download_file → run_code with OCR → extract text → process
- Audio numbers → download_file → run_code with speech recognition → get text → sum
- Complex calculations → run_code with numpy/pandas
- Data transformation → run_code with appropriate libraries

TIME LIMIT RULES:
- Each task has a hard 3-minute limit.
- The server response includes a "delay" field indicating elapsed time.
- If your answer is wrong, retry with a different approach using code.

STOPPING CONDITION:
- Only return "END" when a server response explicitly contains NO new URL.
- DO NOT return END under any other condition.

ADDITIONAL INFORMATION YOU MUST INCLUDE WHEN REQUIRED:
- Email: {EMAIL}
- Secret: {SECRET}

YOUR JOB:
- Follow pages exactly.
- Extract data reliably using tools and code.
- Never guess or directly interpret non-text content.
- Write Python code for all analysis and processing tasks.
- Submit correct answers.
- Continue until no new URL.
- Then respond with: END
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), MessagesPlaceholder(variable_name="messages")]
)

llm_with_prompt = prompt | llm


# -------------------------------------------------
# AGENT NODE
# -------------------------------------------------
def agent_node(state: AgentState):
    result = llm_with_prompt.invoke({"messages": state["messages"]})
    return {"messages": state["messages"] + [result]}


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
