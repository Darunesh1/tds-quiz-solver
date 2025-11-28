import logging
import os
import sys
from typing import Annotated, Any, List, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
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


# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent that solves tasks using tools and code.

CORE WORKFLOW:
1. Scrape the quiz page → extract instructions and endpoint.
2. Analyze what's needed (vision, audio, data, math).
3. Use existing tools OR write code to solve.
4. Submit answer → follow next URL or END.

CRITICAL RULES:

TOOL USAGE:
- Use existing tools when available: get_rendered_html, download_file, post_request, ocr_image, transcribe_audio.
- If a tool doesn't exist for the task → IMMEDIATELY use run_code to create solution.
- NEVER guess answers - always process data programmatically.

SELF-HEALING APPROACH:
When you encounter a task type you can't handle:
1. Identify what's missing (e.g., screenshot capability, image processing).
2. Use run_code to write Python code that solves it.
3. Execute the code and use the output.

TASK-SPECIFIC HANDLING:

VISUAL TASKS (images, canvas, screenshots):
- If you see <img>, <canvas>, or visual elements:
  1. Write code to capture/download the image.
  2. Write code using pytesseract or PIL to extract text via OCR.
  3. Use extracted text as answer.
- NEVER submit guesses for visual content.

AUDIO TASKS:
- Download audio file.
- Use transcribe_audio tool OR write code with speech_recognition.
- Extract and process the spoken content.

DATA/MATH TASKS:
- ALWAYS use run_code for calculations, analysis, transformations.
- Fetch API data, parse JSON, compute results via code.
- NEVER compute manually.

CODE WRITING GUIDELINES:
- Install missing libraries first: run_code("import subprocess; subprocess.run(['uv', 'add ', 'library_name'])")
- Write complete, executable code.
- Handle errors gracefully.
- Print results clearly.

EXAMPLE: Visual Quiz Without Screenshot Tool
Step 1: Write code to screenshot and OCR
run_code('''
from playwright.sync_api import sync_playwright
import pytesseract
from PIL import Image

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("QUIZ_URL")
    page.screenshot(path="LLMFiles/quiz.png")
    browser.close()

text = pytesseract.image_to_string(Image.open("LLMFiles/quiz.png"))
print(text.strip())
''')
Step 2: Use output as answer
    
TIME MANAGEMENT & SKIPPING RULES (CRITICAL):
1. CHECK DELAY: Every server response has a "delay" field (seconds elapsed).
2. IF DELAY > 175s (2.55 minutes):
   - STOP trying to solve the current task.
   - IMMEDIATELY submit a "SKIP" or partial answer to the endpoint.
   - The goal is to get the NEXT URL before the 3-minute hard limit kills the session.
   
3. IF ANSWER IS WRONG 3 TIMES:
   - STOP retrying.
   - Submit a final attempt or move to next step if server allows.

SKIP STRATEGY:
- If you are stuck or time is running out (>175s):
  - Submit a JSON payload with: answer="partial answer you generated", email="{EMAIL}", secret="{SECRET}"
  - This forces a response. If response has new URL, GO TO IT IMMEDIATELY.
  - Do not waste time perfecting one answer if it risks the whole session.

YOUR PRIORITY:
1. Solve correctly (if < 175s delay)
2. Skip/Submit anyway (if > 175s delay)
3. Get next URL -> Keep session alive

RESPONSE FORMAT:
- Continue until server response has no new URL.
- Then respond with: END

CREDENTIALS:
- Email: {EMAIL}
- Secret: {SECRET}
- ENDPOINT VERIFICATION**: 
- Look for <form action="..."> or specific text like "Submit to: /submit/X".
- NEVER submit to .json, .css, .js files. 
- The submission URL usually contains "submit".
- If you get a 404, STOP and re-read the page instructions to find the real endpoint.
IF YOU GET A 404 ERROR:
1. You are sending to the WRONG URL.
2. Re-read the HTML content using get_rendered_html.
3. Look for the text "submit" or "endpoint".
4. Try the correct URL (e.g., change /q1.json to /submit/1).



REMEMBER: Code solves everything. If unsure, write code. Never guess visual/audio content.
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

    # FORCE RESET: Since each task is independent, we only need:
    # 1. The very first message (User URL/Instruction) - actually, usually just the LAST user instruction.
    # But wait, if we are in a loop of steps for ONE task, we need context of that task.

    # If we truly want independence per task, we rely on the System Prompt + Current Input.
    # However, LangGraph accumulates all messages in `state["messages"]`.

    # ROBUST STRATEGY:
    # Find the LAST HumanMessage. That is the start of the CURRENT task.
    # Discard everything before it.

    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].type == "human":
            last_human_idx = i
            break

    if last_human_idx != -1:
        # Keep ONLY from the last human message onwards.
        # This includes the current instructions and any tool steps taken SO FAR for this specific task.
        # It discards all previous solved tasks.
        valid_history = messages[last_human_idx:]
    else:
        # Fallback (shouldn't happen if started correctly)
        valid_history = messages[-1:]

    # Log what we are sending
    logger.info(f"🧠 Agent thinking... (Context size: {len(valid_history)} messages)")

    # Call LLM with only the current task's history
    result = llm_with_prompt.invoke({"messages": valid_history})
    # LOG THE DECISION
    if result.tool_calls:
        tools_called = [t["name"] for t in result.tool_calls]
        logger.info(f"🤖 AI decided to call tools: {tools_called}")
    else:
        logger.info(
            f"🤖 AI Response: {result.content[:200]}..."
        )  # Truncate long thoughts

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
