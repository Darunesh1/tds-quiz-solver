import json
import logging
import os
from typing import Annotated, Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# Imports match your tools/__init__.py exactly
from tools import (
    adddependencies,
    downloadfile,
    getrenderedhtml,
    listdependencies,
    postrequest,
    runcode,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger("agent")

load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")
RECURSION_LIMIT = 50


class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]


class SafeToolNode:
    """Custom ToolNode that handles StructuredTool objects safely."""

    def __init__(self, tools: List[Any]):
        self.tools = {}
        for tool in tools:
            if hasattr(tool, "name"):
                self.tools[tool.name] = tool
            else:
                toolname = getattr(tool, "__name__", "unknown")
                self.tools[toolname] = tool
        logger.info(f"🔧 Loaded tools: {list(self.tools.keys())}")

    async def __call__(self, state: AgentState) -> Dict[str, List[Any]]:
        messages = state["messages"]
        last_message = messages[-1]

        tool_calls = getattr(last_message, "tool_calls", [])
        if not tool_calls:
            return {"messages": messages}

        tool_messages = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            logger.info(f"🚀 Calling tool: {tool_name}")

            if tool_name in self.tools:
                try:
                    tool = self.tools[tool_name]
                    # Handle both LangChain StructuredTools and raw functions
                    result = (
                        tool.invoke(tool_args)
                        if hasattr(tool, "invoke")
                        else tool(tool_args)
                    )

                    result_str = str(result)
                    preview = (
                        result_str[:200] + "..."
                        if len(result_str) > 200
                        else result_str
                    )
                    logger.info(f"🔍 TOOL RESULT: {preview}")

                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": result_str,
                        }
                    )
                except Exception as e:
                    error_msg = f"Tool failed: {str(e)}"
                    logger.error(f"❌ {error_msg}")
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": error_msg,
                        }
                    )
            else:
                msg = (
                    f"Tool {tool_name} not found. Available: {list(self.tools.keys())}"
                )
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": msg,
                    }
                )
                logger.warning(f"⚠️ {msg}")

        return {"messages": messages + tool_messages}


# Initialize LLM
provider = os.getenv("LLM_PROVIDER", "google_genai")

if provider == "aipipe":
    logger.info("🚀 INITIALIZING: Using AI Pipe")
    base_llm = init_chat_model(
        model="google/gemini-2.5-flash",
        model_provider="openai",
        api_key=os.getenv("AIPIPE_APIKEY"),
        base_url="https://aipipe.org/openrouter/v1",
        temperature=0,
    )
else:
    logger.info("🚀 INITIALIZING: Using Direct Google GenAI")
    base_llm = init_chat_model(
        model="gemini-2.5-flash",
        model_provider="google_genai",
        temperature=0,
    )

# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = f"""You are an autonomous quiz-solving agent for TDS Project 2.

Credentials: Email={EMAIL}, Secret={SECRET}

TOOLS:
- getrenderedhtml(url): Fetch HTML.
- listdependencies(): List installed packages.
- adddependencies(packages): Install packages.
- downloadfile(url, filename): Download to LLMFiles/.
- runcode(code): Execute Python.
- postrequest(url, payload): Submit answers.

PROCEDURE:
1. **Analyze**: Use `getrenderedhtml` on the current URL. Find the Task.
2. **Execute**: Use tools to solve it. (Download files, Run code).
3. **Submit**: Use `postrequest` to send the answer.
   Payload keys: "url", "answer", "secret".

LOOP LOGIC (CRITICAL):
- The `postrequest` tool returns a JSON response.
- **IF response contains "url":** You MUST immediately use `getrenderedhtml` on that NEW url. Do NOT stop.
- **IF response has NO "url" AND correct=True:** You are finished. Output "FINISHED_ALL_TASKS".
- **IF response has correct=False:** Analyze why, fix code, and RETRY.

RULES:
- Only output "FINISHED_ALL_TASKS" when the server response explicitly stops giving new URLs.
- Do not chat. Just call tools.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# -------------------------------------------------
# TOOL BINDING FIX
# -------------------------------------------------
safetools = [
    runcode,
    getrenderedhtml,
    downloadfile,
    postrequest,
    adddependencies,
    listdependencies,
]

# FIX: Manually convert tools to OpenAI-format dictionaries first.
# This passes the underlying function (t.func) instead of the StructuredTool object
# to bypass the Python 3.12 inspect.signature error.
try:
    logger.info("🛠️ Binding tools (with safe conversion)...")
    tool_schemas = [
        convert_to_openai_tool(t.func if hasattr(t, "func") else t) for t in safetools
    ]

    # Bind using the schemas
    llm_with_tools = base_llm.bind_tools(tool_schemas)
    logger.info("✅ Tools bound successfully.")
except Exception as e:
    logger.error(f"⚠️ Tool binding failed: {e}. Attempting fallback...")
    # Fallback: Try binding directly (if the specific provider supports it natively)
    llm_with_tools = base_llm.bind_tools(safetools)

llm_with_prompt = prompt | llm_with_tools


def agent_node(state: AgentState) -> Dict[str, List[Any]]:
    logger.info("🤖 Agent processing...")
    result = llm_with_prompt.invoke(state["messages"])

    if result.content:
        logger.info(f"💭 Agent Thought: {result.content}")

    return {"messages": [result]}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]

    # 1. If the LLM wants to call tools, let it.
    tool_calls = getattr(last_message, "tool_calls", [])
    if tool_calls:
        return "tools"

    # 2. STRICT TERMINATION CHECK
    content = str(last_message.content).upper()
    if "FINISHED_ALL_TASKS" in content:
        logger.info("🛑 Termination signal received. Stopping.")
        return END

    # 3. Fallback: If agent is chatting but not calling tools, force it to loop.
    logger.info("🔄 Agent didn't call tools or finish. Looping back.")
    return "agent"


# Build graph
tool_node = SafeToolNode(safetools)

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_conditional_edges(
    "agent", should_continue, {"tools": "tools", "agent": "agent", END: END}
)

app = graph.compile()


async def run_agent(url: str) -> None:
    logger.info(f"🏁 Starting Agent on: {url}")
    try:
        # 2. Use the asynchronous method: await app.ainvoke
        result = await app.ainvoke(
            {"messages": [{"role": "user", "content": f"Start the quiz at: {url}"}]},
            config={"recursion_limit": RECURSION_LIMIT},
        )
        logger.info("🎉 Agent execution finished successfully.")
    except Exception as e:
        logger.error(f"💥 Agent crashed: {e}", exc_info=True)


# Export for main.py import
runagent = run_agent
