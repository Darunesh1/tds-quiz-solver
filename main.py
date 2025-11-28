import asyncio
import logging
import os
import time

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent import runagent
from tools.send_request import postrequest

load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")

# Configure Logging - set DEBUG to get more verbose logs during dev
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DUMMY_SUBMISSION_DEADLINE = 175  # seconds

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()


@app.get("/healthz")
def healthz():
    return {"status": "ok", "uptime_seconds": int(time.time() - START_TIME)}


async def runagent_with_timeout(url: str, secret: str):
    loop = asyncio.get_running_loop()
    agent_future = loop.run_in_executor(None, runagent, url)
    try:
        logger.debug(
            f"⏳ Monitoring agent... Timeout set to {DUMMY_SUBMISSION_DEADLINE}s"
        )
        await asyncio.wait_for(agent_future, timeout=DUMMY_SUBMISSION_DEADLINE)
        logger.info("✅ Agent completed within time limit.")
    except asyncio.TimeoutError:
        logger.warning(
            f"⌛ TIMEOUT REACHED {DUMMY_SUBMISSION_DEADLINE}s. Forcibly stopping agent and sending dummy submission."
        )
        dummy_payload = {
            "url": url,
            "answer": 0,  # Usually 0 or safe partial fail answer
            "secret": secret,
            "notes": "Time limit exceeded, forced submission.",
        }
        try:
            # Use the postrequest tool directly to send a dummy submission to get some partial credit
            postrequest.invoke(url, dummy_payload)
            logger.info("Dummy submission sent successfully.")
        except Exception as e:
            logger.error(f"Failed to send dummy submission: {e}")
    except Exception as e:
        logger.error(f"Error during agent execution: {e}")


@app.post("/solve")
async def solve(request: Request, backgroundtasks: BackgroundTasks):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    url = data.get("url")
    secret = data.get("secret")

    if not url or not secret:
        raise HTTPException(status_code=400, detail="Missing url or secret")
    if secret != SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    logger.info(f"🚀 Received task for URL: {url}")

    # Run the agent in background, allowing immediate response with status ok
    backgroundtasks.add_task(runagent_with_timeout, url, secret)

    return JSONResponse(status_code=200, content={"status": "ok"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
