"""Test foundation files."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.logger import logger, setup_logger
from app.models import SolveRequest, SolveResponse
from app.timer import QuestionTimer
from app.utils.exceptions import QuizSolverError, TimeoutError
from app.utils.helpers import format_json, safe_filename

print("✅ All foundation imports successful!\n")

# Test logger
test_logger = setup_logger("test")
test_logger.info("Logger working!")

# Test timer
timer = QuestionTimer()
timer.start()
print(f"\n⏱️  Timer status: {timer.get_status()}")

# Test models
request = SolveRequest(
    email="test@example.com",
    secret="your-secret-hee",
    url="https://tds-llm-analysis.s-anand.net/demo",
)
print(f"\n📋 Request model: {request.model_dump_json(indent=2)}")

# Test helpers
filename = safe_filename("https://example.com/data.csv")
print(f"\n📄 Safe filename: {filename}")

print("\n✅ All foundation tests passed!")
