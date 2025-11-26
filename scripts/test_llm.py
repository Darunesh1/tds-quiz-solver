"""
Standalone LLM client test - NO FastAPI server needed!
"""

import asyncio
import sys
from pathlib import Path

# Add project root to Python path (BEFORE importing app)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Now imports will work
from app.config import settings
from app.llm import llm_client


async def test_provider():
    print("=" * 60)
    print("🧪 Testing LLM Client (Direct Module Test)")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Primary provider: {settings.llm_provider}")
    print(f"Fallback enabled: {settings.llm_fallback_enabled}")
    print(f"Gemini available: {llm_client.gemini_available}")
    print(f"AIpipe available: {llm_client.aipipe_available}")
    print("=" * 60 + "\n")

    try:
        print("📤 Sending test prompt...")
        response = await llm_client.generate(
            prompt="Say 'Hello! I am working correctly.' in one sentence.",
            system="You are a helpful assistant.",
        )
        print(f"✅ SUCCESS!\n")
        print(f"📥 Response:\n{response}\n")
        print("=" * 60)

    except Exception as e:
        print(f"❌ ERROR: {e}\n")
        print("=" * 60)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print(f"\n🚀 Starting test...\n")
    asyncio.run(test_provider())
