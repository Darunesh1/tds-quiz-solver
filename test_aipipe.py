import os

from dotenv import load_dotenv
from openai import OpenAI

# Load variables from .env file
load_dotenv()

# --- CONFIGURATION ---
# Retrieve the key from the environment variable we defined earlier
API_KEY = os.getenv("AI_PIPE_API_KEY")
BASE_URL = "https://aipipe.org/openrouter/v1"  #

# Simple check to ensure the key loaded
if not API_KEY:
    print("❌ Error: AI_PIPE_API_KEY not found. Please check your .env file.")
    exit(1)

try:
    print(f"🔌 Connecting to AI Pipe at {BASE_URL}...")

    # Initialize the client pointing to AI Pipe
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Simple test request
    response = client.chat.completions.create(
        model="google/gemini-2.5-flash",
        messages=[
            {
                "role": "user",
                "content": "If you are working, reply with 'Connection Successful!'",
            }
        ],
    )

    # Print the result
    print("\n✅ SUCCESS!")
    print(f"Model Used: {response.model}")
    print(f"Response: {response.choices[0].message.content}")

except Exception as e:
    print("\n❌ CONNECTION FAILED")
    print(f"Error: {e}")
