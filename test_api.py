from dotenv import load_dotenv
import os
import sys
from google import genai

# Load from current directory
load_dotenv()

api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    print("ERROR: No API key in .env")
    sys.exit(1)

print(f"Testing with API key: {api_key[:20]}...")

client = genai.Client(api_key=api_key)

# Try different model name formats
models_to_try = [
    "gemini-pro",
    "models/gemini-pro",
    "gemini-pro-vision",
    "models/gemini-pro-vision",
    "gemini-2.0-flash",
    "models/gemini-2.0-flash",
]

print("\nTesting models:")
for model in models_to_try:
    try:
        response = client.models.generate_content(
            model=model,
            contents="test"
        )
        print(f"✓ {model} WORKS!")
        break
    except Exception as e:
        error_msg = str(e)[:100]
        print(f"✗ {model}: {error_msg}")

