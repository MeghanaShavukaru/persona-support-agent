import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    print("ERROR: No API key found in .env")
else:
    c = genai.Client(api_key=api_key)

    # Test common model names
    models_to_test = [
        "models/gemini-pro",
        "models/gemini-1.5-pro",
        "models/gemini-1.5-flash",
        "gemini-pro",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "models/gemini-2.0-flash",
        "gemini-2.0-flash"
    ]

    print("Testing available models:")
    for model_name in models_to_test:
        try:
            response = c.models.generate_content(
                model=model_name,
                contents="test"
            )
            print(f"  ✓ {model_name} - WORKS")
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                print(f"  ✗ {model_name} - not available")
            else:
                print(f"  ? {model_name} - error: {type(e).__name__}")

