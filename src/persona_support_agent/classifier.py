import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class PersonaClassifier:
    PERSONA_OPTIONS = ["Technical Expert", "Frustrated User", "Business Executive"]

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GENAI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)

    def _heuristic_classify(self, user_message: str, reason: str) -> dict:
        message = user_message.lower()

        frustrated_terms = [
            "tried everything",
            "nothing works",
            "hour",
            "stuck",
            "broken",
            "frustrated",
            "angry",
            "demand",
        ]
        executive_terms = [
            "business impact",
            "revenue",
            "uptime",
            "sla",
            "timeline",
            "operations",
            "executive",
        ]
        technical_terms = [
            "api",
            "header",
            "bearer",
            "token",
            "endpoint",
            "database",
            "integration",
            "ssl",
            "tls",
            "error code",
        ]

        scores = {
            "Frustrated User": sum(term in message for term in frustrated_terms),
            "Business Executive": sum(term in message for term in executive_terms),
            "Technical Expert": sum(term in message for term in technical_terms),
        }
        persona = max(scores, key=scores.get)
        confidence = 0.55 if scores[persona] else 0.35

        return {
            "persona": persona,
            "confidence": confidence,
            "reasoning": f"{reason}. Used local keyword fallback.",
        }

    def classify(self, user_message: str) -> dict:
        system_instruction = (
            "You are a customer support persona classifier. Analyze the user's message and return one of three personas: "
            "Technical Expert, Frustrated User, or Business Executive. Respond only in JSON with persona, confidence, and reasoning."
        )

        response_schema = {
            "type": "OBJECT",
            "properties": {
                "persona": {
                    "type": "STRING",
                    "enum": self.PERSONA_OPTIONS,
                },
                "confidence": {"type": "NUMBER"},
                "reasoning": {"type": "STRING"},
            },
            "required": ["persona", "confidence", "reasoning"],
        }

        try:
            response = self.client.models.generate_content(
                model=os.environ.get("GEMINI_MODEL_CLASSIFIER", "gemini-2.0-flash"),
                contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.0,
            ),
        )

            text = getattr(response, "text", None) or response.text
        except Exception as e:
            return self._heuristic_classify(user_message, f"Classifier call failed: {type(e).__name__}")

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "persona": "Technical Expert",
                "confidence": 0.45,
                "reasoning": "Could not parse classifier output, defaulting to Technical Expert persona.",
            }
