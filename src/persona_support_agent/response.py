import json
import os
from typing import Any


from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


class AdaptiveResponder:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GENAI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)

        # Configurable escalation
        self.retrieval_confidence_threshold = _env_float("RETRIEVAL_CONFIDENCE_THRESHOLD", 0.40)

        raw_keywords = os.environ.get(
            "ESCALATION_KEYWORDS",
            "billing,refund,duplicate charge,account,legal,sensitive,chargeback,stolen",
        )
        self.escalation_keywords = [k.strip().lower() for k in raw_keywords.split(",") if k.strip()]

        # How many conversation turns to summarize for the handoff
        self.history_max_turns = int(os.environ.get("HANDOFF_HISTORY_MAX_TURNS", "8"))

    def get_persona_instructions(self, persona: str) -> str:
        if persona == "Technical Expert":
            return (
                "Respond with a technical, detail-rich explanation. Include root cause analysis, API references, "
                "and step-by-step troubleshooting. Use code-style formatting for commands/headers when helpful."
            )
        if persona == "Frustrated User":
            return (
                "Respond with empathy and concise steps. Validate the user's frustration, keep language simple, avoid "
                "jargon, and provide clear action items. Reassure them that the issue will be resolved quickly."
            )
        if persona == "Business Executive":
            return (
                "Respond concisely and impact-focused. Avoid unnecessary technical detail. Emphasize business outcomes, "
                "estimated resolution time, and next steps."
            )
        return "Respond clearly and helpfully."

    def _build_prompt(self, user_query: str, persona: str, retrieved_chunks: list[dict]) -> str:
        # Provide compact context with provenance so the model can cite.
        context_lines: list[str] = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            src = chunk.get("source", "unknown")
            page = chunk.get("page_number")
            section = chunk.get("section")
            provenance = src
            if page is not None:
                provenance += f" (page {page})"
            elif section:
                provenance += f" ({section})"

            text = chunk.get("text", "")
            context_lines.append(f"[{i}] {provenance}:\n{text}")

        instructions = self.get_persona_instructions(persona)

        return (
            f"You are a customer support assistant. The user has the persona: {persona}.\n"
            "CRITICAL GROUNDING RULES:\n"
            "- You must answer using ONLY the retrieved knowledge base snippets included below.\n"
            "- Every factual claim you make MUST include at least one citation reference in the form [<snippet_id>].\n"
            "- Snippet IDs are the bracketed numbers you see in the context (e.g., [1], [2], ...).\n"
            "- If the retrieved snippets do not contain enough information to answer, output exactly:\n"
            "  ESCALATE_REQUIRED\n"
            "  and nothing else.\n\n"
            f"User persona: {persona}\n"
            f"Persona style instructions: {instructions}\n\n"
            "Retrieved knowledge base snippets:\n"
            + "\n\n".join(context_lines)
            + "\n\n"
            f"User question: {user_query}\n\n"
            "Now write the response."
        )

    def _fallback_escalation_answer(
        self,
        user_query: str,
        persona: str,
        retrieved_chunks: list[dict],
        reason: str,
    ) -> str:
        best_score = max((chunk.get("score", 0.0) for chunk in retrieved_chunks), default=0.0)
        relevant_chunks = [
            chunk for chunk in retrieved_chunks if chunk.get("score", 0.0) >= self.retrieval_confidence_threshold
        ]

        if relevant_chunks:
            context_note = "I found some possibly relevant knowledge base material, but not enough to answer confidently."
        elif retrieved_chunks:
            context_note = (
                "The knowledge base matches are weak, so I do not want to guess from unrelated articles."
            )
        else:
            context_note = "I could not find matching knowledge base material for this request."

        if persona == "Frustrated User":
            opener = "I'm sorry this has been so frustrating."
            next_steps = [
                "I'm escalating this to a human support agent with the details you provided.",
                "Please include what page or workflow is failing, any error text, and whether it happens after refresh or sign-in.",
            ]
        elif persona == "Business Executive":
            opener = "I do not have enough verified context to give a reliable resolution timeline."
            next_steps = [
                "I'm escalating this for account-specific review and impact assessment.",
                "Please include the affected users, business impact, and desired resolution window.",
            ]
        else:
            opener = "I do not have enough verified context to provide a grounded technical fix."
            next_steps = [
                "I'm escalating this so support can review account context, logs, and recent errors.",
                "Please include the endpoint/workflow, timestamp, error message, and steps already attempted.",
            ]

        lines = [
            opener,
            "",
            context_note,
            f"Best retrieval confidence: {best_score:.2f}.",
            f"Model status: {reason}.",
            "",
            "Next steps:",
        ]
        lines.extend(f"- {step}" for step in next_steps)
        return "\n".join(lines)

    def generate(
        self,
        user_query: str,
        persona: str,
        retrieved_chunks: list[dict],
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> dict:
        # Note: conversation_history currently not used in the prompt for the LLM itself.
        # It is used for the human handoff summary.
        prompt = self._build_prompt(user_query, persona, retrieved_chunks)
        try:
            response = self.client.models.generate_content(
                model=os.environ.get("GEMINI_MODEL_GENERATOR", "gemini-2.0-flash"),
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=512,
                ),
            )
        except Exception as e:
            # If Gemini is unavailable (quota/rate limits), fall back to KB-grounded escalation.
            reason = f"{type(e).__name__}: {str(e).splitlines()[0][:180]}"
            return {
                "answer": self._fallback_escalation_answer(
                    user_query,
                    persona,
                    retrieved_chunks,
                    reason,
                ),
                "escalate_required": True,
                "error": reason,
            }
        answer = getattr(response, "text", None) or response.text
        answer = (answer or "").strip()

        if answer == "ESCALATE_REQUIRED":
            return {
                "answer": self._fallback_escalation_answer(
                    user_query,
                    persona,
                    retrieved_chunks,
                    "Retrieved context was insufficient",
                ),
                "escalate_required": True,
            }

        return {"answer": answer, "escalate_required": False}

    def should_escalate(self, user_query: str, persona: str, retrieved_chunks: list[dict]) -> bool:
        best_score = max((chunk.get("score", 0.0) for chunk in retrieved_chunks), default=0.0)
        low_confidence = best_score < self.retrieval_confidence_threshold

        triggered_by_keyword = any(keyword in user_query.lower() for keyword in self.escalation_keywords)
        no_context = len(retrieved_chunks) == 0

        return no_context or low_confidence or triggered_by_keyword

    def build_handoff_summary(
        self,
        user_query: str,
        persona: str,
        retrieved_chunks: list[dict],
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> str:
        conversation_history = conversation_history or []
        trimmed_history = conversation_history[-self.history_max_turns :]

        docs_used: list[dict[str, Any]] = []
        for c in retrieved_chunks:
            src = c.get("source", "unknown")
            page = c.get("page_number")
            section = c.get("section")
            provenance = src
            if page is not None:
                provenance += f" (page {page})"
            elif section:
                provenance += f" ({section})"

            docs_used.append(
                {
                    "source": src,
                    "page_number": page,
                    "section": section,
                    "provenance": provenance,
                    "score": c.get("score", 0.0),
                }
            )

        # Actions attempted: best-effort extraction from what the KB snippet likely suggests.
        attempted_steps: list[str] = []
        for c in retrieved_chunks[:3]:
            text = c.get("text", "")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            # Heuristic: pick a few imperative-ish lines.
            for ln in lines[:25]:
                low = ln.lower()
                if any(
                    low.startswith(p)
                    for p in [
                        "try",
                        "restart",
                        "check",
                        "ensure",
                        "verify",
                        "reset",
                        "clear",
                        "update",
                        "contact",
                        "refresh",
                    ]
                ):
                    attempted_steps.append(ln)
                if len(attempted_steps) >= 6:
                    break
            if len(attempted_steps) >= 6:
                break

        if not attempted_steps:
            attempted_steps = ["No explicit troubleshooting steps detected from KB."]

        confidence_score = max((chunk.get("score", 0.0) for chunk in retrieved_chunks), default=0.0)

        handoff_data = {
            "persona": persona,
            "issue": user_query,
            "conversation_history": trimmed_history,
            "documents_used": docs_used,
            "actions_attempted": attempted_steps,
            "confidence_score": confidence_score,
            "recommendation": [
                "Investigate the issue using the user's account context and error logs.",
                "Cross-check the KB steps referenced in the retrieved documents.",
                "If billing/legal/account changes are involved, escalate to the appropriate internal team.",
            ],
        }
        return json.dumps(handoff_data, indent=2)
