import os

import streamlit as st
from dotenv import load_dotenv

from persona_support_agent.classifier import PersonaClassifier
from persona_support_agent.rag import LocalRAGPipeline
from persona_support_agent.response import AdaptiveResponder

load_dotenv()


def main():
    st.set_page_config(page_title="Persona Support Agent", layout="wide")
    st.title("Persona-Adaptive Customer Support Agent")
    st.markdown(
        "Use this interface to send a support message, detect the customer persona, retrieve knowledge base context, "
        "and generate a persona-tailored response."
    )

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GENAI_API_KEY")
    if not api_key:
        st.error("No Gemini API key found. Set `GEMINI_API_KEY` in your .env file.")
        return

    classifier = PersonaClassifier()
    rag = LocalRAGPipeline()
    responder = AdaptiveResponder()

    if "history" not in st.session_state:
        st.session_state.history = []


    user_message = st.text_area("User message", height=160)
    run_button = st.button("Generate response")

    if run_button and user_message.strip():
        with st.spinner("Classifying persona and retrieving knowledge..."):
            st.session_state.history.append({"role": "user", "content": user_message})

            persona_result = classifier.classify(user_message)

            persona = persona_result.get("persona", "Technical Expert")
            confidence = persona_result.get("confidence", 0.0)
            reasoning = persona_result.get("reasoning", "")

            retrieved = rag.retrieve_context(user_message, top_k=4)
            # Add assistant response to history after generation.

            response = responder.generate(
                user_message,
                persona,
                retrieved,
                conversation_history=st.session_state.history,
            )

            answer = response.get("answer", "")
            st.session_state.history.append({"role": "assistant", "content": answer})

            escalate = response.get("escalate_required", False) or responder.should_escalate(
                user_message,
                persona,
                retrieved,
            )
            model_error = response.get("error")
            handoff = (
                responder.build_handoff_summary(
                    user_message,
                    persona,
                    retrieved,
                    conversation_history=st.session_state.history,
                )
                if escalate
                else None
            )


        st.subheader("Persona Detection")
        st.metric(label="Detected persona", value=persona)
        st.metric(label="Confidence", value=f"{confidence:.2f}")
        st.write("**Reasoning:**", reasoning)

        st.subheader("Retrieved Context")
        if retrieved:
            for chunk in retrieved:
                with st.expander(f"Source: {chunk['source']} — score {chunk['score']:.2f}"):
                    st.write(chunk["text"])
        else:
            st.warning("No relevant KB results were found for this query.")

        st.subheader("Generated Response")
        if model_error:
            st.warning(f"Gemini generation fallback used: {model_error}")
        st.write(answer)

        st.subheader("Escalation")
        st.write("**Escalation needed:**", "Yes" if escalate else "No")
        if escalate and handoff:
            with st.expander("Handoff Summary"):
                st.code(handoff, language="json")

    st.sidebar.header("Example Queries")
    st.sidebar.write(
        "- Where is the guide to clear cookies? It's been an hour and nothing is loading on your interface!\n"
        "- What are the header parameter requirements for your bearer token auth implementation?\n"
        "- Our operational uptime is decreasing. We need a timeline of when billing disputes are resolved.\n"
        "- I'm experiencing an issue with your database integration that's causing internal errors.\n"
        "- My billing statement has unexpected duplicate charges. I demand an immediate refund!"
    )


if __name__ == "__main__":
    main()
