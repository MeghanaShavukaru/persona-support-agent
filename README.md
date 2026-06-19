# Persona Support Agent

An adaptive customer support agent built with Google Gemini, ChromaDB, LangChain, and a persona-aware response engine.

## Project Overview

This repository implements a persona-adaptive support assistant that:
- Detects customer personas from incoming messages
- Retrieves relevant content from a knowledge base
- Generates grounded responses that match the detected persona
- Escalates sensitive or low-confidence support requests to a human agent
- Produces a structured handoff summary for human follow-up

## Tech Stack

- Python 3.11+
- Google Gemini via `google-genai`
- ChromaDB for local vector retrieval
- LangChain for document chunking
- Streamlit (optional UI)
- `pypdf` for PDF ingestion
- `python-dotenv` for environment variable loading

## Repository Structure

- `src/persona_support_agent/`: core agent modules
- `data/`: knowledge base documents
- `scripts/`: ingestion and utility scripts
- `streamlit_app.py`: optional web UI interface
- `README.md`: project overview and setup instructions

## Architecture Overview

The system workflow is:
1. User message input
2. Persona classification via Google Gemini
3. Knowledge base retrieval from ChromaDB
4. Adaptive response generation based on persona
5. Escalation detection and human handoff summary creation

## Setup Instructions

1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create a `.env` file with your Gemini API key:

```bash
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL_CLASSIFIER=gemini-2.5-flash-lite
GEMINI_MODEL_GENERATOR=gemini-2.5-flash-lite
```

3. Install the package in editable mode and ingest the data corpus:

```bash
pip install -e .
python scripts/prepare_index.py
```

4. Run the CLI:

```bash
python -m persona_support_agent.cli
```

5. Or start the Streamlit UI:

```bash
streamlit run streamlit_app.py
```

## Streamlit Cloud Deployment

Deploy `streamlit_app.py` from this repository. Add these values in Streamlit Cloud app secrets:

```toml
GEMINI_API_KEY = "your_api_key_here"
GEMINI_MODEL_CLASSIFIER = "gemini-2.5-flash-lite"
GEMINI_MODEL_GENERATOR = "gemini-2.5-flash-lite"
GEMINI_MODEL_EMBEDDING = "models/gemini-embedding-2"
CHROMA_DB_DIR = "./chroma_db"
DATA_DIR = "./data"
```

Do not commit your local `.env` file. The app builds the local Chroma index from `data/` on first run when `chroma_db/` is empty.

## Persona Detection Strategy

- Uses a structured Gemini JSON output to classify the user message into one of three personas.
- The prompt is designed to focus on tone, vocabulary, urgency, and business impact.
- If the model output cannot be parsed, it defaults safely to `Technical Expert`.

## RAG Pipeline Design

- Knowledge base documents are loaded from `data/`.
- Documents are chunked into overlapping text windows.
- Gemini embeddings are stored in a local ChromaDB collection.
- Retrieval uses cosine similarity to return the top relevant chunks.

## Escalation Logic

- The system escalates when retrieval confidence is below 0.40.
- Billing, refund, account, or legal keywords also trigger escalation.
- On escalation, a JSON handoff summary is generated for a human support agent.

## Example Queries

- "Where is the guide to clear cookies? It's been an hour and nothing is loading on your interface!"
- "What are the header parameter requirements for your bearer token auth implementation?"
- "Our operational uptime is decreasing. We need a timeline of when billing disputes are resolved."
- "I'm experiencing an issue with your database integration that's causing internal errors."
- "My billing statement has unexpected duplicate charges. I demand an immediate refund!"

## Known Limitations

- Requires valid Google Gemini credentials
- Retrieval quality depends on the coverage of the knowledge base
- Responses are grounded using retrieved chunks; otherwise escalation is triggered
- Escalation triggers are configurable via environment variables (e.g., `RETRIEVAL_CONFIDENCE_THRESHOLD`)
- Multi-turn memory is implemented in CLI and Streamlit (session history)

