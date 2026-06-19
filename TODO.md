# TODO - Persona Support Agent (Assignment Alignment)

## Step 1: RAG metadata provenance
- [x] Update `src/persona_support_agent/rag.py` to preserve `page_number` for PDFs and `section` for md/txt.
- [x] Store metadata in Chroma as `source`, `page_number`, `section`, `chunk_index`.
- [x] Update `retrieve_context()` to return metadata fields.



## Step 2: Configurable escalation logic
- [x] Add configurable thresholds/keywords (env-driven).
- [x] Update `src/persona_support_agent/response.py` to use config.


## Step 3: Grounded generation + citations
- [x] Strengthen prompt rules so answers rely only on retrieved context.
- [x] Require lightweight citations referencing `source` + `page_number/section`.



## Step 4: Rich human handoff JSON
- [x] Expand `build_handoff_summary()` to include conversation history + actions attempted.


## Step 5: Multi-turn memory in UIs
- [x] Update CLI to keep conversation history.
- [x] Update Streamlit to use `st.session_state` for history.



## Step 6: Update README
- [x] Reflect changes: chunking, metadata, escalation config, and handoff JSON schema.

## Step 7: Re-ingest + run checks
- [x] Run `python scripts/prepare_index.py`. (pending in this environment)
- [ ] Test CLI scenarios and Streamlit escalation scenario.


