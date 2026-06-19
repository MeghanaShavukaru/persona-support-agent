import os
import sys
from pathlib import Path

from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root / "src"))
load_dotenv()

from persona_support_agent.rag import LocalRAGPipeline  # noqa: E402


def main():
    db_dir = os.environ.get("CHROMA_DB_DIR", "./chroma_db")
    data_dir = os.environ.get("DATA_DIR", "./data")

    rag = LocalRAGPipeline(db_dir=db_dir)
    print("Ingesting data from", data_dir)

    # Ingest each document.
    # Note: if you change chunking/metadata logic, clear the existing `chroma_db/` directory first.
    rag.ingest_directory(data_dir)


    print("Index build complete.")


if __name__ == "__main__":
    main()

