import os
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from google import genai
from pypdf import PdfReader

load_dotenv()


def split_text(content: str, chunk_size: int = 450, chunk_overlap: int = 60) -> list[str]:
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    chunks: list[str] = []
    start = 0
    text_length = len(content)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(content[start:end])
        start += chunk_size - chunk_overlap
    return chunks



class LocalRAGPipeline:
    def __init__(self, db_dir: str | None = None, data_dir: str | None = None):
        db_dir = db_dir or os.environ.get("CHROMA_DB_DIR", "./chroma_db")
        self.data_dir = data_dir or os.environ.get("DATA_DIR", "./data")
        self.client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GENAI_API_KEY")
            or ""
        )
        self.chroma_client = chromadb.PersistentClient(path=db_dir)
        self.collection = self.chroma_client.get_or_create_collection(name="support_kb")
        self._ensure_index()

    def _ensure_index(self):
        if self.collection.count() == 0:
            self.ingest_directory(self.data_dir)

    def get_embedding(self, text: str) -> list[float]:
        # Model name for embeddings can differ by API version.
        # Allow override via env; default to a common Gemini embedding model.
        embedding_model = os.environ.get("GEMINI_MODEL_EMBEDDING", "models/gemini-embedding-2")
        response = self.client.models.embed_content(
            model=embedding_model,
            contents=text,
        )
        return response.embeddings[0].values

    def _guess_section_name(self, text: str) -> str:
        # Heuristic: try to find a Markdown heading closest to the beginning of the chunk.
        # If no headings exist, return a fallback.
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("#") and len(s) > 1:
                return s.lstrip("#").strip()[:120]
        return "file"

    def _iter_chunks_with_metadata(self, doc_name: str, content: str, suffix: str) -> list[dict]:
        """Return chunks with provenance metadata.

        Each returned dict has:
        - text
        - metadata: {source, page_number, section, chunk_index}
        """
        chunks_out: list[dict] = []

        if suffix == ".pdf":
            # content is already page-separated in _read_pdf
            pages = content.split("\n\f\n") if content else []
            chunk_index = 0
            for page_idx, page_text in enumerate(pages):
                if not page_text.strip():
                    continue
                page_chunks = split_text(page_text, chunk_size=450, chunk_overlap=60)
                section = "pdf_page"
                for pc in page_chunks:
                    chunks_out.append(
                        {
                            "text": pc,
                            "metadata": {
                                "source": doc_name,
                                "page_number": page_idx + 1,
                                "section": section,
                                "chunk_index": chunk_index,
                            },
                        }
                    )
                    chunk_index += 1
            return chunks_out

        # md/txt: preserve a section label per chunk (best-effort)
        raw_chunks = split_text(content, chunk_size=450, chunk_overlap=60)
        for idx, ch in enumerate(raw_chunks):
            section_name = self._guess_section_name(ch)
            chunks_out.append(
                {
                    "text": ch,
                    "metadata": {
                        "source": doc_name,
                        # Chroma metadata cannot contain None; omit page_number instead.
                        "section": section_name,
                        "chunk_index": idx,
                    },
                }
            )
        return chunks_out

    def ingest_document(self, doc_name: str, content: str, suffix: str):
        chunks = self._iter_chunks_with_metadata(doc_name, content, suffix=suffix)
        for chunk in chunks:
            embedding = self.get_embedding(chunk["text"])
            document_id = f"{doc_name}_chunk_{chunk['metadata']['chunk_index']}"

            # Chroma requires metadata objects to be JSON-serializable.
            self.collection.add(
                ids=[document_id],
                embeddings=[embedding],
                metadatas=[chunk["metadata"]],
                documents=[chunk["text"]],
            )



    def ingest_directory(self, data_dir: str = "./data"):
        data_path = Path(data_dir)
        for path in sorted(data_path.glob("**/*")):
            if not path.is_file():
                continue

            content = self._read_document(path)
            if not content:
                continue

            suffix = path.suffix.lower()
            if suffix not in {".pdf", ".md", ".txt"}:
                continue

            self.ingest_document(path.name, content, suffix=suffix)

    def _read_document(self, path: Path) -> str | None:
        if path.suffix.lower() == ".pdf":
            return self._read_pdf(path)
        if path.suffix.lower() in {".md", ".txt"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        return None


    def _read_pdf(self, path: Path) -> str:
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n\n".join(pages)

    def retrieve_context(self, query: str, top_k: int = 3) -> list[dict]:
        query_vector = self.get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
        )

        retrieved: list[dict] = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for idx, chunk_text in enumerate(docs):
                score = 0.0
                if distances and len(distances) > idx and distances[idx] is not None:
                    score = max(0.0, 1.0 - distances[idx])

                meta = metas[idx] if metas and idx < len(metas) else {}
                retrieved.append(
                    {
                        "text": chunk_text,
                        "source": meta.get("source", "unknown"),
                        "page_number": meta.get("page_number"),
                        "section": meta.get("section"),
                        "chunk_index": meta.get("chunk_index"),
                        "score": score,
                    }
                )
        return retrieved
