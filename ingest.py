from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

_client = chromadb.PersistentClient(path="./chroma_db")
_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def get_collection():
    """The single place that defines our vector collection + embedding model."""
    return _client.get_or_create_collection(
        name="documents",
        embedding_function=_embed_fn,
    )


def load_file(path, display_name=None):
    """Read a PDF or text file into a list of {text, source, page} records."""
    path = Path(path)
    name = display_name or path.name
    records = []

    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                records.append({"text": text, "source": name, "page": page_num})
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
        records.append({"text": text, "source": name, "page": None})

    return records


def chunk_text(text, chunk_size=800, overlap=150):
    """Split text into overlapping chunks of ~chunk_size characters."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def build_chunks(records, chunk_size=800, overlap=150):
    """Turn loaded records into chunk records that remember their source."""
    all_chunks = []
    for rec in records:
        for i, piece in enumerate(chunk_text(rec["text"], chunk_size, overlap)):
            all_chunks.append({
                "text": piece,
                "source": rec["source"],
                "page": rec["page"],
                "chunk_id": f'{rec["source"]}-p{rec["page"]}-{i}',
            })
    return all_chunks


def index_chunks(chunks):
    """Embed chunks and store them in Chroma. Chroma embeds automatically."""
    collection = get_collection()
    collection.upsert(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[
            {"source": c["source"], "page": c["page"] if c["page"] is not None else 0}
            for c in chunks
        ],
    )
    return collection.count()


def ingest_file(path, display_name=None):
    """Full ingestion pipeline: load → chunk → embed → store."""
    records = load_file(path, display_name=display_name)
    chunks = build_chunks(records)
    total = index_chunks(chunks)
    return len(chunks), total


if __name__ == "__main__":
    added, total = ingest_file("docs/sample.txt")
    print(f"Indexed {added} chunks. Collection now holds {total} chunks.")
