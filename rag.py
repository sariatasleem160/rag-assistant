import os

import anthropic

from ingest import get_collection

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context passages.

Rules:
- Use only the information in the numbered context passages below.
- Cite the passages you rely on by their number in square brackets, e.g. [1] or [2].
- If the answer is not contained in the context, say you don't have enough information. Never invent facts.
- Be clear and concise."""


def get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it in the Streamlit sidebar "
            "or run: $env:ANTHROPIC_API_KEY=\"sk-ant-your-key\""
        )
    return anthropic.Anthropic(api_key=api_key)


def retrieve(query, k=4):
    """Return the k most relevant chunks for a query."""
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=k)

    chunks = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": text,
            "source": meta["source"],
            "page": meta["page"],
            "distance": dist,
        })
    return chunks


def build_context(chunks):
    """Format retrieved chunks into a numbered context block for citations."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        loc = c["source"] + (f', p.{c["page"]}' if c["page"] else "")
        blocks.append(f"[{i}] (from {loc})\n{c['text']}")
    return "\n\n".join(blocks)


def answer(query, chunks, history=None):
    history = history or []
    context = build_context(chunks)
    user_message = f"Context passages:\n\n{context}\n\nQuestion: {query}"

    response = get_client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=history + [{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def ask(query, history=None, k=4):
    """Main entry point: retrieve + generate. Returns (answer_text, source_chunks)."""
    chunks = retrieve(query, k=k)
    text = answer(query, chunks, history=history)
    return text, chunks


if __name__ == "__main__":
    text, sources = ask("How does RAG store vectors?")
    print(text, "\n\nSources:")
    for i, c in enumerate(sources, 1):
        loc = c["source"] + (f' p.{c["page"]}' if c["page"] else "")
        print(f"[{i}] {loc}")
