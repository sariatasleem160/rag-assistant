import anthropic

from rag import ask, retrieve

client = anthropic.Anthropic()

TESTS = [
    {
        "question": "What database does RAG use to store vectors?",
        "expected_source": "sample.txt",
        "expected_keyword": "Chroma",
    },
    {
        "question": "What is the first step in the RAG pipeline?",
        "expected_source": "sample.txt",
        "expected_keyword": "Loading",
    },
    {
        "question": "How are documents split before embedding?",
        "expected_source": "sample.txt",
        "expected_keyword": "chunks",
    },
]


def judge(question, answer_text, context_text):
    prompt = (
        f"Question: {question}\n\nAnswer given: {answer_text}\n\n"
        f"Source context:\n{context_text}\n\n"
        "Is the answer fully supported by the context and does it correctly answer "
        "the question? Reply GOOD or BAD, then one sentence of reasoning."
    )
    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text


def evaluate():
    retrieval_hits = answer_hits = 0
    for t in TESTS:
        sources = {c["source"] for c in retrieve(t["question"], k=4)}
        if t["expected_source"] in sources:
            retrieval_hits += 1

        text, _ = ask(t["question"])
        if t["expected_keyword"].lower() in text.lower():
            answer_hits += 1

    n = len(TESTS)
    print(f"Retrieval hit rate:   {retrieval_hits}/{n}")
    print(f"Answer keyword match: {answer_hits}/{n}")


if __name__ == "__main__":
    evaluate()
