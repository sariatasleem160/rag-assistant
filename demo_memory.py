from rag import ask

history = []
for q in ["What is RAG?", "What database does it use for storage?"]:
    text, _ = ask(q, history=history)
    print(f"\nQ: {q}\nA: {text}")
    history += [
        {"role": "user", "content": q},
        {"role": "assistant", "content": text},
    ]
