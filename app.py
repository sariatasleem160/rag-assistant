import os
import tempfile

import streamlit as st

from ingest import get_collection, ingest_file
from rag import ask

st.set_page_config(page_title="Talk to Documents", page_icon="📄")
st.title("📄 Talk to Documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Get one from console.anthropic.com — never commit this key.",
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
    elif not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("Add your API key above to get answers from Claude.")

    st.header("Your documents")
    uploaded = st.file_uploader(
        "Upload PDFs or text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Index documents") and uploaded:
        with st.spinner("Reading and indexing..."):
            total = 0
            for f in uploaded:
                suffix = os.path.splitext(f.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.getvalue())
                    tmp_path = tmp.name
                added, _ = ingest_file(tmp_path, display_name=f.name)
                total += added
                os.unlink(tmp_path)
        st.success(f"Indexed {total} chunks from {len(uploaded)} file(s).")

    try:
        st.caption(f"{get_collection().count()} chunks in the database.")
    except Exception:
        st.caption("No documents indexed yet.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for i, s in enumerate(m["sources"], 1):
                    loc = s["source"] + (f', p.{s["page"]}' if s["page"] else "")
                    st.markdown(f"**[{i}]** {loc}")

if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.error(
                "API key missing. Add your Anthropic key in the sidebar, "
                "or set ANTHROPIC_API_KEY before starting Streamlit."
            )
        else:
            with st.spinner("Thinking..."):
                text, sources = ask(prompt, history=history)
            st.markdown(text)
            with st.expander("Sources"):
                for i, s in enumerate(sources, 1):
                    loc = s["source"] + (f', p.{s["page"]}' if s["page"] else "")
                    st.markdown(f"**[{i}]** {loc}")

            st.session_state.messages.append(
                {"role": "assistant", "content": text, "sources": sources}
            )
