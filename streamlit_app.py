import os
import requests
import streamlit as st


# ----- Config -----
BACKEND_URL = os.getenv("RAG_BACKEND_URL", "http://127.0.0.1:8000")


st.set_page_config(page_title="AI Document Search Assistant", layout="wide")

st.title("📄 AI Document Search Assistant (RAG)")
st.markdown(
    "Upload your PDF documents, then ask questions and get answers grounded in those files."
)

st.sidebar.header("Server status")
with st.sidebar:
    st.write(f"Backend URL: `{BACKEND_URL}`")
    status_placeholder = st.empty()


def check_backend():
    try:
        r = requests.get(f"{BACKEND_URL}/")
        if r.ok:
            data = r.json()
            status_placeholder.success(
                f"✅ Connected: {data.get('message', 'Server is running.')}"
            )
            return True
        else:
            status_placeholder.error(f"❌ Backend error: {r.status_code}")
            return False
    except Exception as e:
        status_placeholder.error(f"❌ Cannot reach backend: {e}")
        return False


backend_ok = check_backend()

tab_upload, tab_chat = st.tabs(["📤 Upload PDFs", "💬 Ask Questions"])


with tab_upload:
    st.subheader("Upload and index PDF documents")
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Upload & Index") and backend_ok:
        if not uploaded_files:
            st.warning("Please select at least one PDF file.")
        else:
            for f in uploaded_files:
                with st.spinner(f"Uploading and indexing `{f.name}`..."):
                    files = {"file": (f.name, f.read(), "application/pdf")}
                    try:
                        resp = requests.post(
                            f"{BACKEND_URL}/upload_doc", files=files, timeout=300
                        )
                        if resp.ok:
                            info = resp.json()
                            st.success(
                                f"Indexed `{info['filename']}` "
                                f"(doc_id={info['doc_id']}, chunks={info['num_chunks']})"
                            )
                        else:
                            st.error(
                                f"Failed for `{f.name}`: {resp.status_code} {resp.text}"
                            )
                    except Exception as e:
                        st.error(f"Error uploading `{f.name}`: {e}")

    st.markdown("---")
    st.subheader("Indexed documents")
    if st.button("Refresh document list") and backend_ok:
        try:
            resp = requests.get(f"{BACKEND_URL}/documents", timeout=60)
            if resp.ok:
                docs = resp.json()
                if not docs:
                    st.info("No documents indexed yet.")
                else:
                    for d in docs:
                        st.write(
                            f"- **{d['filename']}** "
                            f"(doc_id={d['doc_id']}, chunks={d['num_chunks']})"
                        )
            else:
                st.error(f"Error fetching documents: {resp.status_code} {resp.text}")
        except Exception as e:
            st.error(f"Error contacting backend: {e}")


with tab_chat:
    st.subheader("Ask questions about your documents")
    query = st.text_area("Your question", height=100)
    top_k = st.slider("Number of context chunks (top_k)", min_value=1, max_value=10, value=5)

    if st.button("Get Answer") and backend_ok:
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Querying documents and generating answer..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/query",
                        json={"query": query, "top_k": top_k},
                        timeout=300,
                    )
                    if resp.ok:
                        data = resp.json()
                        st.markdown("### 🧠 Answer")
                        st.write(data.get("answer", "No answer returned."))

                        chunks = data.get("chunks", [])
                        if chunks:
                            st.markdown("### 📚 Supporting document chunks")
                            for i, ch in enumerate(chunks, start=1):
                                with st.expander(
                                    f"Chunk {i} — {ch['doc_name']} (score: {ch['score']:.4f})"
                                ):
                                    st.write(ch["text"])
                        else:
                            st.info("No supporting chunks were returned.")
                    else:
                        st.error(
                            f"Backend returned error {resp.status_code}: {resp.text}"
                        )
                except Exception as e:
                    st.error(f"Error contacting backend: {e}")


