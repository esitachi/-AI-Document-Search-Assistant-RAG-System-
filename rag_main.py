from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import uuid
import pickle
from typing import List, Dict, Any
import numpy as np
import openai
from PyPDF2 import PdfReader

# ----- Config -----
INDEX_PATH = "vector_index.pkl"
DOCS_DIR = "uploaded_docs"
CHUNK_SIZE = 800  # characters
CHUNK_OVERLAP = 200  # characters

os.makedirs(DOCS_DIR, exist_ok=True)

# Set your API key in the env OR just paste directly here while testing
# Example (quick hack for today): openai.api_key = "sk-...."
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="AI Document Search Assistant (RAG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy import of sentence transformer model
_sentence_model = None


def get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_model


# ----- Data models -----


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponseChunk(BaseModel):
    doc_id: str
    doc_name: str
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    chunks: List[QueryResponseChunk]


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int


# ----- Simple vector index -----


def load_index() -> Dict[str, Any]:
    if not os.path.exists(INDEX_PATH):
        return {"docs": {}, "embeddings": [], "metadatas": []}
    with open(INDEX_PATH, "rb") as f:
        return pickle.load(f)


def save_index(index: Dict[str, Any]):
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)


def add_document_to_index(doc_id: str, filename: str, chunks: List[str]):
    model = get_sentence_model()
    embeddings = model.encode(chunks, show_progress_bar=False)

    index = load_index()
    for chunk_text, emb in zip(chunks, embeddings):
        index["embeddings"].append(emb.astype(np.float32))
        index["metadatas"].append(
            {
                "doc_id": doc_id,
                "doc_name": filename,
                "text": chunk_text,
            }
        )

    index["docs"][doc_id] = {
        "filename": filename,
        "num_chunks": len(chunks),
    }

    save_index(index)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return np.dot(a_norm, b_norm)


def search_index(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    index = load_index()
    if not index["embeddings"]:
        raise HTTPException(status_code=400, detail="No documents indexed yet.")

    model = get_sentence_model()
    query_emb = model.encode([query])[0].astype(np.float32)

    emb_matrix = np.stack(index["embeddings"], axis=0)
    scores = cosine_similarity(emb_matrix, query_emb)
    top_k = min(top_k, len(scores))
    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        metadata = index["metadatas"][idx]
        results.append(
            {
                "score": float(scores[idx]),
                **metadata,
            }
        )
    return results


# ----- Helpers -----


def read_pdf_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n".join(texts)


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    text = text.replace("\n", " ")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def build_answer_with_openai(query: str, contexts: List[str]) -> str:
    if not openai.api_key:
        return "OPENAI_API_KEY not set. Cannot generate answer."

    system_prompt = (
        "You are a helpful AI assistant. "
        "Answer the user's question using ONLY the provided context from documents. "
        "If the answer is not in the context, say you do not know."
    )

    context_text = "\n\n".join(
        [f"Context {i+1}: {c}" for i, c in enumerate(contexts)]
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {query}",
        },
    ]

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # or "gpt-4o" / "gpt-3.5-turbo"
            messages=messages,
            temperature=0.2,
        )
        return completion.choices[0].message["content"].strip()
    except Exception as e:
        return f"Error while calling OpenAI API: {e}"


# ----- Routes -----


@app.get("/")
def root():
    return {"message": "AI Document Search Assistant is running."}


@app.post("/upload_doc", response_model=DocumentInfo)
async def upload_doc(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    doc_id = str(uuid.uuid4())
    save_path = os.path.join(DOCS_DIR, f"{doc_id}.pdf")

    with open(save_path, "wb") as f:
        f.write(contents)

    try:
        full_text = read_pdf_text(save_path)
        if not full_text.strip():
            raise ValueError("Could not extract any text from the PDF.")

        chunks = chunk_text(full_text)
        if not chunks:
            raise ValueError("No text chunks were created from the PDF.")

        add_document_to_index(doc_id, file.filename, chunks)

        return DocumentInfo(
            doc_id=doc_id,
            filename=file.filename,
            num_chunks=len(chunks),
        )
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {e}")


@app.get("/documents", response_model=List[DocumentInfo])
def list_documents():
    index = load_index()
    docs = []
    for doc_id, info in index["docs"].items():
        docs.append(
            DocumentInfo(
                doc_id=doc_id,
                filename=info["filename"],
                num_chunks=info["num_chunks"],
            )
        )
    return docs


@app.post("/query", response_model=QueryResponse)
def query_documents(body: QueryRequest):
    results = search_index(body.query, body.top_k)
    contexts = [r["text"] for r in results]
    answer = build_answer_with_openai(body.query, contexts)

    chunks = [
        QueryResponseChunk(
            doc_id=r["doc_id"],
            doc_name=r["doc_name"],
            score=r["score"],
            text=r["text"],
        )
        for r in results
    ]

    return QueryResponse(answer=answer, chunks=chunks)
