"""
The NEURA RAG pipeline — indexing, retrieval and generation in one place.

app.py, main.py, create_database.py and the eval suite (evals/) all build on
these functions, so what the evals measure is exactly what the app serves.
"""

import os
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(ROOT, "1_document_loaders", "PDF.pdf")
CHROMA_DIR = os.path.join(ROOT, "chroma_db")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_REPO = "Qwen/Qwen2.5-7B-Instruct"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# The retrieval settings the app ships with.
RETRIEVAL = {"search_type": "mmr", "k": 4, "fetch_k": 10, "lambda_mult": 0.5}

REFUSAL = "I could not find the answer in the document."
SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Use ONLY the provided context to "
    "answer the question. If the answer is not present in the context, "
    f'say: "{REFUSAL}"'
)


# ------------------------------------------------------------------ indexing
def load_pages(pdf_path=PDF_PATH):
    """One document per PDF page; metadata["page"] is 0-based."""
    from langchain_community.document_loaders import PyPDFLoader

    return PyPDFLoader(pdf_path).load()


def load_chunks(pdf_path=PDF_PATH, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """PDF -> page documents -> overlapping text chunks (page metadata kept)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    pages = load_pages(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(pages)
    # Remove invalid Unicode surrogate characters that the PDF extraction
    # leaves behind, otherwise the tokenizer rejects them.
    for chunk in chunks:
        chunk.page_content = chunk.page_content.encode("utf-8", "ignore").decode("utf-8")
    return chunks


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_index(chunks, persist_dir=None, collection_name="langchain"):
    """Embed chunks into Chroma. persist_dir=None keeps the index in memory."""
    from langchain_community.vectorstores import Chroma

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=persist_dir,
        collection_name=collection_name,
    )


def open_index(persist_dir=CHROMA_DIR):
    """Open the index that create_database.py wrote to disk."""
    from langchain_community.vectorstores import Chroma

    return Chroma(persist_directory=persist_dir, embedding_function=get_embeddings())


# ----------------------------------------------------------------- retrieval
def make_retriever(store, search_type="mmr", k=4, fetch_k=10, lambda_mult=0.5):
    search_kwargs = {"k": k}
    if search_type == "mmr":
        search_kwargs.update(fetch_k=fetch_k, lambda_mult=lambda_mult)
    return store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


# ---------------------------------------------------------------- generation
def make_llm(max_new_tokens=400, repo_id=LLM_REPO):
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    return ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id=repo_id,
            task="text-generation",
            max_new_tokens=max_new_tokens,
            do_sample=False,
            huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        )
    )


def make_prompt():
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Context: {context}\nQuestion: {question}"),
        ]
    )


def format_context(docs):
    return "\n\n".join(d.page_content for d in docs)


def generate(llm, prompt, question, docs):
    """Answer `question` grounded in the retrieved `docs`."""
    messages = prompt.invoke({"context": format_context(docs), "question": question})
    return llm.invoke(messages).content
