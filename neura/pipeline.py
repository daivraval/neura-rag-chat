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

# Every provider speaks the OpenAI chat API, so one client covers them all.
# Pick one with LLM_PROVIDER (default groq) and optionally LLM_MODEL in .env.
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
    },
    "huggingface": {
        "base_url": "https://router.huggingface.co/v1",
        "key_env": "HF_TOKEN",
        "model": "Qwen/Qwen2.5-72B-Instruct",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "model": None,  # no default — set LLM_MODEL
    },
}
DEFAULT_PROVIDER = "groq"

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
def resolve_llm(provider=None, model=None):
    """Settle (provider, model): explicit args, then .env, then the preset.

    LLM_MODEL only applies to the provider named in LLM_PROVIDER, so asking
    for another provider explicitly never inherits a model it can't serve.
    """
    env_provider = os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER
    provider = provider or env_provider
    if provider not in PROVIDERS:
        raise ValueError(f"unknown LLM provider {provider!r} — choose from {', '.join(PROVIDERS)}")
    if not model and provider == env_provider:
        model = os.getenv("LLM_MODEL")
    model = model or PROVIDERS[provider]["model"]
    if not model:
        raise ValueError(f"set LLM_MODEL — provider {provider!r} has no default model")
    return provider, model


def make_llm(max_new_tokens=400, provider=None, model=None, **extra):
    """Chat model on any OpenAI-compatible provider, temperature 0 so the
    same question against the same index gives the same answer."""
    from langchain_openai import ChatOpenAI

    provider, model = resolve_llm(provider, model)
    preset = PROVIDERS[provider]
    api_key = os.getenv(preset["key_env"])
    if not api_key:
        raise RuntimeError(f"{preset['key_env']} is not set (needed for LLM provider {provider!r}) — see .env.example")
    return ChatOpenAI(
        model=model,
        base_url=preset["base_url"],
        api_key=api_key,
        temperature=0,
        max_tokens=max_new_tokens,
        max_retries=3,
        **extra,
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
