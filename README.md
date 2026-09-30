<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:6f5bf5,45:3b9dfc,100:2fd6c4&height=190&section=header&text=NEURA&fontSize=78&fontColor=ffffff&animation=fadeIn&fontAlignY=36&desc=a%20document%20RAG%20chat%20engine%20that%20refuses%20to%20guess&descAlignY=57&descSize=17" alt="NEURA" />

<a href="https://github.com/daivraval/neura-rag-chat">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&duration=3200&pause=900&color=2FD6C4&center=true&vCenter=true&width=680&lines=Ask+your+documents+anything+at+all.;MMR+retrieval+over+a+Chroma+vector+store;Grounded+in+your+PDF+%E2%80%94+or+it+says+it+doesn't+know;No+build+step.+No+node_modules.+Just+python+app.py" alt="typing" />
</a>

<br/>

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" />
<img src="https://img.shields.io/badge/Chroma-FF6B4A?style=for-the-badge&logo=chromatic&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/Groq-Llama_3.3_70B-F55036?style=for-the-badge" />
<img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" />

</div>

<br/>

![NEURA UI](assets/neura-ui.png)

---

## The one rule

Most "chat with your PDF" demos will happily answer from the model's own memory when retrieval comes back empty — and you can't tell the difference from the outside. NEURA's system prompt makes the contract explicit:

> Use **ONLY** the provided context to answer the question. If the answer is not present in the context, say: *"I could not find the answer in the document."*

Paired with `temperature=0`, the same question against the same index returns the same answer every time. That's the whole design goal: **a retrieval system you can actually audit**, not a plausible-sounding one.

---

## How it works

```mermaid
flowchart LR
    subgraph IDX["🗂️ Indexing — create_database.py, run once"]
        A[PDF] --> B["RecursiveCharacterTextSplitter<br/>1000 chars · 200 overlap"]
        B --> C["Strip UTF-8 surrogates"]
        C --> D["all-MiniLM-L6-v2<br/>local · 384-dim"]
        D --> E[("Chroma<br/>chroma_db/")]
    end

    subgraph QRY["💬 Query — app.py, per message"]
        F[User question] --> G["MMR retrieval<br/>fetch_k=10 → k=4 · λ=0.5"]
        E -.-> G
        G --> H["Context + grounding prompt"]
        H --> I["Llama 3.3 70B (default)<br/>Groq · any OpenAI-compatible API · temp 0"]
        I --> J[Grounded answer + source snippets]
        J --> K[("SQLite<br/>chat_history.db")]
    end
```

**Why MMR and not plain similarity?** Straight cosine search on a question like *"what is gradient descent"* returns four chunks that are near-duplicates of each other — you burn the context window on one idea restated four times. Maximal Marginal Relevance pulls a wider candidate net (`fetch_k=10`), then greedily picks the 4 that are relevant **and** mutually dissimilar. `lambda_mult=0.5` splits the objective evenly between relevance and diversity.

---

## Retrieval configuration

Every knob that shapes an answer, in one place:

| Stage | Setting | Value | Why |
|---|---|---|---|
| Chunking | `chunk_size` / `chunk_overlap` | `1000` / `200` | Big enough to hold a full argument, 20% overlap so ideas aren't severed at a boundary |
| Embedding | `all-MiniLM-L6-v2` | local, 384-dim | Runs on CPU — indexing costs nothing and never leaves the machine |
| Search | `search_type` | `mmr` | Diversity-aware; kills near-duplicate chunks |
| Search | `fetch_k` → `k` | `10` → `4` | Cast wide, return four distinct passages |
| Search | `lambda_mult` | `0.5` | Balanced relevance ↔ diversity (`1.0` = pure relevance, `0.0` = pure diversity) |
| Generation | `llama-3.3-70b-versatile` | Groq (swappable) | Strong instruction-following, fast, free tier. `LLM_PROVIDER` / `LLM_MODEL` switch to Hugging Face or OpenAI |
| Generation | `max_tokens` | `400` | Long enough to explain, short enough to stay on-context |
| Generation | `temperature` | `0` | Deterministic — reproducible answers |

---

## Quickstart

```bash
git clone https://github.com/daivraval/neura-rag-chat.git
cd neura-rag-chat
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

Copy `.env.example` → `.env`. The default provider is [Groq](https://console.groq.com/keys), which has a free tier, so one key is enough:

```env
GROQ_API_KEY="gsk_..."
```

To use another provider, set `LLM_PROVIDER` to `huggingface` (uses `HF_TOKEN`) or `openai` (uses `OPENAI_API_KEY` plus `LLM_MODEL`). Any OpenAI-compatible API works: add it to `PROVIDERS` in `neura/pipeline.py`.

Drop your PDF at `1_document_loaders/PDF.pdf`, build the index, and run:

```bash
python create_database.py     # PDF → chunks → embeddings → chroma_db/
python app.py                 # http://127.0.0.1:8000
```

First run downloads the MiniLM weights (~90 MB). After that, indexing is fully offline — only generation calls the network.

> **Prefer a terminal?** `python main.py` gives you the same pipeline as a REPL. Type `quit` to exit.

---

## API

The frontend is just a client. Everything is reachable over HTTP:

| Method | Route | Does |
|---|---|---|
| `GET` | `/` | Serves the single-page UI (embedded in `app.py`, no build step) |
| `POST` | `/api/chat` | `{message, session_id?}` → retrieves, generates, persists, returns `{session_id, answer, sources[]}` |
| `GET` | `/api/sessions` | List conversations, newest first |
| `POST` | `/api/sessions` | Start an empty conversation |
| `GET` | `/api/sessions/{id}` | Full transcript with source snippets |
| `DELETE` | `/api/sessions/{id}` | Delete a conversation and its messages |

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What methodology does the paper use?"}'
```

Omit `session_id` and the server mints one, naming the conversation from your first 48 characters. Every reply ships the three passages it was built from, so you can check the answer against the source without leaving the page.

---

## Persistence

Two tables in `chat_history.db`, created on startup — no migration step, no ORM:

```sql
sessions(id TEXT PK, title TEXT, created_at TEXT)
messages(id INTEGER PK, session_id TEXT → sessions(id) ON DELETE CASCADE,
         role TEXT, content TEXT, sources TEXT, created_at TEXT)
```

`sources` holds the retrieved snippets for AI turns, so a reopened conversation still shows what each answer was grounded in.

---

## Repository layout

| Path | What it is |
|---|---|
| `app.py` | The product — FastAPI backend, SQLite history, and the entire frontend in one file |
| `neura/pipeline.py` | The RAG pipeline (chunking, embeddings, retrieval, prompt, LLM) shared by the app, the CLI and the evals |
| `create_database.py` | Indexer: PDF → chunks → embeddings → Chroma |
| `evals/` | Eval suite — golden set, scoring, runner, and the latest reports in `evals/results/` |
| `tests/` | Unit tests for the eval scoring |
| `main.py` | Same pipeline as a CLI REPL |
| `1_document_loaders/` | 🧪 Loader & splitter experiments — PDF, web-page summarization, splitter comparisons |
| `Retrievers/` | 🧪 Retrieval-strategy experiments — MMR, MultiQuery, ArXiv, each with worked examples in the docstrings |

The two 🧪 folders are a **learning lab**, not app dependencies. They're the notes behind the choices in the table above — kept in the repo on purpose, because the reasoning is the interesting part.

---

## Evaluation

"Refuses to guess" is a claim until it's measured. `evals/` holds a **54-question golden set** about the indexed paper, graded against the PDF itself:

- **42 answerable questions** (definitions, method details, numbers, table lookups, metadata). Each one is pinned to its page and a verbatim evidence passage.
- **12 unanswerable traps.** Some are plausible questions the paper never answers ("what batch size?"). Others are general knowledge the model already knows ("what is the capital of France?"). The only correct response is a refusal.

```bash
python -m evals.run_eval validate      # checks every evidence passage is really on its page
python -m evals.run_eval retrieval     # offline, compares retrieval configs, no API calls
python -m evals.run_eval generation --judge   # full pipeline on the configured LLM (--provider / --model to compare)
pytest                                 # unit tests for the scoring
```

**Retrieval** is scored by **evidence hit@k**: did a retrieved chunk contain the exact passage that answers the question? It also reports MRR, page-level hit rate and context precision.

**Generation** is scored by:
- **answer accuracy**: every expected fact is present in the answer
- **refusal accuracy**: unanswerable questions are refused
- **false-refusal rate**
- **faithfulness**, graded by an LLM judge from a different model family (`gpt-oss-120b` on Groq)

Each wrong answer is labelled a *retrieval miss* (the passage never reached the model) or a *generation miss* (it did, and the model still got it wrong).

### Current results: retrieval

Full report: [`evals/results/retrieval.md`](evals/results/retrieval.md).

| Config | Setup | Evidence hit@4 | MRR | Page hit@4 |
|---|---|---|---|---|
| **app (shipped)** | 1000/200 chunks, MMR λ=0.5 | 69.0% | 0.563 | 85.7% |
| similarity | 1000/200 chunks, cosine top-4 | **83.3%** | **0.609** | **88.1%** |
| mmr-0.8 | 1000/200 chunks, MMR λ=0.8 | 76.2% | 0.591 | 83.3% |
| small-chunks | 500/100 chunks, top-8 | 76.2% | 0.512 | 85.7% |

The eval's first finding: **on this paper, MMR costs recall.** Plain similarity finds the answering passage 14 points more often. MMR's diversity term pushes out the second-best chunk, and for pinpoint questions ("what loss function?", "which optimizer?") that chunk is often the one with the answer. Metadata questions (author, grant) miss under every config: the title-page chunk is dominated by the abstract.

Can the top retrieval score detect a question the document can't answer? The **AUROC is 0.71**: it separates them better than chance, but not well enough to refuse on the score alone yet.

---

## The frontend

Particle-field canvas, aurora gradients, glassmorphism panels — written by hand in vanilla JS and CSS and served as one string from `app.py`. No React, no Tailwind, no bundler, no `node_modules`. Clone and run; there is no build step to break.

---

## Known limits

Being straight about what this does and doesn't do:

- **One corpus at a time.** The PDF path is hardcoded in `create_database.py`; swapping documents means re-indexing. No multi-document routing.
- **No streaming.** `/api/chat` blocks until the full generation returns, so long answers sit behind a spinner.
- **Snippets, not citations.** Sources are the first 220 characters of each chunk — no page numbers or highlight-in-PDF.
- **No re-ranking.** MMR output goes straight to the prompt; a cross-encoder pass would sharpen it further.
- **Local and unauthenticated.** Binds to `127.0.0.1` with no auth layer. Don't expose it as-is.

---

## Roadmap

- [ ] Token streaming over SSE
- [ ] Multi-document indexing with per-source filters
- [ ] Page-anchored citations that jump into the PDF
- [ ] Cross-encoder re-ranking after MMR
- [ ] Drag-and-drop upload + in-app re-indexing
- [x] Swappable LLM backends: Groq, Hugging Face, or any OpenAI-compatible API

---

<div align="center">

**LangChain** · **Chroma** · **Groq** · **Hugging Face** · **FastAPI** · **SQLite** · **vanilla JS**

Built by [@daivraval](https://github.com/daivraval) · ⭐ it if the grounding contract is your kind of thing

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:2fd6c4,55:3b9dfc,100:6f5bf5&height=110&section=footer" />

</div>
