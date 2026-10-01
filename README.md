<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:6f5bf5,45:3b9dfc,100:2fd6c4&height=190&section=header&text=NEURA&fontSize=78&fontColor=ffffff&animation=fadeIn&fontAlignY=36&desc=a%20document%20RAG%20chat%20engine%20that%20refuses%20to%20guess&descAlignY=57&descSize=17" alt="NEURA" />

<a href="https://github.com/daivraval/neura-rag-chat">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&duration=3200&pause=900&color=2FD6C4&center=true&vCenter=true&width=680&lines=Ask+your+documents+anything+at+all.;54-question+eval%3A+every+wrong+answer+traced+to+retrieval;Grounded+in+your+PDF+%E2%80%94+or+it+says+it+doesn't+know;No+build+step.+No+node_modules.+Just+python+app.py" alt="typing" />
</a>

<br/>

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" />
<img src="https://img.shields.io/badge/Chroma-FF6B4A?style=for-the-badge&logo=chromatic&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/Groq-gpt--oss--120b-F55036?style=for-the-badge" />
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
        F[User question] --> G["Similarity search<br/>top k=4"]
        E -.-> G
        G --> H["Context + grounding prompt"]
        H --> I["gpt-oss-120b (default)<br/>Groq · any OpenAI-compatible API · temp 0"]
        I --> J[Grounded answer + source snippets]
        J --> K[("SQLite<br/>chat_history.db")]
    end
```

**Why plain similarity and not MMR?** NEURA originally used Maximal Marginal Relevance, which trades some relevance for diversity so the 4 chunks aren't near-duplicates. The [eval](#evaluation) showed that trade was a bad one. On the indexed paper MMR found the answering passage 43% of the time and plain similarity 60%, and answer accuracy is 69.0% with similarity against 61.9% with MMR. For pinpoint questions ("which optimizer?", "what dropout rate?") the chunk MMR drops for being "too similar" is often the one with the answer. MMR is still in the eval as a baseline (`--config mmr`).

---

## Retrieval configuration

Every knob that shapes an answer, in one place:

| Stage | Setting | Value | Why |
|---|---|---|---|
| Chunking | `chunk_size` / `chunk_overlap` | `1000` / `200` | Big enough to hold a full argument, 20% overlap so ideas aren't severed at a boundary |
| Embedding | `all-MiniLM-L6-v2` | local, 384-dim | Runs on CPU — indexing costs nothing and never leaves the machine |
| Search | `search_type` | `similarity` | Beat MMR in the eval: 59.5% evidence hit@4 vs 42.9% |
| Search | `k` | `4` | Four passages, ~4000 characters of context |
| Generation | `openai/gpt-oss-120b` | Groq (swappable) | Strongest production model on Groq free tier; refused 10 of 12 trap questions in the eval. `LLM_PROVIDER` / `LLM_MODEL` switch to Hugging Face or OpenAI |
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
| `GET` | `/api/info` | The indexed document and pipeline settings (chunks, retrieval, model) |
| `GET` | `/api/evals` | Headline numbers from the saved eval reports in `evals/results/` |
| `GET` | `/pdf` | The indexed PDF itself, so `/pdf#page=5` opens a cited page |

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What methodology does the paper use?"}'
```

Omit `session_id` and the server mints one, naming the conversation from your first 48 characters. Every reply ships the top three passages it was built from as `{text, page}`, so you can check the answer against the source without leaving the page.

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
- **12 unanswerable traps.** Some are plausible questions the paper never answers ("what weight decay?", "how does it do on MS MARCO?"). Others are general knowledge the model already knows ("what is the capital of France?"). The only correct response is a refusal.

The indexed document is *[Dense Passage Retrieval for Open-Domain Question Answering](https://aclanthology.org/2020.emnlp-main.550/)* (Karpukhin et al., EMNLP 2020), redistributed under its [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. Swap in any PDF you like; the golden set is specific to this one.

```bash
python -m evals.run_eval validate      # checks every evidence passage is really on its page
python -m evals.run_eval retrieval     # offline, compares retrieval configs, no API calls
python -m evals.run_eval generation --judge --sleep 2   # full pipeline on the configured LLM (--config / --provider / --model to compare)
python -m evals.run_eval rescore evals/results/<run>.json   # re-grade saved answers, no API calls
pytest                                 # unit tests for the scoring
```

**Retrieval** is scored by **evidence hit@k**: did a retrieved chunk contain the exact passage that answers the question? It also reports MRR, page-level hit rate and context precision.

**Generation** is scored by:
- **answer accuracy**: every expected fact is present in the answer
- **refusal accuracy**: unanswerable questions are refused
- **false-refusal rate**
- **faithfulness**, graded by an LLM judge from a different model family (`qwen3.8-27b` on Groq)

Each wrong answer is labelled a *retrieval miss* (the passage never reached the model) or a *generation miss* (it did, and the model still got it wrong).

### Current results: retrieval

Full report: [`evals/results/retrieval.md`](evals/results/retrieval.md).

| Config | Setup | Evidence hit@4 | MRR | Page hit@4 |
|---|---|---|---|---|
| **app (shipped)** | 1000/200 chunks, similarity top-4 | **59.5%** | **0.429** | **83.3%** |
| mmr (previous default) | 1000/200 chunks, MMR λ=0.5 | 42.9% | 0.367 | 73.8% |
| mmr-0.8 | 1000/200 chunks, MMR λ=0.8 | 50.0% | 0.401 | 78.6% |
| small-chunks | 500/100 chunks, similarity top-8 | 73.8% | 0.480 | 90.5% |

Two findings. **MMR costs recall**: its diversity term pushes out the second-best chunk, and for pinpoint questions that chunk is often the one with the answer, so NEURA ships plain similarity. **Smaller chunks find more**: 500-character chunks with top-8 reach 73.8% evidence hit, 14 points above the shipped setup, because the paper's dense tables and numbers get their own chunks instead of sharing one with surrounding prose. The answer eval for that config is still to run (it hit the Groq free tier's daily token limit), so the shipped config stays put until it's measured end to end. Metadata questions (conference, institutions) miss under every config: the title-page chunk is dominated by the author list and the abstract.

Can the top retrieval score detect a question the document can't answer? The **AUROC is 0.80** (0.83 with 500-character chunks): a useful signal, but not yet clean enough to refuse on the score alone.

### Current results: answers

`gpt-oss-120b` on Groq, all 54 questions, faithfulness judged by `qwen3.8-27b`. Full reports: [similarity](evals/results/generation_app_groq_gpt-oss-120b.md) · [mmr](evals/results/generation_mmr_groq_gpt-oss-120b.md).

| Retrieval | Answer accuracy | Refusal accuracy | Hallucination rate | False refusals | Faithfulness |
|---|---|---|---|---|---|
| **similarity (shipped)** | **69.0%** | 83.3% | 16.7% | **21.4%** | **93.9%** |
| mmr (previous default) | 61.9% | **91.7%** | **8.3%** | 26.2% | 90.3% |

- **The refusal contract mostly holds, and its one crack is instructive.** Similarity refused 10 of 12 traps, MMR 11 of 12. Both misses are general-knowledge traps ("capital of France?", "boiling point of water?"). For those, retrieval returns the paper's Table 7, which contains a worked example: "What is the body of water between England and Ireland?". The model answers *that* question ("the Irish Sea") instead of refusing. It didn't invent a fact; a question-and-answer pair inside the document hijacked the answer. The prompt-level fix is on the roadmap.
- **Every wrong answer traces back to retrieval.** In both runs, each miss happened because the answering passage never reached the model (13 misses with similarity, 16 with MMR). When the passage was in context, the model got it right every time, so better search moves the score directly.
- **Faithfulness is high but not the whole story.** The judge finds 93.9% of the similarity run's answers fully supported by their context. A grounded answer to the wrong question, like the Table 7 case, still counts as supported, which is why refusal accuracy is measured separately.

The graders are tested too. The first run scored some correct answers as wrong because the model writes non-breaking hyphens ("top‑k") and thin spaces ("21 015 324"). The normalizer now handles both, and `rescore` re-grades saved answers without calling the model again.

---

## The frontend

Laid out like a listening-library dashboard: the paper is the library, each suggested question is a title on its shelf, and the composer is the player bar, whose progress track runs while NEURA retrieves and fills as the answer types out.

- **Ask** shows nine real questions from the golden set as cards, each with an authored cover for its type (definition, method, numeric, table, trap) and a link that opens the PDF at the cited page.
- **History** shows saved chats as cards with a preview of the latest answer.
- **Evals** reads `evals/results/` through `/api/evals`, so the numbers on screen are the ones the repo can back up. The top bar repeats the headline (traps refused, answers correct).
- **Document** describes the indexed paper and the pipeline settings, via `/api/info`.
- Answers show their retrieved passages with page links; a refusal gets its own "Not in the document" state.

Palette: ground `#16130E`, oxblood `#5B1408` (only on things you can press), rust `#9F2E10`, umber `#4B3F29`, amber `#E8892F`, apricot `#F1B978`. Hanken Grotesk for titles, Courier Prime for everything else, square corners, no gradients. Views are deep-linkable (`/#evals`, `/#doc`, `/#history`). It's written by hand in vanilla JS and CSS and served as one string from `app.py`. No React, no Tailwind, no bundler, no `node_modules`. Clone and run; there is no build step to break.

---

## Known limits

Being straight about what this does and doesn't do:

- **One corpus at a time.** The PDF path is hardcoded in `create_database.py`; swapping documents means re-indexing. No multi-document routing.
- **No streaming.** `/api/chat` blocks until the full generation returns, so long answers sit behind a spinner.
- **Snippets, not highlights.** Sources are the first 220 characters of each chunk with a link to its page; there is no highlight-in-PDF.
- **No re-ranking.** The top-4 similarity results go straight to the prompt; a cross-encoder pass would sharpen them further.
- **Example Q&A in the document can hijack answers.** A worked question-and-answer pair in the indexed text can get answered in place of a refusal (see the eval).
- **Some questions are out of retrieval's reach.** Metadata questions (which conference, which institutions) miss under every config, because the title-page chunk is dominated by the author list and the abstract.
- **Local and unauthenticated.** Binds to `127.0.0.1` with no auth layer. Don't expose it as-is.

---

## Roadmap

- [ ] Token streaming over SSE
- [ ] Multi-document indexing with per-source filters
- [ ] Page-anchored citations that jump into the PDF
- [ ] Cross-encoder re-ranking after retrieval (measured with `evals/`)
- [ ] Refuse when the retrieved context answers a different question than the one asked
- [ ] Answer-eval the 500/100 chunking and ship it if it wins end to end
- [ ] Drag-and-drop upload + in-app re-indexing
- [x] Swappable LLM backends: Groq, Hugging Face, or any OpenAI-compatible API

---

<div align="center">

**LangChain** · **Chroma** · **Groq** · **Hugging Face** · **FastAPI** · **SQLite** · **vanilla JS**

Built by [@daivraval](https://github.com/daivraval) · ⭐ it if the grounding contract is your kind of thing

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:2fd6c4,55:3b9dfc,100:6f5bf5&height=110&section=footer" />

</div>
