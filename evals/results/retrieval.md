# Retrieval eval

_2026-09-30 07:35 UTC · 42 answerable questions · offline (no LLM calls)_

**Evidence hit@k** — a retrieved chunk contains the exact passage that answers the question. **MRR** — 1/rank of that chunk. **Page hit@k** — a chunk from the right page (lenient). **Context precision** — share of retrieved chunks from a right page.

| Config | Setup | Evidence hit@k | MRR | Page hit@k | Context precision | p50 latency |
|---|---|---|---|---|---|---|
| **app** | chunk 1000/200, similarity, k=4 | 59.5% | 0.429 | 83.3% | 34.5% | 8 ms |
| mmr | chunk 1000/200, mmr, k=4, fetch_k=10 lambda=0.5 | 42.9% | 0.367 | 73.8% | 28.0% | 11 ms |
| mmr-0.8 | chunk 1000/200, mmr, k=4, fetch_k=20 lambda=0.8 | 50.0% | 0.401 | 78.6% | 32.1% | 12 ms |
| small-chunks | chunk 500/100, similarity, k=8 | 73.8% | 0.480 | 90.5% | 28.0% | 8 ms |

## Evidence hit@k by question type

| Config | definition (n=8) | metadata (n=2) | method (n=12) | numeric (n=12) | table (n=8) |
|---|---|---|---|---|---|
| app | 75.0% | 0.0% | 66.7% | 66.7% | 37.5% |
| mmr | 25.0% | 0.0% | 83.3% | 41.7% | 12.5% |
| mmr-0.8 | 50.0% | 0.0% | 75.0% | 50.0% | 25.0% |
| small-chunks | 87.5% | 0.0% | 75.0% | 83.3% | 62.5% |

## Can the retriever tell when the answer isn't there?

AUROC of the top-1 relevance score, answerable vs the 12 unanswerable questions (0.5 = coin flip, 1.0 = perfect split). High values mean a score threshold could refuse *before* calling the LLM.

| Chunking | AUROC |
|---|---|
| 1000/200 | 0.804 |
| 500/100 | 0.833 |

## Misses

- **app**: q02, q07, q10, q11, q14, q17, q22, q23, q30, q31, q33, q34, q35, q36, q40, q41, q42
- **mmr**: q02, q04, q05, q06, q07, q08, q11, q17, q21, q22, q23, q25, q29, q30, q31, q33, q34, q35, q36, q37, q39, q40, q41, q42
- **mmr-0.8**: q02, q04, q07, q08, q11, q14, q17, q22, q23, q25, q29, q30, q31, q33, q34, q35, q36, q37, q40, q41, q42
- **small-chunks**: q07, q10, q11, q17, q30, q31, q36, q38, q40, q41, q42
