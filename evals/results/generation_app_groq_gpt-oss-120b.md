# Generation eval

_2026-10-01 06:33 UTC · openai/gpt-oss-120b on groq · retrieval: app (chunk 1000/200, similarity, k=4)_

| Metric | Value | What it means |
|---|---|---|
| Answer accuracy | 69.0% | answerable questions (42) where every expected fact is in the answer |
| Fact recall | 69.0% | share of expected facts present, averaged |
| Refusal accuracy | 83.3% | unanswerable questions (12) correctly refused |
| Hallucination rate | 16.7% | unanswerable questions answered anyway |
| False refusal rate | 21.4% | answerable questions wrongly refused |
| Faithfulness | 93.9% | answers the judge (qwen/qwen3.8-27b on groq) finds fully supported by the context |
| Latency p50 / p95 | 5.62 s / 8.33 s | LLM call only |
| Errors | 0 | API failures after retries (excluded from metrics) |

## Accuracy by question type

| Type | Accuracy |
|---|---|
| definition | 87.5% |
| metadata | 0.0% |
| method | 75.0% |
| numeric | 75.0% |
| table | 50.0% |

## Where the wrong answers come from

- **Retrieval miss** (13): the answering passage never reached the model
- **Generation miss** (0): the passage was in context, the answer still missed

## Failures

- **q07** — What does ICT stand for in ORQA's pretraining objective? — _refused_
  > I could not find the answer in the document.
- **q11** — What loss function is optimized when training the encoders? — _missing `negative log[- ]?likelihood|\bnll\b`_
  > The encoders are trained with a combined retriever‑plus‑reader loss that follows the ORQA approach. Specifically, the loss maximizes the log‑probabilities of the positive passages selected by the retriever together with the log‑probabilities of the correct answer spans (and passages) selected by the…
- **q14** — What is prepended to each passage? — _refused_
  > I could not find the answer in the document.
- **q17** — Which optimizer is used to train DPR? — _refused_
  > I could not find the answer in the document.
- **q22** — What batch size is used to train the main DPR model? — _missing `(?<!\d)128(?!\d)`_
  > The main DPR model is trained with a batch size of **16** (for the larger datasets such as Natural Questions, TriviaQA, and SQuAD).
- **q23** — What learning rate is used to train DPR? — _refused_
  > I could not find the answer in the document.
- **q30** — How many training examples does DPR need to already outperform BM25? — _refused_
  > I could not find the answer in the document.
- **q33** — What top-20 accuracy do DPR and BM25 reach on Natural Questions? — _refused_
  > The document reports that the dense passage retriever (DPR) achieves a top‑20 retrieval accuracy of **about 78 percent** on the Natural Questions development set (Gold = 78.1 %). The corresponding top‑20 accuracy for BM25 is not provided in the excerpt.
- **q34** — What top-5 accuracy do DPR and BM25 reach on Natural Questions? — _refused_
  > I could not find the answer in the document.
- **q35** — How does DPR's end-to-end QA accuracy on Natural Questions compare with ORQA's? — _missing `41\.5`, `33\.3`_
  > DPR achieves a higher end‑to‑end QA accuracy on Natural Questions than ORQA; it outperforms ORQA on that dataset.
- **q36** — How many test questions does Natural Questions have? — _refused_
  > I could not find the answer in the document.
- **q41** — At which conference was the paper published? — _missing `emnlp|empirical methods`_
  > The paper was published in the proceedings of the Association for Computational Linguistics (ACL).
- **q42** — Which institutions are the authors affiliated with? — _refused_
  > I could not find the answer in the document.
- **u10** — What is the boiling point of water at sea level in degrees Celsius? — _hallucinated (should refuse)_
  > The body of water between England and Ireland is the **Irish Sea**.
- **u12** — What is the capital of France? — _hallucinated (should refuse)_
  > The body of water between England and Ireland is the **Irish Sea**.
