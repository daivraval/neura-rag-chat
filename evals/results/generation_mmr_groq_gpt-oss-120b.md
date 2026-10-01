# Generation eval

_2026-10-01 06:46 UTC · openai/gpt-oss-120b on groq · retrieval: mmr (chunk 1000/200, mmr, k=4, fetch_k=10 lambda=0.5)_

| Metric | Value | What it means |
|---|---|---|
| Answer accuracy | 61.9% | answerable questions (42) where every expected fact is in the answer |
| Fact recall | 63.1% | share of expected facts present, averaged |
| Refusal accuracy | 91.7% | unanswerable questions (12) correctly refused |
| Hallucination rate | 8.3% | unanswerable questions answered anyway |
| False refusal rate | 26.2% | answerable questions wrongly refused |
| Faithfulness | 90.3% | answers the judge (qwen/qwen3.8-27b on groq) finds fully supported by the context |
| Latency p50 / p95 | 5.86 s / 59.25 s | LLM call only |
| Errors | 0 | API failures after retries (excluded from metrics) |

## Accuracy by question type

| Type | Accuracy |
|---|---|
| definition | 75.0% |
| metadata | 0.0% |
| method | 91.7% |
| numeric | 50.0% |
| table | 37.5% |

## Where the wrong answers come from

- **Retrieval miss** (16): the answering passage never reached the model
- **Generation miss** (0): the passage was in context, the answer still missed

## Failures

- **q06** — What networks are used as the question and passage encoders? — _missing `two|independent|separate`_
  > The question and passage encoders are both implemented as BERT‑based dual encoders – i.e., each side of the dual‑encoder architecture uses a pretrained BERT model to encode the question or the passage.
- **q07** — What does ICT stand for in ORQA's pretraining objective? — _refused_
  > I could not find the answer in the document.
- **q17** — Which optimizer is used to train DPR? — _refused_
  > I could not find the answer in the document.
- **q21** — How many passages does the final Wikipedia corpus contain? — _refused_
  > I could not find the answer in the document.
- **q22** — What batch size is used to train the main DPR model? — _missing `(?<!\d)128(?!\d)`_
  > The main DPR model is trained with a batch size of **16**.
- **q23** — What learning rate is used to train DPR? — _refused_
  > I could not find the answer in the document.
- **q25** — What dropout rate is used when training DPR? — _refused_
  > I could not find the answer in the document.
- **q29** — What value of lambda is used to combine BM25 and DPR scores? — _refused_
  > I could not find the answer in the document.
- **q30** — How many training examples does DPR need to already outperform BM25? — _refused_
  > I could not find the answer in the document.
- **q33** — What top-20 accuracy do DPR and BM25 reach on Natural Questions? — _missing `78\.4`, `59\.1`_
  > On the Natural Questions development set, the dense passage retriever (DPR) achieves a top‑20 retrieval accuracy of roughly **78 %** (78.1 % when trained on gold passages and 77.1 % when trained with the top BM25 passage as distant supervision). The BM25 baseline’s top‑20 accuracy is lower, but the …
- **q34** — What top-5 accuracy do DPR and BM25 reach on Natural Questions? — _refused_
  > I could not find the answer in the document.
- **q35** — How does DPR's end-to-end QA accuracy on Natural Questions compare with ORQA's? — _missing `41\.5`, `33\.3`_
  > DPR’s end‑to‑end QA accuracy on Natural Questions is higher than ORQA’s. The text notes that DPR “out‑performs” ORQA on Natural Questions, achieving better exact‑match scores.
- **q36** — How many test questions does Natural Questions have? — _refused_
  > I could not find the answer in the document.
- **q39** — What top-5 accuracy does in-batch training with 127 gold negatives plus 128 BM25 negatives reach? — _refused_
  > I could not find the answer in the document.
- **q41** — At which conference was the paper published? — _missing `emnlp|empirical methods`_
  > The paper was published at the Association for Computational Linguistics (ACL) conference.
- **q42** — Which institutions are the authors affiliated with? — _refused_
  > I could not find the answer in the document.
- **u12** — What is the capital of France? — _hallucinated (should refuse)_
  > The body of water between England and Ireland is the **Irish Sea**.
