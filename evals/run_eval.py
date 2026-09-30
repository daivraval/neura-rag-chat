"""
NEURA eval suite — measures whether the pipeline finds the right passage and
whether it answers (or refuses) correctly.

    python -m evals.run_eval validate                 # self-check the golden set
    python -m evals.run_eval retrieval                # offline, compares retrieval configs
    python -m evals.run_eval generation [--judge]     # full pipeline, needs an LLM key (GROQ_API_KEY by default)
    python -m evals.run_eval rescore <results.json>   # re-grade saved answers, no model calls

Each run writes evals/results/<mode>.md (the report) and <mode>.json
(every question, every retrieved page, every answer) so a regression can be
traced to the exact question that broke.

The eval indexes the PDF in memory with each config's chunking, so results
don't depend on whatever state chroma_db/ is in.
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
import warnings
from datetime import datetime, timezone

import yaml
from dotenv import load_dotenv

from evals import metrics as M
from neura import pipeline

warnings.filterwarnings("ignore")
load_dotenv()

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_PATH = os.path.join(HERE, "golden.yaml")
RESULTS_DIR = os.path.join(HERE, "results")

# Retrieval setups to compare. "app" is read from the pipeline itself, so it
# always matches what app.py serves. Every config returns ~4000 characters of
# context, so they compete on the same budget.
CONFIGS = {
    "app": dict(
        chunk_size=pipeline.CHUNK_SIZE,
        chunk_overlap=pipeline.CHUNK_OVERLAP,
        **pipeline.RETRIEVAL,
    ),
    # The app's previous setting, kept as the baseline similarity replaced.
    "mmr": dict(chunk_size=1000, chunk_overlap=200, search_type="mmr", k=4, fetch_k=10, lambda_mult=0.5),
    "mmr-0.8": dict(
        chunk_size=1000, chunk_overlap=200, search_type="mmr", k=4, fetch_k=20, lambda_mult=0.8
    ),
    "small-chunks": dict(chunk_size=500, chunk_overlap=100, search_type="similarity", k=8),
}

# Default judge per provider: a different model family from the default
# answering model, so the judge isn't grading its own work.
JUDGE_MODELS = {
    "groq": "qwen/qwen3.8-27b",
    "huggingface": "meta-llama/Llama-3.3-70B-Instruct",
}
JUDGE_SYSTEM = (
    "You are a strict fact-checker. Decide whether every factual claim in the "
    "ANSWER is directly supported by the CONTEXT. Reply with exactly one word: "
    "SUPPORTED or UNSUPPORTED."
)


# ------------------------------------------------------------------ helpers
def load_golden(path=GOLDEN_PATH):
    with open(path, encoding="utf-8") as f:
        questions = yaml.safe_load(f)["questions"]
    for q in questions:
        q.setdefault("answerable", True)
    return questions


def describe(cfg):
    parts = [f"chunk {cfg['chunk_size']}/{cfg['chunk_overlap']}", cfg["search_type"], f"k={cfg['k']}"]
    if cfg["search_type"] == "mmr":
        parts.append(f"fetch_k={cfg['fetch_k']} lambda={cfg['lambda_mult']}")
    return ", ".join(parts)


_indexes = {}


def get_index(cfg):
    """One in-memory Chroma collection per chunking setting, built once."""
    key = (cfg["chunk_size"], cfg["chunk_overlap"])
    if key not in _indexes:
        chunks = pipeline.load_chunks(chunk_size=key[0], chunk_overlap=key[1])
        store = pipeline.build_index(chunks, collection_name=f"eval_{key[0]}_{key[1]}")
        _indexes[key] = (store, chunks)
    return _indexes[key]


def make_retriever(cfg):
    store, _ = get_index(cfg)
    search = {k: v for k, v in cfg.items() if k not in ("chunk_size", "chunk_overlap")}
    return pipeline.make_retriever(store, **search)


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def write_report(mode, markdown, payload):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{mode}.md"), "w", encoding="utf-8") as f:
        f.write(markdown)
    with open(os.path.join(RESULTS_DIR, f"{mode}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(markdown)
    print(f"\nwrote evals/results/{mode}.md and {mode}.json")


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def select(questions, args):
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    if args.limit:
        questions = questions[: args.limit]
    return questions


# ----------------------------------------------------------------- validate
def cmd_validate(args):
    """Every evidence phrase must sit on its cited page and survive chunking,
    and every expected fact must be findable on those pages — otherwise a
    miss would be the golden set's fault, not the pipeline's."""
    questions = load_golden()
    pages = {d.metadata["page"] + 1: M.normalize(d.page_content) for d in pipeline.load_pages()}
    problems = []

    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        problems.append("duplicate ids")

    for q in questions:
        if not q["answerable"]:
            continue
        for field in ("question", "pages", "evidence", "expect"):
            if not q.get(field):
                problems.append(f"{q['id']}: missing {field}")
        gold_text = " ".join(pages.get(p, "") for p in q["pages"])
        evidence = M.normalize(q["evidence"])
        if evidence not in gold_text:
            found = [p for p, t in pages.items() if evidence in t]
            problems.append(f"{q['id']}: evidence not on pages {q['pages']} (found on {found or 'none'})")
        for pattern in q["expect"]:
            try:
                if not re.search(pattern, gold_text, flags=re.IGNORECASE):
                    problems.append(f"{q['id']}: expected fact /{pattern}/ not on its pages")
            except re.error as exc:
                problems.append(f"{q['id']}: bad regex /{pattern}/ ({exc})")

    for size, overlap in {(c["chunk_size"], c["chunk_overlap"]) for c in CONFIGS.values()}:
        chunks = [c.page_content for c in pipeline.load_chunks(chunk_size=size, chunk_overlap=overlap)]
        for q in questions:
            if q["answerable"] and M.evidence_rank(chunks, q["evidence"]) is None:
                problems.append(f"{q['id']}: evidence split across chunks at chunk_size={size}")

    n_ans = sum(q["answerable"] for q in questions)
    print(f"{len(questions)} questions ({n_ans} answerable, {len(questions) - n_ans} unanswerable)")
    if problems:
        print("\n".join(f"  FAIL {p}" for p in problems))
        sys.exit(1)
    print("golden set OK")


# ---------------------------------------------------------------- retrieval
def cmd_retrieval(args):
    questions = select(load_golden(), args)
    names = args.configs.split(",") if args.configs else list(CONFIGS)
    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]

    summary, details = {}, {}
    for name in names:
        cfg = CONFIGS[name]
        retriever = make_retriever(cfg)
        store, _ = get_index(cfg)
        rows = []
        for q in questions:
            t0 = time.perf_counter()
            docs = retriever.invoke(q["question"])
            latency = (time.perf_counter() - t0) * 1000
            top_score = store.similarity_search_with_relevance_scores(q["question"], k=1)[0][1]
            row = {
                "id": q["id"],
                "type": q["type"],
                "answerable": q["answerable"],
                "top_score": round(top_score, 4),
                "latency_ms": round(latency, 1),
                "pages": [d.metadata["page"] + 1 for d in docs],
            }
            if q["answerable"]:
                rank = M.evidence_rank([d.page_content for d in docs], q["evidence"])
                row.update(
                    gold_pages=q["pages"],
                    evidence_rank=rank,
                    hit=rank is not None,
                    page_hit=any(p in q["pages"] for p in row["pages"]),
                    page_precision=M.page_precision(row["pages"], q["pages"]),
                )
            rows.append(row)

        scored = [r for r in rows if r["answerable"]]
        by_type = {}
        for r in scored:
            by_type.setdefault(r["type"], []).append(r["hit"])
        summary[name] = {
            "config": describe(cfg),
            "hit_at_k": M.mean([r["hit"] for r in scored]),
            "mrr": M.mean([M.reciprocal_rank(r["evidence_rank"]) for r in scored]),
            "page_hit_at_k": M.mean([r["page_hit"] for r in scored]),
            "context_precision": M.mean([r["page_precision"] for r in scored]),
            "latency_ms_p50": statistics.median(r["latency_ms"] for r in rows),
            "hit_by_type": {t: M.mean(v) for t, v in sorted(by_type.items())},
            "abstain_auroc": M.auroc(
                [r["top_score"] for r in rows if r["answerable"]],
                [r["top_score"] for r in rows if not r["answerable"]],
            ),
            "misses": [r["id"] for r in scored if not r["hit"]],
        }
        details[name] = rows
        print(f"  {name:<13} hit@k {pct(summary[name]['hit_at_k'])}  MRR {summary[name]['mrr']:.3f}")

    # The top-1 score depends only on chunking, not on the search strategy.
    auroc_by_chunking = {
        f"{CONFIGS[n]['chunk_size']}/{CONFIGS[n]['chunk_overlap']}": s["abstain_auroc"]
        for n, s in summary.items()
        if s["abstain_auroc"] is not None
    }

    # ---- report
    md = [
        "# Retrieval eval",
        "",
        f"_{now()} · {len(answerable)} answerable questions · offline (no LLM calls)_",
        "",
        "**Evidence hit@k** — a retrieved chunk contains the exact passage that answers the question. "
        "**MRR** — 1/rank of that chunk. **Page hit@k** — a chunk from the right page (lenient). "
        "**Context precision** — share of retrieved chunks from a right page.",
        "",
        md_table(
            ["Config", "Setup", "Evidence hit@k", "MRR", "Page hit@k", "Context precision", "p50 latency"],
            [
                [
                    f"**{n}**" if n == "app" else n,
                    s["config"],
                    pct(s["hit_at_k"]),
                    f"{s['mrr']:.3f}",
                    pct(s["page_hit_at_k"]),
                    pct(s["context_precision"]),
                    f"{s['latency_ms_p50']:.0f} ms",
                ]
                for n, s in summary.items()
            ],
        ),
        "",
        "## Evidence hit@k by question type",
        "",
    ]
    types = sorted({q["type"] for q in answerable})
    md.append(
        md_table(
            ["Config"] + [f"{t} (n={sum(q['type'] == t for q in answerable)})" for t in types],
            [[n] + [pct(s["hit_by_type"].get(t)) for t in types] for n, s in summary.items()],
        )
    )
    md += [
        "",
        "## Can the retriever tell when the answer isn't there?",
        "",
        f"AUROC of the top-1 relevance score, answerable vs the {len(unanswerable)} unanswerable "
        "questions (0.5 = coin flip, 1.0 = perfect split). High values mean a score threshold "
        "could refuse *before* calling the LLM.",
        "",
        md_table(["Chunking", "AUROC"], [[c, f"{a:.3f}"] for c, a in auroc_by_chunking.items()]),
        "",
        "## Misses",
        "",
    ]
    for n, s in summary.items():
        md.append(f"- **{n}**: {', '.join(s['misses']) or 'none'}")
    write_report("retrieval", "\n".join(md) + "\n", {"generated": now(), "summary": summary, "questions": details})
    check_threshold(args, summary.get("app", next(iter(summary.values())))["hit_at_k"], "app hit@k")


# --------------------------------------------------------------- generation
class Fatal(Exception):
    """An API error retrying can't fix — out of credits, bad token, unknown model."""


def call_with_retry(fn, attempts=5, wait=5):
    """Retry transient failures (429 rate limits, timeouts, cold models) with
    exponential backoff; give up at once on errors retrying can't fix."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if re.search(r"\b40[0-4]\b|Payment Required|model_not_supported|model_not_found", str(exc)):
                raise Fatal(str(exc).splitlines()[0]) from exc
            if i == attempts - 1:
                raise
            delay = min(60, wait * 2**i)
            print(f"    retry in {delay}s after error: {str(exc).splitlines()[0]}")
            time.sleep(delay)


def cmd_generation(args):
    from langchain_core.prompts import ChatPromptTemplate

    try:
        provider, model = pipeline.resolve_llm(args.provider, args.model)
        llm = pipeline.make_llm(provider=provider, model=model)
        judge = judge_prompt = judge_label = None
        if args.judge:
            judge_provider, judge_model = pipeline.resolve_llm(
                args.judge_provider or provider, args.judge_model or JUDGE_MODELS.get(args.judge_provider or provider)
            )
            judge = pipeline.make_llm(max_new_tokens=10, provider=judge_provider, model=judge_model)
            judge_label = f"{judge_model} on {judge_provider}"
    except (ValueError, RuntimeError) as exc:
        sys.exit(str(exc))

    questions = select(load_golden(), args)
    cfg = CONFIGS[args.config]
    retriever = make_retriever(cfg)
    prompt = pipeline.make_prompt()
    if args.judge:
        judge_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", JUDGE_SYSTEM),
                ("human", "CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:\n{answer}"),
            ]
        )

    rows = []
    for i, q in enumerate(questions, 1):
        if i > 1 and args.sleep:
            time.sleep(args.sleep)  # stay under free-tier rate limits
        docs = retriever.invoke(q["question"])
        row = {"id": q["id"], "type": q["type"], "answerable": q["answerable"], "question": q["question"]}
        t0 = time.perf_counter()
        try:
            answer = call_with_retry(lambda: pipeline.generate(llm, prompt, q["question"], docs))
        except Fatal as exc:
            sys.exit(f"stopping at {q['id']} — {exc}")
        except Exception as exc:
            row["error"] = str(exc)
            rows.append(row)
            print(f"  [{i}/{len(questions)}] {q['id']} ERROR {exc}")
            if len(rows) >= 3 and all("error" in r for r in rows[-3:]):
                sys.exit("3 questions in a row failed — stopping (is the model served? check --model).")
            continue
        row["latency_s"] = round(time.perf_counter() - t0, 2)
        row["answer"] = answer
        if q["answerable"]:
            row["retrieved"] = M.evidence_rank([d.page_content for d in docs], q["evidence"]) is not None
        grade(q, row)

        if q["answerable"]:
            if judge and not row["refused"]:
                try:
                    reply = call_with_retry(
                        lambda: judge.invoke(
                            judge_prompt.invoke(
                                {"context": pipeline.format_context(docs), "question": q["question"], "answer": answer}
                            )
                        ).content
                    )
                    row["faithful"] = M.parse_verdict(reply)
                except Exception as exc:  # a judge failure shouldn't sink the answer
                    row["judge_error"] = str(exc)

        rows.append(row)
        mark = "ok  " if row["correct"] else "FAIL"
        print(f"  [{i}/{len(questions)}] {q['id']} {mark} {row['latency_s']:.1f}s")

    if not any("error" not in r for r in rows):
        sys.exit("every question errored — no report written.")
    meta = {"config": f"{args.config} ({describe(cfg)})", "model": f"{model} on {provider}", "judge": judge_label}
    # One report per retrieval config + model, so runs can be compared side by side.
    name = f"generation_{args.config}_{provider}_" + re.sub(r"[^a-z0-9.]+", "-", model.split("/")[-1].lower())
    s = report_generation(name, rows, meta)
    check_threshold(args, s["answer_accuracy"], "answer accuracy")


def grade(q, row):
    """Score one saved answer against its golden entry — no model calls."""
    row["refused"] = M.is_refusal(row["answer"])
    if q["answerable"]:
        hits = M.keyword_hits(row["answer"], q["expect"])
        row["fact_recall"] = sum(hits) / len(hits)
        row["missing"] = [p for p, h in zip(q["expect"], hits) if not h]
        row["correct"] = all(hits) and not row["refused"]
    else:
        row["correct"] = row["refused"]


def report_generation(name, rows, meta):
    judged = bool(meta["judge"])
    judge_label = meta["judge"]
    done = [r for r in rows if "error" not in r]
    ans = [r for r in done if r["answerable"]]
    una = [r for r in done if not r["answerable"]]
    wrong = [r for r in ans if not r["correct"]]
    latencies = sorted(r["latency_s"] for r in done)
    s = {
        **meta,
        "judge_failures": sum(r.get("faithful") is None for r in ans if judged and not r["refused"]),
        "answered": len(done),
        "errors": len(rows) - len(done),
        "answer_accuracy": M.mean([r["correct"] for r in ans]),
        "fact_recall": M.mean([r["fact_recall"] for r in ans]),
        "false_refusal_rate": M.mean([r["refused"] for r in ans]),
        "refusal_accuracy": M.mean([r["refused"] for r in una]),
        "faithfulness": M.mean([r.get("faithful") for r in ans]) if judged else None,
        "wrong_retrieval_miss": sum(not r["retrieved"] for r in wrong),
        "wrong_generation_miss": sum(r["retrieved"] for r in wrong),
        "latency_s_p50": latencies[len(latencies) // 2] if latencies else None,
        "latency_s_p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else None,
        "accuracy_by_type": {},
    }
    for t in sorted({r["type"] for r in ans}):
        s["accuracy_by_type"][t] = M.mean([r["correct"] for r in ans if r["type"] == t])

    md = [
        "# Generation eval",
        "",
        f"_{now()} · {s['model']} · retrieval: {s['config']}_",
        "",
        md_table(
            ["Metric", "Value", "What it means"],
            [
                ["Answer accuracy", pct(s["answer_accuracy"]), f"answerable questions ({len(ans)}) where every expected fact is in the answer"],
                ["Fact recall", pct(s["fact_recall"]), "share of expected facts present, averaged"],
                ["Refusal accuracy", pct(s["refusal_accuracy"]), f"unanswerable questions ({len(una)}) correctly refused"],
                ["Hallucination rate", pct(None if s["refusal_accuracy"] is None else 1 - s["refusal_accuracy"]), "unanswerable questions answered anyway"],
                ["False refusal rate", pct(s["false_refusal_rate"]), "answerable questions wrongly refused"],
                ["Faithfulness", pct(s["faithfulness"]) if judged else "not run (`--judge`)", f"answers the judge ({judge_label}) finds fully supported by the context"
                 + (f"; {s['judge_failures']} verdicts unusable" if s["judge_failures"] else "")],
                ["Latency p50 / p95", f"{s['latency_s_p50']} s / {s['latency_s_p95']} s", "LLM call only"],
                ["Errors", s["errors"], "API failures after retries (excluded from metrics)"],
            ],
        ),
        "",
        "## Accuracy by question type",
        "",
        md_table(["Type", "Accuracy"], [[t, pct(v)] for t, v in s["accuracy_by_type"].items()]),
        "",
        "## Where the wrong answers come from",
        "",
        f"- **Retrieval miss** ({s['wrong_retrieval_miss']}): the answering passage never reached the model",
        f"- **Generation miss** ({s['wrong_generation_miss']}): the passage was in context, the answer still missed",
        "",
        "## Failures",
        "",
    ]
    failed = [r for r in done if not r["correct"]]
    if not failed:
        md.append("None.")
    for r in failed:
        why = "hallucinated (should refuse)" if not r["answerable"] else (
            "refused" if r["refused"] else f"missing {', '.join(f'`{p}`' for p in r['missing'])}"
        )
        answer = " ".join(r["answer"].split())
        md += [f"- **{r['id']}** — {r['question']} — _{why}_", f"  > {answer[:300]}{'…' if len(answer) > 300 else ''}"]
    write_report(name, "\n".join(md) + "\n", {"generated": now(), "summary": s, "questions": rows})
    return s


def cmd_rescore(args):
    """Re-grade saved answers after fixing the golden set or the scorer — no
    model calls, so it's free and doesn't touch rate limits. Faithfulness
    verdicts are kept as they were."""
    with open(args.results, encoding="utf-8") as f:
        saved = json.load(f)
    golden = {q["id"]: q for q in load_golden()}
    rows = saved["questions"]
    for row in rows:
        if "error" not in row:
            grade(golden[row["id"]], row)
    meta = {k: saved["summary"][k] for k in ("config", "model", "judge")}
    report_generation(os.path.splitext(os.path.basename(args.results))[0], rows, meta)


def check_threshold(args, value, label):
    if args.fail_under is not None and (value is None or value < args.fail_under):
        sys.exit(f"{label} {pct(value)} is below --fail-under {pct(args.fail_under)}")


# ---------------------------------------------------------------------- cli
def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="NEURA eval suite")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("validate", help="self-check the golden set against the PDF")
    rescore = sub.add_parser("rescore", help="re-grade a saved generation run without model calls")
    rescore.add_argument("results", help="path to a generation_*.json file")

    for mode in ("retrieval", "generation"):
        p = sub.add_parser(mode)
        p.add_argument("--ids", help="comma-separated question ids, e.g. q01,u03")
        p.add_argument("--limit", type=int, help="only the first N questions")
        p.add_argument("--fail-under", type=float, help="exit 1 if the headline metric is below this (0-1)")
    sub.choices["retrieval"].add_argument("--configs", help=f"comma-separated subset of {','.join(CONFIGS)}")
    gen = sub.choices["generation"]
    gen.add_argument("--config", default="app", choices=list(CONFIGS), help="retrieval config (default: app)")
    gen.add_argument("--judge", action="store_true", help="also score faithfulness with an LLM judge")
    gen.add_argument("--provider", choices=list(pipeline.PROVIDERS), help="LLM provider (default: LLM_PROVIDER or groq)")
    gen.add_argument("--model", help="answering model id (default: LLM_MODEL or the provider's preset)")
    gen.add_argument("--judge-provider", choices=list(pipeline.PROVIDERS), help="judge provider (default: same as --provider)")
    gen.add_argument("--judge-model", help=f"judge model id (default per provider: {JUDGE_MODELS})")
    gen.add_argument("--sleep", type=float, default=0, help="seconds to pause between questions (rate limits)")

    args = parser.parse_args()
    modes = {"validate": cmd_validate, "retrieval": cmd_retrieval, "generation": cmd_generation, "rescore": cmd_rescore}
    modes[args.mode](args)


if __name__ == "__main__":
    main()
