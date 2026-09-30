"""
Pure scoring functions for the eval suite — no models, no I/O, unit-tested.
"""

import re
import unicodedata

# Phrasings models use when they decline. The first is the exact contract in
# the system prompt; the rest catch paraphrases so a polite refusal isn't
# scored as a hallucination.
REFUSAL_PATTERNS = [
    r"could not find the answer",
    r"(could not|couldn't|cannot|can't|unable to) (find|locate|determine|answer)",
    r"(is|are) not (mentioned|provided|specified|stated|given|found|available|included|present)",
    r"(does|do) not (mention|provide|specify|state|contain|include|say|give)",
    r"no (information|mention|details?) (about|on|regarding|of)",
]


def normalize(text):
    """Canonical form for substring matching against PDF-extracted text.

    NFKC folds ligatures (the PDF's "ﬁ" -> "fi") and odd spaces, Unicode
    dashes become "-" (models write "K‑means" with a non-breaking hyphen),
    "-\\n" re-joins words the PDF hyphenated across lines, and all whitespace
    collapses to one space.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[‐-―−]", "-", text)
    text = re.sub(r"-\s*\n\s*", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


# ---------------------------------------------------------------- retrieval
def evidence_rank(chunk_texts, evidence):
    """1-based rank of the first chunk containing `evidence`, else None."""
    needle = normalize(evidence)
    for rank, text in enumerate(chunk_texts, start=1):
        if needle in normalize(text):
            return rank
    return None


def reciprocal_rank(rank):
    return 0.0 if rank is None else 1.0 / rank


def page_precision(chunk_pages, gold_pages):
    """Share of retrieved chunks that come from a gold page."""
    if not chunk_pages:
        return 0.0
    gold = set(gold_pages)
    return sum(p in gold for p in chunk_pages) / len(chunk_pages)


# --------------------------------------------------------------- generation
def is_refusal(answer):
    text = normalize(answer)
    return any(re.search(p, text) for p in REFUSAL_PATTERNS)


def keyword_hits(answer, patterns):
    """One bool per required fact. Each pattern is a case-insensitive regex;
    use `a|b` for acceptable alternatives (e.g. "sgd|stochastic gradient")."""
    text = normalize(answer)
    return [re.search(p, text, flags=re.IGNORECASE) is not None for p in patterns]


def parse_verdict(judge_reply):
    """Map a judge reply to True (supported) / False (unsupported) / None."""
    reply = judge_reply.upper()
    if "UNSUPPORTED" in reply:
        return False
    if "SUPPORTED" in reply:
        return True
    return None


# ------------------------------------------------------------- aggregation
def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def auroc(positive_scores, negative_scores):
    """Probability a random positive outscores a random negative (ties = 0.5).

    Used to ask: does the top retrieval score separate answerable questions
    from unanswerable ones well enough to abstain before calling the LLM?
    """
    if not positive_scores or not negative_scores:
        return None
    wins = 0.0
    for p in positive_scores:
        for n in negative_scores:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(positive_scores) * len(negative_scores))
