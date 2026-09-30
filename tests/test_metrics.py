import re

from evals import metrics as M
from evals.run_eval import load_golden


def test_normalize_folds_ligatures_hyphenation_and_whitespace():
    assert M.normalize("ﬁgure-of-merit\n  is  ﬁne") == "figure-of-merit is fine"
    assert M.normalize("re-\ntriever model") == "retriever model"
    # models write non-breaking hyphens and thin spaces
    assert M.normalize("top‑k over 21 015 324 passages") == "top-k over 21 015 324 passages"


def test_evidence_rank_is_one_based_and_normalized():
    chunks = ["unrelated text", "The batch\nsize is 128.", "size is 128 again"]
    assert M.evidence_rank(chunks, "the batch size is 128") == 2
    assert M.evidence_rank(chunks, "not there") is None


def test_reciprocal_rank():
    assert M.reciprocal_rank(1) == 1.0
    assert M.reciprocal_rank(4) == 0.25
    assert M.reciprocal_rank(None) == 0.0


def test_page_precision():
    assert M.page_precision([2, 2, 5, 8], [2]) == 0.5
    assert M.page_precision([], [2]) == 0.0


def test_is_refusal_catches_the_contract_and_paraphrases():
    assert M.is_refusal("I could not find the answer in the document.")
    assert M.is_refusal("The batch size is not mentioned in the provided context.")
    assert M.is_refusal("The document does not specify which framework was used.")
    assert not M.is_refusal("DPR is trained with Adam at a learning rate of 1e-5.")


def test_keyword_hits_uses_regex_alternatives():
    answer = "It is trained with the Adam optimizer at a learning rate of 1e-5."
    assert M.keyword_hits(answer, [r"adam", r"1e-?5", r"\b256\b"]) == [True, True, False]


def test_parse_verdict():
    assert M.parse_verdict("SUPPORTED") is True
    assert M.parse_verdict("unsupported.") is False
    assert M.parse_verdict("maybe") is None


def test_auroc():
    assert M.auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert M.auroc([0.5], [0.5]) == 0.5
    assert M.auroc([0.1], [0.9]) == 0.0
    assert M.auroc([], [0.3]) is None


def test_mean_skips_none():
    assert M.mean([1, None, 0]) == 0.5
    assert M.mean([]) is None


def test_golden_set_is_well_formed():
    questions = load_golden()
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))
    assert any(not q["answerable"] for q in questions)
    for q in questions:
        if q["answerable"]:
            assert q["evidence"] and q["pages"] and q["expect"], q["id"]
            for pattern in q["expect"]:
                re.compile(pattern)
