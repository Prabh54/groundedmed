"""Benchmark-integrity (leakage) tests on the real corpus.

Needs `data/corpus.jsonl` (run `uv run python -m scripts.prepare_dataset`) and the pinned dataset
(downloaded once, then cached). Skipped locally if the corpus is missing; fails in CI.
`long_answer` is read here only to prove it is absent from the corpus.
"""

import hashlib
import json
import os
import re
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from scripts import prepare_dataset as pd

DATA = Path(__file__).resolve().parents[3] / "data"
CORPUS = DATA / "corpus.jsonl"
KNOWN_DUPLICATES = {"18270496", "17491578"}  # copies of labeled paper 18019905
# PMID 24519615's conclusion repeats two of its own RESULTS sentences verbatim. These are
# results, not the conclusion's inference; the context is the dataset's conclusion-free text as-is.
ALLOWED_CONCLUSION_SENTENCE_HITS = {("24519615", "24519615"): 2}
MIN_SENTENCE_CHARS = 60

if not CORPUS.exists():
    if os.environ.get("CI"):
        raise RuntimeError(f"{CORPUS} missing: CI must run scripts.prepare_dataset first")
    pytest.skip("data/corpus.jsonl not built", allow_module_level=True)


@pytest.fixture(scope="module")
def corpus():
    return [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def labeled_raw():
    path = pd.dataset_path(pd.LABELED)
    return pq.read_table(path, columns=["pubid", "question", "long_answer"]).to_pylist()


@pytest.fixture(scope="module")
def corpus_text(corpus):
    """All normalized passages joined by newlines (normalized text never contains one)."""
    return "\n".join(pd.normalize(r["passage"]) for r in corpus)


def sentences(text):
    parts = (pd.normalize(s) for s in re.split(r"(?<=[.!?])\s+", text))
    return [s for s in parts if len(s) >= MIN_SENTENCE_CHARS]


def test_corpus_matches_frozen_manifest(corpus):
    manifest = json.loads((DATA / "manifests" / "corpus.json").read_text(encoding="utf-8"))
    assert manifest["count"] == len(corpus) == 2500
    assert manifest["source_counts"] == {"pqa_l": 1000, "pqa_u": 1500}
    expected = [(d["pmid"], d["source"], d["content_sha256"]) for d in manifest["documents"]]
    assert [(r["pmid"], r["source"], r["content_sha256"]) for r in corpus] == expected


def test_records_hold_only_allowed_fields(corpus):
    allowed = {
        "pmid",
        "source",
        "sections",
        "passage",
        "meshes",
        "content_sha256",
        "corpus_version",
    }
    for record in corpus:
        assert set(record) == allowed
        assert record["corpus_version"] == pd.CORPUS_VERSION
        for section in record["sections"]:
            assert set(section) == {"label", "text"}
            assert "CONCLU" not in section["label"].upper()


def test_passage_is_exactly_the_section_texts(corpus):
    for record in corpus:
        assert record["passage"] == "\n\n".join(s["text"] for s in record["sections"])
        digest = hashlib.sha256(record["passage"].encode("utf-8")).hexdigest()
        assert digest == record["content_sha256"]


# Each check below collects offending IDs and asserts the list is empty. Never assert
# `x not in corpus_text` directly: on failure pytest diffs the multi-MB string and hangs.


def test_no_reference_conclusion_in_indexed_text(corpus, labeled_raw, corpus_text):
    leaked = [
        row["pubid"] for row in labeled_raw if pd.normalize(row["long_answer"]) in corpus_text
    ]
    assert leaked == []

    hits = {}
    for row in labeled_raw:
        for sentence in sentences(row["long_answer"]):
            if sentence not in corpus_text:
                continue
            for record in corpus:
                if sentence in pd.normalize(record["passage"]):
                    key = (str(row["pubid"]), record["pmid"])
                    hits[key] = hits.get(key, 0) + 1
    assert hits == ALLOWED_CONCLUSION_SENTENCE_HITS


def test_no_question_text_as_document_text(labeled_raw, corpus_text):
    leaked = [row["pubid"] for row in labeled_raw if pd.normalize(row["question"]) in corpus_text]
    assert leaked == []


def test_no_answer_labels_in_metadata(corpus):
    serialized = json.dumps([{k: v for k, v in r.items() if k != "passage"} for r in corpus])
    leaked = [field for field in pd.ANSWER_FIELDS if field in serialized]
    assert leaked == []


def test_all_labeled_papers_present_and_unique(corpus):
    pmids = [r["pmid"] for r in corpus]
    assert len(pmids) == len(set(pmids))
    passages = [pd.normalize(r["passage"]) for r in corpus]
    assert len(passages) == len(set(passages))
    labeled = {r["pmid"] for r in corpus if r["source"] == "pqa_l"}
    for name in pd.SPLITS:
        manifest = json.loads((DATA / "manifests" / f"{name}.json").read_text(encoding="utf-8"))
        assert {item["pmid"] for item in manifest["items"]} <= labeled


def test_no_duplicates_of_labeled_papers(corpus, labeled_raw):
    pmids = {r["pmid"] for r in corpus}
    assert not KNOWN_DUPLICATES & pmids

    labeled_ids = {str(row["pubid"]) for row in labeled_raw}
    labeled_questions = {pd.normalize(row["question"]) for row in labeled_raw}
    labeled_prefixes = {
        pd.normalize(r["passage"])[: pd.DUPLICATE_PREFIX_CHARS]
        for r in corpus
        if r["source"] == "pqa_l"
    }
    unlabeled = pd.read_rows(pd.dataset_path(pd.UNLABELED), pd.UNLABELED)
    questions = {row.document.pmid: row.question for row in unlabeled}
    for record in corpus:
        if record["source"] != "pqa_u":
            continue
        assert record["pmid"] not in labeled_ids
        assert pd.normalize(questions[record["pmid"]]) not in labeled_questions
        prefix = pd.normalize(record["passage"])[: pd.DUPLICATE_PREFIX_CHARS]
        assert prefix not in labeled_prefixes
    # Not asserted: a distractor's own question absent from its passage. 2 of 1,500 abstracts
    # contain that wording themselves; we add no question text (passage == section texts, tested
    # above) and distractor questions are never benchmark queries.
