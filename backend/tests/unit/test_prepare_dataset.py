"""Offline tests for scripts/prepare_dataset.py using small synthetic parquet files."""

import dataclasses
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import prepare_dataset as pd

SENTINELS = {
    "long_answer": "SENTINEL_LONG_ANSWER",
    "final_decision": "SENTINEL_DECISION",
    "reasoning_required_pred": "SENTINEL_REASONING_REQUIRED",
    "reasoning_free_pred": "SENTINEL_REASONING_FREE",
}


def raw_row(pubid, *, labeled=True, meshes=("Humans", "Asthma"), label=None):
    label = label or SENTINELS["final_decision"]
    context = {
        "contexts": [f"Background of {pubid}.", f"Results of {pubid}."],
        "labels": ["BACKGROUND", "RESULTS"],
        "meshes": list(meshes),
    }
    row = {"pubid": pubid, "question": f"Question {pubid}?", "context": context}
    row["long_answer"] = SENTINELS["long_answer"]
    if labeled:
        # Real values are char lists (['y','e','s']); a whole-word sentinel makes leaks visible.
        context["reasoning_required_pred"] = [SENTINELS["reasoning_required_pred"]]
        context["reasoning_free_pred"] = [SENTINELS["reasoning_free_pred"]]
        row["final_decision"] = label
    return row


def write_parquet(tmp_path, rows, name="rows.parquet"):
    path = tmp_path / name
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def test_answer_fields_never_reach_document_output(tmp_path):
    path = write_parquet(tmp_path, [raw_row(101), raw_row(102)])
    rows = pd.read_rows(path, pd.LABELED)

    for row in rows:
        document_output = json.dumps(dataclasses.asdict(row.document))
        for field, sentinel in SENTINELS.items():
            assert field not in document_output
            assert sentinel not in document_output
        assert {f.name for f in dataclasses.fields(row.document)} == {
            "pmid",
            "source",
            "sections",
            "meshes",
        }
        # The full row keeps nothing from long_answer or reasoning_* either.
        for field in ("long_answer", "reasoning_required_pred", "reasoning_free_pred"):
            assert SENTINELS[field] not in repr(row)


def test_final_decision_only_on_benchmark_item(tmp_path):
    path = write_parquet(tmp_path, [raw_row(101, label="no")])
    (row,) = pd.read_rows(path, pd.LABELED)
    assert pd.benchmark_item(row) == pd.BenchmarkItem("101", "Question 101?", "no")
    assert "final_decision" not in json.dumps(dataclasses.asdict(row.document))


def test_unlabeled_rows_load_without_label(tmp_path):
    path = write_parquet(tmp_path, [raw_row(201, labeled=False)])
    (row,) = pd.read_rows(path, pd.UNLABELED)
    assert row.final_decision is None
    assert row.document.source == "pqa_u"
    assert row.document.sections[1] == pd.Section("RESULTS", "Results of 201.")


def test_unexpected_label_rejected(tmp_path):
    path = write_parquet(tmp_path, [raw_row(101, label="probably")])
    (row,) = pd.read_rows(path, pd.LABELED)
    with pytest.raises(ValueError, match="unexpected label"):
        pd.benchmark_item(row)


def synthetic_items():
    counts = {"yes": 552, "no": 338, "maybe": 110}
    items, pmid = [], 1000
    for label, n in counts.items():
        for _ in range(n):
            pmid += 7
            items.append(pd.BenchmarkItem(str(pmid), f"Q{pmid}?", label))
    return items


def test_split_is_stratified_and_disjoint():
    splits = pd.split_labeled(synthetic_items())
    assert {name: pd.label_counts(splits[name]) for name in pd.SPLITS} == {
        "test": {"yes": 110, "no": 68, "maybe": 22},
        "dev": {"yes": 55, "no": 34, "maybe": 11},
        "reserve": {"yes": 387, "no": 236, "maybe": 77},
    }
    ids = [{item.pmid for item in splits[name]} for name in pd.SPLITS]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert sum(len(s) for s in ids) == 1000


def test_split_is_deterministic_and_order_independent():
    items = synthetic_items()
    first = pd.split_labeled(items)
    assert pd.split_labeled(list(reversed(items))) == first
    assert pd.split_labeled(items, seed=1) != first


def test_write_frozen_refuses_to_change_existing_file(tmp_path):
    path = tmp_path / "test.json"
    pd.write_frozen({path: "original\n"})
    pd.write_frozen({path: "original\n"})  # identical content is a no-op
    other = tmp_path / "dev.json"
    with pytest.raises(pd.FrozenManifestError):
        pd.write_frozen({other: "new\n", path: "changed\n"})
    assert path.read_text(encoding="utf-8") == "original\n"
    assert not other.exists()  # nothing is written when any file would change


def doc(pmid, meshes, source="pqa_u", text=None):
    sections = (pd.Section("RESULTS", text or f"Text {pmid}."),)
    return pd.Document(str(pmid), source, sections, tuple(meshes))


def test_generic_mesh_terms_by_share():
    docs = [doc(i, ["Humans", f"Topic{i}"]) for i in range(100)]
    docs += [doc(1000 + i, ["Humans", "Rare"]) for i in range(3)]
    # Humans tags all 103 papers; Rare tags 3/103 (2.9%, not generic); Topic<i> 1 each.
    assert pd.generic_mesh_terms(docs) == {"Humans"}


def test_select_distractors_ranks_by_shared_specific_terms_round_robin():
    labeled = [doc(2, ["A", "B", "C"], "pqa_l"), doc(1, ["X", "Y"], "pqa_l")]
    candidates = [
        doc(10, ["A", "B", "C", "G"]),  # 3 shared with paper 2
        doc(11, ["A", "B", "G"]),  # 2 shared with paper 2
        doc(12, ["A", "B"]),  # 2 shared with paper 2; loses PMID tie-break to 11
        doc(13, ["X", "G"]),  # 1 shared with paper 1
        doc(14, ["G"]),  # generic overlap only: never chosen
    ]
    # Round 1: paper 1 (lower PMID) takes 13, paper 2 takes 10. Round 2: paper 2 takes 11.
    assert pd.select_distractors(labeled, candidates, {"G"}, 3) == ["10", "11", "13"]
    with pytest.raises(ValueError, match="only 4 distractor candidates"):
        pd.select_distractors(labeled, candidates, {"G"}, 5)


def test_labeled_duplicates_by_question_or_opening_text():
    def row(pmid, question, text):
        return pd.SourceRow(doc(pmid, ["A"], text=text), question, None)

    labeled = [row(1, "Does X help?", "Original abstract text " * 20)]
    unlabeled = [
        row(2, "does x help", "Different text " * 20),  # same question after normalizing
        row(3, "Other question?", "Original abstract text " * 20 + "extra"),  # same opening
        row(4, "Other question?", "Unrelated " * 20),
    ]
    assert pd.labeled_duplicates(labeled, unlabeled) == {"2", "3"}


def test_corpus_record_contains_only_allowed_fields():
    record = pd.corpus_record(doc(5, ["A"], "pqa_l", text="Body."))
    assert set(record) == {
        "pmid",
        "source",
        "sections",
        "passage",
        "meshes",
        "content_sha256",
        "corpus_version",
    }
    assert record["passage"] == "Body."
    assert record["content_sha256"] == pd.sha256_text("Body.")
