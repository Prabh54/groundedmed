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
