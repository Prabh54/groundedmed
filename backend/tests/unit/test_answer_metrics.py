"""Answer metrics and the eval runner's test-split guard (PLAN.md M2-T5, ARCHITECTURE §9)."""

import pytest

from eval.metrics.answers import (
    accuracy,
    answered_only_accuracy,
    coverage,
    majority_class_accuracy,
)
from eval.runners import run_system
from eval.runners.run_system import check_allowed

IE = "insufficient_evidence"


def test_refusal_counts_as_incorrect_in_overall_accuracy():
    verdicts, gold = ["yes", "no", IE, "maybe"], ["yes", "yes", "no", "maybe"]
    assert accuracy(verdicts, gold) == 0.5  # 2 of 4; the refusal is wrong
    assert coverage(verdicts) == 0.75
    assert answered_only_accuracy(verdicts, gold) == pytest.approx(2 / 3)


def test_answered_only_accuracy_is_none_when_nothing_answered():
    assert answered_only_accuracy([IE, IE], ["yes", "no"]) is None
    assert coverage([IE, IE]) == 0.0


def test_majority_class_reference():
    assert majority_class_accuracy(["yes", "no", "yes", "maybe"]) == 0.5


def test_metrics_reject_mismatched_or_empty_input():
    with pytest.raises(ValueError):
        accuracy(["yes"], ["yes", "no"])
    with pytest.raises(ValueError):
        accuracy([], [])


def test_test_split_needs_flag_and_clean_tree():
    clean, dirty = {"git_dirty": False}, {"git_dirty": True}
    with pytest.raises(SystemExit):
        check_allowed("test", final_test_ok=False, git=clean)
    with pytest.raises(SystemExit):
        check_allowed("test", final_test_ok=True, git=dirty)
    check_allowed("test", final_test_ok=True, git=clean)
    check_allowed("dev", final_test_ok=False, git=dirty)  # dev runs are always allowed


def test_partial_or_invalid_limits_are_refused():
    clean = {"git_dirty": False}
    with pytest.raises(SystemExit):
        check_allowed("test", final_test_ok=True, git=clean, limit=5)
    with pytest.raises(SystemExit):
        check_allowed("dev", final_test_ok=False, git=clean, limit=0)
    check_allowed("dev", final_test_ok=False, git=clean, limit=3)


def test_run_dir_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(run_system, "RESULTS_DIR", tmp_path)
    run_system.create_run_dir("dev-baseline-x")
    with pytest.raises(FileExistsError):
        run_system.create_run_dir("dev-baseline-x")


def test_tampered_test_manifest_is_refused(tmp_path, monkeypatch):
    (tmp_path / "test.json").write_text('{"split": "test", "items": []}', encoding="utf-8")
    (tmp_path / "test.sha256").write_text("0" * 64 + "  test.json\n", encoding="utf-8")
    monkeypatch.setattr(run_system, "MANIFESTS", tmp_path)
    with pytest.raises(SystemExit):
        run_system.load_manifest("test")


def test_summary_handles_a_single_error_row():
    item = {"pmid": "1", "question": "q", "final_decision": "yes"}
    summary = run_system.summarize([run_system.error_row(item, "RetrievalError: down", 12.0)])
    assert summary["errors"] == 1 and summary["accuracy"] == 0.0 and summary["coverage"] == 0.0
    assert summary["latency_ms"]["p50"] == 12.0
