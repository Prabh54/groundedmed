"""Answer metrics and the eval runner's test-split guard (PLAN.md M2-T5, ARCHITECTURE §9)."""

import pytest

from eval.metrics.answers import (
    accuracy,
    answered_only_accuracy,
    coverage,
    majority_class_accuracy,
)
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
