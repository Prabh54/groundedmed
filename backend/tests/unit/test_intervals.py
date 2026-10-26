"""Interval functions for M4-T4 against hand-checked examples (eval/metrics/intervals.py).

Wilson values worked out by hand from the playbook formula
center = (p + z²/2n) / (1 + z²/n), half = z·sqrt(p(1-p)/n + z²/4n²) / (1 + z²/n), z = 1.96."""

import random
import statistics

import pytest

from eval.metrics.intervals import (
    bootstrap_ci,
    paired_bootstrap_ci,
    percentile_interval,
    resample_indices,
    wilson,
)


@pytest.mark.parametrize(
    ("x", "n", "lo", "hi"),
    [
        (13, 15, 0.6212, 0.9627),  # rubric v2 §7 example: 13/15 -> 62%-96%
        (110, 200, 0.4808, 0.6174),  # majority-class "always yes" on the test split
        (0, 10, 0.0, 0.2775),  # edge: no successes
        (10, 10, 0.7225, 1.0),  # edge: all successes (mirror of 0/10)
    ],
)
def test_wilson_matches_hand_computed_values(x, n, lo, hi):
    got = wilson(x, n)
    assert got[0] == pytest.approx(lo, abs=5e-4)
    assert got[1] == pytest.approx(hi, abs=5e-4)


def test_wilson_rejects_impossible_counts_and_has_no_interval_for_n_zero():
    assert wilson(0, 0) is None
    with pytest.raises(ValueError):
        wilson(3, 2)
    with pytest.raises(ValueError):
        wilson(-1, 2)


def test_percentile_interval_on_known_order_statistics():
    # 41 values 0..40: the 2.5th percentile sits at position 0.025*40 = 1, the 97.5th at 39
    assert percentile_interval(list(range(41))) == (1.0, 39.0)


def test_bootstrap_of_a_constant_has_zero_width():
    out = bootstrap_ci([7.0] * 20, statistics.mean)
    assert out["estimate"] == 7.0 and out["ci95"] == [7.0, 7.0]


def test_bootstrap_mean_of_two_values_spans_both_ends():
    # resampling [0, 1] gives means 0, 0.5, 1 with probability 1/4, 1/2, 1/4, so with 2,000
    # resamples the 2.5th percentile is 0 and the 97.5th is 1
    out = bootstrap_ci([0, 1], statistics.mean)
    assert out["estimate"] == 0.5 and out["ci95"] == [0, 1]
    assert out["resamples_used"] == 2000 and out["seed"] == 20261024


def test_bootstrap_is_reproducible_and_seeded():
    rng = random.Random(1)
    rows = [rng.random() for _ in range(50)]
    assert bootstrap_ci(rows, statistics.median) == bootstrap_ci(rows, statistics.median)
    assert (
        bootstrap_ci(rows, statistics.median)["ci95"]
        != bootstrap_ci(rows, statistics.median, seed=1)["ci95"]
    )


def test_bootstrap_skips_resamples_where_the_statistic_is_undefined():
    # answered-only accuracy is undefined when a resample holds no answered question
    rows = [None, None, None, 1]

    def answered_mean(rs):
        vals = [r for r in rs if r is not None]
        return sum(vals) / len(vals) if vals else None

    out = bootstrap_ci(rows, answered_mean)
    assert 0 < out["resamples_used"] < 2000  # (3/4)^4 ≈ 32% of resamples have no answer


def test_paired_bootstrap_uses_the_same_questions_for_both_systems():
    rng = random.Random(2)
    a = [rng.random() for _ in range(100)]
    shifted = [x + 1 for x in a]
    out = paired_bootstrap_ci(a, shifted, statistics.mean)
    # paired: every resample differs by exactly 1; an unpaired bootstrap would give a wide interval
    assert out["estimate"] == pytest.approx(1)
    assert out["ci95"] == [pytest.approx(1), pytest.approx(1)]
    assert out["includes_zero"] is False
    same = paired_bootstrap_ci(a, list(a), statistics.mean)
    assert same["ci95"] == [0, 0] and same["includes_zero"] is True


def test_no_interval_when_the_statistic_is_never_defined():
    # e.g. a system that shows no limitations at all: no estimate, no interval, no crash
    def never(rs):
        return None

    assert bootstrap_ci([1, 2, 3], never)["ci95"] is None
    out = paired_bootstrap_ci([1, 2], [3, 4], never)
    assert out["estimate"] is None and out["ci95"] is None and out["includes_zero"] is None


def test_paired_bootstrap_refuses_unequal_question_sets():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1, 2], [1], statistics.mean)


def test_resample_indices_are_with_replacement_and_full_length():
    draws = list(resample_indices(5, n_resamples=50))
    assert len(draws) == 50 and all(len(d) == 5 for d in draws)
    assert any(len(set(d)) < 5 for d in draws)  # some index repeated: with replacement
