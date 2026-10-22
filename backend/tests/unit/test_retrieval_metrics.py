"""Hand-checked examples for eval/metrics/retrieval.py."""

import pytest

from eval.metrics.retrieval import gold_rank, hit_at_k, mean_hit_at_k


def test_gold_rank():
    assert gold_rank(["a", "b", "c"], "a") == 1
    assert gold_rank(["a", "b", "c"], "c") == 3
    assert gold_rank(["a", "b", "c"], "z") is None


def test_hit_at_k_boundaries():
    ranked = ["a", "b", "c", "d", "e", "f"]
    assert hit_at_k(ranked, "a", 1)
    assert not hit_at_k(ranked, "b", 1)
    assert hit_at_k(ranked, "e", 5)
    assert not hit_at_k(ranked, "f", 5)
    assert not hit_at_k(["a"], "z", 5)
    with pytest.raises(ValueError):
        hit_at_k(ranked, "a", 0)


def test_mean_hit_at_k_hand_example():
    rankings = [["g1", "x"], ["x", "g2"], ["x", "y"], ["g4"]]
    gold = ["g1", "g2", "g3", "g4"]
    assert mean_hit_at_k(rankings, gold, 1) == 0.5  # q1, q4
    assert mean_hit_at_k(rankings, gold, 5) == 0.75  # q1, q2, q4


def test_mean_hit_at_k_rejects_bad_input():
    with pytest.raises(ValueError):
        mean_hit_at_k([["a"]], ["a", "b"], 1)
    with pytest.raises(ValueError):
        mean_hit_at_k([], [], 1)
