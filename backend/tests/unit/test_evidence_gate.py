"""Evidence gate and its dev calibration rule (ARCHITECTURE §7)."""

import json

from backend.app.config import (
    EVIDENCE_GATE_DENSE_THRESHOLD,
    EVIDENCE_GATE_DEV_RUN,
    EVIDENCE_GATE_THRESHOLD,
    REPO_ROOT,
)
from backend.app.guardrails.evidence_gate import gate_passes, top_rerank_score
from backend.app.models.retrieval import Candidate
from eval.runners.gate_calibration import choose_threshold, dense_observations, observations


def ev(*scores):
    return [
        Candidate(pmid=str(i), passage="p", sections=[], dense_score=0.5, dense_rank=i,
                  rerank_score=s, rerank_rank=i)
        for i, s in enumerate(scores, 1)
    ]  # fmt: skip


def test_gate_uses_top_rerank_score():
    assert gate_passes(ev(0.3, 0.8), 0.72, 0.62)
    assert not gate_passes(ev(0.71, 0.5), 0.72, 0.0)  # dense score ignored when reranked
    assert gate_passes(ev(0.72), 0.72, 0.62)  # boundary passes
    assert top_rerank_score(ev(0.3, 0.8)) == 0.8


def test_gate_fails_on_no_evidence():
    assert not gate_passes([], 0.72, 0.62)


def test_gate_falls_back_to_dense_threshold_when_rerank_skipped():
    weak = [c.model_copy(update={"dense_score": 0.4}) for c in ev(None, None)]
    strong = [c.model_copy(update={"dense_score": 0.7}) for c in ev(None, None)]
    assert not gate_passes(weak, 0.72, 0.62)
    assert gate_passes(strong, 0.72, 0.62)


def test_choose_threshold_maximises_youden_j_with_ties_to_lower_t():
    choice = choose_threshold([0.9, 0.8, 0.7], [0.2, 0.3, 0.4])
    assert choice["gate"] == "on" and choice["youden_j"] == 1.0
    assert choice["threshold"] == 0.41  # lowest T that separates perfectly


def test_choose_threshold_turns_gate_off_when_scores_overlap():
    choice = choose_threshold([0.5, 0.6, 0.4], [0.5, 0.6, 0.4])
    assert choice["threshold"] == 0.0 and choice["gate"].startswith("off")


def test_observations_use_best_non_gold_score_as_unanswerable():
    row = {"gold_pmid": "2", "final_pmids": ["2", "1", "3"], "rerank_scores": [0.9, 0.6, 0.1]}
    assert observations([row]) == ([0.9], [0.6])


def test_frozen_thresholds_are_recomputed_from_their_dev_analysis():
    """Recompute T and the dense fallback from the stored score lists with the frozen rule."""
    analysis_path = REPO_ROOT / "eval" / "results" / EVIDENCE_GATE_DEV_RUN / "analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert analysis["split"] == "dev"
    rerank = choose_threshold(
        analysis["answerable_top_scores"], analysis["unanswerable_top_scores"]
    )
    dense = choose_threshold(
        analysis["dense_answerable_top_scores"], analysis["dense_unanswerable_top_scores"]
    )
    assert rerank["threshold"] == analysis["choice"]["threshold"] == EVIDENCE_GATE_THRESHOLD
    assert dense["threshold"] == analysis["dense_fallback_choice"]["threshold"]
    assert dense["threshold"] == EVIDENCE_GATE_DENSE_THRESHOLD
    assert all(
        p["abstains"] and p["dense_fallback_abstains"] for p in analysis["out_of_scope_probes"]
    )


def test_dense_observations_use_best_non_gold_dense_score():
    row = {"gold_pmid": "2", "dense_pmids": ["1", "2", "3"], "dense_scores": [0.8, 0.7, 0.1]}
    assert dense_observations([row]) == ([0.8], [0.8])
