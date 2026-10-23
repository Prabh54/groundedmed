"""Evidence gate (ARCHITECTURE §7, ADR-015): abstain when the best rerank score is below T.

Shared by baseline and agent. T is chosen on dev data only (config.EVIDENCE_GATE_THRESHOLD).
If reranking was skipped there is no rerank score to compare, so the gate passes and the answer
carries `rerank_skipped=True` (visible, not silent).
"""

from backend.app.models.retrieval import Candidate


def top_rerank_score(evidence: list[Candidate]) -> float | None:
    scores = [c.rerank_score for c in evidence if c.rerank_score is not None]
    return max(scores) if scores else None


def gate_passes(evidence: list[Candidate], threshold: float) -> bool:
    if not evidence:
        return False
    top = top_rerank_score(evidence)
    return top is None or top >= threshold
