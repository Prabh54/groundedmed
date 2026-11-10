"""Deterministic agent nodes (ARCHITECTURE §6–7): the evidence gate and the final validator.

Final validator, in order (pure Python, never calls an LLM):
1. A node failure -> abstain `error`. A failed gate -> abstain `weak_evidence`.
2. The citation validator on the synthesizer's draft (same checks and report as the baseline).
3. The critic's removals on what survived (`apply_critic`; claims and limitations).
4. Abstain `no_supported_claims` if a yes/no/maybe verdict has no claim left (validator or critic)
   or the critic said `answer_supported: false` while removing at least one claim (ADR-044 fix B:
   no veto when every citation-valid claim was kept); `model_abstained` if the synthesizer said
   insufficient_evidence. The verdict is always the synthesizer's; nothing is ever added.
"""

from backend.app.agents.critic import CRITIC_VERSION
from backend.app.agents.graph import Node
from backend.app.agents.planner import PLANNER_VERSION
from backend.app.agents.rules import refusal_reason
from backend.app.baseline.pipeline import (
    ERROR_SUMMARY,
    WEAK_EVIDENCE_SUMMARY,
    compose_answer,
)
from backend.app.baseline.synthesizer import SYNTHESIZER_VERSION
from backend.app.config import EVIDENCE_GATE_DENSE_THRESHOLD, EVIDENCE_GATE_THRESHOLD
from backend.app.guardrails.evidence_gate import gate_passes, top_rerank_score
from backend.app.models.agent import ResearchState
from backend.app.models.answer import FinalAnswer, ValidationReport

PROMPT_VERSIONS = {
    "planner": PLANNER_VERSION,
    "synthesizer": SYNTHESIZER_VERSION,
    "critic": CRITIC_VERSION,
}


def make_gate_node(
    threshold: float = EVIDENCE_GATE_THRESHOLD,
    dense_threshold: float = EVIDENCE_GATE_DENSE_THRESHOLD,
) -> Node:
    """Same gate and frozen thresholds as the baseline."""

    def gate(state: ResearchState) -> dict:
        return {"gate_passed": gate_passes(state["selected_evidence"], threshold, dense_threshold)}

    return gate


def _final(state: ResearchState, **fields) -> FinalAnswer:
    evidence = state["selected_evidence"]
    return FinalAnswer(
        question=state["question"],
        evidence=evidence,
        rerank_skipped=state["rerank_skipped"],
        rerank_error=state["rerank_error"],
        gate_passed=state["gate_passed"],
        top_rerank_score=top_rerank_score(evidence),
        timing_ms=dict(state["timings"]),
        usage=list(state["usage"]),
        estimated_cost_usd=sum(u.cost_usd for u in state["usage"]),
        prompt_versions=PROMPT_VERSIONS,
        **fields,
    )


def _abstain(state: ResearchState, reason: str, answer: str, **fields) -> FinalAnswer:
    return _final(state, verdict="insufficient_evidence", answer=answer, claims=[], limitations=[],
                  abstained=True, abstain_reason=reason, **fields)  # fmt: skip


def finalize(state: ResearchState) -> tuple[FinalAnswer, list[str]]:
    """Return (final answer, item IDs the critic removed)."""
    if state["abstain_reason"] == "error" or state["errors"]:
        message = "; ".join(f"{e.stage}: {e.message}" for e in state["errors"]) or None
        return _abstain(state, "error", ERROR_SUMMARY, validation=ValidationReport(),
                        error=message), []  # fmt: skip
    if not state["gate_passed"]:
        return _abstain(state, "weak_evidence", WEAK_EVIDENCE_SUMMARY,
                        validation=ValidationReport()), []  # fmt: skip
    draft = state["draft_answer"]
    if draft is None:  # cannot happen in the fixed graph; never answer without a draft
        raise RuntimeError("validator reached without a draft answer")

    reason, checked, report, critic_removed = refusal_reason(
        draft,
        [c.citation_id for c in state["selected_evidence"]],
        state["critic_results"],
        state["answer_supported"],
    )

    if reason == "no_supported_claims":
        verdict, claims, limitations = "insufficient_evidence", [], []
    else:
        verdict, claims, limitations = checked.verdict, checked.claims, checked.limitations
    final = _final(
        state,
        verdict=verdict,
        answer=compose_answer(verdict, claims, reason),
        claims=claims,
        limitations=limitations,
        abstained=reason is not None,
        abstain_reason=reason,
        validation=report,
    )
    return final, critic_removed


def make_validator_node() -> Node:
    def validator(state: ResearchState) -> dict:
        final, critic_removed = finalize(state)
        return {
            "final_answer": final,
            "abstained": final.abstained,
            "abstain_reason": final.abstain_reason,
            "claims": final.claims,
            "critic_removed": critic_removed,
        }

    return validator
