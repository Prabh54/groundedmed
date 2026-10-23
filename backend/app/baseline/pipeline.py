"""Baseline RAG (ARCHITECTURE §4): retrieve -> evidence gate -> one synthesis call -> validator.

Retrieval failures (embedding or database) raise RetrievalError; the API turns them into 503.
An LLM failure gives a controlled abstention with `abstain_reason="error"` and the error text,
never a partial answer.
"""

import time

import anthropic
import psycopg
import voyageai

from backend.app.baseline.synthesizer import SYNTHESIZER_VERSION, synthesize
from backend.app.config import EVIDENCE_GATE_DENSE_THRESHOLD, EVIDENCE_GATE_THRESHOLD, redact
from backend.app.guardrails.citation_validator import validate
from backend.app.guardrails.evidence_gate import gate_passes, top_rerank_score
from backend.app.llm.client import LLMError, anthropic_client
from backend.app.models.answer import FinalAnswer, ValidationReport
from backend.app.retrieval.retrieve import retrieve

# Says what was measured, not that the corpus lacks the evidence: in all 3 dev gate refusals the
# source paper was retrieved (rank 1 or 2) but scored below the threshold (M2 final review).
WEAK_EVIDENCE_SUMMARY = (
    "No answer was generated: the retrieved evidence scored below the relevance threshold."
)
ERROR_SUMMARY = "The answer could not be generated because a model call failed."
NO_SUPPORTED_CLAIMS_SUMMARY = (
    "The retrieved evidence did not yield any citable claim for this question."
)
MODEL_ABSTAINED_SUMMARY = "None of the retrieved evidence directly addresses this question."


def compose_answer(verdict: str, claims: list, abstain_reason: str | None) -> str:
    """The displayed answer: the validated claims' texts in order (ADR-030), never model prose."""
    if abstain_reason == "no_supported_claims":
        return NO_SUPPORTED_CLAIMS_SUMMARY
    if verdict == "insufficient_evidence" and not claims:
        return MODEL_ABSTAINED_SUMMARY
    return " ".join(c.text.strip() for c in claims)


def answer_baseline(
    question: str,
    *,
    threshold: float = EVIDENCE_GATE_THRESHOLD,
    dense_threshold: float = EVIDENCE_GATE_DENSE_THRESHOLD,
    conn: psycopg.Connection | None = None,
    voyage: voyageai.Client | None = None,
    llm: anthropic.Anthropic | None = None,
) -> FinalAnswer:
    start = time.perf_counter()
    retrieval = retrieve(question, conn=conn, client=voyage)
    evidence = retrieval.evidence
    usage: list = list(retrieval.usage)
    timing = {"retrieval": retrieval.timings_ms["total"], "synthesis": 0.0}
    passed = gate_passes(evidence, threshold, dense_threshold)

    def finish(**fields) -> FinalAnswer:
        timing["total"] = (time.perf_counter() - start) * 1000
        return FinalAnswer(
            question=question,
            evidence=evidence,
            rerank_skipped=retrieval.rerank_skipped,
            rerank_error=retrieval.rerank_error,
            gate_passed=passed,
            top_rerank_score=top_rerank_score(evidence),
            timing_ms=timing,
            usage=usage,
            estimated_cost_usd=sum(u.cost_usd for u in usage),
            prompt_versions={"synthesizer": SYNTHESIZER_VERSION},
            **fields,
        )

    if not passed:
        return finish(
            verdict="insufficient_evidence",
            answer=WEAK_EVIDENCE_SUMMARY,
            claims=[],
            limitations=[],
            abstained=True,
            abstain_reason="weak_evidence",
            validation=ValidationReport(),
        )

    try:
        draft, call_usage, synth_ms = synthesize(llm or anthropic_client(), question, evidence)
    except LLMError as exc:
        if exc.usage:
            usage.append(exc.usage)
        timing["synthesis"] = exc.elapsed_ms
        return finish(
            verdict="insufficient_evidence",
            answer=ERROR_SUMMARY,
            claims=[],
            limitations=[],
            abstained=True,
            abstain_reason="error",
            validation=ValidationReport(),
            error=redact(str(exc)),
        )
    usage.append(call_usage)
    timing["synthesis"] = synth_ms

    cleaned, report = validate(draft, [c.citation_id for c in evidence])
    abstain_reason = None
    if report.downgraded_to_insufficient:
        abstain_reason = "no_supported_claims"
    elif cleaned.verdict == "insufficient_evidence":
        abstain_reason = "model_abstained"
    return finish(
        verdict=cleaned.verdict,
        answer=compose_answer(cleaned.verdict, cleaned.claims, abstain_reason),
        claims=cleaned.claims,
        limitations=cleaned.limitations,
        abstained=abstain_reason is not None,
        abstain_reason=abstain_reason,
        validation=report,
    )
