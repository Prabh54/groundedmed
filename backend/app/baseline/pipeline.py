"""Baseline RAG (ARCHITECTURE §4): retrieve -> evidence gate -> one synthesis call -> validator.

Retrieval failures (embedding or database) raise; the API turns them into 503. An LLM failure gives
a controlled abstention with `abstain_reason="error"` and the error text, never a partial answer.
"""

import time

import anthropic
import psycopg
import voyageai

from backend.app.baseline.synthesizer import SYNTHESIZER_VERSION, synthesize
from backend.app.config import EVIDENCE_GATE_THRESHOLD
from backend.app.guardrails.citation_validator import validate
from backend.app.guardrails.evidence_gate import gate_passes, top_rerank_score
from backend.app.llm.client import LLMError, anthropic_client
from backend.app.models.answer import FinalAnswer, ValidationReport
from backend.app.retrieval.retrieve import retrieve

WEAK_EVIDENCE_SUMMARY = (
    "The corpus does not contain evidence relevant enough to answer this question."
)
ERROR_SUMMARY = "The answer could not be generated because a model call failed."


def answer_baseline(
    question: str,
    *,
    threshold: float = EVIDENCE_GATE_THRESHOLD,
    conn: psycopg.Connection | None = None,
    voyage: voyageai.Client | None = None,
    llm: anthropic.Anthropic | None = None,
) -> FinalAnswer:
    start = time.perf_counter()
    retrieval = retrieve(question, conn=conn, client=voyage)
    evidence = retrieval.evidence
    usage: list = list(retrieval.usage)
    timing = {"retrieval": retrieval.timings_ms["total"], "synthesis": 0.0}
    passed = gate_passes(evidence, threshold)

    def finish(**fields) -> FinalAnswer:
        timing["total"] = (time.perf_counter() - start) * 1000
        return FinalAnswer(
            question=question,
            evidence=evidence,
            rerank_skipped=retrieval.rerank_skipped,
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
            uncertainty="",
            abstained=True,
            abstain_reason="weak_evidence",
            validation=ValidationReport(),
        )

    try:
        draft, call_usage, synth_ms = synthesize(llm or anthropic_client(), question, evidence)
    except LLMError as exc:
        if exc.usage:
            usage.append(exc.usage)
        return finish(
            verdict="insufficient_evidence",
            answer=ERROR_SUMMARY,
            claims=[],
            uncertainty="",
            abstained=True,
            abstain_reason="error",
            validation=ValidationReport(),
            error=str(exc),
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
        answer=cleaned.summary,
        claims=cleaned.claims,
        uncertainty=cleaned.uncertainty,
        abstained=abstain_reason is not None,
        abstain_reason=abstain_reason,
        validation=report,
    )
