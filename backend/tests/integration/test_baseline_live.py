"""Live baseline checks (PLAN.md M2-T3). Needs VOYAGE_API_KEY and DATABASE_URL; costs ~nothing:
the out-of-scope question is refused by the evidence gate before any LLM call."""

import os

import pytest

from backend.app.baseline.pipeline import answer_baseline

pytestmark = pytest.mark.skipif(
    not (os.environ.get("VOYAGE_API_KEY") and os.environ.get("DATABASE_URL")),
    reason="Voyage key or database URL not set",
)


class NoLLM:
    """Fails the test if the pipeline tries to call the LLM."""

    def __getattr__(self, name):
        raise AssertionError("the evidence gate should have abstained before any LLM call")


def test_out_of_scope_question_abstains_at_the_gate():
    answer = answer_baseline("What is the best recipe for sourdough bread?", llm=NoLLM())
    assert answer.abstained and answer.abstain_reason == "weak_evidence"
    assert answer.verdict == "insufficient_evidence" and answer.claims == []
    assert not answer.gate_passed and answer.top_rerank_score < 0.72
