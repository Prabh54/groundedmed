"""Agent-mode types: the graph state passed between nodes (ARCHITECTURE §5) and the planner/critic
outputs it carries (PROMPTS.md Planner v1, Critic v1)."""

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel

from backend.app.models.answer import AbstainReason, Claim, DraftAnswer, FinalAnswer, LLMUsage
from backend.app.models.retrieval import CallUsage, Candidate


class SubQuestion(BaseModel):
    id: str
    query: str


class CriticResult(BaseModel):
    claim_id: str
    supported: bool
    citation_valid: bool
    action: Literal["keep", "revise", "remove"]  # revise is treated as remove in v1
    reason: str


class StageError(BaseModel):
    stage: str
    message: str  # redacted, plain text


def merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}


class ResearchState(TypedDict):
    """Explicit state passed through the fixed graph. Each node returns only the keys it sets.

    `timings`, `usage` and `errors` accumulate across nodes (their reducers merge/append), so a node
    reports only its own entries; every other key is replaced by the node that owns it.
    """

    request_id: str
    mode: Literal["agent"]
    question: str
    subquestions: list[SubQuestion]  # planner output
    retrieved_candidates: list[Candidate]  # all dense hits before rerank (pooled, deduped)
    selected_evidence: list[Candidate]  # final top 5 after rerank
    retrieval_scores: dict[str, float]  # pmid -> rerank score
    planner_added_evidence: int  # final-5 papers not in the question-only top 20 (ADR-023)
    gate_passed: bool
    draft_answer: DraftAnswer | None  # synthesizer output
    claims: list[Claim]
    critic_results: list[CriticResult]
    final_answer: FinalAnswer | None
    abstained: bool
    abstain_reason: AbstainReason | None
    timings: Annotated[dict[str, float], merge_dicts]  # ms per stage, plus total
    usage: Annotated[list[CallUsage | LLMUsage], operator.add]
    errors: Annotated[list[StageError], operator.add]


def initial_state(request_id: str, question: str) -> ResearchState:
    return ResearchState(
        request_id=request_id,
        mode="agent",
        question=question,
        subquestions=[],
        retrieved_candidates=[],
        selected_evidence=[],
        retrieval_scores={},
        planner_added_evidence=0,
        gate_passed=False,
        draft_answer=None,
        claims=[],
        critic_results=[],
        final_answer=None,
        abstained=False,
        abstain_reason=None,
        timings={},
        usage=[],
        errors=[],
    )
