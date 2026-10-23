"""Answer types: the synthesizer's structured output and the validated final answer
(ARCHITECTURE §5–8, PROMPTS.md Synthesizer)."""

from typing import Literal

from pydantic import BaseModel

from backend.app.models.retrieval import CallUsage, Candidate

Verdict = Literal["yes", "no", "maybe", "insufficient_evidence"]
AbstainReason = Literal["weak_evidence", "no_supported_claims", "model_abstained", "error"]


class Claim(BaseModel):
    id: str
    text: str
    citation_ids: list[str]


class DraftAnswer(BaseModel):
    """Exactly the JSON the synthesizer must return (sent to the API as the output schema)."""

    verdict: Verdict
    summary: str
    claims: list[Claim]
    uncertainty: str


class LLMUsage(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int  # includes thinking tokens
    cost_usd: float


class ValidationReport(BaseModel):
    """What the deterministic citation validator removed (ARCHITECTURE §7)."""

    removed_uncited: list[str] = []  # claim IDs
    removed_invalid_citation: list[str] = []  # claim IDs with a malformed or unknown citation ID
    downgraded_to_insufficient: bool = False
    summary_replaced: bool = False  # summary mentioned a paper outside the evidence
    uncertainty_removed: bool = False  # uncertainty mentioned a paper outside the evidence


class FinalAnswer(BaseModel):
    """Validated answer returned by the baseline pipeline (the API response is built from it)."""

    question: str
    verdict: Verdict
    answer: str
    claims: list[Claim]
    uncertainty: str
    abstained: bool
    abstain_reason: AbstainReason | None
    evidence: list[Candidate]  # the 5 passages the model saw (citations resolve to these)
    rerank_skipped: bool
    rerank_error: str | None = None
    gate_passed: bool
    top_rerank_score: float | None
    validation: ValidationReport
    error: str | None = None
    timing_ms: dict[str, float]
    usage: list[CallUsage | LLMUsage]
    estimated_cost_usd: float
    prompt_versions: dict[str, str]
