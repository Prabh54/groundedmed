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
    """Exactly the JSON the synthesizer must return (sent to the API as the output schema).

    No free-text summary (ADR-030): the displayed answer is built from validated claims only, and
    limitations are cited items checked the same way."""

    verdict: Verdict
    claims: list[Claim]
    limitations: list[Claim]


class LLMUsage(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int  # includes thinking tokens
    cost_usd: float


class ValidationReport(BaseModel):
    """What the deterministic citation validator removed (ARCHITECTURE §7)."""

    removed_uncited: list[str] = []  # claim IDs
    removed_invalid_citation: list[str] = []  # claim IDs with a malformed or unknown citation ID
    removed_limitations: list[str] = []  # limitation IDs failing the same checks
    removed_bad_id: list[
        str
    ] = []  # positions ("claims[2]") of items with a malformed or repeated ID
    downgraded_to_insufficient: bool = False


class FinalAnswer(BaseModel):
    """Validated answer returned by the baseline pipeline (the API response is built from it)."""

    question: str
    verdict: Verdict
    answer: str  # validated claim texts in order, or a fixed abstention message
    claims: list[Claim]
    limitations: list[Claim]
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
