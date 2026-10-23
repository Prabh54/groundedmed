"""API request/response schemas (ARCHITECTURE §8)."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from backend.app.models.answer import AbstainReason, Claim, Verdict
from backend.app.models.retrieval import Section


class QueryRequest(BaseModel):
    question: str = Field(min_length=5, max_length=500)
    mode: Literal["baseline", "agent"] = "baseline"

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, value):
        return value.strip() if isinstance(value, str) else value


class Citation(BaseModel):
    id: str
    pmid: str
    passage: str
    sections: list[Section]
    pubmed_url: str
    rerank_score: float | None


class Stage(BaseModel):
    name: str
    status: Literal["done", "skipped", "failed"]
    message: str
    ms: float


class QueryResponse(BaseModel):
    request_id: str
    mode: Literal["baseline", "agent"]
    verdict: Verdict
    answer: str
    claims: list[Claim]
    citations: list[Citation]  # the evidence passages the answer was generated from
    limitations: list[Claim]  # cited, validated like claims (ADR-030)
    abstained: bool
    abstain_reason: AbstainReason | None
    rerank_skipped: bool
    stages: list[Stage]
    timing_ms: dict[str, float]
    estimated_cost_usd: float
    prompt_versions: dict[str, str]


class SourceResponse(BaseModel):
    id: str
    pmid: str
    passage: str
    sections: list[Section]
    meshes: list[str]
    pubmed_url: str


def pubmed_url(pmid: str) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
