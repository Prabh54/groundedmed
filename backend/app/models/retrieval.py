"""Retrieval result types (ARCHITECTURE §4–5)."""

from pydantic import BaseModel


class RetrievalError(RuntimeError):
    """Embedding or database failure: no evidence can be retrieved (the API answers 503)."""


class Section(BaseModel):
    label: str
    text: str


class Candidate(BaseModel):
    """One retrieved paper. Ranks are 1-based; rerank fields stay None if reranking was skipped."""

    pmid: str
    passage: str
    sections: list[Section]
    dense_score: float  # cosine similarity
    dense_rank: int
    rerank_score: float | None = None
    rerank_rank: int | None = None

    @property
    def citation_id(self) -> str:
        return f"PMID:{self.pmid}"


class CallUsage(BaseModel):
    model: str
    tokens: int
    cost_usd: float


class RetrievalResult(BaseModel):
    query: str
    candidates: list[Candidate]  # dense top-k_candidates, in dense order
    evidence: list[Candidate]  # final top-k_final (reranked, or dense order if rerank_skipped)
    rerank_skipped: bool
    rerank_error: str | None = None
    timings_ms: dict[str, float]
    usage: list[CallUsage]
