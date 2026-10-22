"""Voyage rerank-3 over the dense candidates (ARCHITECTURE §4, §13)."""

import voyageai

from backend.app.config import RERANK_MODEL
from backend.app.models.retrieval import Candidate


def rerank_candidates(
    client: voyageai.Client, query: str, candidates: list[Candidate]
) -> tuple[list[Candidate], int]:
    """Return the candidates in their original (dense) order with rerank score and rank filled in,
    plus tokens billed. Ties keep dense order. `truncation=False` fails loudly on over-long input.
    """
    if not candidates:
        return [], 0
    result = client.rerank(
        query, [c.passage for c in candidates], model=RERANK_MODEL, truncation=False
    )
    if sorted(r.index for r in result.results) != list(range(len(candidates))):
        raise RuntimeError(f"{RERANK_MODEL} did not score every candidate exactly once")
    order = sorted(result.results, key=lambda r: (-r.relevance_score, r.index))
    scored = {
        r.index: candidates[r.index].model_copy(
            update={"rerank_score": r.relevance_score, "rerank_rank": rank}
        )
        for rank, r in enumerate(order, start=1)
    }
    return [scored[i] for i in range(len(candidates))], result.total_tokens
