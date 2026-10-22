"""Retrieval metrics: Hit@k over ranked PMID lists (ARCHITECTURE §4, §9)."""


def gold_rank(ranked_pmids: list[str], gold_pmid: str) -> int | None:
    """1-based rank of the gold paper, or None if it was not retrieved."""
    try:
        return ranked_pmids.index(gold_pmid) + 1
    except ValueError:
        return None


def hit_at_k(ranked_pmids: list[str], gold_pmid: str, k: int) -> bool:
    if k < 1:
        raise ValueError("k must be >= 1")
    return gold_pmid in ranked_pmids[:k]


def mean_hit_at_k(rankings: list[list[str]], gold_pmids: list[str], k: int) -> float:
    """Share of questions whose gold paper is in the top k."""
    if len(rankings) != len(gold_pmids):
        raise ValueError("one ranking per gold PMID is required")
    if not rankings:
        raise ValueError("no questions")
    hits = sum(hit_at_k(r, g, k) for r, g in zip(rankings, gold_pmids, strict=True))
    return hits / len(rankings)
