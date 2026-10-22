"""retrieve(): embed query -> exact dense top 20 -> rerank -> top 5 (ARCHITECTURE §4).

If reranking fails, the evidence falls back to dense order with `rerank_skipped=True` (§13):
visible in the result and logs, never silent.
"""

import time

import psycopg
import voyageai

from backend.app.config import EMBEDDING_MODEL, PRICE_PER_M_TOKENS, RERANK_MODEL, redact
from backend.app.models.retrieval import CallUsage, RetrievalResult
from backend.app.retrieval.embed import embed_texts, voyage_client
from backend.app.retrieval.rerank import rerank_candidates
from backend.app.retrieval.vector_search import connect, dense_search

K_CANDIDATES = 20
K_FINAL = 5


def usage(model: str, tokens: int) -> CallUsage:
    return CallUsage(model=model, tokens=tokens, cost_usd=tokens * PRICE_PER_M_TOKENS[model] / 1e6)


def retrieve(
    query: str,
    k_candidates: int = K_CANDIDATES,
    k_final: int = K_FINAL,
    *,
    conn: psycopg.Connection | None = None,
    client: voyageai.Client | None = None,
) -> RetrievalResult:
    """Embedding or database failures raise (the API turns them into 503); rerank failures don't."""
    start = time.perf_counter()
    client = client or voyage_client()
    vectors, embed_tokens = embed_texts(client, [query], input_type="query")
    embedded = time.perf_counter()

    own_conn = conn is None
    conn = conn or connect()
    try:
        candidates = dense_search(conn, vectors[0], k_candidates)
    finally:
        if own_conn:
            conn.close()
    searched = time.perf_counter()

    calls = [usage(EMBEDDING_MODEL, embed_tokens)]
    rerank_error = None
    try:
        candidates, rerank_tokens = rerank_candidates(client, query, candidates)
        calls.append(usage(RERANK_MODEL, rerank_tokens))
        evidence = sorted(candidates, key=lambda c: c.rerank_rank)[:k_final]
    except Exception as exc:  # any rerank failure -> dense order, flagged
        rerank_error = redact(f"{type(exc).__name__}: {exc}")[:300]
        evidence = candidates[:k_final]
    done = time.perf_counter()

    return RetrievalResult(
        query=query,
        candidates=candidates,
        evidence=evidence,
        rerank_skipped=rerank_error is not None,
        rerank_error=rerank_error,
        timings_ms={
            "embed": (embedded - start) * 1000,
            "search": (searched - embedded) * 1000,
            "rerank": (done - searched) * 1000,
            "total": (done - start) * 1000,
        },
        usage=calls,
    )
