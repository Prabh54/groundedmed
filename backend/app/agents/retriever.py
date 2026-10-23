"""Agent retriever (ADR-023, ARCHITECTURE §4): question + planner sub-queries -> dense top 20 each
(one batched embedding call) -> union, dedupe by PMID -> rerank against the question -> top 5.

Built from the same tested pieces as the baseline's `retrieve()` (embed_texts, dense_search,
rerank_candidates), which is left unchanged. No LLM call, no generated text.
Embedding/database failures raise RetrievalError (API 503); a rerank failure falls back to the
question's own dense top 5 — the same evidence the baseline would fall back to — flagged
`rerank_skipped`, so the gate's dense fallback threshold sees question-only cosine scores.
"""

import json
import logging
import time

import psycopg
import voyageai

from backend.app.agents.graph import Node
from backend.app.agents.planner import query_key
from backend.app.config import EMBEDDING_MODEL, RERANK_MODEL, redact
from backend.app.models.agent import PooledRetrieval, ResearchState, SubQuestion
from backend.app.models.retrieval import Candidate, RetrievalError
from backend.app.retrieval.embed import embed_texts, voyage_client
from backend.app.retrieval.rerank import rerank_candidates
from backend.app.retrieval.retrieve import K_CANDIDATES, K_FINAL, usage
from backend.app.retrieval.vector_search import connect, dense_search

log = logging.getLogger(__name__)


def build_queries(question: str, subquestions: list[SubQuestion]) -> list[SubQuestion]:
    """The question first (q0, always searched), then each distinct sub-query once."""
    queries = [SubQuestion(id="q0", query=question)]
    seen = {query_key(question)}
    for sub in subquestions:
        if query_key(sub.query) not in seen:
            seen.add(query_key(sub.query))
            queries.append(sub)
    return queries


def pool_candidates(hits: list[list[Candidate]]) -> list[Candidate]:
    """Union of the per-query hit lists in query order, first occurrence kept: the question's
    top 20 come first, in their own dense order."""
    pooled: dict[str, Candidate] = {}
    for result in hits:
        for c in result:
            pooled.setdefault(c.pmid, c)
    return list(pooled.values())


def retrieve_pooled(
    question: str,
    subquestions: list[SubQuestion],
    k_candidates: int = K_CANDIDATES,
    k_final: int = K_FINAL,
    *,
    conn: psycopg.Connection | None = None,
    client: voyageai.Client | None = None,
) -> PooledRetrieval:
    start = time.perf_counter()
    queries = build_queries(question, subquestions)
    own_conn = conn is None
    try:
        client = client or voyage_client()
        vectors, embed_tokens = embed_texts(client, [q.query for q in queries], input_type="query")
        embedded = time.perf_counter()
        conn = conn or connect()
        hits = [dense_search(conn, v, k_candidates) for v in vectors]
    except RetrievalError:
        raise
    except Exception as exc:
        raise RetrievalError(redact(f"{type(exc).__name__}: {exc}")[:300]) from None
    finally:
        if own_conn and conn is not None:
            conn.close()
    searched = time.perf_counter()

    pool = pool_candidates(hits)
    question_pmids = {c.pmid for c in hits[0]}
    calls = [usage(EMBEDDING_MODEL, embed_tokens)]
    rerank_error = None
    try:
        pool, rerank_tokens = rerank_candidates(client, question, pool)
        calls.append(usage(RERANK_MODEL, rerank_tokens))
        evidence = sorted(pool, key=lambda c: c.rerank_rank)[:k_final]
    except Exception as exc:  # any rerank failure -> the question's dense order, flagged
        rerank_error = redact(f"{type(exc).__name__}: {exc}")[:300]
        evidence = hits[0][:k_final]
    done = time.perf_counter()

    return PooledRetrieval(
        question=question,
        queries=queries,
        query_hits={q.id: [c.pmid for c in h] for q, h in zip(queries, hits, strict=True)},
        candidates=pool,
        evidence=evidence,
        planner_added_evidence=sum(c.pmid not in question_pmids for c in evidence),
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


def make_retriever_node(
    conn: psycopg.Connection | None = None, client: voyageai.Client | None = None
) -> Node:
    def retriever(state: ResearchState) -> dict:
        result = retrieve_pooled(state["question"], state["subquestions"], conn=conn, client=client)
        log.info(
            json.dumps(
                {
                    "request_id": state["request_id"],
                    "stage": "retriever",
                    "queries": len(result.queries),
                    "candidates": len(result.candidates),
                    "evidence": len(result.evidence),
                    "planner_added_evidence": result.planner_added_evidence,
                    "rerank_skipped": result.rerank_skipped,
                    "timings_ms": {k: round(v) for k, v in result.timings_ms.items()},
                }
            )
        )
        return {
            "retrieved_candidates": result.candidates,
            "selected_evidence": result.evidence,
            "retrieval_scores": {
                c.pmid: c.rerank_score for c in result.evidence if c.rerank_score is not None
            },
            "planner_added_evidence": result.planner_added_evidence,
            "rerank_skipped": result.rerank_skipped,
            "rerank_error": result.rerank_error,
            "usage": result.usage,
            "timings": {f"retriever_{k}": v for k, v in result.timings_ms.items() if k != "total"},
        }

    return retriever
