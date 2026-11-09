"""Agent retriever with fakes for Voyage and the database (no network)."""

import json
import logging
import random
from types import SimpleNamespace

import pytest

from backend.app.agents import retriever as retriever_module
from backend.app.agents.retriever import (
    build_queries,
    make_retriever_node,
    retrieve_pooled,
)
from backend.app.models.agent import SubQuestion, initial_state
from backend.app.models.retrieval import Candidate, RetrievalError

QUESTION = "Does early mobilization reduce delirium in ICU patients?"


def cand(pmid: str, rank: int) -> Candidate:
    return Candidate(
        pmid=pmid,
        passage=f"passage {pmid}",
        sections=[],
        dense_score=1 - rank / 100,
        dense_rank=rank,
    )


def hit_list(pmids: list[str]) -> list[Candidate]:
    return [cand(p, i) for i, p in enumerate(pmids, start=1)]


QUESTION_TOP20 = [str(1000 + i) for i in range(20)]


class FakeVoyage:
    """embed(): the vector encodes which call (= which query) it is; rerank(): scores by PMID."""

    def __init__(self, scores: dict[str, float] | None = None, rerank_error=None, embed_error=None):
        self.scores, self.rerank_error, self.embed_error = scores or {}, rerank_error, embed_error
        self.embed_calls, self.rerank_calls = [], []

    def embed(self, texts, **kwargs):
        self.embed_calls.append(list(texts))
        if self.embed_error:
            raise self.embed_error
        index = len(self.embed_calls) - 1  # one text per call: vector i = the i-th query
        return SimpleNamespace(embeddings=[[float(index)] * 1024 for _ in texts],
                               total_tokens=5 * len(texts))  # fmt: skip

    def rerank(self, query, documents, **kwargs):
        self.rerank_calls.append((query, list(documents)))
        if self.rerank_error:
            raise self.rerank_error
        results = [
            SimpleNamespace(index=i, relevance_score=self.scores.get(d.split()[-1], 0.1))
            for i, d in enumerate(documents)
        ]
        return SimpleNamespace(results=results, total_tokens=30 * len(documents))


@pytest.fixture
def search(monkeypatch):
    """Route query i (by its vector) to hits[i]; the test sets hits."""
    state = {"hits": [QUESTION_TOP20]}

    def fake_dense_search(conn, vector, k):
        return hit_list(state["hits"][int(vector[0])][:k])

    monkeypatch.setattr(retriever_module, "dense_search", fake_dense_search)
    return state


def subs(*queries: str) -> list[SubQuestion]:
    return [SubQuestion(id=f"q{i}", query=q) for i, q in enumerate(queries, start=1)]


def run(client, subquestions):
    return retrieve_pooled(QUESTION, subquestions, conn=object(), client=client)


# --- queries -----------------------------------------------------------------------------------


def test_question_is_always_the_first_query_and_duplicates_are_searched_once():
    queries = build_queries(QUESTION, subs(QUESTION.lower(), "a", "A ", "b"))
    assert [(q.id, q.query) for q in queries] == [("q0", QUESTION), ("q2", "a"), ("q4", "b")]


def test_each_query_is_embedded_alone_and_the_pool_reranked_once_against_the_question(search):
    search["hits"] = [QUESTION_TOP20, ["2001", "1000"], ["2002"]]
    client = FakeVoyage()
    result = run(client, subs("sub a", "sub b"))
    assert client.embed_calls == [[QUESTION], ["sub a"], ["sub b"]]  # no multi-input batch
    assert len(client.rerank_calls) == 1 and client.rerank_calls[0][0] == QUESTION
    assert len(client.rerank_calls[0][1]) == 22  # 20 + 2 new papers, 1000 deduped
    assert result.query_hits == {"q0": QUESTION_TOP20, "q1": ["2001", "1000"], "q2": ["2002"]}


# --- the pool always contains the question-only top 20 --------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_pool_always_contains_the_question_top_20_first(search, seed):
    rng = random.Random(seed)
    corpus = [str(1000 + i) for i in range(60)]
    search["hits"] = [QUESTION_TOP20] + [rng.sample(corpus, 20) for _ in range(rng.randint(0, 3))]
    result = run(FakeVoyage(), subs(*[f"sub {i}" for i in range(len(search["hits"]) - 1)]))
    pmids = [c.pmid for c in result.candidates]
    assert pmids[:20] == QUESTION_TOP20  # same papers, same dense order, same scores as baseline
    assert [c.dense_score for c in result.candidates[:20]] == [
        c.dense_score for c in hit_list(QUESTION_TOP20)
    ]
    assert len(pmids) == len(set(pmids))  # deduped by PMID
    assert set(pmids) == set().union(*search["hits"])


def test_no_subqueries_gives_the_baseline_pool(search):
    result = run(FakeVoyage(), [])
    assert [c.pmid for c in result.candidates] == QUESTION_TOP20
    assert result.planner_added_evidence == 0


# --- top 5 and planner_added_evidence ---------------------------------------------------------


def test_reranked_top_5_and_planner_added_evidence(search):
    search["hits"] = [QUESTION_TOP20, ["2001", "2002", "1003"]]
    scores = {"2001": 0.95, "1003": 0.9, "2002": 0.8, "1005": 0.7, "1007": 0.6}
    result = run(FakeVoyage(scores), subs("sub a"))
    assert [c.pmid for c in result.evidence] == ["2001", "1003", "2002", "1005", "1007"]
    assert [c.rerank_rank for c in result.evidence] == [1, 2, 3, 4, 5]
    assert result.planner_added_evidence == 2  # 2001 and 2002 are not in the question's top 20
    assert not result.rerank_skipped


def test_planner_added_evidence_is_zero_when_question_papers_win(search):
    search["hits"] = [QUESTION_TOP20, ["2001"]]
    result = run(FakeVoyage({"1000": 0.9, "2001": 0.05}), subs("sub a"))
    assert "2001" not in [c.pmid for c in result.evidence]
    assert result.planner_added_evidence == 0


# --- failures ---------------------------------------------------------------------------------


def test_rerank_failure_falls_back_to_the_question_dense_top_5(search, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-fake-secret-value-123")
    search["hits"] = [QUESTION_TOP20, ["2001", "2002"]]
    client = FakeVoyage(rerank_error=RuntimeError("boom pa-fake-secret-value-123"))
    result = run(client, subs("sub a"))
    assert [c.pmid for c in result.evidence] == QUESTION_TOP20[:5]
    assert all(c.rerank_score is None for c in result.evidence)
    assert result.rerank_skipped and "secret" not in result.rerank_error
    assert result.planner_added_evidence == 0
    assert [u.model for u in result.usage] == ["voyage-4"]  # no rerank usage recorded


def test_embedding_failure_is_a_retrieval_error(search):
    with pytest.raises(RetrievalError):
        run(FakeVoyage(embed_error=RuntimeError("voyage down")), subs("a"))


def test_database_failure_is_a_retrieval_error(search, monkeypatch):
    def broken(conn, vector, k):
        raise OSError("connection reset")

    monkeypatch.setattr(retriever_module, "dense_search", broken)
    with pytest.raises(RetrievalError):
        run(FakeVoyage(), subs("a"))


# --- graph node -------------------------------------------------------------------------------


def test_node_fills_state_and_logs_counts(search, caplog):
    search["hits"] = [QUESTION_TOP20, ["2001", "1000"]]
    scores = {"2001": 0.95, "1001": 0.9, "1002": 0.8, "1003": 0.7, "1004": 0.6}
    node = make_retriever_node(conn=object(), client=FakeVoyage(scores))
    state = initial_state("req-7", QUESTION)
    state["subquestions"] = subs("sub a")
    with caplog.at_level(logging.INFO, logger=retriever_module.__name__):
        update = node(state)

    assert len(update["retrieved_candidates"]) == 21
    assert [c.pmid for c in update["selected_evidence"]] == ["2001", "1001", "1002", "1003", "1004"]
    assert update["retrieval_scores"] == scores
    assert update["planner_added_evidence"] == 1
    assert update["rerank_skipped"] is False and update["rerank_error"] is None
    assert [u.model for u in update["usage"]] == ["voyage-4", "rerank-3"]
    assert set(update["timings"]) == {"retriever_embed", "retriever_search", "retriever_rerank"}

    record = json.loads(caplog.records[-1].getMessage())
    assert record["request_id"] == "req-7"
    assert (record["queries"], record["candidates"], record["evidence"]) == (2, 21, 5)
    assert record["planner_added_evidence"] == 1
    assert set(record["timings_ms"]) == {"embed", "search", "rerank", "total"}
