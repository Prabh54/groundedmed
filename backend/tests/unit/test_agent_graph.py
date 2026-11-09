"""The fixed agent graph compiles and passes typed state through its nodes in order
(ARCHITECTURE §5–6), using stub nodes (no network, no model calls)."""

import pytest

from backend.app.agents.graph import NODE_ORDER, build_graph
from backend.app.llm.client import LLMError
from backend.app.models.agent import CriticResult, SubQuestion, initial_state
from backend.app.models.answer import Claim, DraftAnswer, LLMUsage
from backend.app.models.retrieval import Candidate, RetrievalError

QUESTION = "Does drug X reduce mortality in sepsis?"


def usage(model: str) -> LLMUsage:
    return LLMUsage(model=model, input_tokens=10, output_tokens=5, cost_usd=0.001)


def candidate(pmid: str, score: float) -> Candidate:
    return Candidate(
        pmid=pmid,
        passage="text",
        sections=[],
        dense_score=0.5,
        dense_rank=1,
        rerank_score=score,
        rerank_rank=1,
    )


def stub_nodes(visited: list[str], *, gate_passes: bool = True, fail: dict | None = None) -> dict:
    """Stubs that record their visit, check they see the previous node's output, and set their
    own fields. `fail` maps a node name to the exception it raises."""
    fail = fail or {}

    def visit(name):
        visited.append(name)
        if name in fail:
            raise fail[name]

    def planner(state):
        visit("planner")
        assert state["question"] == QUESTION
        return {
            "subquestions": [SubQuestion(id="q1", query="drug X sepsis mortality")],
            "usage": [usage("planner-model")],
        }

    def retriever(state):
        visit("retriever")
        assert [s.query for s in state["subquestions"]] == ["drug X sepsis mortality"]
        evidence = [candidate("111", 0.9)]
        return {
            "retrieved_candidates": evidence,
            "selected_evidence": evidence,
            "retrieval_scores": {"111": 0.9},
            "planner_added_evidence": 0,
        }

    def gate(state):
        visit("gate")
        assert [c.pmid for c in state["selected_evidence"]] == ["111"]
        return {"gate_passed": gate_passes}

    def synthesizer(state):
        visit("synthesizer")
        assert state["gate_passed"]
        claim = Claim(id="c1", text="X lowered mortality.", citation_ids=["PMID:111"])
        draft = DraftAnswer(verdict="yes", claims=[claim], limitations=[])
        return {"draft_answer": draft, "claims": [claim], "usage": [usage("synth-model")]}

    def critic(state):
        visit("critic")
        assert [c.id for c in state["claims"]] == ["c1"]
        result = CriticResult(
            claim_id="c1", supported=True, citation_valid=True, action="keep", reason="ok"
        )
        return {"critic_results": [result]}

    def validator(state):
        visit("validator")
        if state["abstained"]:
            return {}
        if not state["gate_passed"]:
            return {"abstained": True, "abstain_reason": "weak_evidence"}
        assert [r.action for r in state["critic_results"]] == ["keep"]
        return {"abstained": False}

    return dict(
        planner=planner,
        retriever=retriever,
        gate=gate,
        synthesizer=synthesizer,
        critic=critic,
        validator=validator,
    )


def run(**kwargs):
    visited: list[str] = []
    graph = build_graph(**stub_nodes(visited, **kwargs))
    final = graph.invoke(initial_state("req-1", QUESTION))
    return visited, final


def test_graph_compiles_with_exactly_the_fixed_nodes():
    graph = build_graph(**stub_nodes([]))
    nodes = set(graph.get_graph().nodes) - {"__start__", "__end__"}
    assert nodes == set(NODE_ORDER)


def test_stub_run_passes_state_through_all_nodes_in_order():
    visited, final = run()
    assert visited == list(NODE_ORDER)
    assert final["request_id"] == "req-1" and final["mode"] == "agent"
    assert final["subquestions"][0].query == "drug X sepsis mortality"
    assert final["selected_evidence"][0].pmid == "111"
    assert final["gate_passed"] is True
    assert final["draft_answer"].verdict == "yes"
    assert final["critic_results"][0].action == "keep"
    assert final["abstained"] is False and final["abstain_reason"] is None
    assert final["errors"] == []


def test_stream_reports_stages_in_order():
    """The API builds stage events from these per-node updates."""
    graph = build_graph(**stub_nodes([]))
    updates = graph.stream(initial_state("req-1", QUESTION), stream_mode="updates")
    assert [next(iter(u)) for u in updates] == list(NODE_ORDER)


def test_every_visited_node_is_timed_and_usage_accumulates():
    _, final = run()
    assert set(final["timings"]) == set(NODE_ORDER)
    assert all(ms >= 0 for ms in final["timings"].values())
    # Planner and synthesizer usage are both kept (append reducer), not overwritten.
    assert [u.model for u in final["usage"]] == ["planner-model", "synth-model"]


def test_failed_gate_skips_synthesizer_and_critic():
    visited, final = run(gate_passes=False)
    assert visited == ["planner", "retriever", "gate", "validator"]
    assert final["abstained"] and final["abstain_reason"] == "weak_evidence"
    assert final["draft_answer"] is None


@pytest.mark.parametrize("stage", ["planner", "retriever", "gate", "synthesizer"])
def test_llm_failure_routes_to_validator_as_error_abstention(stage, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-secretsecretsecret")
    failed_usage = usage("failed-call")
    exc = LLMError("model refused; key=sk-ant-fake-secretsecretsecret", usage=failed_usage)
    visited, final = run(fail={stage: exc})
    assert visited == [*NODE_ORDER[: NODE_ORDER.index(stage) + 1], "validator"]
    assert final["abstained"] and final["abstain_reason"] == "error"
    assert [e.stage for e in final["errors"]] == [stage]
    assert "secretsecret" not in final["errors"][0].message
    assert failed_usage in final["usage"]  # a failed call's cost is still counted
    assert stage in final["timings"]


def test_critic_failure_still_reaches_validator():
    visited, final = run(fail={"critic": LLMError("timeout")})
    assert visited == list(NODE_ORDER)
    assert final["abstain_reason"] == "error"


@pytest.mark.parametrize("exc", [RetrievalError("db down"), KeyError("bug")])
def test_non_provider_errors_propagate(exc):
    """Retrieval/DB failures (API 503) and bugs (500) are not hidden as abstentions."""
    with pytest.raises(type(exc)):
        run(fail={"retriever": exc})
