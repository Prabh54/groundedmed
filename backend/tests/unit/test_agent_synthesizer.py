"""Synthesizer node with fixed evidence and a fake model (no network). The model request
must be byte-identical to the baseline's for the same evidence."""

import json
from types import SimpleNamespace

from backend.app.agents.graph import build_graph
from backend.app.agents.synthesizer import make_synthesizer_node
from backend.app.baseline import pipeline
from backend.app.models.agent import initial_state
from backend.app.models.retrieval import Candidate, RetrievalResult, Section

QUESTION = "Does early mobilization reduce delirium in ICU patients?"

DRAFT = {
    "verdict": "yes",
    "claims": [
        {
            "id": "c1",
            "text": "Delirium days fell with early mobilization.",
            "citation_ids": ["PMID:101"],
        },
        {"id": "c2", "text": "ICU stay was shorter.", "citation_ids": ["PMID:102"]},
    ],
    "limitations": [{"id": "l1", "text": "Single-centre trial.", "citation_ids": ["PMID:101"]}],
}


def evidence() -> list[Candidate]:
    return [
        Candidate(pmid=str(100 + i), passage=f"passage {i}",
                  sections=[Section(label="RESULTS", text=f"Result text {i}.")],
                  dense_score=0.8, dense_rank=i, rerank_score=0.95 - i / 100, rerank_rank=i)
        for i in range(1, 6)
    ]  # fmt: skip


class FakeLLM:
    def __init__(self, text: str = json.dumps(DRAFT), stop_reason: str = "end_turn"):
        self.text, self.stop_reason, self.calls = text, stop_reason, []
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)],
            stop_reason=self.stop_reason,
            stop_details=None,
            usage=SimpleNamespace(input_tokens=3000, output_tokens=400),
        )


def state_with_evidence():
    state = initial_state("req-1", QUESTION)
    state["selected_evidence"] = evidence()
    state["gate_passed"] = True
    return state


def test_node_returns_draft_claims_and_usage():
    update = make_synthesizer_node(FakeLLM())(state_with_evidence())
    assert update["draft_answer"].verdict == "yes"
    assert [c.id for c in update["claims"]] == ["c1", "c2"]
    assert [item.id for item in update["draft_answer"].limitations] == ["l1"]
    assert update["usage"][0].cost_usd == (3000 * 2 + 400 * 10) / 1e6


def test_model_request_is_identical_to_the_baseline_for_the_same_evidence(monkeypatch):
    agent_llm, baseline_llm = FakeLLM(), FakeLLM()
    make_synthesizer_node(agent_llm)(state_with_evidence())

    ev = evidence()
    monkeypatch.setattr(
        pipeline, "retrieve",
        lambda q, conn=None, client=None: RetrievalResult(
            query=q, candidates=ev, evidence=ev, rerank_skipped=False, timings_ms={"total": 1.0},
            usage=[]),
    )  # fmt: skip
    pipeline.answer_baseline(QUESTION, llm=baseline_llm)

    assert len(agent_llm.calls) == len(baseline_llm.calls) == 1
    agent_call, baseline_call = agent_llm.calls[0], baseline_llm.calls[0]
    # ADR-044: same system prompt, user turn, schema and model; only the effort differs
    assert agent_call["output_config"]["effort"] == "high"
    assert baseline_call["output_config"]["effort"] == "medium"
    strip = lambda call: {**call, "output_config": {**call["output_config"], "effort": None}}  # noqa: E731
    assert strip(agent_call) == strip(baseline_call)
    assert "tools" not in agent_call


def test_only_the_selected_evidence_is_sent():
    llm = FakeLLM()
    state = state_with_evidence()
    state["retrieved_candidates"] = [*evidence(), *[
        Candidate(pmid="999", passage="pool only", sections=[Section(label="X", text="pool only")],
                  dense_score=0.1, dense_rank=30)
    ]]  # fmt: skip
    make_synthesizer_node(llm)(state)
    user = llm.calls[0]["messages"][0]["content"]
    assert user.count("<EVIDENCE id=") == 5
    assert "PMID:999" not in user and "pool only" not in user


def test_synthesizer_in_graph_and_its_failure_becomes_error_abstention():
    def nothing(state):
        return {}

    def fixed_retrieval(state):
        return {"selected_evidence": evidence()}

    def gate(state):
        return {"gate_passed": True}

    visited = []

    def critic(state):
        visited.append("critic")
        return {}

    def build(llm):
        return build_graph(planner=nothing, retriever=fixed_retrieval, gate=gate,
                           synthesizer=make_synthesizer_node(llm), critic=critic,
                           validator=nothing)  # fmt: skip

    final = build(FakeLLM()).invoke(initial_state("r", QUESTION))
    assert final["draft_answer"].verdict == "yes" and visited == ["critic"]
    assert "synthesizer" in final["timings"]

    visited.clear()
    refused = FakeLLM(text="I can't help.", stop_reason="refusal")
    final = build(refused).invoke(initial_state("r", QUESTION))
    assert final["abstain_reason"] == "error" and final["draft_answer"] is None
    assert [e.stage for e in final["errors"]] == ["synthesizer"]
    assert final["usage"][0].input_tokens == 3000  # refused call still costed
    assert visited == []  # critic skipped
