"""Final validator + gate nodes, and the whole fixed graph wired with real nodes over fake
model / Voyage / database (no network)."""

import json
from types import SimpleNamespace

import pytest

from backend.app.agents import critic as critic_module
from backend.app.agents import planner as planner_module
from backend.app.agents import retriever as retriever_module
from backend.app.agents.critic import make_critic_node
from backend.app.agents.graph import build_graph
from backend.app.agents.planner import make_planner_node
from backend.app.agents.retriever import make_retriever_node
from backend.app.agents.synthesizer import make_synthesizer_node
from backend.app.agents.validator import finalize, make_gate_node, make_validator_node
from backend.app.baseline import synthesizer as synth_module
from backend.app.baseline.pipeline import (
    ERROR_SUMMARY,
    MODEL_ABSTAINED_SUMMARY,
    NO_SUPPORTED_CLAIMS_SUMMARY,
    WEAK_EVIDENCE_SUMMARY,
)
from backend.app.guardrails.citation_validator import validate
from backend.app.models.agent import CriticResult, StageError, initial_state
from backend.app.models.answer import Claim, DraftAnswer, LLMUsage
from backend.app.models.retrieval import Candidate, Section

QUESTION = "Does early mobilization reduce delirium in ICU patients?"


def cand(pmid: str, score: float = 0.9) -> Candidate:
    return Candidate(pmid=pmid, passage=f"passage {pmid}",
                     sections=[Section(label="RESULTS", text=f"Result of paper {pmid}.")],
                     dense_score=0.8, dense_rank=1, rerank_score=score, rerank_rank=1)  # fmt: skip


EVIDENCE = [cand(str(p)) for p in (101, 102, 103, 104, 105)]


def claim(cid: str, text: str, *cites: str) -> Claim:
    return Claim(id=cid, text=text, citation_ids=list(cites))


def draft(**overrides) -> DraftAnswer:
    fields = dict(
        verdict="yes",
        claims=[claim("c1", "Delirium was shorter.", "PMID:101"),
                claim("c2", "ICU stay was shorter.", "PMID:102")],
        limitations=[claim("l1", "Single centre.", "PMID:101")],
    )  # fmt: skip
    fields.update(overrides)
    return DraftAnswer(**fields)


def keep(*ids: str) -> list[CriticResult]:
    return [CriticResult(claim_id=i, supported=True, citation_valid=True, action="keep",
                         reason="ok") for i in ids]  # fmt: skip


def remove(i: str) -> CriticResult:
    return CriticResult(claim_id=i, supported=False, citation_valid=True, action="remove",
                        reason="Mortality is not mentioned; new claim: X cures Y.")  # fmt: skip


def state(d: DraftAnswer | None = None, results=None, answer_supported=True, **overrides):
    s = initial_state("req-1", QUESTION)
    s.update(selected_evidence=EVIDENCE, gate_passed=True, draft_answer=d or draft(),
             critic_results=keep("c1", "c2", "l1") if results is None else results,
             answer_supported=answer_supported,
             usage=[LLMUsage(model="m", input_tokens=1, output_tokens=1, cost_usd=0.01),
                    LLMUsage(model="m", input_tokens=1, output_tokens=1, cost_usd=0.02)],
             timings={"planner": 5.0})  # fmt: skip
    s.update(overrides)
    return s


# --- finalize -------------------------------------------------------------------------------


def test_supported_answer_passes_unchanged():
    final, removed = finalize(state())
    assert final.verdict == "yes" and not final.abstained and final.abstain_reason is None
    assert final.answer == "Delirium was shorter. ICU stay was shorter."
    assert [c.id for c in final.claims] == ["c1", "c2"] and [i.id for i in final.limitations] == [
        "l1"
    ]
    assert removed == []
    assert final.estimated_cost_usd == pytest.approx(0.03)
    assert final.prompt_versions == {"planner": "v1", "synthesizer": "v3", "critic": "v1"}
    assert final.evidence == EVIDENCE and final.gate_passed


def test_invalid_citation_is_removed_and_counted():
    d = draft(claims=[claim("c1", "Delirium was shorter.", "PMID:101"),
                      claim("c2", "Invented.", "PMID:999")])  # fmt: skip
    final, _ = finalize(state(d, keep("c1", "c2", "l1")))
    assert [c.id for c in final.claims] == ["c1"]
    assert final.validation.removed_invalid_citation == ["c2"]
    assert final.validation == validate(d, [c.citation_id for c in EVIDENCE])[1]  # same as baseline


def test_only_invalid_citations_abstain():
    d = draft(claims=[claim("c1", "Invented.", "PMID:999"), claim("c2", "Uncited.")])
    final, _ = finalize(state(d, keep("c1", "c2", "l1")))
    assert final.abstained and final.abstain_reason == "no_supported_claims"
    assert final.verdict == "insufficient_evidence" and final.claims == final.limitations == []
    assert final.answer == NO_SUPPORTED_CLAIMS_SUMMARY
    assert final.validation.removed_invalid_citation == ["c1"]
    assert final.validation.removed_uncited == ["c2"]


def test_critic_removal_reaches_the_final_answer():
    final, removed = finalize(state(results=[*keep("c1"), remove("c2"), remove("l1")]))
    assert [c.id for c in final.claims] == ["c1"] and final.limitations == []
    assert final.answer == "Delirium was shorter." and removed == ["c2", "l1"]
    assert "cures" not in final.model_dump_json()  # critic reasons never reach the answer


def test_critic_removing_every_claim_abstains():
    final, removed = finalize(state(results=[remove("c1"), remove("c2"), *keep("l1")]))
    assert final.abstained and final.abstain_reason == "no_supported_claims"
    assert final.claims == [] and final.limitations == []  # limitations dropped with the verdict
    assert final.validation.downgraded_to_insufficient is False  # citations were fine
    assert removed == ["c1", "c2"]


def test_answer_not_supported_abstains_even_with_kept_claims():
    final, _ = finalize(state(answer_supported=False))
    assert final.abstained and final.abstain_reason == "no_supported_claims"
    assert final.verdict == "insufficient_evidence" and final.claims == []


def test_model_abstention():
    d = DraftAnswer(verdict="insufficient_evidence", claims=[], limitations=[])
    final, _ = finalize(state(d, results=[], answer_supported=None))
    assert final.abstain_reason == "model_abstained" and final.answer == MODEL_ABSTAINED_SUMMARY


def test_weak_evidence_abstains():
    final, _ = finalize(state(gate_passed=False, draft_answer=None))
    assert final.abstain_reason == "weak_evidence" and final.answer == WEAK_EVIDENCE_SUMMARY


def test_node_error_abstains_with_the_stage_named():
    s = state(errors=[StageError(stage="critic", message="LLMError: timeout")],
              abstained=True, abstain_reason="error")  # fmt: skip
    final, _ = finalize(s)
    assert final.abstain_reason == "error" and final.answer == ERROR_SUMMARY
    assert final.claims == [] and final.error == "critic: LLMError: timeout"


def test_validator_node_updates_state():
    update = make_validator_node()(state(results=[*keep("c1", "l1"), remove("c2")]))
    assert update["final_answer"].claims == update["claims"]
    assert update["abstained"] is False and update["critic_removed"] == ["c2"]


@pytest.mark.parametrize("score,passed", [(0.9, True), (0.72, True), (0.71, False)])
def test_gate_node_uses_the_frozen_threshold(score, passed):
    s = initial_state("r", QUESTION)
    s["selected_evidence"] = [cand("101", score)]
    assert make_gate_node()(s) == {"gate_passed": passed}


# --- the whole graph with real nodes over fakes ----------------------------------------------

PLAN_OUT = {"subquestions": [{"id": "q1", "query": "mobilization delirium trial"}]}
DRAFT_OUT = {
    "verdict": "yes",
    "claims": [
        {"id": "c1", "text": "Delirium was shorter.", "citation_ids": ["PMID:101"]},
        {"id": "c2", "text": "Mortality fell by half.", "citation_ids": ["PMID:102"]},
    ],
    "limitations": [{"id": "l1", "text": "Single centre.", "citation_ids": ["PMID:101"]}],
}
CRITIC_OUT = {"results": [r.model_dump() for r in [*keep("c1", "l1"), remove("c2")]],
              "answer_supported": True}  # fmt: skip


class FakeLLM:
    """Answers by system prompt; `fail` names a stage whose call is refused."""

    def __init__(self, fail: str | None = None, draft_out=DRAFT_OUT):
        self.fail, self.calls = fail, []
        self.outputs = {
            planner_module.SYSTEM_PROMPT: ("planner", PLAN_OUT),
            synth_module.SYSTEM_PROMPT: ("synthesizer", draft_out),
            critic_module.SYSTEM_PROMPT: ("critic", CRITIC_OUT),
        }
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        stage, out = self.outputs[kwargs["system"]]
        self.calls.append(stage)
        refused = stage == self.fail
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="no" if refused else json.dumps(out))],
            stop_reason="refusal" if refused else "end_turn", stop_details=None,
            usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
        )  # fmt: skip


class FakeVoyage:
    def __init__(self, score: float):
        self.score = score

    def embed(self, texts, **kwargs):
        # vector 0 = the question, 1 = a sub-query (dense_search routes on it)
        return SimpleNamespace(embeddings=[[0.0 if t == QUESTION else 1.0] * 1024 for t in texts],
                               total_tokens=10)  # fmt: skip

    def rerank(self, query, documents, **kwargs):
        results = [SimpleNamespace(index=i, relevance_score=self.score - i / 100)
                   for i in range(len(documents))]  # fmt: skip
        return SimpleNamespace(results=results, total_tokens=100)


@pytest.fixture(autouse=True)
def fake_db(monkeypatch):
    def dense_search(conn, vector, k):
        pmids = ["101", "102", "103", "104", "105"] if vector[0] == 0 else ["106", "101"]
        return [Candidate(pmid=p, passage=f"passage {p}",
                          sections=[Section(label="RESULTS", text=f"Result of paper {p}.")],
                          dense_score=0.8, dense_rank=i)
                for i, p in enumerate(pmids, start=1)]  # fmt: skip

    monkeypatch.setattr(retriever_module, "dense_search", dense_search)


def agent_graph(llm, score=0.9):
    return build_graph(planner=make_planner_node(llm),
                       retriever=make_retriever_node(conn=object(), client=FakeVoyage(score)),
                       gate=make_gate_node(), synthesizer=make_synthesizer_node(llm),
                       critic=make_critic_node(llm), validator=make_validator_node())  # fmt: skip


def run(llm, score=0.9):
    return agent_graph(llm, score).invoke(initial_state("req-9", QUESTION))


def test_graph_end_to_end_removes_the_unsupported_claim():
    llm = FakeLLM()
    final = run(llm)
    answer = final["final_answer"]
    assert llm.calls == ["planner", "synthesizer", "critic"]
    assert answer.verdict == "yes" and not answer.abstained
    assert [c.id for c in answer.claims] == ["c1"] and answer.answer == "Delirium was shorter."
    assert final["critic_removed"] == ["c2"]
    assert [s.query for s in final["subquestions"]] == ["mobilization delirium trial"]
    assert len(final["retrieved_candidates"]) == 6  # 5 + paper 106 from the sub-query
    llm_cost = sum(u.cost_usd for u in answer.usage if isinstance(u, LLMUsage))
    assert llm_cost == pytest.approx(3 * (1000 * 2 + 100 * 10) / 1e6)  # planner, synth, critic
    assert answer.estimated_cost_usd > llm_cost  # + Voyage embedding and rerank (list price)
    stages = {"planner", "retriever", "gate", "synthesizer", "critic", "validator"}
    assert stages <= set(final["timings"])
    assert stages - set(answer.timing_ms) == {"validator"}  # answer built before its own timing


def test_graph_low_evidence_refusal_makes_no_synthesis_or_critic_call():
    llm = FakeLLM()
    answer = run(llm, score=0.3)["final_answer"]
    assert llm.calls == ["planner"]
    assert answer.abstain_reason == "weak_evidence" and answer.claims == []


def test_graph_invalid_citation_from_the_synthesizer_is_removed():
    bad = {**DRAFT_OUT, "claims": [{"id": "c1", "text": "X.", "citation_ids": ["PMID:999"]}]}
    answer = run(FakeLLM(draft_out=bad))["final_answer"]
    assert answer.abstain_reason == "no_supported_claims"
    assert answer.validation.removed_invalid_citation == ["c1"]


@pytest.mark.parametrize("stage", ["planner", "synthesizer", "critic"])
def test_graph_provider_failure_is_a_controlled_error(stage):
    llm = FakeLLM(fail=stage)
    final = run(llm)
    answer = final["final_answer"]
    assert answer.abstained and answer.abstain_reason == "error" and answer.claims == []
    assert answer.answer == ERROR_SUMMARY and stage in answer.error
    assert answer.estimated_cost_usd > 0  # the refused call is still counted
    assert llm.calls[-1] == stage  # nothing after the failing stage called the model
