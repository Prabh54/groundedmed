"""Live agent check (M3 review S2): an off-topic question is refused by the evidence gate in agent
mode too, although the agent reranks a larger pool than T was calibrated on. Needs Voyage, the
database and Anthropic; opt-in with RUN_LIVE_LLM=1 (~$0.003: only the planner is called)."""

import os

import pytest

from backend.app.agents import critic as critic_module
from backend.app.agents import planner as planner_module
from backend.app.agents.pipeline import answer_agent
from backend.app.baseline import synthesizer as synth_module
from backend.app.config import EVIDENCE_GATE_THRESHOLD
from backend.app.llm.client import anthropic_client
from backend.app.retrieval.embed import voyage_client
from backend.app.retrieval.vector_search import connect

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_LLM") != "1", reason="set RUN_LIVE_LLM=1 (~$0.003)"
)


class PlannerOnlyLLM:
    """Real client for the planner; fails the test if the synthesizer or critic is called."""

    def __init__(self):
        self.real = anthropic_client()
        self.messages = self
        self.forbidden = {
            synth_module.SYSTEM_PROMPT: "synthesizer",
            critic_module.SYSTEM_PROMPT: "critic",
        }

    def create(self, **kwargs):
        if kwargs["system"] in self.forbidden:
            raise AssertionError(f"{self.forbidden[kwargs['system']]} called after a refusal")
        assert kwargs["system"] == planner_module.SYSTEM_PROMPT
        return self.real.messages.create(**kwargs)


def test_off_topic_question_is_refused_at_the_gate_in_agent_mode():
    with connect() as conn:
        answer, state = answer_agent("What is the best recipe for sourdough bread?", conn=conn,
                                     voyage=voyage_client(), llm=PlannerOnlyLLM())  # fmt: skip
    queries = [s.query for s in state["subquestions"]]
    pool = len(state["retrieved_candidates"])
    print(f"\nLIVE AGENT: top {answer.top_rerank_score:.3f}, pool {pool}, sub-queries {queries}, "
          f"cost ${answer.estimated_cost_usd:.4f}")  # fmt: skip
    assert answer.abstained and answer.abstain_reason == "weak_evidence"
    assert answer.claims == [] and not answer.gate_passed
    assert answer.top_rerank_score < EVIDENCE_GATE_THRESHOLD
    assert len(state["subquestions"]) >= 1  # the planner really ran and widened the pool
