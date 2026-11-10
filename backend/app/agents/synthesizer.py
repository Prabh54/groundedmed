"""Synthesizer node: the baseline's `synthesize()` (same prompt, user turn and schema) over the
retriever's selected evidence, at the agent's own effort (AGENT_SYNTHESIZER_EFFORT, ADR-044).

An LLMError propagates to the graph wrapper, which turns it into an `error` abstention with the
failed call's cost counted.
"""

import anthropic

from backend.app.agents.graph import Node
from backend.app.agents.reviser import AGENT_SYNTHESIZER_EFFORT
from backend.app.baseline.synthesizer import synthesize
from backend.app.models.agent import ResearchState


def make_synthesizer_node(client: anthropic.Anthropic) -> Node:
    def synthesizer(state: ResearchState) -> dict:
        draft, usage, _ = synthesize(
            client, state["question"], state["selected_evidence"], effort=AGENT_SYNTHESIZER_EFFORT
        )
        return {"draft_answer": draft, "claims": draft.claims, "usage": [usage]}

    return synthesizer
