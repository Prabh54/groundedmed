"""Synthesizer node: the baseline's `synthesize()` (same prompt, user turn, schema and effort)
over the retriever's selected evidence. Only the evidence passages differ between modes.

An LLMError propagates to the graph wrapper, which turns it into an `error` abstention with the
failed call's cost counted.
"""

import anthropic

from backend.app.agents.graph import Node
from backend.app.baseline.synthesizer import synthesize
from backend.app.models.agent import ResearchState


def make_synthesizer_node(client: anthropic.Anthropic) -> Node:
    def synthesizer(state: ResearchState) -> dict:
        draft, usage, _ = synthesize(client, state["question"], state["selected_evidence"])
        return {"draft_answer": draft, "claims": draft.claims, "usage": [usage]}

    return synthesizer
