"""Fixed agent graph (ARCHITECTURE §5–6): planner -> retriever -> gate -> synthesizer -> critic ->
validator -> END. A failed gate or a failed node goes straight to the validator, which abstains.

Nodes are passed in, so tests can run the graph with stubs and each M3 task swaps in a real node.
"""

import time
from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from backend.app.config import redact
from backend.app.llm.client import LLMError
from backend.app.models.agent import ResearchState, StageError

Node = Callable[[ResearchState], dict]

NODE_ORDER = ("planner", "retriever", "gate", "synthesizer", "critic", "validator")

# Failures that end the run as a controlled abstention (ARCHITECTURE §13). Retrieval/database
# errors and programming bugs are not caught: they propagate so the API answers 503/500, as in
# baseline mode.
NODE_FAILURES: tuple[type[Exception], ...] = (LLMError,)


def _guarded(name: str, node: Node) -> Node:
    """Time the node and turn a known provider failure into an `error` abstention."""

    def run(state: ResearchState) -> dict:
        start = time.perf_counter()
        try:
            update = dict(node(state))
        except NODE_FAILURES as exc:
            update = {
                "errors": [StageError(stage=name, message=redact(str(exc)))],
                "abstained": True,
                "abstain_reason": "error",
            }
            if getattr(exc, "usage", None):
                update["usage"] = [exc.usage]
        elapsed_ms = (time.perf_counter() - start) * 1000
        update["timings"] = {**update.get("timings", {}), name: elapsed_ms}
        return update

    return run


def _then(next_node: str) -> Callable[[ResearchState], str]:
    def route(state: ResearchState) -> str:
        return "validator" if state["errors"] else next_node

    return route


def _after_gate(state: ResearchState) -> str:
    return "synthesizer" if state["gate_passed"] and not state["errors"] else "validator"


def build_graph(
    *,
    planner: Node,
    retriever: Node,
    gate: Node,
    synthesizer: Node,
    critic: Node,
    validator: Node,
):
    given = (planner, retriever, gate, synthesizer, critic, validator)
    nodes = dict(zip(NODE_ORDER, given, strict=True))
    graph = StateGraph(ResearchState)
    for name, node in nodes.items():
        graph.add_node(name, _guarded(name, node))

    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", _then("retriever"), ["retriever", "validator"])
    graph.add_conditional_edges("retriever", _then("gate"), ["gate", "validator"])
    graph.add_conditional_edges("gate", _after_gate, ["synthesizer", "validator"])
    graph.add_conditional_edges("synthesizer", _then("critic"), ["critic", "validator"])
    graph.add_edge("critic", "validator")
    graph.add_edge("validator", END)
    return graph.compile()
