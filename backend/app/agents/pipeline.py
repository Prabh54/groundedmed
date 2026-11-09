"""Agent mode end to end (ARCHITECTURE §5–6, §8): the fixed graph with the real nodes, total timing,
and stage events for `/query/stream`.

Stage messages state only observable facts (counts, pass/fail), never model reasoning or critic
reasons. Retrieval/database failures raise RetrievalError (API 503); a provider failure ends as an
`error` abstention inside the answer, like the baseline.
"""

import time
import uuid
from collections.abc import Callable

import anthropic
import psycopg
import voyageai

from backend.app.agents.critic import make_critic_node
from backend.app.agents.graph import NODE_ORDER, Node, build_graph
from backend.app.agents.planner import make_planner_node
from backend.app.agents.retriever import make_retriever_node
from backend.app.agents.synthesizer import make_synthesizer_node
from backend.app.agents.validator import make_gate_node, make_validator_node
from backend.app.llm.client import Cancelled
from backend.app.models.agent import ResearchState, initial_state
from backend.app.models.answer import FinalAnswer

# (name, status, message, ms); status is started | done | failed
StageEvent = tuple[str, str, str, float]
Emit = Callable[[StageEvent], None]
ShouldStop = Callable[[], bool]

STARTED = {
    "planner": "Planning evidence needs",
    "retriever": "Searching the corpus",
    "gate": "Checking evidence relevance",
    "synthesizer": "Writing a cited answer",
    "validator": "Validating citations",
}
FAILED = "This step failed"
NO_ANSWER = {  # validator "done" message per abstain reason (observable facts only)
    "weak_evidence": "No answer: the evidence scored below the relevance threshold",
    "no_supported_claims": "No answer: no claim was supported by its cited evidence",
    "model_abstained": "No answer: the evidence does not address the question",
    "error": "No answer: a processing step failed",
}


def _items(state: ResearchState) -> int:
    draft = state["draft_answer"]
    return len(draft.claims) + len(draft.limitations) if draft else 0


def started_message(name: str, state: ResearchState) -> str:
    if name == "critic":
        return f"Checking {_items(state)} claims and limitations"
    return STARTED[name]


def done_message(name: str, update: dict, state: ResearchState) -> str:
    if name == "planner":
        return f"Planned {len(update['subquestions'])} search queries"
    if name == "retriever":
        note = " (reranker unavailable: similarity order)" if update["rerank_skipped"] else ""
        return (
            f"Retrieved {len(update['retrieved_candidates'])} candidate passages; "
            f"selected {len(update['selected_evidence'])} evidence passages{note}"
        )
    if name == "gate":
        return (
            "Evidence passed the relevance threshold"
            if update["gate_passed"]
            else "Evidence scored below the relevance threshold"
        )
    if name == "synthesizer":
        draft = update["draft_answer"]
        return f"Drafted {len(draft.claims)} claims and {len(draft.limitations)} limitations"
    if name == "critic":
        return f"Checked {_items(state)} claims and limitations"
    final: FinalAnswer = update["final_answer"]
    if final.abstained:
        return NO_ANSWER[final.abstain_reason]
    removed = len(update["critic_removed"]) + _validator_removed(final)
    return f"Kept {len(final.claims)} cited claims; removed {removed} unsupported items"


def _validator_removed(final: FinalAnswer) -> int:
    v = final.validation
    return sum(
        map(len, (v.removed_uncited, v.removed_invalid_citation, v.removed_limitations,
                  v.removed_bad_id))
    )  # fmt: skip


def _announced(name: str, node: Node, emit: Emit) -> Node:
    def run(state: ResearchState) -> dict:
        emit((name, "started", started_message(name, state), 0.0))
        start = time.perf_counter()
        try:
            update = node(state)
        except Exception:
            emit((name, "failed", FAILED, (time.perf_counter() - start) * 1000))
            raise
        emit(
            (name, "done", done_message(name, update, state), (time.perf_counter() - start) * 1000)
        )
        return update

    return run


def _stoppable(name: str, node: Node, should_stop: ShouldStop) -> Node:
    """Start the node only if the visitor is still there; a running node always finishes."""

    def run(state: ResearchState) -> dict:
        if should_stop():
            raise Cancelled(name)
        return node(state)

    return run


def build_agent_graph(
    llm: anthropic.Anthropic,
    voyage: voyageai.Client,
    conn: psycopg.Connection,
    emit: Emit | None = None,
    should_stop: ShouldStop | None = None,
):
    nodes = {
        "planner": make_planner_node(llm),
        "retriever": make_retriever_node(conn=conn, client=voyage),
        "gate": make_gate_node(),
        "synthesizer": make_synthesizer_node(llm),
        "critic": make_critic_node(llm),
        "validator": make_validator_node(),
    }
    if emit is not None:
        nodes = {name: _announced(name, node, emit) for name, node in nodes.items()}
    if should_stop is not None:  # outermost, so a skipped node is not even announced
        nodes = {name: _stoppable(name, node, should_stop) for name, node in nodes.items()}
    return build_graph(**nodes)


def answer_agent(
    question: str,
    *,
    conn: psycopg.Connection,
    voyage: voyageai.Client,
    llm: anthropic.Anthropic,
    request_id: str | None = None,
    emit: Emit | None = None,
    should_stop: ShouldStop | None = None,
) -> tuple[FinalAnswer, ResearchState]:
    """Run the graph; return the final answer (timing_ms completed with every stage and the total)
    and the final state (sub-queries, candidates, critic results for logs and eval).

    An exception that escapes the graph (retrieval/database failure, bug) gets `usage`: the calls
    already made, e.g. the planner's, so a failed question still counts its cost.
    That includes `Cancelled`, raised when `should_stop()` is true before a node starts."""
    start = time.perf_counter()
    graph = build_agent_graph(llm, voyage, conn, emit, should_stop)
    state = initial_state(request_id or str(uuid.uuid4()), question)
    try:
        for step_state in graph.stream(state, stream_mode="values"):  # state after each step
            state = step_state
    except Exception as exc:
        exc.usage = list(state["usage"])
        raise
    total = (time.perf_counter() - start) * 1000
    timing = {name: 0.0 for name in NODE_ORDER} | state["timings"] | {"total": total}
    final = state["final_answer"].model_copy(update={"timing_ms": timing})
    return final, state
