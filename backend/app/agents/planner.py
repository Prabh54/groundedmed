"""Planner (version in PLANNER_VERSION): question -> 1–3 search sub-queries.

The model only proposes queries; the cleanup below is deterministic. If no usable query survives,
the planner falls back to the original question (ARCHITECTURE §13). The retriever always searches
with the original question as well, so the planner can only widen the candidate pool.
"""

import re

import anthropic
from pydantic import BaseModel

from backend.app.agents.graph import Node
from backend.app.baseline.synthesizer import _neutralize
from backend.app.config import PLANNER_MODEL, PROMPT_CACHE_STAGES, REPO_ROOT
from backend.app.llm.client import Effort, structured_call
from backend.app.models.agent import ResearchState, SubQuestion
from backend.app.models.answer import LLMUsage

PLANNER_VERSION = "v1"
PLANNER_EFFORT: Effort = "low"  # starting point
SYSTEM_PROMPT = (
    REPO_ROOT / "backend" / "app" / "prompts" / f"planner_{PLANNER_VERSION}.txt"
).read_text(encoding="utf-8")

# The user turn (versioned with the prompt). The question is user input, defused like in the
# synthesizer so it cannot open or close a wrapper element.
USER_TEMPLATE = (
    "<QUESTION>\n{question}\n</QUESTION>\n\n"
    "List the evidence needs for this QUESTION as search queries. Do not answer it."
)

MAX_SUBQUESTIONS = 3
MAX_QUERY_CHARS = 300

# A query that is, or opens with, a verdict ("Yes", "No, ...", "maybe.") is an answer, not a search.
# "No-reflow after PCI" or "Yesterday's ..." are not matched.
_VERDICT = re.compile(r"^\s*(yes|no|maybe|insufficient[_ ]evidence)\s*([.,:;!]|$)", re.IGNORECASE)


class PlannerOutput(BaseModel):
    """Exactly the JSON the planner must return (sent to the API as the output schema)."""

    subquestions: list[SubQuestion]


def query_key(text: str) -> str:
    return " ".join(text.lower().split()).rstrip("?.!")


def clean_subquestions(raw: list[SubQuestion], question: str) -> list[SubQuestion]:
    """Keep at most 3 usable queries, renumbered q1..qn; fall back to the original question.

    Dropped: blank, too long, verdict-like, duplicates (of each other or of the question, which the
    retriever searches anyway)."""
    seen = {query_key(question)}
    kept: list[str] = []
    for sub in raw:
        query = " ".join(sub.query.split())
        if not query or len(query) > MAX_QUERY_CHARS or _VERDICT.match(query):
            continue
        if query_key(query) in seen:
            continue
        seen.add(query_key(query))
        kept.append(query)
        if len(kept) == MAX_SUBQUESTIONS:
            break
    if not kept:
        kept = [question.strip()]
    return [SubQuestion(id=f"q{i}", query=q) for i, q in enumerate(kept, start=1)]


def render_user_message(question: str) -> str:
    return USER_TEMPLATE.format(question=_neutralize(question))


def plan(
    client: anthropic.Anthropic, question: str
) -> tuple[list[SubQuestion], list[SubQuestion], LLMUsage, float]:
    """Return (cleaned sub-queries, raw model sub-queries, usage, elapsed ms)."""
    output, usage, elapsed_ms = structured_call(
        client,
        system=SYSTEM_PROMPT,
        user=render_user_message(question),
        schema=PlannerOutput,
        effort=PLANNER_EFFORT,
        model=PLANNER_MODEL,
        cache="planner" in PROMPT_CACHE_STAGES,
    )
    return clean_subquestions(output.subquestions, question), output.subquestions, usage, elapsed_ms


def make_planner_node(client: anthropic.Anthropic) -> Node:
    def planner(state: ResearchState) -> dict:
        subquestions, _, usage, _ = plan(client, state["question"])
        return {"subquestions": subquestions, "usage": [usage]}

    return planner
