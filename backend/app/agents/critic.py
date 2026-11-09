"""Critic (version in CRITIC_VERSION): checks each claim and limitation against
only the passages it cites. It can only remove items, never add or rewrite text (ARCHITECTURE §6).

The model returns a verdict per item ID; `apply_critic` (deterministic) keeps an item only if its
result says keep + supported + citation_valid. `revise` counts as remove, an item without a result
is removed (unverified), and results for unknown IDs are ignored. Kept items are the synthesizer's
original objects, so critic text never reaches the answer.
"""

import html
import re
import unicodedata

import anthropic
from pydantic import BaseModel

from backend.app.agents.graph import Node
from backend.app.config import REPO_ROOT
from backend.app.llm.client import Effort, structured_call
from backend.app.models.agent import CriticResult, ResearchState
from backend.app.models.answer import Claim, DraftAnswer, LLMUsage
from backend.app.models.retrieval import Candidate

CRITIC_VERSION = "v1"
CRITIC_EFFORT: Effort = "low"  # starting point
SYSTEM_PROMPT = (
    REPO_ROOT / "backend" / "app" / "prompts" / f"critic_{CRITIC_VERSION}.txt"
).read_text(encoding="utf-8")

USER_TEMPLATE = (
    "<QUESTION>\n{question}\n</QUESTION>\n\n"
    "<VERDICT>{verdict}</VERDICT>\n\n"
    "{items}\n\n"
    "Check each ITEM (claims c<n>, limitations l<n>) only against the EVIDENCE inside that ITEM. "
    "Return exactly one result per ITEM, with its id as claim_id. Set answer_supported to false "
    "if the supported claims do not support the VERDICT."
)

# Model-written item text and untrusted passages must not open or close a wrapper element.
_TAG = re.compile(r"<\s*(/?)\s*(evidence|question|item|verdict)\b", re.IGNORECASE)


class CriticOutput(BaseModel):
    """Exactly the JSON the critic must return (sent to the API as the output schema)."""

    results: list[CriticResult]
    answer_supported: bool


def _defuse(text: str) -> str:
    """Same folding as the synthesizer (entities, NFKC, invisible characters), wider tag set."""
    folded = unicodedata.normalize("NFKC", html.unescape(text))
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Cf")
    if not _TAG.search(folded):
        return text
    return _TAG.sub(lambda m: f"[{m.group(1)}{m.group(2)}", folded)


def _attr(value: str) -> str:
    """Model-written IDs go into attributes: keep only ID-like characters (no quotes, no tags)."""
    return re.sub(r"[^\w:.-]", "_", value)


def _render_item(kind: str, item: Claim, by_id: dict[str, Candidate]) -> str:
    blocks = []
    for cid in item.citation_ids:
        c = by_id.get(cid)
        if c is None:
            blocks.append(
                f'<EVIDENCE id="{_attr(cid)}">\n(not in the retrieved evidence)\n</EVIDENCE>'
            )
            continue
        body = "\n".join(f"{_defuse(s.label)}: {_defuse(s.text)}" for s in c.sections)
        blocks.append(f'<EVIDENCE id="{c.citation_id}">\n{body}\n</EVIDENCE>')
    evidence = "\n".join(blocks) if blocks else "(no citation)"
    head = f'<ITEM id="{_attr(item.id)}" kind="{kind}">'
    return f"{head}\nTEXT: {_defuse(item.text)}\n{evidence}\n</ITEM>"


def render_user_message(question: str, draft: DraftAnswer, evidence: list[Candidate]) -> str:
    by_id = {c.citation_id: c for c in evidence}
    items = [_render_item("claim", c, by_id) for c in draft.claims]
    items += [_render_item("limitation", item, by_id) for item in draft.limitations]
    return USER_TEMPLATE.format(
        question=_defuse(question), verdict=draft.verdict, items="\n\n".join(items)
    )


def critique(
    client: anthropic.Anthropic, question: str, draft: DraftAnswer, evidence: list[Candidate]
) -> tuple[CriticOutput, LLMUsage, float]:
    return structured_call(
        client,
        system=SYSTEM_PROMPT,
        user=render_user_message(question, draft, evidence),
        schema=CriticOutput,
        effort=CRITIC_EFFORT,
    )


def _kept(result: CriticResult | None) -> bool:
    return (
        result is not None
        and result.action == "keep"
        and result.supported
        and result.citation_valid
    )


def apply_critic(draft: DraftAnswer, results: list[CriticResult]) -> tuple[DraftAnswer, list[str]]:
    """Return (draft with only kept items, removed item IDs). Never adds or edits text: the verdict
    and every kept item are the synthesizer's own."""
    by_id: dict[str, CriticResult] = {}
    for r in results:
        by_id.setdefault(r.claim_id, r)  # first result per ID wins; duplicates can't overrule it
    removed = [i.id for i in draft.claims + draft.limitations if not _kept(by_id.get(i.id))]
    cleaned = DraftAnswer(
        verdict=draft.verdict,
        claims=[c for c in draft.claims if c.id not in removed],
        limitations=[item for item in draft.limitations if item.id not in removed],
    )
    return cleaned, removed


def make_critic_node(client: anthropic.Anthropic) -> Node:
    def critic(state: ResearchState) -> dict:
        draft = state["draft_answer"]
        if draft is None or not (draft.claims or draft.limitations):
            return {"critic_results": [], "answer_supported": None}  # nothing to check, no call
        output, usage, _ = critique(client, state["question"], draft, state["selected_evidence"])
        cleaned, _ = apply_critic(draft, output.results)
        return {
            "critic_results": output.results,
            "answer_supported": output.answer_supported,
            "claims": cleaned.claims,
            "usage": [usage],
        }

    return critic
