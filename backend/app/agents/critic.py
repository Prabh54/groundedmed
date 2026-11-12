"""Critic (version in CRITIC_VERSION): checks each claim and limitation against
only the passages it cites. It can only remove items, never add or rewrite text (ARCHITECTURE §6).

The model returns a verdict per item ID; `apply_critic` (deterministic) keeps an item only if its
result says keep + supported + citation_valid. `revise` counts as remove, an item without a result
is removed (unverified), and results for unknown IDs are ignored. Kept items are the synthesizer's
original objects, so critic text never reaches the answer.

ADR-044/045: when the critic flags a draft that would be refused, the critic stage runs one
synthesizer retry with the critic's notes (agents/reviser.py) and checks the new draft once.
The retry is the synthesizer writing, not the critic; its draft passes the same citation validator
and critic rules. The deterministic rules live in agents/rules.py.
"""

import html
import re
import unicodedata

import anthropic
from pydantic import BaseModel

from backend.app.agents.graph import Node
from backend.app.agents.reviser import needs_retry, revise
from backend.app.agents.rules import _kept, apply_critic, refusal_reason  # noqa: F401 (re-exported)
from backend.app.config import CRITIC_MODEL, PROMPT_CACHE_STAGES, REPO_ROOT
from backend.app.llm.client import Effort, LLMError, structured_call
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
        model=CRITIC_MODEL,
        cache="critic" in PROMPT_CACHE_STAGES,
    )


def make_critic_node(client: anthropic.Anthropic) -> Node:
    """Check the draft. If the critic flags it AND the first pass would be refused, one synthesizer
    retry with the critic's notes and one re-check (ADR-044, ADR-045: a first pass that would be
    answered is kept as it is). The retry runs inside this stage, so the graph's stages are
    unchanged. If the retry fails, the first pass stands (its cost is still counted)."""

    def critic(state: ResearchState) -> dict:
        draft = state["draft_answer"]
        if draft is None or not (draft.claims or draft.limitations):
            return {"critic_results": [], "answer_supported": None}  # nothing to check, no call
        question, evidence = state["question"], state["selected_evidence"]
        output, usage, _ = critique(client, question, draft, evidence)
        update: dict = {"usage": [usage]}
        final = (draft, output.results, output.answer_supported)
        evidence_ids = [c.citation_id for c in evidence]
        first = refusal_reason(draft, evidence_ids, output.results, output.answer_supported)
        if first[0] is not None and needs_retry(
            draft, output.results, output.answer_supported, _kept
        ):
            update |= {"retry_used": True, "first_draft_answer": draft,
                       "first_critic_results": output.results,
                       "first_answer_supported": output.answer_supported}  # fmt: skip
            try:
                new_draft, retry_usage, _ = revise(
                    client, question, draft, output.results, output.answer_supported, evidence
                )
                update["usage"].append(retry_usage)
                if new_draft.claims or new_draft.limitations:
                    recheck, recheck_usage, _ = critique(client, question, new_draft, evidence)
                    update["usage"].append(recheck_usage)
                    final = (new_draft, recheck.results, recheck.answer_supported)
                else:  # e.g. the retry now says insufficient_evidence: nothing to re-check
                    final = (new_draft, [], None)
            except LLMError as exc:  # the first pass stands
                update["retry_error"] = str(exc)[:300]
                if exc.usage is not None:
                    update["usage"].append(exc.usage)
        final_draft, final_results, final_supported = final
        cleaned, _ = apply_critic(final_draft, final_results)
        return update | {
            "draft_answer": final_draft,
            "critic_results": final_results,
            "answer_supported": final_supported,
            "claims": cleaned.claims,
        }

    return critic
