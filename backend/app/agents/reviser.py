"""Agent-only second synthesizer pass (ADR-044, M15): one retry with the critic's feedback.

When the critic does not keep a claim, or says the claims do not support the verdict, the
synthesizer writes ONE new answer from the same evidence, seeing its previous answer and the
critic's notes. It may change the verdict (including to insufficient_evidence). The critic then
re-checks the new answer once; there is never a second retry. The new answer goes through the
same citation validator and critic rules as any draft, so nothing uncited or unchecked can reach
the user. The baseline never uses this.

Critic notes and previous item text are model-written, derived from untrusted passages: they are
wrapped as data with wrapper-like tags defused, exactly like evidence.
"""

import html
import re
import unicodedata

import anthropic

from backend.app.baseline.synthesizer import SYSTEM_PROMPT, render_evidence
from backend.app.llm.client import Effort, structured_call
from backend.app.models.agent import CriticResult
from backend.app.models.answer import DraftAnswer, LLMUsage
from backend.app.models.retrieval import Candidate

RETRY_VERSION = "v1"
AGENT_SYNTHESIZER_EFFORT: Effort = "high"  # agent first pass and retry; the baseline keeps medium

RETRY_TEMPLATE = (
    "{evidence}\n\n<QUESTION>\n{question}\n</QUESTION>\n\n"
    "<PREVIOUS_ANSWER>\n<VERDICT>{verdict}</VERDICT>\n{items}\n</PREVIOUS_ANSWER>\n\n"
    "<REVIEW>\n{notes}\n<ANSWER_SUPPORTED>{answer_supported}</ANSWER_SUPPORTED>\n</REVIEW>\n\n"
    "A reviewer checked your previous answer against the EVIDENCE. Write a new answer to the "
    "QUESTION using only the EVIDENCE above. For every claim the REVIEW flags, reword it so its "
    "cited evidence fully supports it, or drop it. If ANSWER_SUPPORTED is false, choose the "
    "verdict (yes, no, maybe or insufficient_evidence) that your supported claims actually "
    'support. Cite evidence by its id, e.g. "{example_id}".'
)

_TAG = re.compile(
    r"<\s*(/?)\s*(evidence|question|previous_answer|verdict|item|review|note|answer_supported)\b",
    re.IGNORECASE,
)


def _defuse(text: str) -> str:
    """Same folding as the critic (entities, NFKC, invisible characters), wider tag set."""
    folded = unicodedata.normalize("NFKC", html.unescape(text))
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Cf")
    if not _TAG.search(folded):
        return text
    return _TAG.sub(lambda m: f"[{m.group(1)}{m.group(2)}", folded)


def _attr(value: str) -> str:
    return re.sub(r"[^\w:.-]", "_", value)


def needs_retry(draft: DraftAnswer, results: list[CriticResult], answer_supported: bool | None,
                kept) -> bool:  # fmt: skip
    """True if the critic did not keep some claim, or said the claims don't support the verdict."""
    by_id: dict[str, CriticResult] = {}
    for r in results:
        by_id.setdefault(r.claim_id, r)
    return answer_supported is False or any(not kept(by_id.get(c.id)) for c in draft.claims)


def render_retry_message(
    question: str,
    draft: DraftAnswer,
    results: list[CriticResult],
    answer_supported: bool | None,
    evidence: list[Candidate],
) -> str:
    items = [
        f'<ITEM id="{_attr(i.id)}" kind="{kind}" cites="{_attr(" ".join(i.citation_ids))}">'
        f"\n{_defuse(i.text)}\n</ITEM>"
        for kind, group in (("claim", draft.claims), ("limitation", draft.limitations))
        for i in group
    ]
    notes = [
        f'<NOTE id="{_attr(r.claim_id)}" action="{r.action}" supported="{str(r.supported).lower()}"'
        f' citation_valid="{str(r.citation_valid).lower()}">\n{_defuse(r.reason)}\n</NOTE>'
        for r in results
    ]
    return RETRY_TEMPLATE.format(
        evidence=render_evidence(evidence),
        question=_defuse(question),
        verdict=draft.verdict,
        items="\n".join(items) or "(no items)",
        notes="\n".join(notes) or "(no notes)",
        answer_supported=str(answer_supported).lower(),
        example_id=evidence[0].citation_id if evidence else "PMID:12345",
    )


def revise(
    client: anthropic.Anthropic,
    question: str,
    draft: DraftAnswer,
    results: list[CriticResult],
    answer_supported: bool | None,
    evidence: list[Candidate],
) -> tuple[DraftAnswer, LLMUsage, float]:
    return structured_call(
        client,
        system=SYSTEM_PROMPT,
        user=render_retry_message(question, draft, results, answer_supported, evidence),
        schema=DraftAnswer,
        effort=AGENT_SYNTHESIZER_EFFORT,
    )
