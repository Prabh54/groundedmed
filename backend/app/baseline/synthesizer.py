"""Synthesizer (PROMPTS.md, version in SYNTHESIZER_VERSION): question + evidence -> DraftAnswer.

Shared by the baseline and, from M3, the agent graph (ADR-015). Evidence is wrapped as untrusted
data (ARCHITECTURE §14.1); the call has no tools.
"""

import html
import re
import unicodedata

import anthropic

from backend.app.config import REPO_ROOT
from backend.app.llm.client import Effort, structured_call
from backend.app.models.answer import DraftAnswer, LLMUsage
from backend.app.models.retrieval import Candidate

SYNTHESIZER_VERSION = "v2"  # v1 kept on disk for earlier runs (ADR-029)
SYNTHESIZER_EFFORT: Effort = "medium"  # ADR-022 starting point
SYSTEM_PROMPT = (
    REPO_ROOT / "backend" / "app" / "prompts" / f"synthesizer_{SYNTHESIZER_VERSION}.txt"
).read_text(encoding="utf-8")

# The user turn (versioned with the prompt; its hash is recorded in every eval run).
USER_TEMPLATE = (
    "{evidence}\n\n<QUESTION>\n{question}\n</QUESTION>\n\n"
    "Answer the QUESTION using only the EVIDENCE above. "
    'Cite evidence by its id, e.g. "{example_id}".'
)

# Untrusted text must not be able to close its own EVIDENCE element or open a fake one.
_TAG = re.compile(r"<\s*(/?)\s*(evidence|question)\b", re.IGNORECASE)


def _normalize(text: str) -> str:
    """Fold look-alikes before matching: HTML entities, full-width forms (NFKC), invisible
    format characters such as zero-width spaces."""
    text = unicodedata.normalize("NFKC", html.unescape(text))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def _neutralize(text: str) -> str:
    """Text without wrapper-like tags is passed through byte-for-byte; otherwise the normalized
    text is used with every such tag defused."""
    folded = _normalize(text)
    if not _TAG.search(folded):
        return text
    return _TAG.sub(lambda m: f"[{m.group(1)}{m.group(2)}", folded)


def render_evidence(evidence: list[Candidate]) -> str:
    blocks = []
    for c in evidence:
        body = "\n".join(f"{_neutralize(s.label)}: {_neutralize(s.text)}" for s in c.sections)
        blocks.append(f'<EVIDENCE id="{c.citation_id}">\n{body}\n</EVIDENCE>')
    return "\n\n".join(blocks)


def render_user_message(question: str, evidence: list[Candidate]) -> str:
    return USER_TEMPLATE.format(
        evidence=render_evidence(evidence),
        question=_neutralize(question),
        example_id=evidence[0].citation_id if evidence else "PMID:12345",
    )


def synthesize(
    client: anthropic.Anthropic, question: str, evidence: list[Candidate]
) -> tuple[DraftAnswer, LLMUsage, float]:
    return structured_call(
        client,
        system=SYSTEM_PROMPT,
        user=render_user_message(question, evidence),
        schema=DraftAnswer,
        effort=SYNTHESIZER_EFFORT,
    )
