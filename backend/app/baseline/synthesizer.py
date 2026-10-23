"""Synthesizer (PROMPTS.md Synthesizer v1): question + selected evidence -> DraftAnswer.

Shared by the baseline and, from M3, the agent graph (ADR-015). Evidence is wrapped as untrusted
data (ARCHITECTURE §14.1); the call has no tools.
"""

import re

import anthropic

from backend.app.config import REPO_ROOT
from backend.app.llm.client import Effort, structured_call
from backend.app.models.answer import DraftAnswer, LLMUsage
from backend.app.models.retrieval import Candidate

SYNTHESIZER_VERSION = "v1"
SYNTHESIZER_EFFORT: Effort = "medium"  # ADR-022 starting point
SYSTEM_PROMPT = (
    REPO_ROOT / "backend" / "app" / "prompts" / f"synthesizer_{SYNTHESIZER_VERSION}.txt"
).read_text(encoding="utf-8")

# A passage must not be able to close its own EVIDENCE element or open a fake one.
_TAG = re.compile(r"<\s*(/?)\s*(evidence|question)\b", re.IGNORECASE)


def _neutralize(text: str) -> str:
    return _TAG.sub(lambda m: f"[{m.group(1)}{m.group(2)}", text)


def render_evidence(evidence: list[Candidate]) -> str:
    blocks = []
    for c in evidence:
        body = "\n".join(f"{s.label}: {_neutralize(s.text)}" for s in c.sections)
        blocks.append(f'<EVIDENCE id="{c.citation_id}">\n{body}\n</EVIDENCE>')
    return "\n\n".join(blocks)


def render_user_message(question: str, evidence: list[Candidate]) -> str:
    return (
        f"{render_evidence(evidence)}\n\n"
        f"<QUESTION>\n{_neutralize(question)}\n</QUESTION>\n\n"
        "Answer the QUESTION using only the EVIDENCE above. Cite evidence by its id, e.g. "
        f'"{evidence[0].citation_id if evidence else "PMID:12345"}".'
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
