"""Closed-book control (ADR-010, PLAN M4-T3b): the runtime model answers the question with no
retrieved evidence, verdict only (yes/no/maybe; it must answer). Same model and effort as the
synthesizer, so the comparison isolates what retrieved evidence adds. Lives in eval/, not in the
product: it is a measurement, never served. Prompt: eval/prompts/closed_book_v1.txt (PROMPTS.md).
"""

from typing import Literal

import anthropic
from pydantic import BaseModel

from backend.app.baseline.synthesizer import SYNTHESIZER_EFFORT, _neutralize
from backend.app.config import REPO_ROOT
from backend.app.llm.client import LLMError, structured_call
from backend.app.models.answer import LLMUsage

CLOSED_BOOK_VERSION = "v1"
CLOSED_BOOK_EFFORT = SYNTHESIZER_EFFORT  # same model and effort as the evidence-based answer
SYSTEM_PROMPT = (
    REPO_ROOT / "eval" / "prompts" / f"closed_book_{CLOSED_BOOK_VERSION}.txt"
).read_text(encoding="utf-8")
USER_TEMPLATE = "<QUESTION>\n{question}\n</QUESTION>"


class ClosedBookAnswer(BaseModel):
    verdict: Literal["yes", "no", "maybe"]


def answer_closed_book(client: anthropic.Anthropic, question: str) -> tuple[str, LLMUsage, float]:
    """(verdict, usage, elapsed ms). A failed call raises LLMError with `usage` as a list, the
    shape the eval runner's error rows expect, so its cost is still counted."""
    try:
        out, usage, elapsed_ms = structured_call(
            client,
            system=SYSTEM_PROMPT,
            user=USER_TEMPLATE.format(question=_neutralize(question)),
            schema=ClosedBookAnswer,
            effort=CLOSED_BOOK_EFFORT,
        )
    except LLMError as exc:
        exc.usage = [exc.usage] if exc.usage else []
        raise
    return out.verdict, usage, elapsed_ms
