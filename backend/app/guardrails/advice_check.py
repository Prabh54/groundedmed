"""Personal-advice screen for the public API (version in ADVICE_VERSION):
question -> is it asking for personal medical advice?

Runs in front of the agent and the baseline, in `backend/app/main.py` only. The benchmark runners
call the pipelines directly, so this screen is not part of any measured result. A failed check is
never treated as "not advice": callers refuse (fail closed).
"""

import anthropic
from pydantic import BaseModel

from backend.app.baseline.synthesizer import _neutralize
from backend.app.config import ADVICE_MODEL, REPO_ROOT, env
from backend.app.llm.client import Effort, structured_call
from backend.app.models.answer import LLMUsage

ADVICE_VERSION = "v1"
ADVICE_EFFORT: Effort = "low"
ADVICE_TIMEOUT_S = 10.0  # per attempt; one retry, so at most ~20 s before the refusal
ADVICE_MAX_TOKENS = 2048  # room for thinking plus the one-field answer; truncation fails closed
SYSTEM_PROMPT = (
    REPO_ROOT / "backend" / "app" / "prompts" / f"advice_check_{ADVICE_VERSION}.txt"
).read_text(encoding="utf-8")

# The question is user input, defused like in the planner so it cannot close its wrapper.
USER_TEMPLATE = (
    "<QUESTION>\n{question}\n</QUESTION>\n\n"
    "Does this QUESTION ask for personal medical advice? Do not answer it."
)


class AdviceOutput(BaseModel):
    """Exactly the JSON the screen must return (sent to the API as the output schema)."""

    personal_advice: bool


def advice_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key=env("ANTHROPIC_API_KEY"), max_retries=1, timeout=ADVICE_TIMEOUT_S
    )


def render_user_message(question: str) -> str:
    return USER_TEMPLATE.format(question=_neutralize(question))


def check_advice(client: anthropic.Anthropic, question: str) -> tuple[bool, LLMUsage, float]:
    """Return (personal advice?, usage, elapsed ms). Any failure raises (LLMError for the model)."""
    output, usage, elapsed_ms = structured_call(
        client,
        system=SYSTEM_PROMPT,
        user=render_user_message(question),
        schema=AdviceOutput,
        effort=ADVICE_EFFORT,
        max_tokens=ADVICE_MAX_TOKENS,
        model=ADVICE_MODEL,
    )
    return output.personal_advice, usage, elapsed_ms
