"""Thin Anthropic wrapper: one structured-output call, usage, cost, timing (ARCHITECTURE §10).

No call is ever given tools (ARCHITECTURE §14.2). Refusals and truncated output raise LLMError,
which callers turn into a controlled abstention (`abstain_reason="error"`), never a partial answer.
"""

import time
from typing import Literal

import anthropic
from pydantic import BaseModel

from backend.app.config import LLM_MODEL, LLM_PRICE_PER_M_TOKENS, env
from backend.app.models.answer import LLMUsage

Effort = Literal["low", "medium", "high"]


class LLMError(RuntimeError):
    """The LLM call failed, was refused, or returned unusable output.

    `usage` is set when the call reached the model, so a failed call's cost is still counted.
    """

    def __init__(self, message: str, usage: LLMUsage | None = None):
        super().__init__(message)
        self.usage = usage


def anthropic_client() -> anthropic.Anthropic:
    # SDK retries 429/5xx/connection errors twice (ARCHITECTURE §13).
    return anthropic.Anthropic(api_key=env("ANTHROPIC_API_KEY"), max_retries=2, timeout=90)


def llm_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price = LLM_PRICE_PER_M_TOKENS[model]
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1e6


def structured_call[T: BaseModel](
    client: anthropic.Anthropic,
    *,
    system: str,
    user: str,
    schema: type[T],
    effort: Effort,
    max_tokens: int = 16000,
) -> tuple[T, LLMUsage, float]:
    """Return (parsed output, usage, elapsed ms). API errors propagate as LLMError."""
    start = time.perf_counter()
    try:
        response = client.messages.parse(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort},
            output_format=schema,
        )
    except anthropic.APIError as exc:
        raise LLMError(f"{type(exc).__name__}: {getattr(exc, 'message', exc)}"[:300]) from None
    elapsed_ms = (time.perf_counter() - start) * 1000

    usage = LLMUsage(
        model=LLM_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost_usd=llm_cost(LLM_MODEL, response.usage.input_tokens, response.usage.output_tokens),
    )
    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None)
        raise LLMError(f"model declined the request (category: {category})", usage)
    if response.stop_reason == "max_tokens":
        raise LLMError("model output was truncated (max_tokens)", usage)
    if response.parsed_output is None:
        raise LLMError("model returned no parseable structured output", usage)
    return response.parsed_output, usage, elapsed_ms
