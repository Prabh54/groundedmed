"""Thin Anthropic wrapper: one structured-output call, usage, cost, timing (ARCHITECTURE §10).

No call is ever given tools (ARCHITECTURE §14.2). Refusals and truncated output raise LLMError,
which callers turn into a controlled abstention (`abstain_reason="error"`), never a partial answer.
"""

import time
from typing import Any, Literal

import anthropic
from pydantic import BaseModel, ValidationError

from backend.app.config import LLM_MODEL, LLM_PRICE_PER_M_TOKENS, env
from backend.app.models.answer import LLMUsage

Effort = Literal["low", "medium", "high"]


class LLMError(RuntimeError):
    """The LLM call failed, was refused, or returned unusable output.

    `usage` is set when the call reached the model, so a failed call's cost is still counted.
    """

    def __init__(self, message: str, usage: LLMUsage | None = None, elapsed_ms: float = 0.0):
        super().__init__(message)
        self.usage, self.elapsed_ms = usage, elapsed_ms


class Cancelled(RuntimeError):
    """The visitor left mid-answer, so step `before` and everything after it never started.

    `usage` is set by the caller to the calls already made, so their cost is still logged.
    """

    def __init__(self, before: str):
        super().__init__(f"visitor disconnected; stopped before {before}")
        self.before, self.usage = before, []


def anthropic_client() -> anthropic.Anthropic:
    # SDK retries 429/5xx/connection errors twice (ARCHITECTURE §13).
    return anthropic.Anthropic(api_key=env("ANTHROPIC_API_KEY"), max_retries=2, timeout=90)


def llm_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """`input_tokens` is the uncached remainder; cache reads and writes bill at their own rates."""
    price = LLM_PRICE_PER_M_TOKENS[model]
    return (
        input_tokens * price["input"]
        + output_tokens * price["output"]
        + cache_read_tokens * price["cache_read"]
        + cache_write_tokens * price["cache_write"]
    ) / 1e6


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The model's JSON schema in the form structured outputs requires: references inlined, every
    object closed (`additionalProperties: false`) with all properties required, no titles."""
    root = model.model_json_schema()
    defs = root.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(defs[node["$ref"].rsplit("/", 1)[-1]])
        out = {k: walk(v) for k, v in node.items() if k != "title"}
        if out.get("type") == "object":
            out["additionalProperties"] = False
            out["required"] = list(out.get("properties", {}))
        return out

    return walk(root)


def structured_call[T: BaseModel](
    client: anthropic.Anthropic,
    *,
    system: str,
    user: str,
    schema: type[T],
    effort: Effort,
    max_tokens: int = 16000,
    model: str = LLM_MODEL,
    cache: bool = False,
) -> tuple[T, LLMUsage, float]:
    """Return (parsed output, usage, elapsed ms). Every failure raises LLMError (ARCHITECTURE §13).

    Uses `messages.create` and validates afterwards, so the stop reason and usage are read before
    parsing: a refused or truncated response is reported as such, with its cost counted.
    """
    start = time.perf_counter()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=(
                [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
                if cache
                else system
            ),
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": strict_schema(schema)},
            },
        )
    except anthropic.APIError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        message = f"{type(exc).__name__}: {getattr(exc, 'message', exc)}"[:300]
        raise LLMError(message, elapsed_ms=elapsed_ms) from None
    elapsed_ms = (time.perf_counter() - start) * 1000

    cache_read = getattr(response.usage, "cache_read_input_tokens", None) or 0
    cache_write = getattr(response.usage, "cache_creation_input_tokens", None) or 0
    usage = LLMUsage(
        model=model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost_usd=llm_cost(
            model,
            response.usage.input_tokens,
            response.usage.output_tokens,
            cache_read,
            cache_write,
        ),
        cache_read_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
    )
    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None)
        raise LLMError(f"model declined the request (category: {category})", usage, elapsed_ms)
    if response.stop_reason == "max_tokens":
        raise LLMError("model output was truncated (max_tokens)", usage, elapsed_ms)
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise LLMError("model returned no text output", usage, elapsed_ms)
    try:
        return schema.model_validate_json(text), usage, elapsed_ms
    except ValidationError as exc:
        raise LLMError(f"output failed schema validation: {exc.error_count()} errors", usage,
                       elapsed_ms) from None  # fmt: skip
