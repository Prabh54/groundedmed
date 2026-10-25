"""Gemini REST client for the independent judge (rubric v2 §4). Offline evaluation only; never
deployed.

Plain HTTPS to the Gemini API (`generateContent`), key `EVALUATOR_API_KEY` in a header (never in a
URL). Every result records the exact model string requested, the `modelVersion` Google reports for
the response and the model's metadata, so a silent change behind the name can be seen afterwards,
as far as Google exposes it.

Pinning: `gemini-2.5-pro` is Google's stable endpoint ("Stable release (June 17th, 2025)" in its
model metadata, version "2.5"). Google publishes no dated snapshot (e.g. `-001`) for it, so this
name is the most exact pin available; Google says stable models "usually don't change", which is
not a guarantee.
"""

import hashlib
import json
import urllib.error
import urllib.request

from backend.app.config import env, redact

JUDGE_PROVIDER = "google"
# ADR-014 choice. 2026-10-24: Google refused it here (HTTP 404 "no longer available to new
# users", on the free countTokens call; nothing billed). Replacement pending the owner's decision.
JUDGE_MODEL = "gemini-2.5-pro"
EXPECTED_MODEL_VERSION = "2.5"  # `version` in Google's metadata for JUDGE_MODEL on 2026-10-24
# USD per million tokens, paid tier, prompts <= 200k tokens (Google pricing page, 2026-10-24).
# Thinking tokens are billed as output.
JUDGE_PRICE_PER_M_TOKENS = {"input": 1.25, "output": 10.00}
API = "https://generativelanguage.googleapis.com/v1beta"
FLOATING_MARKERS = ("latest", "preview", "exp")


class JudgeAPIError(RuntimeError):
    pass


def check_pinned(model: str) -> None:
    """Refuse floating aliases and preview/experimental names (they can change under us)."""
    if any(marker in model for marker in FLOATING_MARKERS):
        raise ValueError(
            f"{model!r} is a floating or preview model name; the judge must be pinned."
        )


def judge_cost(input_tokens: int, output_tokens_incl_thinking: int) -> float:
    price = JUDGE_PRICE_PER_M_TOKENS
    return (input_tokens * price["input"] + output_tokens_incl_thinking * price["output"]) / 1e6


def _request(path: str, body: dict | None = None, timeout: float = 120) -> dict:
    data = None if body is None else json.dumps(body).encode()
    headers = {"x-goog-api-key": env("EVALUATOR_API_KEY"), "Content-Type": "application/json"}
    req = urllib.request.Request(f"{API}/{path}", data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = redact(exc.read().decode(errors="replace"))[:800]
        raise JudgeAPIError(f"HTTP {exc.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise JudgeAPIError(redact(f"{type(exc).__name__}: {exc}")) from None


def model_metadata(model: str = JUDGE_MODEL) -> dict:
    """Google's metadata for the model (free call)."""
    meta = _request(f"models/{model}")
    return {k: meta.get(k) for k in ("name", "version", "displayName", "description")}


def build_body(system_instruction: str, user_text: str, schema: dict, *, temperature: float,
               seed: int, max_output_tokens: int, thinking_budget: int | None) -> dict:  # fmt: skip
    config = {
        "temperature": temperature,
        "seed": seed,
        "maxOutputTokens": max_output_tokens,
        "responseMimeType": "application/json",
        "responseJsonSchema": schema,
    }
    if thinking_budget is not None:
        config["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    return {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": config,
    }


def count_tokens(body: dict, model: str = JUDGE_MODEL) -> int:
    """Input tokens for this request (free call)."""
    request = {k: body[k] for k in ("systemInstruction", "contents")} | {"model": f"models/{model}"}
    return _request(f"models/{model}:countTokens", {"generateContentRequest": request})[
        "totalTokens"
    ]


def generate(body: dict, model: str = JUDGE_MODEL) -> dict:
    """One paid generateContent call. Returns the raw response plus usage, cost and fingerprint."""
    check_pinned(model)
    raw = _request(f"models/{model}:generateContent", body)
    usage = raw.get("usageMetadata", {})
    out = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
    return {
        "request_sha256": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest(),
        "model_requested": model,
        "model_version_reported": raw.get("modelVersion"),
        "response": raw,
        "usage": {
            "input_tokens": usage.get("promptTokenCount", 0),
            "output_tokens": usage.get("candidatesTokenCount", 0),
            "thinking_tokens": usage.get("thoughtsTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
        },
        "cost_usd": judge_cost(usage.get("promptTokenCount", 0), out),
    }


def response_text(raw: dict) -> str:
    parts = raw["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))
