"""Gemini REST client for the independent judge (rubric v2 §4). Offline evaluation only; never
deployed.

Plain HTTPS to the Gemini API (`generateContent`), key `EVALUATOR_API_KEY` in a header (never in a
URL). Every result records the exact model string requested, the `modelVersion` Google reports for
the response and the model's metadata, so a silent change behind the name can be seen afterwards,
as far as Google exposes it.

Pinning (ADR-035): `gemini-3.1-pro-preview`, whose metadata reports the dated version
"3.1-pro-preview-01-2026". No frozen snapshot name is reachable for this account, so the pin is
the name plus that version string: `verify_model` refuses to judge if Google's metadata version
differs (the judge runner checks it before, during and after each run), and `generate` refuses a
response whose reported `modelVersion` differs from the one observed at the M4-T0 test call. That
per-response value is only "gemini-3.1-pro-preview" (undated), so the dated check lives in the
metadata. A swap that changes neither string cannot be detected.
"""

import hashlib
import json
import urllib.error
import urllib.request

from backend.app.config import env, redact

JUDGE_PROVIDER = "google"
JUDGE_MODEL = "gemini-3.1-pro-preview"  # ADR-035 (gemini-2.5-pro refused for this account)
EXPECTED_MODEL_VERSION = "3.1-pro-preview-01-2026"  # Google's metadata `version`, 2026-10-24
# `modelVersion` reported at the M4-T0 test call (judge-testcall-20261025-190152); undated
EXPECTED_RESPONSE_MODEL_VERSION = "gemini-3.1-pro-preview"
# USD per million tokens, paid tier, prompts <= 200k tokens (Google pricing page, 2026-10-24).
# Thinking tokens are billed as output.
JUDGE_PRICE_PER_M_TOKENS = {"input": 2.00, "output": 12.00}
API = "https://generativelanguage.googleapis.com/v1beta"
FLOATING_MARKERS = ("latest", "exp")  # aliases that move by design


class JudgeAPIError(RuntimeError):
    pass


class JudgeModelChanged(JudgeAPIError):
    """Google reports a different model version than the pinned one (methodology contingency)."""


def check_pinned(model: str) -> None:
    """Refuse floating aliases ("latest") and experimental names (they change by design)."""
    if any(marker in model for marker in FLOATING_MARKERS):
        raise ValueError(f"{model!r} is a floating or experimental name; the judge must be pinned.")


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


def verify_model(model: str = JUDGE_MODEL, expected: str = EXPECTED_MODEL_VERSION) -> dict:
    """Free check before judging: refuse if Google's metadata version is not the pinned one."""
    check_pinned(model)
    meta = model_metadata(model)
    if meta["version"] != expected:
        raise JudgeModelChanged(f"{model}: Google reports version {meta['version']!r}, pinned "
                                f"{expected!r}. Stop: see eval/methodology.md.")  # fmt: skip
    return meta


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
    reported = raw.get("modelVersion")
    if EXPECTED_RESPONSE_MODEL_VERSION is not None and reported != EXPECTED_RESPONSE_MODEL_VERSION:
        raise JudgeModelChanged(f"response came from {reported!r}, pinned "
                                f"{EXPECTED_RESPONSE_MODEL_VERSION!r}; stop.")  # fmt: skip
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
