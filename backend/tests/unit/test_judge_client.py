"""Judge client and response check (M4-T0/T2): offline, with a fake HTTP layer. No paid calls."""

import io
import json

import pytest

from backend.app.agents import critic
from backend.app.config import LLM_MODEL
from eval import judge_client as jc
from eval import judge_rubric as jr


def test_judge_provider_and_model_differ_from_the_runtime_critic():
    assert jc.JUDGE_PROVIDER == "google" and not jc.JUDGE_MODEL.startswith("claude")
    assert LLM_MODEL.startswith("claude") and jc.JUDGE_MODEL != LLM_MODEL
    assert critic.SYSTEM_PROMPT != jr.load_prompt()["SYSTEM_INSTRUCTION"]


@pytest.mark.parametrize("name", ["gemini-pro-latest", "gemini-flash-latest", "gemini-2.0-exp"])
def test_floating_model_names_are_refused(name):
    with pytest.raises(ValueError, match="pinned"):
        jc.check_pinned(name)


def test_cost_arithmetic():
    # 1,000,000 input + 1,000,000 output(incl. thinking) at the table prices
    price = jc.JUDGE_PRICE_PER_M_TOKENS
    assert jc.judge_cost(1_000_000, 1_000_000) == pytest.approx(price["input"] + price["output"])
    assert jc.judge_cost(0, 0) == 0


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_generate_sends_key_in_header_only_and_counts_thinking_as_output(monkeypatch):
    monkeypatch.setenv("EVALUATOR_API_KEY", "fake-key-123456")
    sent = {}
    reply = {
        "modelVersion": "fake-version",
        "candidates": [{"content": {"parts": [{"text": "thinking", "thought": True},
                                              {"text": '{"judgments": []}'}]},
                        "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 20,
                          "thoughtsTokenCount": 30, "totalTokenCount": 150},
    }  # fmt: skip

    def urlopen(req, timeout):
        sent["url"], sent["headers"] = req.full_url, dict(req.header_items())
        sent["body"] = json.loads(req.data)
        return FakeResponse(json.dumps(reply).encode())

    monkeypatch.setattr(jc.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(jc, "JUDGE_MODEL", "gemini-test-model")
    body = jc.build_body("sys", "user", jr.load_schema(), temperature=0, seed=jr.SEED,
                         max_output_tokens=100, thinking_budget=None)  # fmt: skip
    out = jc.generate(body, model="gemini-test-model")
    assert "fake-key-123456" not in sent["url"]
    assert sent["headers"]["X-goog-api-key"] == "fake-key-123456"
    assert sent["body"]["generationConfig"]["responseJsonSchema"] == jr.load_schema()
    assert out["model_version_reported"] == "fake-version"
    assert out["usage"]["thinking_tokens"] == 30
    assert out["cost_usd"] == pytest.approx(jc.judge_cost(100, 50))
    assert jc.response_text(out["response"]) == '{"judgments": []}'


def judgment(item_id, label="supported"):
    return {"item_id": item_id, "evidence_quote": "q", "reason": "r", "label": label}


def test_check_response_accepts_exactly_one_valid_judgment_per_item():
    assert jr.check_response({"judgments": [judgment("A.c1"), judgment("B.c1")]},
                             ["A.c1", "B.c1"]) == []  # fmt: skip


@pytest.mark.parametrize(
    "obj,expected",
    [
        ({"judgments": [judgment("A.c1")]}, "B.c1: 0 judgments"),
        ({"judgments": [judgment("A.c1"), judgment("A.c1"), judgment("B.c1")]}, "A.c1: 2"),
        ({"judgments": [judgment("A.c1", "maybe"), judgment("B.c1")]}, "unknown label"),
        ({"judgments": [judgment("A.c1"), judgment("B.c1"), judgment("Z.c9")]}, "unknown item_id"),
        ({"judgments": [{"item_id": "A.c1"}, judgment("B.c1")]}, "malformed"),
        ({"results": []}, "exactly one key"),
    ],
)
def test_check_response_reports_each_problem(obj, expected):
    problems = jr.check_response(obj, ["A.c1", "B.c1"])
    assert any(expected in p for p in problems), problems
