"""structured_call failure paths (ARCHITECTURE §13): every failure is an LLMError, cost kept."""

from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from backend.app.llm.client import LLMError, strict_schema, structured_call
from backend.app.models.answer import DraftAnswer

VALID = '{"verdict": "yes", "summary": "S.", "claims": [], "uncertainty": "U."}'


class FakeClient:
    def __init__(self, text=VALID, stop_reason="end_turn", error=None, category=None):
        self.text, self.stop_reason, self.error, self.category = text, stop_reason, error, category
        self.messages = SimpleNamespace(create=self.create)
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        content = [SimpleNamespace(type="thinking", thinking="")]
        if self.text is not None:
            content.append(SimpleNamespace(type="text", text=self.text))
        return SimpleNamespace(
            content=content, stop_reason=self.stop_reason,
            stop_details=SimpleNamespace(category=self.category) if self.category else None,
            usage=SimpleNamespace(input_tokens=3000, output_tokens=500),
        )  # fmt: skip


def call(client):
    return structured_call(client, system="s", user="u", schema=DraftAnswer, effort="medium")


def test_valid_output_is_parsed_with_usage_and_no_tools():
    client = FakeClient()
    draft, usage, ms = call(client)
    assert draft.verdict == "yes" and usage.cost_usd == pytest.approx(0.011)
    assert "tools" not in client.kwargs
    assert client.kwargs["output_config"]["format"]["type"] == "json_schema"


@pytest.mark.parametrize(
    "client,fragment",
    [
        (FakeClient(text="I can't help with that.", stop_reason="refusal", category="bio"),
         "declined"),
        (FakeClient(text='{"verdict": "yes", "summ', stop_reason="max_tokens"), "truncated"),
        (FakeClient(text="not json"), "schema validation"),
        (FakeClient(text='{"verdict": "probably", "summary": "", "claims": [], "uncertainty": ""}'),
         "schema validation"),
        (FakeClient(text=None), "no text"),
    ],
)  # fmt: skip
def test_unusable_output_raises_llm_error_with_cost(client, fragment):
    with pytest.raises(LLMError) as info:
        call(client)
    assert fragment in str(info.value)
    assert info.value.usage is not None and info.value.usage.cost_usd > 0


def test_api_error_raises_llm_error_without_usage():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    with pytest.raises(LLMError) as info:
        call(FakeClient(error=anthropic.APIConnectionError(request=request)))
    assert info.value.usage is None and "APIConnectionError" in str(info.value)


def test_strict_schema_closes_every_object_and_inlines_refs():
    schema = strict_schema(DraftAnswer)
    claim = schema["properties"]["claims"]["items"]
    assert "$defs" not in str(schema) and "$ref" not in str(schema)
    assert schema["additionalProperties"] is False and claim["additionalProperties"] is False
    assert set(schema["required"]) == {"verdict", "summary", "claims", "uncertainty"}
    assert set(claim["required"]) == {"id", "text", "citation_ids"}
