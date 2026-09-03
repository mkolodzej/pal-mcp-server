"""default_reasoning_effort must reach the chat-completions request for reasoning models."""

from types import SimpleNamespace
from unittest.mock import Mock

from providers.azure_openai import AzureOpenAIProvider


def _run(model: str) -> dict:
    client = Mock()
    resp = Mock()
    resp.choices = [Mock()]
    resp.choices[0].message.content = "ok"
    resp.choices[0].finish_reason = "stop"
    resp.model = model
    resp.id = "x"
    resp.created = 1
    resp.usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    client.chat.completions.create.return_value = resp
    p = AzureOpenAIProvider(api_key="k", azure_endpoint="https://x.openai.azure.com")
    p._client = client  # the provider builds a real AzureOpenAI lazily; inject instead
    p.generate_content(prompt="hi", model_name=model, temperature=0.2, max_output_tokens=10)
    return client.chat.completions.create.call_args.kwargs


def test_sol_sends_high_effort_and_no_sampling_params():
    kw = _run("sol")
    assert kw["reasoning_effort"] == "high"
    assert "temperature" not in kw and "max_tokens" not in kw


def test_oss_keeps_sampling_and_sends_no_effort():
    kw = _run("oss")
    assert "reasoning_effort" not in kw
    assert kw["temperature"] == 0.2 and kw["max_tokens"] == 10
