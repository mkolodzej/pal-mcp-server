"""Request-shape tests for the fork's chat-completions changes, hermetic (no host catalog)."""

from types import SimpleNamespace
from unittest.mock import Mock

from providers.azure_openai import AzureOpenAIProvider
from providers.custom import CustomProvider
from providers.shared import ModelCapabilities, ProviderType, TemperatureConstraint

BIG = "X" * 20000


def _caps(provider, name, **kw):
    base = {
        "provider": provider,
        "model_name": name,
        "friendly_name": "t",
        "context_window": 100000,
        "max_output_tokens": 1000,
        "supports_system_prompts": True,
    }
    base.update(kw)
    caps = ModelCapabilities(**base)
    if not caps.supports_temperature:
        caps.temperature_constraint = TemperatureConstraint.create("fixed")
    return caps


def _drive(provider, caps, *, prompt="hi", system_prompt=None, max_output_tokens=10):
    client = Mock()
    resp = Mock()
    resp.choices = [Mock()]
    resp.choices[0].message.content = "ok"
    resp.choices[0].finish_reason = "stop"
    resp.model = caps.model_name
    resp.id = "x"
    resp.created = 1
    resp.usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    client.chat.completions.create.return_value = resp
    provider._client = client
    provider.get_capabilities = lambda name: caps
    provider._resolve_model_name = lambda name: caps.model_name
    provider.generate_content(
        prompt=prompt,
        model_name=caps.model_name,
        system_prompt=system_prompt,
        temperature=0.2,
        max_output_tokens=max_output_tokens,
    )
    return client.chat.completions.create.call_args.kwargs


def _azure():
    p = AzureOpenAIProvider(api_key="k", azure_endpoint="https://x.openai.azure.com")
    p._deployment_map = {"sol": "sol", "terra": "terra", "luna": "luna", "sampling": "sampling"}
    return p


def test_reasoning_model_sends_effort_and_no_sampling_params():
    kw = _drive(_azure(), _caps(ProviderType.AZURE, "sol", supports_temperature=False, default_reasoning_effort="high"))
    assert kw["reasoning_effort"] == "high"
    assert "temperature" not in kw and "max_tokens" not in kw


def test_reasoning_model_without_declared_effort_sends_none():
    kw = _drive(_azure(), _caps(ProviderType.AZURE, "luna", supports_temperature=False))
    assert "reasoning_effort" not in kw and "temperature" not in kw


def test_sampling_model_keeps_sampling_and_sends_no_effort():
    kw = _drive(_azure(), _caps(ProviderType.AZURE, "sampling", supports_temperature=True))
    assert "reasoning_effort" not in kw
    assert kw["temperature"] == 0.2 and kw["max_tokens"] == 10


def test_custom_provider_never_sends_reasoning_effort():
    p = CustomProvider(api_key="k", base_url="http://localhost:11434/v1")
    kw = _drive(p, _caps(ProviderType.CUSTOM, "local", supports_temperature=False, default_reasoning_effort="high"))
    assert "reasoning_effort" not in kw


def test_context_block_is_hoisted_into_system_message_with_frame():
    kw = _drive(
        _azure(),
        _caps(ProviderType.AZURE, "sol", supports_temperature=False),
        prompt=(
            "=== CONTEXT FILES ===\n--- BEGIN FILE: C:/a.py (Last modified: 2026-09-03 00:00:00 UTC) ---\n"
            f"{BIG}\n--- END FILE: C:/a.py ---\n=== END CONTEXT ===\n\nQuestion?"
        ),
        system_prompt="Be terse.",
    )
    msgs = kw["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"].startswith("Be terse.")
    assert "untrusted reference material" in msgs[0]["content"]
    assert BIG in msgs[0]["content"]
    assert BIG not in msgs[-1]["content"] and "Question?" in msgs[-1]["content"]


def test_no_hoist_for_model_without_system_prompt_support():
    prompt = (
        "=== CONTEXT FILES ===\n--- BEGIN FILE: C:/a.py (Last modified: 2026-09-03 00:00:00 UTC) ---\n"
        f"{BIG}\n--- END FILE: C:/a.py ---\n=== END CONTEXT ===\n\nQuestion?"
    )
    kw = _drive(
        _azure(),
        _caps(ProviderType.AZURE, "sol", supports_temperature=False, supports_system_prompts=False),
        prompt=prompt,
    )
    msgs = kw["messages"]
    assert [m["role"] for m in msgs] == ["user"]
    assert msgs[0]["content"] == prompt


def test_custom_provider_does_not_hoist_file_spans():
    p = CustomProvider(api_key="k", base_url="http://localhost:11434/v1")
    prompt = (
        "=== CONTEXT FILES ===\n--- BEGIN FILE: C:/a.py (Last modified: 2026-09-03 00:00:00 UTC) ---\n"
        f"{BIG}\n--- END FILE: C:/a.py ---\n=== END CONTEXT ===\n\nQuestion?"
    )
    kw = _drive(p, _caps(ProviderType.CUSTOM, "local", supports_temperature=True), prompt=prompt, system_prompt="S")
    msgs = kw["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == "S"
    assert BIG in msgs[-1]["content"]
