"""Keep initialization guidance consistent with the tool schemas clients receive."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from jsonschema import ValidationError, validate

import config
import server
from providers.registry import ModelProviderRegistry
from tools import ChallengeTool, ConsensusTool, ListModelsTool, LookupTool, VersionTool


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["fixed", "auto", "auto_empty", "auto_error"])
async def test_initialization_routes_model_selection_by_tool_schema(monkeypatch, caplog, mode):
    """Inspect the real main() initialization options, not a detached text helper."""
    monkeypatch.setattr(config, "IS_AUTO_MODE", mode != "fixed")
    monkeypatch.setattr(server, "DEFAULT_MODEL", "sol")
    monkeypatch.setattr(server, "configure_providers", lambda: None)

    def available_models():
        if mode == "auto_error":
            raise RuntimeError("sensitive-provider-diagnostic")
        return [] if mode == "auto_empty" else ["test-model"]

    monkeypatch.setattr(ModelProviderRegistry, "get_available_models", available_models)
    provider = SimpleNamespace(
        get_capabilities=lambda name: SimpleNamespace(model_name="test-model", description="Test roster model")
    )
    monkeypatch.setattr(ModelProviderRegistry, "get_provider_for_model", lambda name: provider)

    @asynccontextmanager
    async def fake_stdio():
        yield object(), object()

    run = AsyncMock()
    monkeypatch.setattr(server, "stdio_server", fake_stdio)
    monkeypatch.setattr(server.server, "run", run)
    await server.main()

    run.assert_awaited_once()
    options = run.await_args.args[2]
    instructions = options.instructions
    assert "inputSchema" in instructions
    assert "Only tools whose inputSchema exposes `model`" in instructions
    assert "consensus" in instructions and "`models` array" in instructions
    assert "no model selector" in instructions
    assert "do not add `model` or `models`" in instructions
    assert "exact model" in instructions
    if mode == "fixed":
        assert "default to 'sol'" in instructions
    elif mode == "auto":
        assert "test-model (Test roster model)" in instructions
        assert "only if a model name is rejected" in instructions
        assert "No model roster is available" not in instructions
        assert "call `listmodels` to discover" not in instructions
    else:
        assert "No model roster is available" in instructions
        assert "call `listmodels` to discover available models" in instructions
        assert "only if a model name is rejected" not in instructions
        assert "choose from the models listed here" not in instructions
    if mode == "auto_error":
        assert "Could not build initialization model roster (RuntimeError)" in caplog.text
        assert "sensitive-provider-diagnostic" not in caplog.text
        assert "sensitive-provider-diagnostic" not in instructions


@pytest.mark.asyncio
async def test_advertised_no_model_inputs_validate_without_selector(monkeypatch):
    """Validate representative inputs against the actual discovery response."""
    tools = [ChallengeTool(), LookupTool(), VersionTool(), ListModelsTool(), ConsensusTool()]
    monkeypatch.setattr(server, "TOOLS", {tool.name: tool for tool in tools})
    # Model summaries are irrelevant to the input shape and must not consult providers.
    monkeypatch.setattr(ConsensusTool, "_get_ranked_model_summaries", lambda self: ([], 0, False))
    inputs = {
        "challenge": {"prompt": "Review this assumption."},
        "apilookup": {"prompt": "Find the official API documentation."},
        "version": {},
        "listmodels": {},
        "consensus": {
            "step": "Assess the proposed change.",
            "step_number": 1,
            "total_steps": 3,
            "next_step_required": True,
            "findings": "The interfaces must agree.",
            "models": [{"model": "test-one", "stance": "for"}, {"model": "test-two", "stance": "against"}],
        },
    }
    advertised = await server.handle_list_tools()
    assert {tool.name for tool in advertised} == set(inputs)
    for tool in advertised:
        assert "model" not in tool.inputSchema["properties"]
        validate(instance=inputs[tool.name], schema=tool.inputSchema)
        if tool.name != "consensus":
            assert "models" not in tool.inputSchema["properties"]
        if tool.name in {"consensus", "version", "listmodels"}:
            # These schemas reject the exact extraneous selector the old
            # server-wide initialization guidance encouraged callers to add.
            with pytest.raises(ValidationError, match="Additional properties"):
                validate(instance={**inputs[tool.name], "model": "test-one"}, schema=tool.inputSchema)
