"""Keep initialization guidance consistent with the tool schemas clients receive."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from jsonschema import ValidationError, validate

import config
import server
from providers.registry import ModelProviderRegistry
from tools import ChallengeTool, ChatTool, ConsensusTool, ListModelsTool, LookupTool, VersionTool
from utils.model_guidance import get_model_selection_guidance


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["fixed", "auto", "auto_empty", "auto_error"])
@pytest.mark.parametrize(
    "routing", [None, "   ", "Use terra for routine reviews; sol for consequential scoped reviews."]
)
async def test_initialization_routes_model_selection_by_tool_schema(monkeypatch, caplog, mode, routing):
    """Inspect the real main() initialization options, not a detached text helper."""
    monkeypatch.setattr(config, "IS_AUTO_MODE", mode != "fixed")
    monkeypatch.setattr(config, "DEFAULT_MODEL", "sol")
    monkeypatch.setattr(server, "DEFAULT_MODEL", "sol")
    monkeypatch.setattr(server, "configure_providers", lambda: None)
    if routing is None:
        monkeypatch.delenv("PAL_MODEL_ROUTING_GUIDANCE", raising=False)
    else:
        monkeypatch.setenv("PAL_MODEL_ROUTING_GUIDANCE", routing)

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
    # Both serialized discovery surfaces carry the same selection policy.
    monkeypatch.setattr(ChatTool, "is_effective_auto_mode", lambda self: mode != "fixed")
    monkeypatch.setattr(ChatTool, "_get_ranked_model_summaries", lambda self: ([], 0, False))
    monkeypatch.setattr(server, "TOOLS", {"chat": ChatTool()})
    advertised = (await server.handle_list_tools())[0].model_dump(mode="json")
    description = advertised["inputSchema"]["properties"]["model"]["description"]
    continuation = advertised["inputSchema"]["properties"]["continuation_id"]["description"]
    assert "only when" in continuation
    assert "still count as input tokens" in continuation
    assert "does not guarantee a cache hit" in continuation
    assert "ALWAYS reuse" not in continuation
    attachments = advertised["inputSchema"]["properties"]["absolute_file_paths"]["description"]
    assert "smallest relevant" in attachments
    assert "still consume input tokens" in attachments
    shared_guidance = get_model_selection_guidance("sol", mode != "fixed")
    assert shared_guidance in instructions
    assert shared_guidance in description
    for text in (instructions, description):
        assert "Override only" not in text
        assert "surface the server error instead of substituting another model" in text
        if routing and routing.strip():
            assert routing in text
            assert text.index("exact model") < text.index(routing)
            assert "When the user has not named a model" in text
        else:
            assert "deployment routing guidance" not in text
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
