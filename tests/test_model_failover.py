"""Runtime retry is limited to auto-selected Azure availability failures."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

import server
from providers.registry import ModelProviderRegistry
from providers.shared import ProviderType
from tools.models import ToolModelCategory
from utils.model_failover import generate_with_auto_failover


class HTTPFailure(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def providers():
    azure = Mock()
    azure.get_provider_type.return_value = ProviderType.AZURE
    gemini = Mock()
    gemini.get_provider_type.return_value = ProviderType.GOOGLE
    gemini.get_preferred_model.return_value = "gemini-test"
    gemini.get_capabilities.return_value = SimpleNamespace(
        supports_images=True, supports_extended_thinking=False, allow_code_generation=False
    )
    gemini.generate_content.return_value = SimpleNamespace(content="fallback answer", usage=None, metadata={})
    context = SimpleNamespace(model_name="azure-test", provider=azure)
    tool = Mock()
    tool.get_model_category.return_value = ToolModelCategory.BALANCED
    tool.get_validated_temperature.return_value = (0.4, [])
    return azure, gemini, context, tool


def test_auto_azure_401_retries_allowed_gemini_once():
    azure, gemini, context, tool = providers()
    azure.generate_content.side_effect = RuntimeError("provider wrapper")
    azure.generate_content.side_effect.__cause__ = HTTPFailure(401)
    arguments = {"_auto_selected_model": True}
    with (
        patch.object(ModelProviderRegistry, "get_provider", return_value=gemini),
        patch.object(ModelProviderRegistry, "_get_allowed_models_for_provider", return_value=["gemini-test"]),
    ):
        result = generate_with_auto_failover(tool, arguments, Mock(), context, prompt="hello", temperature=0.2)
    assert result.response.content == "fallback answer"
    assert result.model_name == "gemini-test"
    assert result.fallback_reason == "azure_http_401"
    assert arguments["_resolved_model_name"] == "gemini-test"
    assert arguments["_model_fallback_reason"] == "azure_http_401"
    gemini.generate_content.assert_called_once()
    assert gemini.generate_content.call_args.kwargs["temperature"] == 0.4


def test_explicit_azure_401_never_substitutes():
    azure, gemini, context, tool = providers()
    azure.generate_content.side_effect = HTTPFailure(401)
    with patch.object(ModelProviderRegistry, "get_provider") as get_provider:
        with pytest.raises(HTTPFailure):
            generate_with_auto_failover(tool, {}, Mock(), context, prompt="hello")
    get_provider.assert_not_called()
    gemini.generate_content.assert_not_called()


def test_auto_azure_401_without_allowed_gemini_preserves_failure():
    azure, gemini, context, tool = providers()
    azure.generate_content.side_effect = HTTPFailure(401)
    with (
        patch.object(ModelProviderRegistry, "get_provider", return_value=gemini),
        patch.object(ModelProviderRegistry, "_get_allowed_models_for_provider", return_value=[]),
    ):
        with pytest.raises(HTTPFailure):
            generate_with_auto_failover(tool, {"_auto_selected_model": True}, Mock(), context, prompt="hello")
    gemini.generate_content.assert_not_called()


def test_auto_azure_401_without_gemini_provider_preserves_failure():
    azure, _, context, tool = providers()
    azure.generate_content.side_effect = HTTPFailure(401)
    with patch.object(ModelProviderRegistry, "get_provider", return_value=None):
        with pytest.raises(HTTPFailure):
            generate_with_auto_failover(tool, {"_auto_selected_model": True}, Mock(), context, prompt="hello")


def test_non_availability_error_does_not_retry():
    azure, gemini, context, tool = providers()
    azure.generate_content.side_effect = ValueError("invalid model parameters")
    with patch.object(ModelProviderRegistry, "get_provider") as get_provider:
        with pytest.raises(ValueError):
            generate_with_auto_failover(tool, {"_auto_selected_model": True}, Mock(), context, prompt="hello")
    get_provider.assert_not_called()


def test_content_policy_http_400_does_not_retry():
    azure, _, context, tool = providers()
    azure.generate_content.side_effect = HTTPFailure(400)
    with patch.object(ModelProviderRegistry, "get_provider") as get_provider:
        with pytest.raises(HTTPFailure):
            generate_with_auto_failover(tool, {"_auto_selected_model": True}, Mock(), context, prompt="hello")
    get_provider.assert_not_called()


@pytest.mark.asyncio
async def test_server_boundary_marks_auto_but_overrides_spoofed_named_flag():
    tool = Mock()
    tool.requires_model.return_value = True
    tool.get_model_category.return_value = ToolModelCategory.BALANCED
    tool.execute = AsyncMock(return_value=[])
    context = SimpleNamespace(capabilities=SimpleNamespace(context_window=100000))
    with (
        patch.dict(server.TOOLS, {"failover_probe": tool}),
        patch.object(ModelProviderRegistry, "get_preferred_fallback_model", return_value="azure-test"),
        patch.object(ModelProviderRegistry, "get_provider_for_model", return_value=Mock()),
        patch("utils.model_context.ModelContext", return_value=context),
    ):
        await server.handle_call_tool("failover_probe", {"model": "auto"})
        auto_arguments = tool.execute.await_args.args[0]
        assert auto_arguments["_auto_selected_model"] is True
        assert auto_arguments["model"] == "azure-test"

        await server.handle_call_tool("failover_probe", {"model": "azure-test", "_auto_selected_model": True})
        explicit_arguments = tool.execute.await_args.args[0]
        assert explicit_arguments["_auto_selected_model"] is False


@pytest.mark.asyncio
async def test_simple_tool_reports_actual_gemini_model_and_reason(tmp_path):
    from tools.chat import ChatTool

    azure, gemini, context, _ = providers()
    azure.generate_content.side_effect = HTTPFailure(401)
    tool = ChatTool()
    tool.prepare_prompt = AsyncMock(return_value="question")
    tool._augment_system_prompt_with_capabilities = Mock(return_value="system")
    tool.get_validated_temperature = Mock(return_value=(0.3, []))
    context.capabilities = SimpleNamespace(supports_extended_thinking=False)
    arguments = {
        "prompt": "question",
        "model": "azure-test",
        "working_directory_absolute_path": str(tmp_path),
        "_model_context": context,
        "_auto_selected_model": True,
    }
    with (
        patch.object(ModelProviderRegistry, "get_provider", return_value=gemini),
        patch.object(ModelProviderRegistry, "_get_allowed_models_for_provider", return_value=["gemini-test"]),
    ):
        result = await tool.execute(arguments)
    output = json.loads(result[0].text)
    assert output["metadata"]["model_used"] == "gemini-test"
    assert output["metadata"]["provider_used"] == "google"
    assert output["metadata"]["fallback_reason"] == "azure_http_401"


@pytest.mark.asyncio
async def test_simple_empty_response_retry_can_fail_over(tmp_path):
    from tools.chat import ChatTool

    azure, gemini, context, _ = providers()
    azure.generate_content.side_effect = [
        SimpleNamespace(content="", metadata={"finish_reason": "STOP"}),
        HTTPFailure(401),
    ]
    tool = ChatTool()
    tool.prepare_prompt = AsyncMock(return_value="question")
    tool._augment_system_prompt_with_capabilities = Mock(return_value="system")
    tool.get_validated_temperature = Mock(return_value=(0.3, []))
    context.capabilities = SimpleNamespace(supports_extended_thinking=False)
    arguments = {
        "prompt": "question",
        "model": "azure-test",
        "working_directory_absolute_path": str(tmp_path),
        "_model_context": context,
        "_auto_selected_model": True,
    }
    with (
        patch.object(ModelProviderRegistry, "get_provider", return_value=gemini),
        patch.object(ModelProviderRegistry, "_get_allowed_models_for_provider", return_value=["gemini-test"]),
    ):
        result = await tool.execute(arguments)
    output = json.loads(result[0].text)
    assert azure.generate_content.call_count == 2
    gemini.generate_content.assert_called_once()
    assert output["metadata"]["provider_used"] == "google"
    assert output["metadata"]["fallback_reason"] == "azure_http_401"


@pytest.mark.asyncio
async def test_workflow_expert_analysis_uses_gemini_and_updates_metadata():
    from tools.debug import DebugInvestigationRequest, DebugIssueTool

    azure, gemini, context, _ = providers()
    azure.generate_content.side_effect = HTTPFailure(401)
    tool = DebugIssueTool()
    tool._model_context = context
    tool._current_model_name = "azure-test"
    tool._augment_system_prompt_with_capabilities = Mock(return_value="system")
    tool.get_validated_temperature = Mock(return_value=(0.3, []))
    tool.should_include_files_in_expert_prompt = Mock(return_value=False)
    gemini.generate_content.return_value = SimpleNamespace(content='{"finding":"ok"}')
    request = DebugInvestigationRequest(
        step="inspect",
        step_number=1,
        total_steps=1,
        next_step_required=False,
        findings="found",
        model="azure-test",
    )
    arguments = {
        "model": "azure-test",
        "_model_context": context,
        "_resolved_model_name": "azure-test",
        "_auto_selected_model": True,
    }
    with (
        patch.object(ModelProviderRegistry, "get_provider", return_value=gemini),
        patch.object(ModelProviderRegistry, "_get_allowed_models_for_provider", return_value=["gemini-test"]),
    ):
        analysis = await tool._call_expert_analysis(arguments, request)
    assert analysis == {"finding": "ok"}
    response = {}
    tool._add_workflow_metadata(response, arguments)
    assert response["metadata"]["model_used"] == "gemini-test"
    assert response["metadata"]["provider_used"] == "google"
    assert response["metadata"]["fallback_reason"] == "azure_http_401"
