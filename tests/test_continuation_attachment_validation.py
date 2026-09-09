"""Boundary rejections must never attach files to a continuation's history."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import server
from providers.registry import ModelProviderRegistry
from utils import conversation_memory as memory


@pytest.fixture
def continuation_boundary(monkeypatch):
    # Serialized storage exercises real thread reads/writes without shared test state.
    stored = {}
    storage = Mock()
    storage.get.side_effect = stored.get
    storage.setex.side_effect = lambda key, ttl, value: stored.__setitem__(key, value)
    monkeypatch.setattr(memory, "get_storage", lambda: storage)

    provider = Mock()
    provider.get_capabilities.return_value = SimpleNamespace(context_window=100_000)
    monkeypatch.setattr(
        ModelProviderRegistry, "get_provider_for_model", lambda model: provider if model == "test-model" else None
    )
    monkeypatch.setattr(ModelProviderRegistry, "get_available_models", lambda **kwargs: {"test-model": provider})
    monkeypatch.setattr(ModelProviderRegistry, "get_preferred_fallback_model", lambda category: "test-model")

    async def execute(arguments):
        # Mirror the real tool's response persistence, checking user-before-assistant order.
        thread_id = arguments["continuation_id"]
        assert memory.get_thread(thread_id).turns[-1].role == "user"
        memory.add_turn(thread_id, "assistant", "Accepted response", model_name="test-model")
        return []

    tool = Mock()
    tool.requires_model.return_value = True
    tool.execute = AsyncMock(side_effect=execute)
    monkeypatch.setitem(server.TOOLS, "chat", tool)
    return stored, storage, tool


@pytest.mark.asyncio
@pytest.mark.parametrize("has_parent", [False, True])
async def test_rejected_attachments_do_not_leak_into_diff_retry(continuation_boundary, tmp_path, has_parent):
    stored, storage, tool = continuation_boundary
    parent_id = memory.create_thread("chat", {}) if has_parent else None
    if parent_id:
        memory.add_turn(parent_id, "assistant", "Parent response", model_name="test-model")
    thread_id = memory.create_thread("chat", {}, parent_thread_id=parent_id)
    memory.add_turn(thread_id, "assistant", "Initial response", model_name="test-model")

    # Each file fits the history budget individually, but together they exceed the
    # existing boundary limit. This reproduces retained files surfacing on retries.
    rejected_files = []
    for index in range(4):
        path = tmp_path / f"rejected-{index}.txt"
        path.write_text(f"REJECTED_ATTACHMENT_SENTINEL_{index}\n" + "context " * 2000, encoding="utf-8")
        rejected_files.append(path.as_posix())
    diff = tmp_path / "accepted.diff"
    diff.write_text("ACCEPTED_DIFF_SENTINEL\n+ fixed attachment persistence\n", encoding="utf-8")

    before = stored.copy()
    storage.setex.reset_mock()
    with pytest.raises(server.ToolExecutionError, match="code_too_large"):
        await server.handle_call_tool(
            "chat",
            {
                "continuation_id": thread_id,
                "prompt": "Rejected review request",
                "absolute_file_paths": rejected_files,
            },
        )
    assert stored == before  # Includes timestamps, file references, and all parent state.
    storage.setex.assert_not_called()  # Rejection must not refresh TTL either.
    tool.execute.assert_not_awaited()

    await server.handle_call_tool(
        "chat",
        {"continuation_id": thread_id, "prompt": "Review only this diff", "absolute_file_paths": [diff.as_posix()]},
    )
    retry = tool.execute.call_args.args[0]
    assert retry["model"] == "test-model"  # Inherited model still owns validation/execution.
    assert "REJECTED_ATTACHMENT_SENTINEL" not in retry["prompt"]
    assert "Rejected review request" not in retry["prompt"]
    assert retry["absolute_file_paths"] == [diff.as_posix()]
    turns = memory.get_thread(thread_id).turns
    assert [turn.role for turn in turns] == ["assistant", "user", "assistant"]
    assert turns[1].content == "Review only this diff"
    assert turns[1].files == [diff.as_posix()]

    await server.handle_call_tool("chat", {"continuation_id": thread_id, "prompt": "Follow up"})
    following = tool.execute.call_args.args[0]["prompt"]
    assert "ACCEPTED_DIFF_SENTINEL" in following
    assert "REJECTED_ATTACHMENT_SENTINEL" not in following
    assert following.index("Initial response") < following.index("Review only this diff")
    assert following.index("Review only this diff") < following.index("Accepted response")
    turns = memory.get_thread(thread_id).turns
    assert sum(turn.files == [diff.as_posix()] for turn in turns) == 1


@pytest.mark.asyncio
async def test_invalid_model_does_not_persist_continuation(continuation_boundary, tmp_path):
    stored, storage, tool = continuation_boundary
    thread_id = memory.create_thread("chat", {})
    memory.add_turn(thread_id, "assistant", "Initial response", model_name="test-model")
    attachment = tmp_path / "rejected.txt"
    attachment.write_text("INVALID_MODEL_ATTACHMENT_SENTINEL", encoding="utf-8")
    before = stored.copy()
    storage.setex.reset_mock()

    with pytest.raises(server.ToolExecutionError, match="not available"):
        await server.handle_call_tool(
            "chat",
            {
                "continuation_id": thread_id,
                "prompt": "Invalid model request",
                "model": "unavailable-model",
                "absolute_file_paths": [attachment.as_posix()],
            },
        )
    assert stored == before
    storage.setex.assert_not_called()
    tool.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_tool_without_model_still_records_accepted_turn(continuation_boundary):
    _, _, tool = continuation_boundary
    tool.requires_model.return_value = False
    thread_id = memory.create_thread("chat", {})
    await server.handle_call_tool("chat", {"continuation_id": thread_id, "prompt": "Accepted without model"})
    turns = memory.get_thread(thread_id).turns
    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert turns[0].content == "Accepted without model"


@pytest.mark.asyncio
async def test_inherited_initial_files_are_not_recorded_as_new_attachments(continuation_boundary, tmp_path):
    _, _, tool = continuation_boundary
    attachment = tmp_path / "initial.txt"
    attachment.write_text("Initial context", encoding="utf-8")
    thread_id = memory.create_thread("chat", {"absolute_file_paths": [attachment.as_posix()]})
    memory.add_turn(thread_id, "assistant", "Initial response", model_name="test-model")

    await server.handle_call_tool("chat", {"continuation_id": thread_id, "prompt": "No new attachments"})
    assert tool.execute.call_args.args[0]["absolute_file_paths"] == [attachment.as_posix()]
    assert memory.get_thread(thread_id).turns[1].files == []
