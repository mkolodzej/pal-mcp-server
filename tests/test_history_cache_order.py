"""Continued threads must be cache-prefix stable: turn N's history block is a byte prefix of
turn N+1's, and the tool detects prepended history (so it is not doubled). The provider-side
hoist is covered in test_cacheable_context_split.py."""

import tempfile
from pathlib import Path

from utils.conversation_memory import ConversationTurn, ThreadContext, build_conversation_history


class _MC:
    """Minimal model context: the builder only needs a token estimator and a budget."""

    model_name = "sol"

    def estimate_tokens(self, text):
        return max(1, len(text) // 4)

    def calculate_token_allocation(self):
        from types import SimpleNamespace

        return SimpleNamespace(file_tokens=200_000, history_tokens=200_000)


MC = _MC()

BIG = "X" * 20000


def _ctx(turns, files=None):
    return ThreadContext(
        thread_id="t1",
        created_at="2026-09-03T00:00:00Z",
        last_updated_at="2026-09-03T00:00:00Z",
        tool_name="chat",
        turns=[
            ConversationTurn(role="user", content=c, timestamp="2026-09-03T00:00:00Z", files=files or []) for c in turns
        ],
        initial_context={},
    )


def test_history_block_of_turn_n_is_prefix_of_turn_n_plus_1_without_files():
    long = "remember this " * 600
    h1, _ = build_conversation_history(_ctx([long]), MC)
    h2, _ = build_conversation_history(_ctx([long, "follow-up"]), MC)
    end = "=== END CONVERSATION HISTORY ==="
    assert h2.startswith(h1[: h1.index(end)])
    assert "Turn 1/" not in h1[: h1.index(end)]  # the per-turn counter lives in the trailer


def test_history_block_prefix_holds_when_a_file_is_added_later():
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d, "a.py"), Path(d, "b.py")
        a.write_text(BIG, encoding="utf-8")
        b.write_text(BIG[::-1], encoding="utf-8")
        c1 = ThreadContext(
            thread_id="t1",
            created_at="x",
            last_updated_at="x",
            tool_name="chat",
            initial_context={},
            turns=[ConversationTurn(role="user", content="q1", timestamp="x", files=[str(a)])],
        )
        c2 = ThreadContext(
            thread_id="t1",
            created_at="x",
            last_updated_at="x",
            tool_name="chat",
            initial_context={},
            turns=[
                ConversationTurn(role="user", content="q1", timestamp="x", files=[str(a)]),
                ConversationTurn(role="user", content="q2", timestamp="x", files=[str(b)]),
            ],
        )
        h1, _ = build_conversation_history(c1, MC)
        h2, _ = build_conversation_history(c2, MC)
        files_end = "=== END REFERENCED FILES ==="
        # everything up to and including file a is unchanged; file b was appended after it
        assert h2.startswith(h1[: h1.index(files_end) - 2])
        assert h2.index(str(a)) < h2.index(str(b))


def test_tool_detects_prepended_history_marker():
    """tools/simple/base.py must recognise the builder's output, else it re-adds the turn."""
    h, _ = build_conversation_history(_ctx(["q1"]), MC)
    src = Path("tools/simple/base.py").read_text(encoding="utf-8")
    assert '"=== END CONVERSATION HISTORY ===" in field_value' in src
    assert "=== END CONVERSATION HISTORY ===" in h
