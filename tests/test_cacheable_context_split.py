"""Tests for OpenAICompatibleProvider._split_cacheable_context (hoist v2).

The provider lifts embedded FILE spans (and, on continued turns, the conversation-history
block) into the system message so provider prompt caches see a byte-identical prefix across
turns of one thread and across independent calls that attach the same files. These cases pin:
identical output for the same file under different tool wrappers, path canonicalisation, the
size floor, and that prompts with nothing to lift pass through untouched.
"""

import pytest

from providers.openai_compatible import OpenAICompatibleProvider as Provider

BODY = "X" * 20000
MT = "2026-09-03 17:27:01 UTC"


def span(path, body=BODY):
    return f"\n--- BEGIN FILE: {path} (Last modified: {MT}) ---\n{body}\n--- END FILE: {path} ---\n"


def test_first_turn_and_continued_turn_yield_identical_system_context():
    win = "C:\\src\\a.md"
    posix = "C:/src/a.md"
    turn1 = f"=== USER REQUEST ===\n=== CONTEXT FILES ==={span(win)}=== END CONTEXT ===\n\nQuestion one?"
    turn2 = (
        "=== CONVERSATION HISTORY (CONTINUATION) ===\nThread: t\nTool: chat\n\n"
        f"=== FILES REFERENCED IN THIS CONVERSATION ===\nRefer to these:\n{span(posix)}\n=== END REFERENCED FILES ===\n\n"
        "Previous conversation turns:\n\n--- Turn 1 (Agent) ---\nQuestion one?\n\n--- Turn 2 (sol) ---\n42\n\n"
        "=== END CONVERSATION HISTORY ===\n\nIMPORTANT: continue.\n\n=== NEW USER INPUT ===\nQuestion two?"
    )
    c1, r1 = Provider._split_cacheable_context(turn1)
    c2, r2 = Provider._split_cacheable_context(turn2)
    assert c1.startswith(f"--- BEGIN FILE: {posix}")  # canonical slashes on both
    assert c2 == c1  # the system message must be byte-identical across the thread
    assert "=== CONVERSATION HISTORY (CONTINUATION) ===" in r2  # history stays in the user message
    assert BODY not in r1 and BODY not in r2
    assert r1.startswith("=== USER REQUEST ===") and "Question one?" in r1
    assert "Question two?" in r2


def test_multiple_files_keep_prompt_order():
    p = f"=== CONTEXT FILES ==={span('C:/a.py', 'A' * 5000)}{span('C:/b.py', 'B' * 5000)}=== END CONTEXT ===\n\nQ"
    ctx, _ = Provider._split_cacheable_context(p)
    assert ctx.index("C:/a.py") < ctx.index("C:/b.py")


def test_below_size_floor_is_left_alone():
    p = f"=== CONTEXT FILES ==={span('C:/a.py', 'small')}=== END CONTEXT ===\n\nQ"
    assert Provider._split_cacheable_context(p) == ("", p)


def test_prompt_without_files_or_history_is_unchanged():
    p = "plain prompt, no markers"
    assert Provider._split_cacheable_context(p) == ("", p)


def test_quoted_end_marker_inside_a_file_does_not_end_the_span():
    body = f"{BODY}\n    print('--- END FILE: C:/a.py ---')\n{BODY}"
    p = f"=== CONTEXT FILES ==={span('C:/a.py', body)}=== END CONTEXT ===\n\nQ"
    ctx, rest = Provider._split_cacheable_context(p)
    # regex anchors the END marker to a line that is followed by newline/end; a quoted marker
    # inside a print() is not at line start so the span runs to the real end marker
    assert ctx.count("--- BEGIN FILE") == 1 and ctx.count(BODY) == 2
    assert "=== END CONTEXT ===" in rest


@pytest.mark.parametrize("prompt", ["", "short", "--- BEGIN FILE: x (Last modified: y) ---"])
def test_never_raises_on_degenerate_input(prompt):
    ctx, rest = Provider._split_cacheable_context(prompt)
    assert isinstance(ctx, str) and isinstance(rest, str)


def test_embedded_begin_marker_inside_a_file_stays_one_span():
    """A file that quotes PAL's own marker format (docs, fixtures) must not be split at it."""
    inner = "\n--- BEGIN FILE: C:/other.py (Last modified: 2026-01-01 00:00:00 UTC) ---\n"
    body = f"{BODY}{inner}{BODY}"
    p = f"=== CONTEXT FILES ==={span('C:/a.py', body)}=== END CONTEXT ===\n\nQ"
    ctx, rest = Provider._split_cacheable_context(p)
    assert ctx.count("--- BEGIN FILE: C:/a.py") == 1 and ctx.count(BODY) == 2
    assert "--- BEGIN FILE: C:/other.py" in ctx  # the quoted marker travelled with the body
    assert "--- END FILE" not in rest and "--- BEGIN FILE" not in rest


def test_indented_or_quoted_end_marker_for_same_path_does_not_end_span():
    body = f"{BODY}\n    --- END FILE: C:/a.py ---\n{BODY}"
    p = f"=== CONTEXT FILES ==={span('C:/a.py', body)}=== END CONTEXT ===\n\nQ"
    ctx, rest = Provider._split_cacheable_context(p)
    assert ctx.count(BODY) == 2 and "--- END FILE" not in rest
