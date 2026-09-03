"""Tests for OpenAICompatibleProvider._split_cacheable_context.

The splitter hoists a large, leading, delimited file-context block out of the
prompt so it can be placed in the system message, where providers will cache it.
These cases pin the conservative behaviour: anything that is not a big leading
terminated block must pass through untouched, so existing prompts are unaffected.
"""

import pytest

from providers.openai_compatible import OpenAICompatibleProvider as Provider

BIG = "X" * 20000


def test_leading_context_block_is_split_off():
    ctx, rest = Provider._split_cacheable_context(f"=== CONTEXT FILES ===\n{BIG}\n=== END CONTEXT ===\n\nMy question?")
    assert ctx.startswith("=== CONTEXT FILES ===")
    assert BIG in ctx
    assert rest == "My question?"


def test_workflow_marker_is_also_recognised():
    ctx, rest = Provider._split_cacheable_context(
        f"=== ESSENTIAL FILES ===\n{BIG}\n=== END ESSENTIAL FILES ===\n\nFindings here"
    )
    assert ctx.startswith("=== ESSENTIAL FILES ===")
    assert rest == "Findings here"


def test_block_must_lead_the_prompt():
    """A block behind variable text cannot form a cacheable prefix."""
    ctx, rest = Provider._split_cacheable_context(
        f"Question first\n\n=== CONTEXT FILES ===\n{BIG}\n=== END CONTEXT ==="
    )
    assert ctx == ""
    assert rest.startswith("Question first")


def test_block_below_size_floor_is_left_alone():
    """Under the provider minimum, a separate message buys nothing."""
    ctx, _ = Provider._split_cacheable_context("=== CONTEXT FILES ===\nsmall\n=== END CONTEXT ===\n\nQ")
    assert ctx == ""


def test_prompt_without_markers_is_unchanged():
    ctx, rest = Provider._split_cacheable_context("plain prompt, no markers")
    assert ctx == ""
    assert rest == "plain prompt, no markers"


def test_empty_prompt_is_safe():
    assert Provider._split_cacheable_context("") == ("", "")


def test_unterminated_block_is_left_alone():
    ctx, _ = Provider._split_cacheable_context(f"=== CONTEXT FILES ===\n{BIG} (unterminated)")
    assert ctx == ""


@pytest.mark.parametrize("prompt", ["", "short", "=== CONTEXT FILES ==="])
def test_never_raises_on_degenerate_input(prompt):
    ctx, rest = Provider._split_cacheable_context(prompt)
    assert isinstance(ctx, str) and isinstance(rest, str)


def test_quoted_end_marker_inside_a_file_does_not_end_the_block():
    """A reviewed file may contain the marker as a string literal; only a full marker line counts."""
    body = f'{BIG}\n    ("=== ESSENTIAL FILES ===", "=== END ESSENTIAL FILES ==="),\n{BIG}'
    ctx, rest = Provider._split_cacheable_context(
        f"=== ESSENTIAL FILES ===\n{body}\n=== END ESSENTIAL FILES ===\n\nFindings"
    )
    assert ctx.endswith("=== END ESSENTIAL FILES ===")
    assert ctx.count(BIG) == 2
    assert rest == "Findings"
