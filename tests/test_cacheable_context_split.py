#!/usr/bin/env python3
"""Edge-case tests for _split_cacheable_context, run in the installed PAL env."""
import sys

sys.path.insert(0, "C:/source/pal-mcp-server")

from providers.openai_compatible import OpenAICompatibleProvider as P  # noqa: E402

BIG = "X" * 20000
fails = 0


def check(label, cond):
    global fails
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        fails += 1


ctx, rest = P._split_cacheable_context(
    f"=== CONTEXT FILES ===\n{BIG}\n=== END CONTEXT ===\n\nMy question?")
check("leading big block splits", ctx.startswith("=== CONTEXT FILES ===") and len(ctx) > 19000)
check("remainder is just the question", rest == "My question?")

ctx2, rest2 = P._split_cacheable_context(
    f"=== ESSENTIAL FILES ===\n{BIG}\n=== END ESSENTIAL FILES ===\n\nFindings here")
check("workflow marker also splits", ctx2.startswith("=== ESSENTIAL FILES ===") and rest2 == "Findings here")

ctx3, rest3 = P._split_cacheable_context(f"Question first\n\n=== CONTEXT FILES ===\n{BIG}\n=== END CONTEXT ===")
check("block not leading -> no split", ctx3 == "")

ctx4, _ = P._split_cacheable_context("=== CONTEXT FILES ===\nsmall\n=== END CONTEXT ===\n\nQ")
check("below size threshold -> no split", ctx4 == "")

ctx5, rest5 = P._split_cacheable_context("plain prompt, no markers")
check("no markers -> unchanged", ctx5 == "" and rest5 == "plain prompt, no markers")

ctx6, rest6 = P._split_cacheable_context("")
check("empty prompt safe", ctx6 == "" and rest6 == "")

ctx7, rest7 = P._split_cacheable_context(f"=== CONTEXT FILES ===\n{BIG} (unterminated)")
check("missing end marker -> no split", ctx7 == "")

print(f"\n{'ALL PASS' if not fails else f'{fails} FAILURE(S)'}")
sys.exit(1 if fails else 0)
