"""_extract_usage must surface prompt-cache read/write counts, not just totals."""

from types import SimpleNamespace

from providers.openai_compatible import OpenAICompatibleProvider


def _extract(resp):
    # _extract_usage never touches self; call it unbound so the abstract class need not be built.
    return OpenAICompatibleProvider._extract_usage(None, resp)


def test_cache_fields_extracted_from_details():
    resp = SimpleNamespace(
        model="sol",
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=50,
            total_tokens=1050,
            prompt_tokens_details=SimpleNamespace(cached_tokens=900, cache_write_tokens=100),
        ),
    )
    usage = _extract(resp)
    assert usage["cached_tokens"] == 900
    assert usage["cache_write_tokens"] == 100
    assert usage["input_tokens"] == 1000


def test_cache_fields_default_to_zero_without_details():
    resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=1, total_tokens=11))
    usage = _extract(resp)
    assert usage["cached_tokens"] == 0
    assert usage["cache_write_tokens"] == 0


def test_no_usage_returns_empty():
    assert _extract(SimpleNamespace()) == {}
