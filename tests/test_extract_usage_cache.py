"""_extract_usage must surface prompt-cache read/write counts, not just totals."""

from types import SimpleNamespace

from providers.openai_compatible import OpenAICompatibleProvider


class _Provider(OpenAICompatibleProvider):
    """Minimal concrete subclass: _extract_usage never touches provider state."""

    def __init__(self):  # noqa: D401 - skip the network client setup entirely
        pass

    def get_provider_type(self):  # abstract in the base
        return None


def _extract(resp):
    return _Provider()._extract_usage(resp)


def test_cache_fields_extracted_from_sdk_object():
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


def test_cache_fields_extracted_from_dict_details():
    resp = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=1,
            total_tokens=11,
            prompt_tokens_details={"cached_tokens": 7, "cache_write_tokens": None},
        )
    )
    usage = _extract(resp)
    assert usage["cached_tokens"] == 7
    assert usage["cache_write_tokens"] == 0


def test_cache_fields_default_to_zero_without_details():
    resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=1, total_tokens=11))
    usage = _extract(resp)
    assert usage["cached_tokens"] == 0
    assert usage["cache_write_tokens"] == 0


def test_no_usage_returns_empty():
    assert _extract(SimpleNamespace()) == {}


def test_mock_placeholder_details_do_not_break_logging():
    """Upstream tests hand _extract_usage a Mock() usage; unset fields must read as 0, not Mock."""
    from unittest.mock import Mock

    resp = Mock()
    resp.usage = Mock()
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 50
    resp.usage.total_tokens = 150
    usage = _extract(resp)
    assert usage["cached_tokens"] == 0
    assert usage["cache_write_tokens"] == 0
