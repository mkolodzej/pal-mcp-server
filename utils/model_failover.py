"""Bounded Azure-to-Gemini retry for requests whose model was selected by auto mode."""

import functools
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from providers.registry import ModelProviderRegistry
from providers.shared import ProviderType
from utils.model_context import ModelContext

logger = logging.getLogger(__name__)

# Auth-class Azure failures (401/403, AuthenticationError, PermissionDeniedError) are negative-cached
# in-process for this long so auto-mode requests skip the doomed Azure attempt. Transient failures
# (408/429/5xx/transport) are never cached. Override with PAL_AZURE_AUTH_COOLDOWN_SECONDS.
AZURE_AUTH_COOLDOWN_SECONDS = 600.0
AZURE_AUTH_COOLDOWN_ENV = "PAL_AZURE_AUTH_COOLDOWN_SECONDS"
AZURE_AUTH_COOLDOWN_REASON = "azure_auth_cooldown"

_now = time.monotonic  # Module attribute so tests can substitute a fake clock.
_azure_auth_lock = threading.Lock()
_azure_auth_failed_at: Optional[float] = None


@dataclass
class ModelCallResult:
    response: Any
    context: ModelContext
    model_name: str
    fallback_reason: Optional[str] = None


def _azure_failure_reason(error: Exception) -> Optional[str]:
    """Classify provider availability failures without copying sensitive exception text."""
    seen = set()
    current: Optional[BaseException] = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = getattr(current, "status_code", None)
        if status in (401, 403, 404, 408, 429) or isinstance(status, int) and 500 <= status <= 599:
            return f"azure_http_{status}"
        name = type(current).__name__.lower()
        if any(
            part in name
            for part in (
                "authenticationerror",
                "permissiondeniederror",
                "apiconnectionerror",
                "apitimeouterror",
                "ratelimiterror",
                "connecterror",
                "timeout",
            )
        ):
            return f"azure_{name}"
        if isinstance(current, (ConnectionError, TimeoutError)):
            return "azure_transport_error"
        current = current.__cause__ or current.__context__
    return None


def _is_azure_auth_failure_reason(reason: Optional[str]) -> bool:
    """True only for reason codes that mean the Azure credentials or subscription are rejected."""
    if not reason:
        return False
    if reason in ("azure_http_401", "azure_http_403"):
        return True
    return reason.endswith(("authenticationerror", "permissiondeniederror"))


@functools.lru_cache(maxsize=4)
def _parse_cooldown_env(raw: Optional[str]) -> float:
    """Parse the TTL override once per distinct value so a malformed setting warns once, not per request."""
    if raw is None or raw.strip() == "":
        return AZURE_AUTH_COOLDOWN_SECONDS
    try:
        return max(float(raw), 0.0)
    except ValueError:
        logger.warning(
            "Ignoring non-numeric %s; using default %.0fs", AZURE_AUTH_COOLDOWN_ENV, AZURE_AUTH_COOLDOWN_SECONDS
        )
        return AZURE_AUTH_COOLDOWN_SECONDS


def _azure_auth_cooldown_seconds() -> float:
    return _parse_cooldown_env(os.environ.get(AZURE_AUTH_COOLDOWN_ENV))


def azure_auth_cooldown_active() -> bool:
    """Whether a recent auth-class Azure failure should suppress the next auto-mode Azure attempt."""
    global _azure_auth_failed_at
    with _azure_auth_lock:
        if _azure_auth_failed_at is None:
            return False
        if _now() - _azure_auth_failed_at < _azure_auth_cooldown_seconds():
            return True
        _azure_auth_failed_at = None  # Expired: let the next request probe Azure again.
        return False


def record_azure_auth_failure() -> None:
    global _azure_auth_failed_at
    with _azure_auth_lock:
        _azure_auth_failed_at = _now()


def clear_azure_auth_failure() -> None:
    global _azure_auth_failed_at
    with _azure_auth_lock:
        _azure_auth_failed_at = None


def _prepare_gemini_fallback(tool: Any, request: Any, kwargs: dict) -> Optional[tuple]:
    """Resolve the allowed Gemini replacement, or None when no substitution is permitted."""
    gemini = ModelProviderRegistry.get_provider(ProviderType.GOOGLE)
    if gemini is None:
        return None
    allowed = ModelProviderRegistry._get_allowed_models_for_provider(gemini, ProviderType.GOOGLE)
    if not allowed:
        return None
    replacement = gemini.get_preferred_model(tool.get_model_category(), allowed)
    if replacement not in allowed:
        return None
    fallback_context = ModelContext(replacement)
    fallback_context._provider = gemini  # Preserve the restriction-checked provider instance.
    capabilities = fallback_context.capabilities
    if kwargs.get("images") and not capabilities.supports_images:
        return None
    temperature, warnings = tool.get_validated_temperature(request, fallback_context)
    for warning in warnings:
        logger.warning(warning)
    fallback_kwargs = dict(kwargs)
    fallback_kwargs["temperature"] = temperature
    if not capabilities.supports_extended_thinking:
        fallback_kwargs["thinking_mode"] = None
    return gemini, replacement, fallback_context, fallback_kwargs


def _run_gemini_fallback(tool: Any, arguments: dict, fallback: tuple, reason: str) -> ModelCallResult:
    gemini, replacement, fallback_context, fallback_kwargs = fallback
    response = gemini.generate_content(model_name=replacement, **fallback_kwargs)
    arguments["_model_context"] = fallback_context
    arguments["_resolved_model_name"] = replacement
    arguments["_model_fallback_reason"] = reason
    tool._model_context = fallback_context
    tool._current_model_name = replacement
    return ModelCallResult(response, fallback_context, replacement, reason)


def generate_with_auto_failover(
    tool: Any, arguments: dict, request: Any, context: ModelContext, **kwargs: Any
) -> ModelCallResult:
    """Try the selected model once; retry qualifying Azure failures once on an allowed Gemini model."""
    model_name = context.model_name
    provider = context.provider
    auto_azure = bool(arguments.get("_auto_selected_model")) and provider.get_provider_type() == ProviderType.AZURE

    if auto_azure and azure_auth_cooldown_active():
        fallback = _prepare_gemini_fallback(tool, request, kwargs)
        if fallback is not None:
            logger.info(
                "Auto-selected Azure model %s skipped (%s); using Gemini %s",
                model_name,
                AZURE_AUTH_COOLDOWN_REASON,
                fallback[1],
            )
            return _run_gemini_fallback(tool, arguments, fallback, AZURE_AUTH_COOLDOWN_REASON)
        # No permitted substitute: fall through and let Azure report its own outcome.

    try:
        response = provider.generate_content(model_name=model_name, **kwargs)
    except Exception as error:
        if not auto_azure:
            raise
        reason = _azure_failure_reason(error)
        if reason is None:
            raise
        if _is_azure_auth_failure_reason(reason):
            record_azure_auth_failure()
        fallback = _prepare_gemini_fallback(tool, request, kwargs)
        if fallback is None:
            raise
        logger.warning(
            "Auto-selected Azure model %s failed (%s); retrying once with Gemini %s", model_name, reason, fallback[1]
        )
        return _run_gemini_fallback(tool, arguments, fallback, reason)

    if provider.get_provider_type() == ProviderType.AZURE:
        clear_azure_auth_failure()
    return ModelCallResult(response, context, model_name)
