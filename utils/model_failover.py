"""Bounded Azure-to-Gemini retry for requests whose model was selected by auto mode."""

import logging
from dataclasses import dataclass
from typing import Any, Optional

from providers.registry import ModelProviderRegistry
from providers.shared import ProviderType
from utils.model_context import ModelContext

logger = logging.getLogger(__name__)


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


def generate_with_auto_failover(
    tool: Any, arguments: dict, request: Any, context: ModelContext, **kwargs: Any
) -> ModelCallResult:
    """Try the selected model once; retry qualifying Azure failures once on an allowed Gemini model."""
    model_name = context.model_name
    provider = context.provider
    try:
        response = provider.generate_content(model_name=model_name, **kwargs)
        return ModelCallResult(response, context, model_name)
    except Exception as error:
        if not arguments.get("_auto_selected_model") or provider.get_provider_type() != ProviderType.AZURE:
            raise
        reason = _azure_failure_reason(error)
        if reason is None:
            raise

        gemini = ModelProviderRegistry.get_provider(ProviderType.GOOGLE)
        if gemini is None:
            raise
        allowed = ModelProviderRegistry._get_allowed_models_for_provider(gemini, ProviderType.GOOGLE)
        if not allowed:
            raise
        replacement = gemini.get_preferred_model(tool.get_model_category(), allowed)
        if replacement not in allowed:
            raise
        fallback_context = ModelContext(replacement)
        fallback_context._provider = gemini  # Preserve the restriction-checked provider instance.
        capabilities = fallback_context.capabilities
        if kwargs.get("images") and not capabilities.supports_images:
            raise
        temperature, warnings = tool.get_validated_temperature(request, fallback_context)
        for warning in warnings:
            logger.warning(warning)
        fallback_kwargs = dict(kwargs)
        fallback_kwargs["temperature"] = temperature
        if not capabilities.supports_extended_thinking:
            fallback_kwargs["thinking_mode"] = None
        logger.warning(
            "Auto-selected Azure model %s failed (%s); retrying once with Gemini %s", model_name, reason, replacement
        )
        response = gemini.generate_content(model_name=replacement, **fallback_kwargs)
        arguments["_model_context"] = fallback_context
        arguments["_resolved_model_name"] = replacement
        arguments["_model_fallback_reason"] = reason
        tool._model_context = fallback_context
        tool._current_model_name = replacement
        return ModelCallResult(response, fallback_context, replacement, reason)
