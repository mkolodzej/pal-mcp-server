"""Shared discovery guidance for tools exposing a single model selector."""

from utils.env import get_env


def get_model_selection_guidance(default_model: str, auto_mode: bool) -> str:
    """Describe caller selection without changing runtime defaults or inference settings.

    PAL_MODEL_ROUTING_GUIDANCE is optional deployment guidance for requests where
    the user has not named a model. Explicit user selections remain authoritative.
    """
    guidance = (
        "When the user names a specific model, send that exact model in the tool call. "
        "If the requested model fails validation, surface the server error instead of substituting another model. "
    )
    routing = (get_env("PAL_MODEL_ROUTING_GUIDANCE", "") or "").strip()
    if routing:
        guidance += f"When the user has not named a model, follow this deployment routing guidance: {routing} "
        if not auto_mode:
            guidance += f"If that guidance does not select a model for the task, default to '{default_model}'. "
    elif not auto_mode:
        guidance += f"When no model is mentioned, default to '{default_model}'. "
    if auto_mode:
        guidance += "Currently in auto model selection mode; select an appropriate available model. "
    guidance += "Use `listmodels` if the advertised roster is unavailable or a model name is rejected."
    return guidance.strip()
