"""Shared config helpers for the idiom-index pipeline.

Every script that accepts --model uses these two functions to resolve the
model-specific config block and substitute the {model} placeholder in paths.

Usage
-----
    from config_utils import resolve_paths, get_model_cfg

    model_key = args.model                          # e.g. "bert" or "macberth"
    mcfg      = get_model_cfg(cfg, model_key)       # {"name": ..., "n_layers": 12, ...}
    paths     = resolve_paths(cfg, model_key)       # all {model} placeholders filled
"""

from __future__ import annotations

_VALID_MODELS = {"bert", "macberth"}


def get_model_cfg(cfg: dict, model_key: str) -> dict:
    """Return the model-specific config block for *model_key*.

    Raises KeyError with a helpful message if the key is missing.
    """
    models = cfg.get("models", {})
    if model_key not in models:
        known = ", ".join(sorted(models.keys())) or "(none defined)"
        raise KeyError(
            f"Unknown model key {model_key!r}.  Known models: {known}.  "
            f"Add a '{model_key}' entry under 'models:' in config/config.yaml."
        )
    return models[model_key]


def resolve_paths(cfg: dict, model_key: str) -> dict:
    """Return the paths dict with every {{model}} placeholder substituted.

    Only string values are processed; other types (int, None, list) are
    passed through unchanged.
    """
    raw = cfg.get("paths", {})
    return {
        k: v.format(model=model_key) if isinstance(v, str) else v
        for k, v in raw.items()
    }
