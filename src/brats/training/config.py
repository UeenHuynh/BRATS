"""Training configuration resolution."""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml


def resolve_path(value: str | None, root: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(
                cast(Mapping[str, Any], result[key]),
                cast(Mapping[str, Any], value),
            )
        else:
            result[key] = copy.deepcopy(cast(Any, value))
    return result


def load_training_spec(path: Path) -> dict[str, Any]:
    loaded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Training spec must be a YAML mapping")
    return cast(dict[str, Any], loaded)


def effective_training_profile(spec: Mapping[str, Any], profile_id: str, chain: tuple[str, ...] = ()) -> dict[str, Any]:
    profiles = cast(Mapping[str, Any], spec["profiles"])
    if profile_id not in profiles:
        raise KeyError(f"Unknown profile: {profile_id}")
    if profile_id in chain:
        raise ValueError(f"Profile inheritance cycle: {' -> '.join(chain + (profile_id,))}")
    profile = copy.deepcopy(cast(dict[str, Any], profiles[profile_id]))
    # Merge top-level sections from spec into profile
    merged: dict[str, Any] = {}
    for key in ("project", "data", "model", "loss", "optimizer", "scheduler", "training"):
        if key in spec:
            merged[key] = copy.deepcopy(spec[key])
    return deep_merge(merged, profile)
