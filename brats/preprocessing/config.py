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


def load_spec(path: Path) -> dict[str, Any]:
    loaded: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Spec must be a YAML mapping")
    spec = cast(dict[str, Any], loaded)
    validate_spec(spec)
    return spec


def validate_spec(spec: Mapping[str, Any]) -> None:
    missing = [key for key in ("project", "manifest", "pipeline", "profiles") if key not in spec]
    if missing:
        raise ValueError(f"Missing spec sections: {missing}")
    pipeline = spec["pipeline"]
    if not isinstance(spec["profiles"], Mapping) or not spec["profiles"]:
        raise ValueError("profiles must be a non-empty mapping")
    orientation = str(pipeline.get("orientation", "RAS")).upper()
    if len(orientation) != 3 or any(axis not in "RLAPSI" for axis in orientation):
        raise ValueError(f"Invalid orientation: {orientation}")
    clipping = pipeline["clipping"]
    lower = float(clipping["lower_percentile"])
    upper = float(clipping["upper_percentile"])
    if not 0 <= lower < upper <= 100:
        raise ValueError("Clipping percentiles must satisfy 0 <= lower < upper <= 100")
    if float(pipeline["crop"].get("margin_mm", 0)) < 0:
        raise ValueError("Crop margin cannot be negative")
    profiles = cast(Mapping[str, Any], spec["profiles"])
    for profile_id in profiles:
        effective_profile(spec, profile_id)


def effective_profile(spec: Mapping[str, Any], profile_id: str, chain: tuple[str, ...] = ()) -> dict[str, Any]:
    profiles = cast(Mapping[str, Any], spec["profiles"])
    if profile_id not in profiles:
        raise KeyError(f"Unknown profile: {profile_id}")
    if profile_id in chain:
        raise ValueError(f"Profile inheritance cycle: {' -> '.join(chain + (profile_id,))}")
    profile = copy.deepcopy(cast(dict[str, Any], profiles[profile_id]))
    parent = profile.pop("inherits", None)
    if parent:
        profile = deep_merge(effective_profile(spec, str(parent), chain + (profile_id,)), profile)
    return deep_merge(spec["pipeline"], profile)
