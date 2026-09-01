from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import rtoml


# --------------- PATHS --------------------------------

CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "packed_spine.toml"
)


# --------------- SCHEMA --------------------------------

@dataclass(frozen=True)
class PackedSpineConfig:
    target_fragment_bytes: int
    max_fragment_bytes: int
    max_estimated_tokens: int
    max_spine_nodes: int
    max_heading_count: int
    max_heading_depth_variance: int
    never_cross_h1: bool

    enforce_safe_boundaries: bool = True
    protect_code_fences: bool = True
    protect_html_blocks: bool = True
    protect_mdx_components: bool = True
    protect_tables: bool = True


# --------------- VALIDATION --------------------------------

def _require_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)

    if not isinstance(value, Mapping):
        raise ValueError(f"Expected TOML table: {key!r}")

    return value


def _require_int(
    data: Mapping[str, Any],
    key: str,
    *,
    min_value: int = 1,
) -> int:
    value = data.get(key)

    if not isinstance(value, int):
        raise ValueError(f"Expected integer config value: {key!r}")

    if value < min_value:
        raise ValueError(f"Expected {key!r} >= {min_value}, got {value}")

    return value


def _require_bool(data: Mapping[str, Any], key: str) -> bool:
    value = data.get(key)

    if not isinstance(value, bool):
        raise ValueError(f"Expected boolean config value: {key!r}")

    return value


def _optional_bool(
    data: Mapping[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    value = data.get(key, default)

    if not isinstance(value, bool):
        raise ValueError(f"Expected boolean config value: {key!r}")

    return value


def _validate_config(config: PackedSpineConfig) -> PackedSpineConfig:
    if config.target_fragment_bytes > config.max_fragment_bytes:
        raise ValueError(
            "packed_spine.target_fragment_bytes must be <= max_fragment_bytes"
        )

    return config


# --------------- PUBLIC LOADER --------------------------------

def load_packed_spine_config(
    config_path: Path | None = None,
) -> PackedSpineConfig:
    path = config_path or CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Packed spine config not found: {path}")

    data = rtoml.loads(path.read_text())
    root = _require_mapping(data, "packed_spine")

    return _validate_config(
        PackedSpineConfig(
            target_fragment_bytes=_require_int(root, "target_fragment_bytes"),
            max_fragment_bytes=_require_int(root, "max_fragment_bytes"),
            max_estimated_tokens=_require_int(root, "max_estimated_tokens"),
            max_spine_nodes=_require_int(root, "max_spine_nodes"),
            max_heading_count=_require_int(root, "max_heading_count"),
            max_heading_depth_variance=_require_int(
                root,
                "max_heading_depth_variance",
                min_value=0,
            ),
            never_cross_h1=_require_bool(root, "never_cross_h1"),
            enforce_safe_boundaries=_optional_bool(
                root,
                "enforce_safe_boundaries",
                default=True,
            ),
            protect_code_fences=_optional_bool(
                root,
                "protect_code_fences",
                default=True,
            ),
            protect_html_blocks=_optional_bool(
                root,
                "protect_html_blocks",
                default=True,
            ),
            protect_mdx_components=_optional_bool(
                root,
                "protect_mdx_components",
                default=True,
            ),
            protect_tables=_optional_bool(
                root,
                "protect_tables",
                default=True,
            ),
        )
    )
