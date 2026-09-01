from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import rtoml


CONFIG_FILE_NAME = "thresholds.toml"
CONFIG_ROOT_KEY = "markdown_spine_thresholds"
CONFIG_PATH = Path(__file__).parent / "config" / CONFIG_FILE_NAME


@dataclass(frozen=True)
class MarkdownRoutingConfig:
    normal_queue: str
    spine_queue: str
    quarantine_queue: str


@dataclass(frozen=True)
class MarkdownSpineThresholds:
    small_bytes: int
    normal_bytes: int
    large_bytes: int
    huge_bytes: int

    small_tokens: int
    normal_tokens: int
    large_tokens: int
    huge_tokens: int

    large_section_tokens: int

    routing: MarkdownRoutingConfig


def _load_toml(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Markdown spine threshold config not found: {path}")

    try:
        data = rtoml.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse TOML config: {path} | {exc}") from exc

    if not isinstance(data, Mapping):
        raise ValueError(f"Markdown spine threshold config must be a mapping: {path}")

    root = data.get(CONFIG_ROOT_KEY)

    if not isinstance(root, Mapping):
        raise ValueError(
            f"Markdown spine threshold config missing root key: {CONFIG_ROOT_KEY!r}"
        )

    return root


def _require_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)

    if not isinstance(value, Mapping):
        raise ValueError(f"Missing or invalid config section: {key}")

    return value


def _require_int(
    data: Mapping[str, Any],
    key: str,
    *,
    min_value: int = 0,
) -> int:
    value = data.get(key)

    if not isinstance(value, int):
        raise ValueError(f"Missing or invalid integer config value: {key}")

    if value < min_value:
        raise ValueError(f"Config value must be >= {min_value}: {key}")

    return value


def _require_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid string config value: {key}")

    return value.strip()


def _validate_threshold_order(
    *,
    small: int,
    normal: int,
    large: int,
    huge: int,
    label: str,
) -> None:
    values = [small, normal, large, huge]

    if values != sorted(values):
        raise ValueError(
            f"{label} thresholds must be ordered as "
            "small <= normal <= large <= huge"
        )


def load_markdown_spine_thresholds(
    path: Path = CONFIG_PATH,
) -> MarkdownSpineThresholds:
    config = _load_toml(path)

    bytes_config = _require_mapping(config, "bytes")
    tokens_config = _require_mapping(config, "estimated_tokens")
    sections_config = _require_mapping(config, "sections")
    routing_config = _require_mapping(config, "routing")

    small_bytes = _require_int(bytes_config, "small")
    normal_bytes = _require_int(bytes_config, "normal")
    large_bytes = _require_int(bytes_config, "large")
    huge_bytes = _require_int(bytes_config, "huge")

    small_tokens = _require_int(tokens_config, "small")
    normal_tokens = _require_int(tokens_config, "normal")
    large_tokens = _require_int(tokens_config, "large")
    huge_tokens = _require_int(tokens_config, "huge")

    _validate_threshold_order(
        small=small_bytes,
        normal=normal_bytes,
        large=large_bytes,
        huge=huge_bytes,
        label="byte",
    )

    _validate_threshold_order(
        small=small_tokens,
        normal=normal_tokens,
        large=large_tokens,
        huge=huge_tokens,
        label="estimated token",
    )

    return MarkdownSpineThresholds(
        small_bytes=small_bytes,
        normal_bytes=normal_bytes,
        large_bytes=large_bytes,
        huge_bytes=huge_bytes,

        small_tokens=small_tokens,
        normal_tokens=normal_tokens,
        large_tokens=large_tokens,
        huge_tokens=huge_tokens,

        large_section_tokens=_require_int(
            sections_config,
            "large_section_tokens",
        ),

        routing=MarkdownRoutingConfig(
            normal_queue=_require_str(routing_config, "normal_queue"),
            spine_queue=_require_str(routing_config, "spine_queue"),
            quarantine_queue=_require_str(routing_config, "quarantine_queue"),
        ),
    )