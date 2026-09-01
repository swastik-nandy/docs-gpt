from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import rtoml

from rag.ingestion.data_transformation.parser.adapters.markdown.execution.schema import (
    DemandLevel,
    ExecutionMode,
    ResourceClass,
    UserTier,
)


CONFIG_FILE_NAME = "policy.toml"
CONFIG_ROOT_KEY = "markdown_execution_policy"


VALID_USER_TIERS: set[str] = {
    "free",
    "standard",
    "premium",
    "internal",
}

VALID_DEMAND_LEVELS: set[str] = {
    "low",
    "normal",
    "high",
    "critical",
}

VALID_RESOURCE_CLASSES: tuple[str, ...] = (
    "tiny",
    "small",
    "medium",
    "large",
    "xlarge",
)

VALID_EXECUTION_MODES: set[str] = {
    "sequential",
    "process_pool",
}

VALID_PARSE_STRATEGIES: set[str] = {
    "single_document",
    "full_parse_then_chunk_fanout",
    "spine_fragmented",
    "quarantine",
}

VALID_SIZE_CLASSES: set[str] = {
    "small",
    "normal",
    "large",
    "huge",
    "pathological",
}


@dataclass(frozen=True)
class MarkdownResourceClassConfig:
    name: ResourceClass
    max_workers: int
    relative_cpu: float


@dataclass(frozen=True)
class MarkdownUserTierConfig:
    name: UserTier
    max_resource_class: ResourceClass
    max_workers: int
    allow_process_pool: bool
    prefer_fast_execution: bool


@dataclass(frozen=True)
class MarkdownDemandLevelConfig:
    name: DemandLevel
    resource_pressure: str
    worker_multiplier: float
    downgrade_resource_class_by: int


@dataclass(frozen=True)
class MarkdownStrategyConfig:
    name: str
    execution_mode: ExecutionMode
    min_resource_class: ResourceClass
    preferred_resource_class: ResourceClass
    process_pool_allowed: bool
    should_parse: bool = True


@dataclass(frozen=True)
class MarkdownSizeClassOverride:
    name: str
    preferred_resource_class: ResourceClass
    max_workers: int


@dataclass(frozen=True)
class MarkdownExecutionDefaults:
    user_tier: UserTier
    demand_level: DemandLevel
    local_cpu_count: int
    max_local_workers: int
    allow_process_pool: bool
    prefer_fast_execution: bool
    available_resource_classes: tuple[ResourceClass, ...]


@dataclass(frozen=True)
class MarkdownExecutionPolicyConfig:
    defaults: MarkdownExecutionDefaults
    resource_classes: dict[ResourceClass, MarkdownResourceClassConfig]
    user_tiers: dict[UserTier, MarkdownUserTierConfig]
    demand_levels: dict[DemandLevel, MarkdownDemandLevelConfig]
    strategy_defaults: dict[str, MarkdownStrategyConfig]
    size_class_overrides: dict[str, MarkdownSizeClassOverride]


def _config_path() -> Path:
    return Path(__file__).resolve().parent / "config" / CONFIG_FILE_NAME


def _load_toml(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Markdown execution policy config not found: {path}")

    try:
        data = rtoml.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to parse TOML config: {path} | {exc}") from exc

    if not isinstance(data, Mapping):
        raise ValueError(f"Markdown execution policy config must be a mapping: {path}")

    root = data.get(CONFIG_ROOT_KEY)

    if not isinstance(root, Mapping):
        raise ValueError(
            f"Markdown execution policy config missing root key: {CONFIG_ROOT_KEY!r}"
        )

    return root


def _require_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)

    if not isinstance(value, Mapping):
        raise ValueError(f"Expected mapping at {key!r}")

    return value


def _require_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string at {key!r}")

    return value.strip()


def _require_int(
    data: Mapping[str, Any],
    key: str,
    *,
    min_value: int = 0,
) -> int:
    value = data.get(key)

    if not isinstance(value, int):
        raise ValueError(f"Expected integer at {key!r}")

    if value < min_value:
        raise ValueError(f"Expected {key!r} >= {min_value}, got {value}")

    return value


def _require_float(
    data: Mapping[str, Any],
    key: str,
    *,
    min_value: float = 0.0,
) -> float:
    value = data.get(key)

    if isinstance(value, int):
        value = float(value)

    if not isinstance(value, float):
        raise ValueError(f"Expected number at {key!r}")

    if value < min_value:
        raise ValueError(f"Expected {key!r} >= {min_value}, got {value}")

    return value


def _require_bool(data: Mapping[str, Any], key: str) -> bool:
    value = data.get(key)

    if not isinstance(value, bool):
        raise ValueError(f"Expected boolean at {key!r}")

    return value


def _require_string_list(data: Mapping[str, Any], key: str) -> list[str]:
    value = data.get(key)

    if not isinstance(value, list):
        raise ValueError(f"Expected list at {key!r}")

    result: list[str] = []

    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Expected non-empty string items at {key!r}")

        result.append(item.strip())

    return result


def _validate_choice(
    value: str,
    allowed: set[str] | tuple[str, ...],
    label: str,
) -> str:
    if value not in allowed:
        raise ValueError(
            f"Invalid {label}: {value!r}. Allowed values: {sorted(allowed)}"
        )

    return value


def _resource_rank(resource_class: str) -> int:
    try:
        return VALID_RESOURCE_CLASSES.index(resource_class)
    except ValueError as exc:
        raise ValueError(f"Unknown resource class: {resource_class!r}") from exc


def _parse_defaults(data: Mapping[str, Any]) -> MarkdownExecutionDefaults:
    defaults = _require_mapping(data, "defaults")

    user_tier = _validate_choice(
        _require_string(defaults, "user_tier"),
        VALID_USER_TIERS,
        "user_tier",
    )

    demand_level = _validate_choice(
        _require_string(defaults, "demand_level"),
        VALID_DEMAND_LEVELS,
        "demand_level",
    )

    available = _require_string_list(defaults, "available_resource_classes")

    if not available:
        raise ValueError("defaults.available_resource_classes cannot be empty")

    for resource_class in available:
        _validate_choice(
            resource_class,
            VALID_RESOURCE_CLASSES,
            "available_resource_class",
        )

    return MarkdownExecutionDefaults(
        user_tier=user_tier,  # type: ignore[arg-type]
        demand_level=demand_level,  # type: ignore[arg-type]
        local_cpu_count=_require_int(defaults, "local_cpu_count", min_value=0),
        max_local_workers=_require_int(defaults, "max_local_workers", min_value=0),
        allow_process_pool=_require_bool(defaults, "allow_process_pool"),
        prefer_fast_execution=_require_bool(defaults, "prefer_fast_execution"),
        available_resource_classes=tuple(available),  # type: ignore[arg-type]
    )


def _parse_resource_classes(
    data: Mapping[str, Any],
) -> dict[ResourceClass, MarkdownResourceClassConfig]:
    raw = _require_mapping(data, "resource_classes")
    parsed: dict[ResourceClass, MarkdownResourceClassConfig] = {}

    for name, raw_config in raw.items():
        if not isinstance(name, str):
            raise ValueError("resource_classes keys must be strings")

        _validate_choice(name, VALID_RESOURCE_CLASSES, "resource_class")

        if not isinstance(raw_config, Mapping):
            raise ValueError(f"resource_classes.{name} must be a mapping")

        parsed[name] = MarkdownResourceClassConfig(  # type: ignore[index]
            name=name,  # type: ignore[arg-type]
            max_workers=_require_int(raw_config, "max_workers", min_value=0),
            relative_cpu=_require_float(raw_config, "relative_cpu", min_value=0.1),
        )

    missing = set(VALID_RESOURCE_CLASSES) - set(parsed)

    if missing:
        raise ValueError(f"Missing resource_classes config for: {sorted(missing)}")

    return parsed


def _parse_user_tiers(
    data: Mapping[str, Any],
) -> dict[UserTier, MarkdownUserTierConfig]:
    raw = _require_mapping(data, "user_tiers")
    parsed: dict[UserTier, MarkdownUserTierConfig] = {}

    for name, raw_config in raw.items():
        if not isinstance(name, str):
            raise ValueError("user_tiers keys must be strings")

        _validate_choice(name, VALID_USER_TIERS, "user_tier")

        if not isinstance(raw_config, Mapping):
            raise ValueError(f"user_tiers.{name} must be a mapping")

        max_resource_class = _validate_choice(
            _require_string(raw_config, "max_resource_class"),
            VALID_RESOURCE_CLASSES,
            f"user_tiers.{name}.max_resource_class",
        )

        parsed[name] = MarkdownUserTierConfig(  # type: ignore[index]
            name=name,  # type: ignore[arg-type]
            max_resource_class=max_resource_class,  # type: ignore[arg-type]
            max_workers=_require_int(raw_config, "max_workers", min_value=0),
            allow_process_pool=_require_bool(raw_config, "allow_process_pool"),
            prefer_fast_execution=_require_bool(raw_config, "prefer_fast_execution"),
        )

    missing = VALID_USER_TIERS - set(parsed)

    if missing:
        raise ValueError(f"Missing user_tiers config for: {sorted(missing)}")

    return parsed


def _parse_demand_levels(
    data: Mapping[str, Any],
) -> dict[DemandLevel, MarkdownDemandLevelConfig]:
    raw = _require_mapping(data, "demand_levels")
    parsed: dict[DemandLevel, MarkdownDemandLevelConfig] = {}

    for name, raw_config in raw.items():
        if not isinstance(name, str):
            raise ValueError("demand_levels keys must be strings")

        _validate_choice(name, VALID_DEMAND_LEVELS, "demand_level")

        if not isinstance(raw_config, Mapping):
            raise ValueError(f"demand_levels.{name} must be a mapping")

        parsed[name] = MarkdownDemandLevelConfig(  # type: ignore[index]
            name=name,  # type: ignore[arg-type]
            resource_pressure=_require_string(raw_config, "resource_pressure"),
            worker_multiplier=_require_float(
                raw_config,
                "worker_multiplier",
                min_value=0.0,
            ),
            downgrade_resource_class_by=_require_int(
                raw_config,
                "downgrade_resource_class_by",
                min_value=0,
            ),
        )

    missing = VALID_DEMAND_LEVELS - set(parsed)

    if missing:
        raise ValueError(f"Missing demand_levels config for: {sorted(missing)}")

    return parsed


def _parse_strategy_defaults(
    data: Mapping[str, Any],
) -> dict[str, MarkdownStrategyConfig]:
    raw = _require_mapping(data, "strategy_defaults")
    parsed: dict[str, MarkdownStrategyConfig] = {}

    for name, raw_config in raw.items():
        if not isinstance(name, str):
            raise ValueError("strategy_defaults keys must be strings")

        _validate_choice(name, VALID_PARSE_STRATEGIES, "parse_strategy")

        if not isinstance(raw_config, Mapping):
            raise ValueError(f"strategy_defaults.{name} must be a mapping")

        execution_mode = _validate_choice(
            _require_string(raw_config, "execution_mode"),
            VALID_EXECUTION_MODES,
            f"strategy_defaults.{name}.execution_mode",
        )

        min_resource_class = _validate_choice(
            _require_string(raw_config, "min_resource_class"),
            VALID_RESOURCE_CLASSES,
            f"strategy_defaults.{name}.min_resource_class",
        )

        preferred_resource_class = _validate_choice(
            _require_string(raw_config, "preferred_resource_class"),
            VALID_RESOURCE_CLASSES,
            f"strategy_defaults.{name}.preferred_resource_class",
        )

        if _resource_rank(preferred_resource_class) < _resource_rank(min_resource_class):
            raise ValueError(
                f"strategy_defaults.{name}.preferred_resource_class "
                "cannot be below min_resource_class"
            )

        should_parse = raw_config.get("should_parse", True)

        if not isinstance(should_parse, bool):
            raise ValueError(f"strategy_defaults.{name}.should_parse must be boolean")

        parsed[name] = MarkdownStrategyConfig(
            name=name,
            execution_mode=execution_mode,  # type: ignore[arg-type]
            min_resource_class=min_resource_class,  # type: ignore[arg-type]
            preferred_resource_class=preferred_resource_class,  # type: ignore[arg-type]
            process_pool_allowed=_require_bool(raw_config, "process_pool_allowed"),
            should_parse=should_parse,
        )

    missing = VALID_PARSE_STRATEGIES - set(parsed)

    if missing:
        raise ValueError(f"Missing strategy_defaults config for: {sorted(missing)}")

    return parsed


def _parse_size_class_overrides(
    data: Mapping[str, Any],
) -> dict[str, MarkdownSizeClassOverride]:
    raw = _require_mapping(data, "size_class_overrides")
    parsed: dict[str, MarkdownSizeClassOverride] = {}

    for name, raw_config in raw.items():
        if not isinstance(name, str):
            raise ValueError("size_class_overrides keys must be strings")

        _validate_choice(name, VALID_SIZE_CLASSES, "size_class")

        if not isinstance(raw_config, Mapping):
            raise ValueError(f"size_class_overrides.{name} must be a mapping")

        preferred_resource_class = _validate_choice(
            _require_string(raw_config, "preferred_resource_class"),
            VALID_RESOURCE_CLASSES,
            f"size_class_overrides.{name}.preferred_resource_class",
        )

        parsed[name] = MarkdownSizeClassOverride(
            name=name,
            preferred_resource_class=preferred_resource_class,  # type: ignore[arg-type]
            max_workers=_require_int(raw_config, "max_workers", min_value=0),
        )

    missing = VALID_SIZE_CLASSES - set(parsed)

    if missing:
        raise ValueError(f"Missing size_class_overrides config for: {sorted(missing)}")

    return parsed


@lru_cache(maxsize=1)
def load_markdown_execution_policy_config() -> MarkdownExecutionPolicyConfig:
    """
    Load and validate Markdown execution policy config.

    Uses rtoml so dev/prod policy knobs can be changed without touching
    parser execution code.
    """

    data = _load_toml(_config_path())

    return MarkdownExecutionPolicyConfig(
        defaults=_parse_defaults(data),
        resource_classes=_parse_resource_classes(data),
        user_tiers=_parse_user_tiers(data),
        demand_levels=_parse_demand_levels(data),
        strategy_defaults=_parse_strategy_defaults(data),
        size_class_overrides=_parse_size_class_overrides(data),
    )


def reload_markdown_execution_policy_config() -> MarkdownExecutionPolicyConfig:
    """
    Clear cache and reload config.

    Useful for tests and local policy tuning.
    """

    load_markdown_execution_policy_config.cache_clear()
    return load_markdown_execution_policy_config()