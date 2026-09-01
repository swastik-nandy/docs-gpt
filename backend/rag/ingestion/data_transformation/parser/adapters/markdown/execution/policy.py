from __future__ import annotations

import os
from typing import Any

from rag.ingestion.data_transformation.parser.adapters.markdown.execution.config_loader import (
    VALID_RESOURCE_CLASSES,
    MarkdownExecutionPolicyConfig,
    load_markdown_execution_policy_config,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.execution.schema import (
    ExecutionMode,
    MarkdownExecutionPlan,
    MarkdownExecutionPolicyContext,
    ResourceClass,
)


_RESOURCE_ORDER: tuple[str, ...] = tuple(VALID_RESOURCE_CLASSES)


def _resource_rank(resource_class: str) -> int:
    try:
        return _RESOURCE_ORDER.index(resource_class)
    except ValueError as exc:
        raise ValueError(f"Unknown resource class: {resource_class!r}") from exc


def _resource_at_rank(rank: int) -> ResourceClass:
    bounded_rank = max(0, min(rank, len(_RESOURCE_ORDER) - 1))
    return _RESOURCE_ORDER[bounded_rank]  # type: ignore[return-value]


def _min_resource_class(
    left: ResourceClass,
    right: ResourceClass,
) -> ResourceClass:
    """
    Return the weaker/lower resource class.
    """

    return left if _resource_rank(left) <= _resource_rank(right) else right


def _max_resource_class(
    left: ResourceClass,
    right: ResourceClass,
) -> ResourceClass:
    """
    Return the stronger/higher resource class.
    """

    return left if _resource_rank(left) >= _resource_rank(right) else right


def _downgrade_resource_class(
    resource_class: ResourceClass,
    downgrade_by: int,
) -> ResourceClass:
    rank = _resource_rank(resource_class)
    return _resource_at_rank(rank - max(0, downgrade_by))


def _strongest_available_resource_class(
    available: tuple[ResourceClass, ...],
) -> ResourceClass:
    if not available:
        return "tiny"

    return max(available, key=_resource_rank)


def _weakest_available_resource_class(
    available: tuple[ResourceClass, ...],
) -> ResourceClass:
    if not available:
        return "tiny"

    return min(available, key=_resource_rank)


def _clamp_to_available_resources(
    desired: ResourceClass,
    available: tuple[ResourceClass, ...],
) -> ResourceClass:
    """
    Pick the strongest available class not above desired.

    If all available classes are above desired, return the weakest available.
    """

    if desired in available:
        return desired

    desired_rank = _resource_rank(desired)

    candidates = [
        resource_class
        for resource_class in available
        if _resource_rank(resource_class) <= desired_rank
    ]

    if candidates:
        return max(candidates, key=_resource_rank)

    return _weakest_available_resource_class(available)


def _effective_policy_context(
    context: MarkdownExecutionPolicyContext | None,
    config: MarkdownExecutionPolicyConfig,
) -> MarkdownExecutionPolicyContext:
    """
    Merge optional caller context with config defaults.

    Auto mode:
      local_cpu_count = 0 means os.cpu_count()
      max_local_workers = 0 means max(1, local_cpu_count - 1)
    """

    if context is not None:
        return context

    defaults = config.defaults
    detected_cpu_count = os.cpu_count() or 1

    if defaults.local_cpu_count > 0:
        local_cpu_count = defaults.local_cpu_count
    else:
        local_cpu_count = detected_cpu_count

    local_cpu_count = max(1, local_cpu_count)

    if defaults.max_local_workers > 0:
        max_local_workers = min(defaults.max_local_workers, local_cpu_count)
    else:
        max_local_workers = max(1, local_cpu_count - 1)

    return MarkdownExecutionPolicyContext(
        user_tier=defaults.user_tier,
        demand_level=defaults.demand_level,
        local_cpu_count=local_cpu_count,
        max_local_workers=max(1, max_local_workers),
        allow_process_pool=defaults.allow_process_pool,
        available_resource_classes=defaults.available_resource_classes,
        prefer_fast_execution=defaults.prefer_fast_execution,
    )

def _resolve_desired_resource_class(
    *,
    parse_strategy: str,
    size_class: str,
    context: MarkdownExecutionPolicyContext,
    config: MarkdownExecutionPolicyConfig,
    reasons: list[str],
) -> ResourceClass:
    strategy_config = config.strategy_defaults[parse_strategy]
    tier_config = config.user_tiers[context.user_tier]
    demand_config = config.demand_levels[context.demand_level]

    desired = strategy_config.preferred_resource_class

    size_override = config.size_class_overrides.get(size_class)

    if size_override:
        desired = _max_resource_class(
            desired,
            size_override.preferred_resource_class,
        )

    if context.prefer_fast_execution or tier_config.prefer_fast_execution:
        desired = _max_resource_class(
            desired,
            strategy_config.preferred_resource_class,
        )

    desired = _max_resource_class(
        desired,
        strategy_config.min_resource_class,
    )

    desired = _min_resource_class(
        desired,
        tier_config.max_resource_class,
    )

    if demand_config.downgrade_resource_class_by > 0:
        desired = _downgrade_resource_class(
            desired,
            demand_config.downgrade_resource_class_by,
        )
        reasons.append(f"demand_downgrade_{context.demand_level}")

    desired = _clamp_to_available_resources(
        desired,
        context.available_resource_classes,
    )

    return desired


def _auto_worker_count(context: MarkdownExecutionPolicyContext) -> int:
    """
    Reserve one CPU for OS/editor/runtime side work.
    """

    return max(1, context.local_cpu_count - 1)


def _resolve_worker_cap(
    configured_workers: int,
    context: MarkdownExecutionPolicyContext,
) -> int:
    """
    Resolve worker cap.

    0 means auto:
      max(1, local_cpu_count - 1)

    Positive values are explicit caps.
    """

    if configured_workers > 0:
        return configured_workers

    return _auto_worker_count(context)

def _resolve_execution_mode(
    *,
    parse_strategy: str,
    context: MarkdownExecutionPolicyContext,
    config: MarkdownExecutionPolicyConfig,
    resource_class: ResourceClass,
    reasons: list[str],
) -> ExecutionMode:
    strategy_config = config.strategy_defaults[parse_strategy]
    tier_config = config.user_tiers[context.user_tier]

    if strategy_config.execution_mode == "sequential":
        return "sequential"

    if not strategy_config.process_pool_allowed:
        reasons.append("strategy_disallows_process_pool")
        return "sequential"

    if not tier_config.allow_process_pool:
        reasons.append("tier_disallows_process_pool")
        return "sequential"

    if not context.allow_process_pool:
        reasons.append("context_disallows_process_pool")
        return "sequential"

    resource_worker_cap = _resolve_worker_cap(
        config.resource_classes[resource_class].max_workers,
        context,
    )

    if resource_worker_cap <= 1:
        reasons.append("resource_class_single_worker")
        return "sequential"

    return "process_pool"

def _resolve_max_workers(
    *,
    parse_strategy: str,
    size_class: str,
    execution_mode: ExecutionMode,
    resource_class: ResourceClass,
    context: MarkdownExecutionPolicyContext,
    config: MarkdownExecutionPolicyConfig,
) -> int:
    if execution_mode == "sequential":
        return 1

    tier_config = config.user_tiers[context.user_tier]
    demand_config = config.demand_levels[context.demand_level]
    resource_config = config.resource_classes[resource_class]
    strategy_config = config.strategy_defaults[parse_strategy]
    size_override = config.size_class_overrides.get(size_class)

    candidates = [
        _resolve_worker_cap(resource_config.max_workers, context),
        _resolve_worker_cap(tier_config.max_workers, context),
        context.max_local_workers,
    ]

    if size_override:
        candidates.append(_resolve_worker_cap(size_override.max_workers, context))

    if not strategy_config.process_pool_allowed:
        candidates.append(1)

    max_workers = max(1, min(candidates))
    max_workers = int(max_workers * demand_config.worker_multiplier)

    return max(1, max_workers)

def _reason_from_size_class(size_class: str) -> str:
    if size_class == "small":
        return "small_document"

    if size_class == "normal":
        return "normal_document"

    if size_class == "large":
        return "large_document"

    if size_class == "huge":
        return "huge_document"

    if size_class == "pathological":
        return "pathological_document"

    return f"size_class_{size_class}"


def _reasons_for_policy_context(
    context: MarkdownExecutionPolicyContext,
) -> list[str]:
    reasons: list[str] = []

    if context.user_tier == "free":
        reasons.append("free_tier")
    elif context.user_tier == "premium":
        reasons.append("premium_tier")
    elif context.user_tier == "internal":
        reasons.append("internal_tier")
    else:
        reasons.append(f"{context.user_tier}_tier")

    if context.demand_level == "high":
        reasons.append("high_platform_demand")
    elif context.demand_level == "critical":
        reasons.append("critical_platform_demand")
    else:
        reasons.append(f"{context.demand_level}_platform_demand")

    return reasons


def build_markdown_execution_plan(
    *,
    inspection: Any,
    policy_context: MarkdownExecutionPolicyContext | None = None,
    config: MarkdownExecutionPolicyConfig | None = None,
) -> MarkdownExecutionPlan:
    """
    Build a resource-aware execution plan for Markdown parsing.

    This function does not parse.
    This function does not write artifacts.

    It converts:
      document inspection + user/runtime policy context + TOML config

    into:
      parse strategy + execution mode + resource class + worker count
    """

    resolved_config = config or load_markdown_execution_policy_config()
    context = _effective_policy_context(policy_context, resolved_config)

    parse_strategy = str(getattr(inspection, "parse_strategy", "") or "")
    size_class = str(getattr(inspection, "size_class", "") or "")
    risk_level = str(getattr(inspection, "risk_level", "") or "")

    if parse_strategy not in resolved_config.strategy_defaults:
        raise ValueError(f"Unsupported markdown parse strategy: {parse_strategy!r}")

    strategy_config = resolved_config.strategy_defaults[parse_strategy]

    reasons = _reasons_for_policy_context(context)
    reasons.append(_reason_from_size_class(size_class))
    reasons.append(f"{parse_strategy}_strategy")

    if risk_level:
        reasons.append(f"{risk_level}_risk")

    if not strategy_config.should_parse:
        return MarkdownExecutionPlan(
            parse_strategy=parse_strategy,
            execution_mode="sequential",
            resource_class=strategy_config.preferred_resource_class,
            max_workers=1,
            should_parse=False,
            reasons=reasons + ["quarantine_strategy"],
            meta={
                "user_tier": context.user_tier,
                "demand_level": context.demand_level,
                "size_class": size_class,
                "risk_level": risk_level,
            },
        )

    resource_class = _resolve_desired_resource_class(
        parse_strategy=parse_strategy,
        size_class=size_class,
        context=context,
        config=resolved_config,
        reasons=reasons,
    )

    execution_mode = _resolve_execution_mode(
        parse_strategy=parse_strategy,
        context=context,
        config=resolved_config,
        resource_class=resource_class,
        reasons=reasons,
    )

    max_workers = _resolve_max_workers(
        parse_strategy=parse_strategy,
        size_class=size_class,
        execution_mode=execution_mode,
        resource_class=resource_class,
        context=context,
        config=resolved_config,
    )

    return MarkdownExecutionPlan(
        parse_strategy=parse_strategy,
        execution_mode=execution_mode,
        resource_class=resource_class,
        max_workers=max_workers,
        should_parse=True,
        reasons=reasons,
        meta={
            "user_tier": context.user_tier,
            "demand_level": context.demand_level,
            "size_class": size_class,
            "risk_level": risk_level,
            "local_cpu_count": context.local_cpu_count,
            "max_local_workers": context.max_local_workers,
            "allow_process_pool": context.allow_process_pool,
            "available_resource_classes": list(context.available_resource_classes),
        },
    )