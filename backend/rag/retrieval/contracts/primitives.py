from __future__ import annotations

import math
from enum import Enum
from typing import Any, Mapping, Sequence


Payload = dict[str, Any]
Vector = list[float]


def optional_payload_str(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)

    if value is None:
        return None

    cleaned = str(value).strip()
    return cleaned or None


def validate_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean.")


def validate_enum_instance(name: str, value: Any, enum_type: type[Enum]) -> None:
    if not isinstance(value, enum_type):
        allowed = ", ".join(str(member.value) for member in enum_type)
        raise ValueError(
            f"{name} must be {enum_type.__name__}. Allowed values: {allowed}."
        )


def validate_optional_clean_str(name: str, value: str | None) -> None:
    if value is None:
        return

    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string when provided.")

    if not value.strip():
        raise ValueError(f"{name} cannot be blank when provided.")


def validate_mapping(name: str, value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")


def validate_mapping_size(name: str, value: Mapping[str, Any], max_keys: int | None) -> None:
    validate_mapping(name, value)

    if max_keys is None:
        return

    validate_positive_int(f"{name}.max_keys", max_keys)

    if len(value) > max_keys:
        raise ValueError(f"{name} cannot contain more than {max_keys} keys.")


def validate_non_empty_str(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")

    if not value.strip():
        raise ValueError(f"{name} cannot be empty.")


def validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")


def validate_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")

    if value < 0:
        raise ValueError(f"{name} cannot be negative.")


def validate_optional_positive_int(name: str, value: int | None) -> None:
    if value is None:
        return

    validate_positive_int(name, value)


def validate_finite_number(name: str, value: Any) -> None:
    if not is_finite_number(value):
        raise ValueError(f"{name} must be a finite number.")


def validate_optional_sequence_limit(
    name: str,
    values: Sequence[Any],
    max_count: int | None,
) -> None:
    if max_count is None:
        return

    validate_non_negative_int(f"{name}.max_count", max_count)

    if len(values) > max_count:
        raise ValueError(f"{name} cannot contain more than {max_count} items.")


def is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False

    if not isinstance(value, (int, float)):
        return False

    return math.isfinite(float(value))


def clean_tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()

    if isinstance(values, str):
        cleaned = values.strip()
        return (cleaned,) if cleaned else ()

    if not isinstance(values, (list, tuple)):
        return ()

    return tuple(
        cleaned
        for item in values
        if (cleaned := str(item).strip())
    )