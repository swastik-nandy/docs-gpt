from __future__ import annotations


def validate_non_empty(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")

    if not value.strip():
        raise ValueError(f"{name} cannot be empty.")


def validate_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean.")


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