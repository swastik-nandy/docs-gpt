# rag/ingestion/data_transformation/chunking/utils.py

from __future__ import annotations

from collections.abc import Sequence

from .tokenization import count_tokens


#--------------------TOKEN COUNTING--------------------

def estimate_tokens(
    text: str,
) -> int:
    if not isinstance(
        text,
        str,
    ):
        raise TypeError(
            "Token counting input must be a string"
        )

    if not text:
        return 0

    return count_tokens(
        text
    )


#--------------------TEXT NORMALIZATION--------------------

def clean_text(
    text: object,
) -> str:
    if text is None:
        return ""

    if isinstance(
        text,
        str,
    ):
        value = text
    else:
        value = str(
            text
        )

    if not value:
        return ""

    return (
        value
        .replace(
            "\r\n",
            "\n",
        )
        .replace(
            "\r",
            "\n",
        )
        .strip()
    )


#--------------------SAFE JOIN--------------------

def join_text(
    parts: Sequence[str],
) -> str:
    if not parts:
        return ""

    cleaned: list[str] = []

    for part in parts:
        if not isinstance(
            part,
            str,
        ):
            raise TypeError(
                "Text parts must be strings"
            )

        normalized = clean_text(
            part
        )

        if normalized:
            cleaned.append(
                normalized
            )

    return "\n\n".join(
        cleaned
    )


#--------------------SEMANTIC GROUPING--------------------

def split_by_token_limit(
    parts: Sequence[str],
    max_tokens: int,
) -> list[str]:
    """
    Group already-semantic text units without cutting their contents.

    This function is intentionally not a general-purpose text splitter.
    Oversized individual units are preserved intact so the authoritative
    semantic token-enforcement stage can handle or quarantine them later.
    """
    _validate_token_limit(
        max_tokens
    )

    if not parts:
        return []

    output: list[str] = []
    buffer: list[str] = []

    for raw_part in parts:
        if not isinstance(
            raw_part,
            str,
        ):
            raise TypeError(
                "Text parts must be strings"
            )

        part = clean_text(
            raw_part
        )

        if not part:
            continue

        part_tokens = estimate_tokens(
            part
        )

        if part_tokens > max_tokens:
            _flush_buffer(
                buffer,
                output,
            )

            output.append(
                part
            )

            continue

        if not buffer:
            buffer.append(
                part
            )
            continue

        candidate = join_text(
            [
                *buffer,
                part,
            ]
        )

        candidate_tokens = estimate_tokens(
            candidate
        )

        if candidate_tokens > max_tokens:
            _flush_buffer(
                buffer,
                output,
            )

            buffer.append(
                part
            )

            continue

        buffer.append(
            part
        )

    _flush_buffer(
        buffer,
        output,
    )

    return output


#--------------------BUFFER HELPERS--------------------

def _flush_buffer(
    buffer: list[str],
    output: list[str],
) -> None:
    if not buffer:
        return

    text = join_text(
        buffer
    )

    if text:
        output.append(
            text
        )

    buffer.clear()


#--------------------VALIDATION--------------------

def _validate_token_limit(
    max_tokens: int,
) -> None:
    if isinstance(
        max_tokens,
        bool,
    ):
        raise TypeError(
            "max_tokens must be an integer"
        )

    if not isinstance(
        max_tokens,
        int,
    ):
        raise TypeError(
            "max_tokens must be an integer"
        )

    if max_tokens <= 0:
        raise ValueError(
            "max_tokens must be greater than zero"
        )