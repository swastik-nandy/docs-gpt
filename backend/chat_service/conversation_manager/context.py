from __future__ import annotations

from collections.abc import Sequence

from .models import Message


# --------- RECENT HISTORY ------------------------

def select_recent_messages(
    messages: Sequence[Message],
    *,
    max_chars: int,
) -> list[Message]:
    selected: list[Message] = []

    used_chars = 0

    for message in reversed(
        messages
    ):
        size = len(
            message.content
        )

        if (
            selected
            and used_chars + size
            > max_chars
        ):
            break

        selected.append(
            message
        )

        used_chars += size

    selected.reverse()

    return selected


# --------- MODEL CONTEXT ------------------------

def build_model_messages(
    *,
    system_prompt: str,
    summary: str | None,
    messages: Sequence[Message],
) -> list[dict[str, str]]:
    result = [
        {
            "role": "system",
            "content": system_prompt.strip(),
        }
    ]

    if summary:
        result.append(
            {
                "role": "system",
                "content": (
                    "Conversation summary:\n"
                    f"{summary.strip()}"
                ),
            }
        )

    for message in messages:
        result.append(
            {
                "role": message.role,
                "content": message.content,
            }
        )

    return result


# --------- SUMMARY SEGMENT ------------------------

def build_summary_segment(
    messages: Sequence[Message],
    *,
    max_chars: int,
) -> tuple[str, int] | None:
    if not messages:
        return None

    parts: list[str] = []
    used_chars = 0
    through_sequence = 0

    for message in messages:
        line = (
            f"{message.role.upper()}: "
            f"{message.content.strip()}"
        )

        if (
            parts
            and used_chars + len(line)
            > max_chars
        ):
            break

        parts.append(
            line
        )

        used_chars += len(
            line
        )

        through_sequence = (
            message.sequence_number
        )

    if not parts:
        return None

    return (
        "\n\n".join(parts),
        through_sequence,
    )
