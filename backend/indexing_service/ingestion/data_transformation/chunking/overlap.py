from __future__ import annotations

import re
from collections.abc import Sequence

from .models import Chunk
from .config import ChunkingConfig
from .utils import estimate_tokens


#--------------------SENTENCE BOUNDARY--------------------

_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s+"
)


#--------------------PUBLIC API--------------------

def apply_overlap(
    chunks: Sequence[Chunk],
    *,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    if not chunks:
        return []

    output: list[Chunk] = [
        chunks[0]
    ]

    for current in chunks[1:]:
        previous = output[-1]

        if not _eligible_pair(
            previous,
            current,
            config=config,
        ):
            output.append(
                current
            )
            continue

        overlap = _last_complete_sentence(
            previous.text
        )

        if not overlap:
            output.append(
                current
            )
            continue

        if _already_present(
            current.text,
            overlap,
        ):
            output.append(
                current
            )
            continue

        text = (
            f"{overlap}\n\n"
            f"{current.text}"
        )

        metadata = dict(
            current.metadata
        )

        metadata[
            "overlap_applied"
        ] = True

        metadata[
            "overlap_source_chunk_id"
        ] = previous.chunk_id

        metadata[
            "overlap_token_count"
        ] = estimate_tokens(
            overlap
        )

        output.append(
            current.copy_with(
                text=text,
                tokens=estimate_tokens(
                    text
                ),
                metadata=metadata,
            )
        )

    return output


#--------------------ELIGIBILITY--------------------

def _eligible_pair(
    previous: Chunk,
    current: Chunk,
    *,
    config: ChunkingConfig | None,
) -> bool:
    if not previous.text.strip():
        return False

    if not current.text.strip():
        return False

    atomic_subtypes = (
        config.atomic_policy.subtypes
        if config is not None
        else frozenset(
            {
                "code",
                "table",
                "list",
                "component",
            }
        )
    )

    if previous.subtype in atomic_subtypes:
        return False

    if current.subtype in atomic_subtypes:
        return False

    return _same_scope(
        previous,
        current,
    )


def _same_scope(
    previous: Chunk,
    current: Chunk,
) -> bool:
    if (
        previous.source
        and current.source
        and previous.source
        != current.source
    ):
        return False

    previous_section = (
        previous.metadata.get(
            "section_id"
        )
    )

    current_section = (
        current.metadata.get(
            "section_id"
        )
    )

    if (
        previous_section is not None
        and current_section is not None
    ):
        return (
            previous_section
            == current_section
        )

    if (
        previous.heading.strip()
        and current.heading.strip()
    ):
        return (
            previous.heading.strip()
            == current.heading.strip()
            and previous.context.strip()
            == current.context.strip()
        )

    return False


#--------------------EXTRACTION--------------------

def _last_complete_sentence(
    text: str,
) -> str:
    value = text.strip()

    if not value:
        return ""

    blocks = [
        block.strip()
        for block in value.split(
            "\n\n"
        )
        if block.strip()
    ]

    if not blocks:
        return ""

    last_block = blocks[-1]

    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY.split(
            last_block
        )
        if sentence.strip()
    ]

    if not sentences:
        return ""

    candidate = sentences[-1]

    if not _has_terminal_punctuation(
        candidate
    ):
        return ""

    return candidate


def _has_terminal_punctuation(
    text: str,
) -> bool:
    stripped = text.rstrip(
        "\"'”’)]}"
    )

    return stripped.endswith(
        (
            ".",
            "!",
            "?",
        )
    )


def _already_present(
    text: str,
    overlap: str,
) -> bool:
    return text.lstrip().startswith(
        overlap
    )
