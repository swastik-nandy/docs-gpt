from __future__ import annotations

from typing import Any

from .config import get_embedding_config


# --------- BUILD EMBEDDING TEXT ------------------------

def build_embedding_text(
    chunk: dict[str, Any],
) -> str:
    config = (
        get_embedding_config()
        .embedder
        .input
    )

    metadata = dict(
        chunk.get("metadata")
        or {}
    )

    parts: list[str] = []

    # --------- SECTION PATH ------------------------

    if config.include_heading_path:
        raw_path = metadata.get(
            "heading_path"
        )

        if isinstance(
            raw_path,
            list,
        ):
            heading_path = [
                str(value).strip()
                for value in raw_path
                if str(value).strip()
            ]

            if heading_path:
                parts.append(
                    "Section: "
                    + " > ".join(
                        heading_path
                    )
                )

    # --------- FALLBACK HEADING ------------------------

    if not parts:
        heading = str(
            chunk.get("heading")
            or ""
        ).strip()

        context = str(
            chunk.get("context")
            or ""
        ).strip()

        hierarchy = []

        if context:
            hierarchy.append(
                context
            )

        if (
            heading
            and heading not in hierarchy
        ):
            hierarchy.append(
                heading
            )

        if hierarchy:
            parts.append(
                "Section: "
                + " > ".join(
                    hierarchy
                )
            )

    # --------- TYPE ------------------------

    chunk_type = str(
        chunk.get("chunk_type")
        or chunk.get("type")
        or ""
    ).strip()

    if (
        config.include_chunk_type
        and chunk_type
        and chunk_type != "text"
    ):
        parts.append(
            f"Type: {chunk_type}"
        )

    # --------- CODE CONTEXT ------------------------

    if (
        config.include_parent_context_for_code
        and chunk_type == "code"
    ):
        parent_context = str(
            metadata.get(
                "parent_context"
            )
            or ""
        ).strip()

        if parent_context:
            parts.append(
                "Context: "
                + parent_context
            )

    # --------- CONTENT ------------------------

    text = str(
        chunk.get("text")
        or ""
    ).strip()

    if not text:
        raise ValueError(
            "Chunk has no text"
        )

    parts.append(
        text
    )

    return "\n\n".join(
        parts
    )
