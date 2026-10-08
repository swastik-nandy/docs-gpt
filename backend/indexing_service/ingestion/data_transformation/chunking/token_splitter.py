from __future__ import annotations

from collections.abc import Sequence

from .config import get_chunking_config
from .models import Chunk
from .semantic_splitter import SemanticTextSplitter
from .config import ChunkingConfig
from .split_materializer import ChunkSplitMaterializer


#--------------------ERRORS--------------------

class TokenLimitEnforcementError(RuntimeError):
    pass


#--------------------PUBLIC API--------------------

def enforce_token_limit(
    chunks: Sequence[Chunk],
    *,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    if not chunks:
        return []

    active_config = (
        config
        if config is not None
        else get_chunking_config()
    )

    if not active_config.tokenization.enabled:
        return list(
            chunks
        )

    splitter = SemanticTextSplitter(
        config=active_config
    )

    materializer = ChunkSplitMaterializer(
        config=active_config,
        splitter=splitter,
    )

    output: list[Chunk] = []

    for chunk in chunks:
        if not isinstance(
            chunk,
            Chunk,
        ):
            raise TypeError(
                "Token enforcement requires Chunk objects"
            )

        if (
            chunk.metadata.get(
                "semantic_split_materialized"
            )
            is True
        ):
            output.append(
                chunk
            )
            continue

        children = materializer.materialize(
            chunk
        )

        _validate_children(
            parent=chunk,
            children=children,
        )

        output.extend(
            children
        )

    return output


#--------------------VALIDATION--------------------

def _validate_children(
    *,
    parent: Chunk,
    children: Sequence[Chunk],
) -> None:
    if parent.text and not children:
        raise TokenLimitEnforcementError(
            "Non-empty chunk produced no "
            "materialized output: "
            f"{parent.chunk_id!r}"
        )

    seen_ids: set[str] = set()

    for child in children:
        chunk_id = child.chunk_id

        if not isinstance(
            chunk_id,
            str,
        ) or not chunk_id.strip():
            raise TokenLimitEnforcementError(
                "Materialized chunk has "
                "no valid chunk_id"
            )

        if chunk_id in seen_ids:
            raise TokenLimitEnforcementError(
                "Duplicate materialized "
                f"chunk_id: {chunk_id!r}"
            )

        seen_ids.add(
            chunk_id
        )

        if child.tokens < 0:
            raise TokenLimitEnforcementError(
                "Materialized chunk has "
                "negative token count"
            )

        if (
            child.text
            and child.tokens == 0
        ):
            raise TokenLimitEnforcementError(
                "Non-empty materialized "
                "chunk has zero tokens"
            )
