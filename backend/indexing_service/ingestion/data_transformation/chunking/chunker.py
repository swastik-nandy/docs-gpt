from __future__ import annotations

from collections.abc import Sequence

from .config import get_chunking_config
from .models import Chunk
from .overlap import apply_overlap
from .quality import evaluate_chunks_quality
from .section_splitter import split_sections
from .config import ChunkingConfig
from .token_splitter import enforce_token_limit
from .validator import ChunkValidationResult
from .validator import validate_chunks_detailed


#--------------------PUBLIC PIPELINE--------------------

def chunk_document(
    sections,
    source: str = "",
    *,
    enable_overlap: bool = False,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    result = chunk_document_detailed(
        sections,
        source=source,
        enable_overlap=enable_overlap,
        config=config,
    )

    return list(
        result.accepted
    )


def chunk_document_detailed(
    sections,
    source: str = "",
    *,
    enable_overlap: bool = False,
    config: ChunkingConfig | None = None,
) -> ChunkValidationResult:
    if not sections:
        return _empty_result()

    active_config = (
        config
        if config is not None
        else get_chunking_config()
    )

    base_chunks = split_sections(
        sections,
        source=source,
    )

    if not base_chunks:
        return _empty_result()

    _require_chunks(
        base_chunks,
        stage="section_splitter",
    )

    retrieval_chunks = (
        apply_overlap(
            base_chunks,
            config=active_config,
        )
        if enable_overlap
        else list(
            base_chunks
        )
    )

    token_safe_chunks = enforce_token_limit(
        retrieval_chunks,
        config=active_config,
    )

    if not token_safe_chunks:
        return _empty_result()

    scored_chunks = evaluate_chunks_quality(
        token_safe_chunks,
        config=active_config,
    )

    return validate_chunks_detailed(
        scored_chunks,
        config=active_config,
    )


#--------------------STAGE VALIDATION--------------------

def _require_chunks(
    chunks: Sequence[Chunk],
    *,
    stage: str,
) -> None:
    for index, chunk in enumerate(
        chunks
    ):
        if not isinstance(
            chunk,
            Chunk,
        ):
            raise TypeError(
                f"{stage} produced non-Chunk "
                f"value at index {index}"
            )


#--------------------EMPTY RESULT--------------------

def _empty_result(
) -> ChunkValidationResult:
    return ChunkValidationResult(
        accepted=(),
        quarantined=(),
        rejected=(),
    )
