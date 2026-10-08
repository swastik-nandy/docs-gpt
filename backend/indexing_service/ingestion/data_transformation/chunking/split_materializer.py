# rag/ingestion/data_transformation/chunking/split_materializer.py

from __future__ import annotations

from typing import Any
from typing import Sequence
from typing import TypedDict

from .config import get_chunking_config
from .models import Chunk
from .semantic_splitter import SemanticSplitFragment
from .semantic_splitter import SemanticSplitResult
from .semantic_splitter import SemanticTextSplitter
from .config import ChunkingConfig
from .split_identity import build_split_child_identity
from .split_identity import build_split_provenance


#--------------------ERRORS--------------------

class SplitMaterializationError(RuntimeError):
    pass


#--------------------METADATA--------------------

class SemanticSplitMetadata(TypedDict, total=False):
    semantic_split_materialized: bool
    semantic_split_was_split: bool
    semantic_split_paragraph_index: int
    oversized_sentence: bool
    oversized_atomic: bool
    requires_quarantine: bool


#--------------------MATERIALIZER--------------------

class ChunkSplitMaterializer:
    def __init__(
        self,
        *,
        config: ChunkingConfig | None = None,
        splitter: SemanticTextSplitter | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else get_chunking_config()
        )

        self.splitter = (
            splitter
            if splitter is not None
            else SemanticTextSplitter(
                config=self.config
            )
        )


#--------------------PUBLIC API--------------------

    def materialize(
        self,
        chunk: Chunk,
        *,
        density: str | None = None,
    ) -> tuple[Chunk, ...]:
        self._validate_chunk(
            chunk
        )

        self._ensure_not_materialized(
            chunk
        )

        subtype = self._resolve_subtype(
            chunk
        )

        resolved_density = self._resolve_density(
            chunk,
            density=density,
        )

        result = self.splitter.split(
            chunk.text,
            subtype=subtype,
            density=resolved_density,
        )

        if not result.fragments:
            return ()

        return self._materialize_result(
            chunk=chunk,
            result=result,
        )


    def materialize_many(
        self,
        chunks: Sequence[Chunk],
    ) -> tuple[Chunk, ...]:
        output: list[Chunk] = []

        for chunk in chunks:
            output.extend(
                self.materialize(
                    chunk
                )
            )

        return tuple(
            output
        )


#--------------------RESULT MATERIALIZATION--------------------

    def _materialize_result(
        self,
        *,
        chunk: Chunk,
        result: SemanticSplitResult,
    ) -> tuple[Chunk, ...]:
        parent_chunk_id = self._parent_chunk_id(
            chunk
        )

        source_ir_id = self._source_ir_id(
            chunk,
            parent_chunk_id=parent_chunk_id,
        )

        split_count = len(
            result.fragments
        )

        children: list[
            Chunk
        ] = []

        for fragment_position, fragment in enumerate(
            result.fragments
        ):
            identity = build_split_child_identity(
                parent_chunk_id=parent_chunk_id,
                source_ir_id=source_ir_id,
                fragment_position=fragment_position,
                split_count=split_count,
                config=self.config.split_identity,
            )

            metadata = self._build_child_metadata(
                chunk=chunk,
                result=result,
                fragment=fragment,
                identity=identity,
            )

            child = chunk.copy_with(
                text=fragment.text,
                subtype=result.subtype,
                metadata=metadata,
                tokens=fragment.token_count,
                chunk_id=identity.chunk_id,
                preserve_chunk_id=False,
            )

            children.append(
                child
            )

        self._validate_materialized_children(
            parent=chunk,
            children=children,
            result=result,
        )

        return tuple(
            children
        )


#--------------------CHILD METADATA--------------------

    def _build_child_metadata(
        self,
        *,
        chunk: Chunk,
        result: SemanticSplitResult,
        fragment: SemanticSplitFragment,
        identity,
    ) -> dict[str, Any]:
        metadata = dict(
            chunk.metadata
        )

        runtime_metadata = SemanticSplitMetadata(
            semantic_split_materialized=True,
            semantic_split_was_split=result.was_split,
            semantic_split_paragraph_index=(
                fragment.paragraph_index
            ),
        )

        if fragment.oversized_sentence:
            runtime_metadata[
                "oversized_sentence"
            ] = True

        if fragment.oversized_atomic:
            runtime_metadata[
                "oversized_atomic"
            ] = True

        if fragment.requires_quarantine:
            runtime_metadata[
                "requires_quarantine"
            ] = True

        metadata.update(
            runtime_metadata
        )

        if result.was_split:
            provenance = build_split_provenance(
                identity=identity,
                original_token_count=(
                    result.original_token_count
                ),
                split_token_count=(
                    fragment.token_count
                ),
                split_reason=(
                    fragment.split_reason
                ),
                config=(
                    self.config
                    .split_provenance
                ),
                identity_config=(
                    self.config
                    .split_identity
                ),
            )

            metadata.update(
                provenance
            )

        elif fragment.split_reason:
            metadata[
                "split_reason"
            ] = fragment.split_reason

        return metadata


#--------------------IDENTITY RESOLUTION--------------------

    @staticmethod
    def _parent_chunk_id(
        chunk: Chunk,
    ) -> str:
        chunk_id = chunk.chunk_id

        if not isinstance(
            chunk_id,
            str,
        ):
            raise SplitMaterializationError(
                "Chunk has no valid chunk_id"
            )

        normalized = chunk_id.strip()

        if not normalized:
            raise SplitMaterializationError(
                "Chunk has no valid chunk_id"
            )

        return normalized


    @staticmethod
    def _source_ir_id(
        chunk: Chunk,
        *,
        parent_chunk_id: str,
    ) -> str:
        candidates = (
            chunk.metadata.get(
                "source_ir_id"
            ),
            chunk.metadata.get(
                "ir_id"
            ),
        )

        for candidate in candidates:
            if (
                isinstance(
                    candidate,
                    str,
                )
                and candidate.strip()
            ):
                return candidate.strip()

        return parent_chunk_id


#--------------------SUBTYPE RESOLUTION--------------------

    @staticmethod
    def _resolve_subtype(
        chunk: Chunk,
    ) -> str:
        candidates = (
            chunk.subtype,
            chunk.metadata.get(
                "subtype"
            ),
        )

        for candidate in candidates:
            if not isinstance(
                candidate,
                str,
            ):
                continue

            normalized = (
                candidate
                .strip()
                .lower()
            )

            if normalized:
                return normalized

        return "unknown"


#--------------------DENSITY RESOLUTION--------------------

    @staticmethod
    def _resolve_density(
        chunk: Chunk,
        *,
        density: str | None,
    ) -> str:
        candidate = density

        if candidate is None:
            candidate = chunk.metadata.get(
                "density"
            )

        if not isinstance(
            candidate,
            str,
        ):
            return "low"

        normalized = (
            candidate
            .strip()
            .lower()
        )

        if normalized not in {
            "low",
            "medium",
            "high",
        }:
            return "low"

        return normalized


#--------------------DOUBLE APPLICATION GUARD--------------------

    @staticmethod
    def _ensure_not_materialized(
        chunk: Chunk,
    ) -> None:
        if (
            chunk.metadata.get(
                "semantic_split_materialized"
            )
            is True
        ):
            raise SplitMaterializationError(
                "Semantic splitting has already been "
                f"materialized for chunk {chunk.chunk_id!r}"
            )


#--------------------VALIDATION--------------------

    @staticmethod
    def _validate_chunk(
        chunk: Chunk,
    ) -> None:
        if not isinstance(
            chunk,
            Chunk,
        ):
            raise TypeError(
                "ChunkSplitMaterializer requires Chunk objects"
            )


    @staticmethod
    def _validate_materialized_children(
        *,
        parent: Chunk,
        children: Sequence[Chunk],
        result: SemanticSplitResult,
    ) -> None:
        if len(
            children
        ) != len(
            result.fragments
        ):
            raise SplitMaterializationError(
                "Materialized child count does not match "
                "semantic fragment count"
            )

        child_ids = [
            child.chunk_id
            for child in children
        ]

        if any(
            not child_id
            for child_id in child_ids
        ):
            raise SplitMaterializationError(
                "Materialized split child has no identity"
            )

        if len(
            set(
                child_ids
            )
        ) != len(
            child_ids
        ):
            raise SplitMaterializationError(
                "Materialized split children contain "
                "duplicate chunk identities"
            )

        if (
            result.was_split
            and parent.chunk_id in child_ids
        ):
            raise SplitMaterializationError(
                "Split child retained the parent chunk id"
            )


#--------------------CONVENIENCE API--------------------

def materialize_chunk_splits(
    chunk: Chunk,
    *,
    config: ChunkingConfig | None = None,
    splitter: SemanticTextSplitter | None = None,
    density: str | None = None,
) -> tuple[Chunk, ...]:
    materializer = ChunkSplitMaterializer(
        config=config,
        splitter=splitter,
    )

    return materializer.materialize(
        chunk,
        density=density,
    )


def materialize_chunk_splits_many(
    chunks: Sequence[Chunk],
    *,
    config: ChunkingConfig | None = None,
    splitter: SemanticTextSplitter | None = None,
) -> tuple[Chunk, ...]:
    materializer = ChunkSplitMaterializer(
        config=config,
        splitter=splitter,
    )

    return materializer.materialize_many(
        chunks
    )