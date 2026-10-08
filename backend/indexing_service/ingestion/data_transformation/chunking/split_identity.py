# rag/ingestion/data_transformation/chunking/split_identity.py

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from .config import SplitIdentityConfig
from .config import SplitProvenanceConfig


#--------------------ERRORS--------------------

class SplitIdentityError(RuntimeError):
    pass


#--------------------METADATA--------------------

class SplitProvenanceMetadata(TypedDict, total=False):
    parent_chunk_id: str
    source_ir_id: str
    split_index: int
    split_count: int
    original_token_count: int
    split_token_count: int
    split_reason: str


#--------------------RUNTIME MODEL--------------------

@dataclass(frozen=True)
class SplitChildIdentity:
    chunk_id: str
    parent_chunk_id: str
    source_ir_id: str
    split_index: int
    split_count: int

    @property
    def is_split_child(
        self,
    ) -> bool:
        return self.split_count > 1


#--------------------IDENTITY BUILDING--------------------

def build_split_child_identity(
    *,
    parent_chunk_id: str,
    source_ir_id: str,
    fragment_position: int,
    split_count: int,
    config: SplitIdentityConfig,
) -> SplitChildIdentity:
    normalized_parent_id = _require_identity(
        parent_chunk_id,
        field="parent_chunk_id",
    )

    normalized_source_ir_id = _require_identity(
        source_ir_id,
        field="source_ir_id",
    )

    if fragment_position < 0:
        raise SplitIdentityError(
            "fragment_position cannot be negative"
        )

    if split_count <= 0:
        raise SplitIdentityError(
            "split_count must be greater than zero"
        )

    if fragment_position >= split_count:
        raise SplitIdentityError(
            "fragment_position must be smaller than split_count"
        )

    split_index = (
        config.split_index_start
        + fragment_position
    )

    if (
        config.enabled
        and config.generate_unique_child_ids
        and split_count > 1
    ):
        chunk_id = _render_child_id(
            parent_chunk_id=normalized_parent_id,
            split_index=split_index,
            template=config.child_id_template,
        )
    else:
        chunk_id = normalized_parent_id

    return SplitChildIdentity(
        chunk_id=chunk_id,
        parent_chunk_id=normalized_parent_id,
        source_ir_id=normalized_source_ir_id,
        split_index=split_index,
        split_count=split_count,
    )


#--------------------PROVENANCE BUILDING--------------------

def build_split_provenance(
    *,
    identity: SplitChildIdentity,
    original_token_count: int,
    split_token_count: int,
    split_reason: str | None,
    config: SplitProvenanceConfig,
    identity_config: SplitIdentityConfig,
) -> SplitProvenanceMetadata:
    metadata = SplitProvenanceMetadata()

    if not config.enabled:
        return metadata

    _validate_token_count(
        original_token_count,
        field="original_token_count",
    )

    _validate_token_count(
        split_token_count,
        field="split_token_count",
    )

    if (
        identity_config.preserve_parent_chunk_id
        and config.record_parent_chunk_id
    ):
        metadata[
            "parent_chunk_id"
        ] = identity.parent_chunk_id

    if (
        identity_config.preserve_source_ir_id
        and config.record_source_ir_id
    ):
        metadata[
            "source_ir_id"
        ] = identity.source_ir_id

    if config.record_split_index:
        metadata[
            "split_index"
        ] = identity.split_index

    if config.record_split_count:
        metadata[
            "split_count"
        ] = identity.split_count

    if config.record_original_token_count:
        metadata[
            "original_token_count"
        ] = original_token_count

    if config.record_split_token_count:
        metadata[
            "split_token_count"
        ] = split_token_count

    if (
        config.record_split_reason
        and split_reason
    ):
        metadata[
            "split_reason"
        ] = split_reason

    return metadata


#--------------------CHILD ID RENDERING--------------------

def _render_child_id(
    *,
    parent_chunk_id: str,
    split_index: int,
    template: str,
) -> str:
    try:
        child_id = template.format(
            parent_id=parent_chunk_id,
            split_index=split_index,
        )
    except Exception as exc:
        raise SplitIdentityError(
            "Unable to render split child id from template"
        ) from exc

    normalized = child_id.strip()

    if not normalized:
        raise SplitIdentityError(
            "Rendered split child id cannot be empty"
        )

    if normalized == parent_chunk_id:
        raise SplitIdentityError(
            "Split child id must differ from parent chunk id"
        )

    return normalized


#--------------------VALIDATION--------------------

def _require_identity(
    value: str,
    *,
    field: str,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{field} must be a string"
        )

    normalized = value.strip()

    if not normalized:
        raise SplitIdentityError(
            f"{field} cannot be empty"
        )

    return normalized


def _validate_token_count(
    value: int,
    *,
    field: str,
) -> None:
    if isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{field} must be an integer"
        )

    if not isinstance(
        value,
        int,
    ):
        raise TypeError(
            f"{field} must be an integer"
        )

    if value < 0:
        raise SplitIdentityError(
            f"{field} cannot be negative"
        )