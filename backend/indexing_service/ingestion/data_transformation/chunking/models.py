# rag/ingestion/data_transformation/chunking/models.py

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import TypedDict


#--------------------METADATA SHAPES--------------------

class SplitMetadata(TypedDict, total=False):
    parent_chunk_id: str
    source_ir_id: str
    split_index: int
    split_count: int
    original_token_count: int
    split_token_count: int
    split_reason: str


class ChunkQualityMetadata(TypedDict, total=False):
    quality_score: float
    validated: bool
    oversized_sentence: bool
    oversized_atomic: bool
    requires_quarantine: bool


#--------------------CHUNK MODEL--------------------

@dataclass
class Chunk:
    text: str
    chunk_type: str

    heading: str = ""
    context: str = ""

    subtype: str = ""
    order: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    tokens: int = 0
    source: str = ""

    chunk_id: str | None = None


#--------------------NORMALIZATION--------------------

    def __post_init__(
        self,
    ) -> None:
        self.text = self._normalize_text(
            self.text
        )

        self.chunk_type = self._normalize_required_text(
            self.chunk_type,
            field_name="chunk_type",
        )

        self.heading = self._normalize_text(
            self.heading
        )

        self.context = self._normalize_text(
            self.context
        )

        self.source = self._normalize_text(
            self.source
        )

        self.metadata = self._normalize_metadata(
            self.metadata
        )

        self.subtype = self._resolve_subtype()

        self.order = self._normalize_non_negative_integer(
            self.order,
            field_name="order",
        )

        self.tokens = self._normalize_non_negative_integer(
            self.tokens,
            field_name="tokens",
        )

        self._synchronize_metadata()

        if self.chunk_id is not None:
            self.chunk_id = self._normalize_optional_identity(
                self.chunk_id
            )

        if not self.chunk_id:
            self.chunk_id = self._generate_id()

        self.metadata[
            "chunk_id"
        ] = self.chunk_id


#--------------------IDENTITY--------------------

    def _generate_id(
        self,
    ) -> str:
        explicit_ir_id = self._metadata_identity(
            "ir_id"
        )

        if explicit_ir_id:
            return explicit_ir_id

        parser_unit_id = self._metadata_identity(
            "parser_unit_id"
        )

        section_id = self._metadata_identity(
            "section_id"
        )

        source_doc_id = (
            self._metadata_identity(
                "parser_source_doc_id"
            )
            or self._metadata_identity(
                "source_doc_id"
            )
        )

        identity_parts = (
            self.source,
            source_doc_id,
            parser_unit_id,
            section_id,
            self.chunk_type,
            self.subtype,
            str(
                self.order
            ),
            self.heading,
            self.context,
            self.text,
        )

        payload = "\x1f".join(
            identity_parts
        )

        digest = hashlib.sha256(
            payload.encode(
                "utf-8"
            )
        ).hexdigest()

        return (
            "chunk_"
            f"{digest}"
        )


    def _metadata_identity(
        self,
        key: str,
    ) -> str:
        value = self.metadata.get(
            key
        )

        if not isinstance(
            value,
            str,
        ):
            return ""

        return value.strip()


#--------------------METADATA--------------------

    def _synchronize_metadata(
        self,
    ) -> None:
        if self.heading:
            self.metadata.setdefault(
                "heading",
                self.heading,
            )

        if self.context:
            self.metadata.setdefault(
                "context",
                self.context,
            )

        if self.subtype:
            self.metadata[
                "subtype"
            ] = self.subtype

        self.metadata[
            "chunk_type"
        ] = self.chunk_type

        self.metadata[
            "order"
        ] = self.order

        self.metadata[
            "tokens"
        ] = self.tokens


    def merged_metadata(
        self,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(
            self.metadata
        )

        merged.update(
            values
        )

        return merged


#--------------------COPYING--------------------

    def copy_with(
        self,
        *,
        text: str | None = None,
        chunk_type: str | None = None,
        heading: str | None = None,
        context: str | None = None,
        subtype: str | None = None,
        order: int | None = None,
        metadata: dict[str, Any] | None = None,
        tokens: int | None = None,
        source: str | None = None,
        chunk_id: str | None = None,
        preserve_chunk_id: bool = True,
    ) -> Chunk:
        resolved_chunk_id = (
            self.chunk_id
            if (
                preserve_chunk_id
                and chunk_id is None
            )
            else chunk_id
        )

        return Chunk(
            text=(
                self.text
                if text is None
                else text
            ),
            chunk_type=(
                self.chunk_type
                if chunk_type is None
                else chunk_type
            ),
            heading=(
                self.heading
                if heading is None
                else heading
            ),
            context=(
                self.context
                if context is None
                else context
            ),
            subtype=(
                self.subtype
                if subtype is None
                else subtype
            ),
            order=(
                self.order
                if order is None
                else order
            ),
            metadata=(
                dict(
                    self.metadata
                )
                if metadata is None
                else dict(
                    metadata
                )
            ),
            tokens=(
                self.tokens
                if tokens is None
                else tokens
            ),
            source=(
                self.source
                if source is None
                else source
            ),
            chunk_id=resolved_chunk_id,
        )


#--------------------SERIALIZATION--------------------

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "id": self.chunk_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "heading": self.heading,
            "context": self.context,
            "type": self.chunk_type,
            "chunk_type": self.chunk_type,
            "subtype": self.subtype,
            "order": self.order,
            "tokens": self.tokens,
            "source": self.source,
            "metadata": dict(
                self.metadata
            ),
        }


#--------------------NORMALIZATION HELPERS--------------------

    def _resolve_subtype(
        self,
    ) -> str:
        explicit = self._normalize_text(
            self.subtype
        ).lower()

        if explicit:
            return explicit

        metadata_subtype = self.metadata.get(
            "subtype"
        )

        if not isinstance(
            metadata_subtype,
            str,
        ):
            return ""

        return (
            metadata_subtype
            .strip()
            .lower()
        )


    @staticmethod
    def _normalize_metadata(
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if metadata is None:
            return {}

        if not isinstance(
            metadata,
            dict,
        ):
            raise TypeError(
                "Chunk metadata must be a dictionary"
            )

        return dict(
            metadata
        )


    @staticmethod
    def _normalize_text(
        value: str | None,
    ) -> str:
        if value is None:
            return ""

        if not isinstance(
            value,
            str,
        ):
            raise TypeError(
                "Chunk text fields must be strings"
            )

        return value.strip()


    @classmethod
    def _normalize_required_text(
        cls,
        value: str | None,
        *,
        field_name: str,
    ) -> str:
        normalized = cls._normalize_text(
            value
        )

        if not normalized:
            raise ValueError(
                f"{field_name} cannot be empty"
            )

        return normalized


    @staticmethod
    def _normalize_optional_identity(
        value: str,
    ) -> str | None:
        if not isinstance(
            value,
            str,
        ):
            raise TypeError(
                "chunk_id must be a string"
            )

        normalized = value.strip()

        return (
            normalized
            or None
        )


    @staticmethod
    def _normalize_non_negative_integer(
        value: int,
        *,
        field_name: str,
    ) -> int:
        if isinstance(
            value,
            bool,
        ):
            raise TypeError(
                f"{field_name} must be an integer"
            )

        if not isinstance(
            value,
            int,
        ):
            raise TypeError(
                f"{field_name} must be an integer"
            )

        if value < 0:
            raise ValueError(
                f"{field_name} cannot be negative"
            )

        return value


#--------------------DEBUGGING--------------------

    def __repr__(
        self,
    ) -> str:
        preview = (
            self.text
            .replace(
                "\n",
                " ",
            )[:50]
        )

        identity = (
            self.chunk_id[:12]
            if self.chunk_id
            else "none"
        )

        return (
            "<Chunk "
            f"id={identity} "
            f"type={self.chunk_type}/{self.subtype} "
            f"tokens={self.tokens} "
            f"text={preview!r}>"
        )