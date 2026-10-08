# rag/ingestion/data_transformation/chunking/validator.py

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from typing import TypedDict

from .config import get_chunk_token_budget
from .config import get_chunking_config
from .models import Chunk
from .config import ChunkingConfig
from .config import ChunkTokenBudget
from .tokenization import Tokenizer
from .tokenization import get_tokenizer


#--------------------ERRORS--------------------

class ChunkValidationError(RuntimeError):
    pass


#--------------------METADATA--------------------

class ValidationMetadata(TypedDict, total=False):
    validated: bool
    validation_status: str
    validation_reasons: list[str]
    token_count_corrected: bool
    previous_token_count: int
    exact_token_count: int


#--------------------RESULT MODELS--------------------

@dataclass(frozen=True)
class RejectedChunk:
    chunk: Chunk
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class QuarantinedChunk:
    chunk: Chunk
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ChunkValidationResult:
    accepted: tuple[Chunk, ...]
    quarantined: tuple[QuarantinedChunk, ...]
    rejected: tuple[RejectedChunk, ...]

    @property
    def accepted_count(
        self,
    ) -> int:
        return len(
            self.accepted
        )

    @property
    def quarantined_count(
        self,
    ) -> int:
        return len(
            self.quarantined
        )

    @property
    def rejected_count(
        self,
    ) -> int:
        return len(
            self.rejected
        )


#--------------------VALIDATOR--------------------

class ChunkValidator:
    def __init__(
        self,
        *,
        config: ChunkingConfig | None = None,
        budget: ChunkTokenBudget | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else get_chunking_config()
        )

        self.budget = (
            budget
            if budget is not None
            else get_chunk_token_budget(
                self.config
            )
        )

        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else get_tokenizer(
                role=self.config.tokenization.role
            )
        )


#--------------------PUBLIC VALIDATION--------------------

    def validate(
        self,
        chunks: Sequence[Chunk],
    ) -> ChunkValidationResult:
        if not chunks:
            return ChunkValidationResult(
                accepted=(),
                quarantined=(),
                rejected=(),
            )

        if not (
            self.config
            .validation
            .enabled
        ):
            return ChunkValidationResult(
                accepted=tuple(
                    chunks
                ),
                quarantined=(),
                rejected=(),
            )

        accepted: list[Chunk] = []
        quarantined: list[
            QuarantinedChunk
        ] = []
        rejected: list[
            RejectedChunk
        ] = []

        seen_chunk_ids: dict[
            str,
            str
        ] = {}

        seen_deduplication_keys: set[
            tuple[Any, ...]
        ] = set()

        for chunk in chunks:
            validated_chunk = (
                self._synchronize_token_count(
                    chunk
                )
            )

            reasons = self._structural_reasons(
                validated_chunk
            )

            if self._requires_quarantine(
                validated_chunk
            ):
                quarantine_reasons = list(
                    reasons
                )

                quarantine_reasons.append(
                    "requires_quarantine"
                )

                quarantined_chunk = (
                    self._with_validation_metadata(
                        validated_chunk,
                        validated=False,
                        status="quarantined",
                        reasons=quarantine_reasons,
                    )
                )

                quarantined.append(
                    QuarantinedChunk(
                        chunk=quarantined_chunk,
                        reasons=tuple(
                            quarantine_reasons
                        ),
                    )
                )

                continue

            self._assert_hard_limit(
                validated_chunk
            )

            if reasons:
                rejected_chunk = (
                    self._with_validation_metadata(
                        validated_chunk,
                        validated=False,
                        status="rejected",
                        reasons=reasons,
                    )
                )

                rejected.append(
                    RejectedChunk(
                        chunk=rejected_chunk,
                        reasons=tuple(
                            reasons
                        ),
                    )
                )

                continue

            self._assert_unique_chunk_identity(
                validated_chunk,
                seen=seen_chunk_ids,
            )

            if self._is_duplicate(
                validated_chunk,
                seen=seen_deduplication_keys,
            ):
                duplicate_reason = (
                    "duplicate_chunk_text"
                )

                rejected_chunk = (
                    self._with_validation_metadata(
                        validated_chunk,
                        validated=False,
                        status="rejected",
                        reasons=[
                            duplicate_reason
                        ],
                    )
                )

                rejected.append(
                    RejectedChunk(
                        chunk=rejected_chunk,
                        reasons=(
                            duplicate_reason,
                        ),
                    )
                )

                continue

            accepted.append(
                self._with_validation_metadata(
                    validated_chunk,
                    validated=True,
                    status="accepted",
                    reasons=[],
                )
            )

        return ChunkValidationResult(
            accepted=tuple(
                accepted
            ),
            quarantined=tuple(
                quarantined
            ),
            rejected=tuple(
                rejected
            ),
        )


#--------------------TOKEN VALIDATION--------------------

    def _synchronize_token_count(
        self,
        chunk: Chunk,
    ) -> Chunk:
        if not isinstance(
            chunk,
            Chunk,
        ):
            raise TypeError(
                "ChunkValidator requires Chunk objects"
            )

        if not (
            self.config
            .validation
            .validate_token_counts
        ):
            return chunk

        exact_count = self.tokenizer.count(
            chunk.text,
            add_special_tokens=(
                self.config
                .tokenization
                .add_special_tokens
            ),
        )

        if chunk.tokens == exact_count:
            return chunk

        metadata = dict(
            chunk.metadata
        )

        metadata[
            "token_count_corrected"
        ] = True

        metadata[
            "previous_token_count"
        ] = chunk.tokens

        metadata[
            "exact_token_count"
        ] = exact_count

        return chunk.copy_with(
            tokens=exact_count,
            metadata=metadata,
        )


#--------------------STRUCTURAL VALIDATION--------------------

    def _structural_reasons(
        self,
        chunk: Chunk,
    ) -> list[str]:
        reasons: list[str] = []

        validation = (
            self.config
            .validation
        )

        if (
            validation.reject_empty_text
            and not chunk.text.strip()
        ):
            reasons.append(
                "empty_text"
            )

        if (
            validation.reject_missing_identity
            and not self._valid_identity(
                chunk.chunk_id
            )
        ):
            reasons.append(
                "missing_chunk_identity"
            )

        if (
            validation.validate_split_identity
        ):
            reasons.extend(
                self._split_identity_reasons(
                    chunk
                )
            )

        if (
            validation.validate_split_provenance
        ):
            reasons.extend(
                self._split_provenance_reasons(
                    chunk
                )
            )

        if (
            validation.strict_limit_is_error
            and chunk.tokens
            > self.budget.strict_tokens
            and not self._allowed_above_strict(
                chunk
            )
        ):
            reasons.append(
                "strict_token_limit_exceeded"
            )

        return reasons


#--------------------SPLIT IDENTITY--------------------

    def _split_identity_reasons(
        self,
        chunk: Chunk,
    ) -> list[str]:
        metadata = chunk.metadata

        split_count = self._metadata_integer(
            metadata,
            "split_count",
        )

        if split_count is None:
            return []

        if split_count <= 1:
            return []

        reasons: list[str] = []

        split_index = self._metadata_integer(
            metadata,
            "split_index",
        )

        parent_chunk_id = metadata.get(
            "parent_chunk_id"
        )

        if not self._valid_identity(
            parent_chunk_id
        ):
            reasons.append(
                "missing_parent_chunk_id"
            )

        source_ir_id = metadata.get(
            "source_ir_id"
        )

        if not self._valid_identity(
            source_ir_id
        ):
            reasons.append(
                "missing_source_ir_id"
            )

        if split_index is None:
            reasons.append(
                "missing_split_index"
            )
        else:
            start = (
                self.config
                .split_identity
                .split_index_start
            )

            maximum = (
                start
                + split_count
                - 1
            )

            if not (
                start
                <= split_index
                <= maximum
            ):
                reasons.append(
                    "invalid_split_index"
                )

        if (
            isinstance(
                parent_chunk_id,
                str,
            )
            and parent_chunk_id.strip()
            and chunk.chunk_id
            == parent_chunk_id.strip()
        ):
            reasons.append(
                "split_child_retained_parent_id"
            )

        return reasons


#--------------------SPLIT PROVENANCE--------------------

    def _split_provenance_reasons(
        self,
        chunk: Chunk,
    ) -> list[str]:
        metadata = chunk.metadata

        split_count = self._metadata_integer(
            metadata,
            "split_count",
        )

        if (
            split_count is None
            or split_count <= 1
        ):
            return []

        config = (
            self.config
            .split_provenance
        )

        if not config.enabled:
            return []

        reasons: list[str] = []

        if (
            config.record_parent_chunk_id
            and not self._valid_identity(
                metadata.get(
                    "parent_chunk_id"
                )
            )
        ):
            reasons.append(
                "missing_split_parent_provenance"
            )

        if (
            config.record_source_ir_id
            and not self._valid_identity(
                metadata.get(
                    "source_ir_id"
                )
            )
        ):
            reasons.append(
                "missing_split_source_ir_provenance"
            )

        if (
            config.record_split_index
            and self._metadata_integer(
                metadata,
                "split_index",
            )
            is None
        ):
            reasons.append(
                "missing_split_index_provenance"
            )

        if (
            config.record_split_count
            and self._metadata_integer(
                metadata,
                "split_count",
            )
            is None
        ):
            reasons.append(
                "missing_split_count_provenance"
            )

        if (
            config.record_original_token_count
            and self._metadata_integer(
                metadata,
                "original_token_count",
            )
            is None
        ):
            reasons.append(
                "missing_original_token_count"
            )

        if (
            config.record_split_token_count
            and self._metadata_integer(
                metadata,
                "split_token_count",
            )
            is None
        ):
            reasons.append(
                "missing_split_token_count"
            )

        if (
            config.record_split_reason
            and not self._valid_identity(
                metadata.get(
                    "split_reason"
                )
            )
        ):
            reasons.append(
                "missing_split_reason"
            )

        split_token_count = (
            self._metadata_integer(
                metadata,
                "split_token_count",
            )
        )

        if (
            split_token_count is not None
            and split_token_count
            != chunk.tokens
        ):
            reasons.append(
                "split_token_count_mismatch"
            )

        return reasons


#--------------------HARD LIMIT--------------------

    def _assert_hard_limit(
        self,
        chunk: Chunk,
    ) -> None:
        if not (
            self.config
            .validation
            .enforce_model_hard_limit
        ):
            return

        hard_limit = (
            self.budget
            .effective_chunk_hard_tokens
        )

        if chunk.tokens <= hard_limit:
            return

        if (
            self.config
            .validation
            .hard_limit_is_error
        ):
            raise ChunkValidationError(
                "Unquarantined chunk exceeds the "
                "effective embedding hard limit: "
                f"chunk_id={chunk.chunk_id!r}, "
                f"tokens={chunk.tokens}, "
                f"hard_limit={hard_limit}"
            )


#--------------------QUARANTINE--------------------

    @staticmethod
    def _requires_quarantine(
        chunk: Chunk,
    ) -> bool:
        return (
            chunk.metadata.get(
                "requires_quarantine"
            )
            is True
        )


#--------------------STRICT LIMIT--------------------

    def _allowed_above_strict(
        self,
        chunk: Chunk,
    ) -> bool:
        metadata = chunk.metadata

        if (
            metadata.get(
                "oversized_sentence"
            )
            is True
            and self.config
            .hard_limit_policy
            .allow_sentence_above_strict
        ):
            return True

        if (
            metadata.get(
                "oversized_atomic"
            )
            is True
            and self.config
            .hard_limit_policy
            .allow_atomic_above_strict
        ):
            return True

        return False


#--------------------IDENTITY UNIQUENESS--------------------

    @staticmethod
    def _assert_unique_chunk_identity(
        chunk: Chunk,
        *,
        seen: dict[str, str],
    ) -> None:
        if not chunk.chunk_id:
            return

        previous_text = seen.get(
            chunk.chunk_id
        )

        if previous_text is None:
            seen[
                chunk.chunk_id
            ] = chunk.text

            return

        if previous_text != chunk.text:
            raise ChunkValidationError(
                "Two different chunks share the same "
                f"chunk_id: {chunk.chunk_id!r}"
            )


#--------------------DEDUPLICATION--------------------

    def _is_duplicate(
        self,
        chunk: Chunk,
        *,
        seen: set[tuple[Any, ...]],
    ) -> bool:
        validation = (
            self.config
            .validation
        )

        dedup = (
            self.config
            .text_deduplication
        )

        if not validation.deduplicate:
            return False

        if not dedup.enabled:
            return False

        if dedup.scope == "none":
            return False

        key = self._dedupe_key(
            chunk
        )

        if key in seen:
            return True

        seen.add(
            key
        )

        return False


    def _dedupe_key(
        self,
        chunk: Chunk,
    ) -> tuple[Any, ...]:
        dedup = (
            self.config
            .text_deduplication
        )

        if (
            self.config
            .validation
            .deduplicate_by_ir_id
        ):
            ir_id = chunk.metadata.get(
                "ir_id"
            )

            if self._valid_identity(
                ir_id
            ):
                return (
                    "ir_id",
                    str(
                        ir_id
                    ).strip(),
                )

        text = self._normalize_dedupe_text(
            chunk.text
        )

        key: list[Any] = [
            "text",
            text,
        ]

        if dedup.scope == "section":
            key.extend(
                [
                    "section",
                    chunk.metadata.get(
                        "section_id"
                    ),
                ]
            )

        if (
            dedup.preserve_distinct_split_children
            and self._is_split_child(
                chunk
            )
        ):
            key.extend(
                [
                    "split_child",
                    chunk.chunk_id,
                ]
            )

        if (
            dedup.preserve_distinct_source_spans
        ):
            source_span = self._source_span_key(
                chunk
            )

            if source_span is not None:
                key.extend(
                    [
                        "source_span",
                        source_span,
                    ]
                )

        return tuple(
            key
        )


    def _normalize_dedupe_text(
        self,
        text: str,
    ) -> str:
        config = (
            self.config
            .text_deduplication
        )

        value = text

        if config.normalize_outer_whitespace:
            value = value.strip()

        if config.normalize_internal_whitespace:
            value = re.sub(
                r"\s+",
                " ",
                value,
            )

        if not config.case_sensitive:
            value = value.casefold()

        return value


#--------------------SOURCE SPAN--------------------

    @staticmethod
    def _source_span_key(
        chunk: Chunk,
    ) -> tuple[Any, ...] | None:
        metadata = chunk.metadata

        direct_span = metadata.get(
            "source_span"
        )

        if direct_span is not None:
            return (
                "source_span",
                repr(
                    direct_span
                ),
            )

        source = metadata.get(
            "source"
        )

        if isinstance(
            source,
            dict,
        ):
            fragments = source.get(
                "fragments"
            )

            if isinstance(
                fragments,
                list,
            ):
                normalized_fragments: list[
                    tuple[Any, ...]
                ] = []

                for fragment in fragments:
                    if not isinstance(
                        fragment,
                        dict,
                    ):
                        continue

                    normalized_fragments.append(
                        (
                            fragment.get(
                                "block_index"
                            ),
                            fragment.get(
                                "line_start"
                            ),
                            fragment.get(
                                "line_end"
                            ),
                        )
                    )

                if normalized_fragments:
                    return (
                        "source_fragments",
                        tuple(
                            normalized_fragments
                        ),
                    )

        start = ChunkValidator._first_not_none(
            metadata.get(
                "source_start"
            ),
            metadata.get(
                "start_offset"
            ),
            metadata.get(
                "start_line"
            ),
        )

        end = ChunkValidator._first_not_none(
            metadata.get(
                "source_end"
            ),
            metadata.get(
                "end_offset"
            ),
            metadata.get(
                "end_line"
            ),
        )

        if (
            start is None
            and end is None
        ):
            return None

        return (
            start,
            end,
        )


    @staticmethod
    def _first_not_none(
        *values: Any,
    ) -> Any:
        for value in values:
            if value is not None:
                return value

        return None


#--------------------VALIDATION METADATA--------------------

    @staticmethod
    def _with_validation_metadata(
        chunk: Chunk,
        *,
        validated: bool,
        status: str,
        reasons: Sequence[str],
    ) -> Chunk:
        metadata = dict(
            chunk.metadata
        )

        validation_metadata = ValidationMetadata(
            validated=validated,
            validation_status=status,
            validation_reasons=list(
                reasons
            ),
        )

        metadata.update(
            validation_metadata
        )

        return chunk.copy_with(
            metadata=metadata
        )


#--------------------HELPERS--------------------

    @staticmethod
    def _valid_identity(
        value: Any,
    ) -> bool:
        return (
            isinstance(
                value,
                str,
            )
            and bool(
                value.strip()
            )
        )


    @staticmethod
    def _metadata_integer(
        metadata: dict[str, Any],
        key: str,
    ) -> int | None:
        value = metadata.get(
            key
        )

        if isinstance(
            value,
            bool,
        ):
            return None

        if not isinstance(
            value,
            int,
        ):
            return None

        return value


    @staticmethod
    def _is_split_child(
        chunk: Chunk,
    ) -> bool:
        split_count = (
            ChunkValidator
            ._metadata_integer(
                chunk.metadata,
                "split_count",
            )
        )

        return (
            split_count is not None
            and split_count > 1
        )


#--------------------COMPATIBILITY API--------------------

def validate_chunks_detailed(
    chunks: Sequence[Chunk],
    *,
    config: ChunkingConfig | None = None,
) -> ChunkValidationResult:
    validator = ChunkValidator(
        config=config
    )

    return validator.validate(
        chunks
    )


def validate_chunks(
    chunks: Sequence[Chunk],
    *,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    result = validate_chunks_detailed(
        chunks,
        config=config,
    )

    return list(
        result.accepted
    )