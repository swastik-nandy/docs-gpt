from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

from rag.retrieval.contracts.candidates import CandidateSourceRef, RerankedCandidate
from rag.retrieval.contracts.config import (
    RetrievalContractConfig,
    get_retrieval_contract_config,
)
from rag.retrieval.contracts.primitives import (
    Payload,
    optional_payload_str,
    validate_finite_number,
    validate_mapping_size,
    validate_non_empty_str,
    validate_non_negative_int,
    validate_optional_clean_str,
    validate_optional_positive_int,
    validate_optional_sequence_limit,
    validate_positive_int,
)


@dataclass(frozen=True)
class ContextBlock:
    block_id: str
    text: str
    source_ref: CandidateSourceRef
    rank: int
    score: float
    token_count: int | None = None
    heading: str | None = None
    heading_path: tuple[str, ...] = field(default_factory=tuple)
    chunk_type: str | None = None
    source_title: str | None = None
    payload: Payload = field(default_factory=dict)
    metadata: Payload = field(default_factory=dict)

    @classmethod
    def from_reranked_candidate(
        cls,
        candidate: RerankedCandidate,
        *,
        token_count: int | None = None,
        metadata: Payload | None = None,
    ) -> Self:
        retrieval_candidate = candidate.candidate
        payload = retrieval_candidate.payload

        return cls(
            block_id=_context_block_id(retrieval_candidate.ref),
            text=retrieval_candidate.text,
            source_ref=retrieval_candidate.ref,
            rank=candidate.rank,
            score=candidate.rerank_score,
            token_count=token_count,
            heading=retrieval_candidate.heading,
            heading_path=retrieval_candidate.heading_path,
            chunk_type=retrieval_candidate.chunk_type,
            source_title=(
                optional_payload_str(payload, "source_title")
                or optional_payload_str(payload, "title")
            ),
            payload=dict(payload),
            metadata=metadata or {},
        )

    @property
    def point_id(self) -> str:
        return self.source_ref.point_id

    @property
    def chunk_id(self) -> str | None:
        return self.source_ref.chunk_id

    @property
    def source_id(self) -> str | None:
        return self.source_ref.source_id

    @property
    def citation_label(self) -> str:
        if self.source_title:
            return self.source_title

        if self.heading:
            return self.heading

        return self.chunk_id or self.point_id

    def normalized(self) -> Self:
        return type(self)(
            block_id=self.block_id.strip(),
            text=self.text.strip(),
            source_ref=self.source_ref.normalized(),
            rank=self.rank,
            score=float(self.score),
            token_count=self.token_count,
            heading=_clean_optional(self.heading),
            heading_path=tuple(
                item.strip() for item in self.heading_path if item.strip()
            ),
            chunk_type=_clean_optional(self.chunk_type),
            source_title=_clean_optional(self.source_title),
            payload=dict(self.payload),
            metadata=dict(self.metadata),
        )

    def validate(
        self,
        *,
        config: RetrievalContractConfig | None = None,
    ) -> None:
        resolved_config = config or get_retrieval_contract_config()
        max_metadata_keys = resolved_config.request_limits.max_metadata_keys

        validate_non_empty_str("context_block.block_id", self.block_id)
        validate_non_empty_str("context_block.text", self.text)

        if not isinstance(self.source_ref, CandidateSourceRef):
            raise ValueError("context_block.source_ref must be CandidateSourceRef.")

        self.source_ref.validate()

        validate_positive_int("context_block.rank", self.rank)
        validate_finite_number("context_block.score", self.score)
        validate_optional_positive_int("context_block.token_count", self.token_count)
        validate_optional_clean_str("context_block.heading", self.heading)
        validate_optional_clean_str("context_block.chunk_type", self.chunk_type)
        validate_optional_clean_str("context_block.source_title", self.source_title)

        validate_optional_sequence_limit(
            "context_block.heading_path",
            self.heading_path,
            resolved_config.source_constraints.max_heading_path_depth,
        )

        for index, heading in enumerate(self.heading_path):
            validate_non_empty_str(f"context_block.heading_path[{index}]", heading)

        validate_mapping_size("context_block.payload", self.payload, None)
        validate_mapping_size(
            "context_block.metadata",
            self.metadata,
            max_metadata_keys,
        )


@dataclass(frozen=True)
class PackedContext:
    query: str
    blocks: tuple[ContextBlock, ...]
    total_tokens: int
    max_total_tokens: int
    reserve_tokens_for_answer: int
    metadata: Payload = field(default_factory=dict)

    @property
    def has_blocks(self) -> bool:
        return bool(self.blocks)

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def remaining_token_budget(self) -> int:
        return max(
            0,
            self.max_total_tokens - self.reserve_tokens_for_answer - self.total_tokens,
        )

    @property
    def source_refs(self) -> tuple[CandidateSourceRef, ...]:
        return tuple(block.source_ref for block in self.blocks)

    def normalized(self) -> Self:
        return type(self)(
            query=self.query.strip(),
            blocks=tuple(block.normalized() for block in self.blocks),
            total_tokens=int(self.total_tokens),
            max_total_tokens=int(self.max_total_tokens),
            reserve_tokens_for_answer=int(self.reserve_tokens_for_answer),
            metadata=dict(self.metadata),
        )

    def validate(
        self,
        *,
        config: RetrievalContractConfig | None = None,
        allow_empty: bool = False,
    ) -> None:
        resolved_config = config or get_retrieval_contract_config()
        context_defaults = resolved_config.context_defaults
        context_limits = resolved_config.context_limits

        validate_non_empty_str("packed_context.query", self.query)
        validate_non_negative_int("packed_context.total_tokens", self.total_tokens)
        validate_positive_int("packed_context.max_total_tokens", self.max_total_tokens)
        validate_non_negative_int(
            "packed_context.reserve_tokens_for_answer",
            self.reserve_tokens_for_answer,
        )

        if (
            resolved_config.validation.reject_empty_context_blocks
            and not self.blocks
            and not allow_empty
        ):
            raise ValueError("packed_context.blocks cannot be empty.")

        validate_optional_sequence_limit(
            "packed_context.blocks",
            self.blocks,
            context_limits.max_context_blocks,
        )

        if self.max_total_tokens > context_limits.max_total_tokens:
            raise ValueError(
                "packed_context.max_total_tokens cannot exceed "
                f"{context_limits.max_total_tokens}."
            )

        if (
            self.reserve_tokens_for_answer
            > context_limits.max_reserve_tokens_for_answer
        ):
            raise ValueError(
                "packed_context.reserve_tokens_for_answer cannot exceed "
                f"{context_limits.max_reserve_tokens_for_answer}."
            )

        if self.total_tokens > self.max_total_tokens:
            raise ValueError(
                "packed_context.total_tokens cannot exceed "
                "packed_context.max_total_tokens."
            )

        usable_budget = self.max_total_tokens - self.reserve_tokens_for_answer
        if self.total_tokens > usable_budget:
            raise ValueError(
                "packed_context.total_tokens cannot exceed usable context budget."
            )

        if self.block_count > context_defaults.max_context_blocks:
            raise ValueError(
                "packed_context.blocks exceeds configured default max_context_blocks."
            )

        for index, block in enumerate(self.blocks):
            if not isinstance(block, ContextBlock):
                raise ValueError(f"packed_context.blocks[{index}] must be ContextBlock.")

            block.validate(config=resolved_config)

            if (
                block.token_count is not None
                and block.token_count > context_defaults.max_block_tokens
            ):
                raise ValueError(
                    f"packed_context.blocks[{index}].token_count exceeds "
                    "context.defaults.max_block_tokens."
                )

        validate_mapping_size(
            "packed_context.metadata",
            self.metadata,
            resolved_config.request_limits.max_metadata_keys,
        )


def build_empty_packed_context(
    *,
    query: str,
    config: RetrievalContractConfig | None = None,
    metadata: Payload | None = None,
) -> PackedContext:
    resolved_config = config or get_retrieval_contract_config()
    defaults = resolved_config.context_defaults

    return PackedContext(
        query=query,
        blocks=(),
        total_tokens=0,
        max_total_tokens=defaults.max_total_tokens,
        reserve_tokens_for_answer=defaults.reserve_tokens_for_answer,
        metadata=metadata or {},
    )


def _context_block_id(ref: CandidateSourceRef) -> str:
    return ref.chunk_id or ref.point_id


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None

    cleaned = value.strip()
    return cleaned or None