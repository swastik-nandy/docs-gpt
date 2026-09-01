# rag/ingestion/data_transformation/chunking/semantic_splitter.py

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .config import get_chunk_token_budget
from .config import get_chunking_config
from .config import ChunkingConfig
from .config import ChunkTokenBudget
from .tokenization import Tokenizer
from .tokenization import get_tokenizer


#--------------------ERRORS--------------------

class SemanticSplitError(RuntimeError):
    pass


#--------------------RESULT MODELS--------------------

@dataclass(frozen=True)
class SemanticSplitFragment:
    text: str
    token_count: int
    paragraph_index: int
    split_reason: str | None = None
    oversized_sentence: bool = False
    oversized_atomic: bool = False
    requires_quarantine: bool = False


@dataclass(frozen=True)
class SemanticSplitResult:
    original_text: str
    original_token_count: int
    subtype: str
    fragments: tuple[SemanticSplitFragment, ...]

    @property
    def was_split(
        self,
    ) -> bool:
        return len(
            self.fragments
        ) > 1

    @property
    def requires_quarantine(
        self,
    ) -> bool:
        return any(
            fragment.requires_quarantine
            for fragment in self.fragments
        )


#--------------------INTERNAL MODELS--------------------

@dataclass(frozen=True)
class _Sentence:
    text: str
    token_count: int


@dataclass(frozen=True)
class _Limits:
    preferred: int
    strict: int
    hard: int


#--------------------SENTENCE BOUNDARIES--------------------

_SENTENCE_TERMINATORS = frozenset(
    {
        ".",
        "!",
        "?",
    }
)

_SENTENCE_CLOSERS = frozenset(
    {
        '"',
        "'",
        "”",
        "’",
        ")",
        "]",
        "}",
    }
)

_COMMON_ABBREVIATIONS = frozenset(
    {
        "e.g.",
        "i.e.",
        "etc.",
        "vs.",
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "sr.",
        "jr.",
        "st.",
        "no.",
        "fig.",
        "eq.",
    }
)

_PARAGRAPH_BOUNDARY = re.compile(
    r"\n[ \t]*\n+"
)


#--------------------PUBLIC SPLITTER--------------------

class SemanticTextSplitter:
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


    def split(
        self,
        text: str,
        *,
        subtype: str,
        density: str = "low",
    ) -> SemanticSplitResult:
        source_text = self._validate_text(
            text
        )

        normalized_subtype = self._normalize_subtype(
            subtype
        )

        original_token_count = (
            self.tokenizer.count(
                source_text,
                add_special_tokens=(
                    self.config
                    .tokenization
                    .add_special_tokens
                ),
            )
        )

        if not source_text:
            return SemanticSplitResult(
                original_text=source_text,
                original_token_count=0,
                subtype=normalized_subtype,
                fragments=(),
            )

        limits = self._resolve_limits(
            subtype=normalized_subtype,
            density=density,
        )

        if self._is_atomic(
            normalized_subtype
        ):
            fragments = self._handle_atomic(
                text=source_text,
                token_count=original_token_count,
                limits=limits,
            )

            return self._result(
                text=source_text,
                token_count=original_token_count,
                subtype=normalized_subtype,
                fragments=fragments,
            )

        if (
            original_token_count
            <= limits.strict
        ):
            return self._result(
                text=source_text,
                token_count=original_token_count,
                subtype=normalized_subtype,
                fragments=(
                    SemanticSplitFragment(
                        text=source_text,
                        token_count=original_token_count,
                        paragraph_index=0,
                    ),
                ),
            )

        if not self._is_splittable(
            normalized_subtype
        ):
            fragments = self._handle_unsplittable(
                text=source_text,
                token_count=original_token_count,
                limits=limits,
            )

            return self._result(
                text=source_text,
                token_count=original_token_count,
                subtype=normalized_subtype,
                fragments=fragments,
            )

        fragments = self._split_paragraphs(
            source_text,
            limits=limits,
        )

        return self._result(
            text=source_text,
            token_count=original_token_count,
            subtype=normalized_subtype,
            fragments=fragments,
        )


#--------------------LIMIT RESOLUTION--------------------

    def _resolve_limits(
        self,
        *,
        subtype: str,
        density: str,
    ) -> _Limits:
        ratio = (
            self.config
            .block_budget
            .ratio_for(
                subtype,
                density,
            )
        )

        preferred = max(
            1,
            int(
                self.budget.preferred_tokens
                * ratio
            ),
        )

        strict = max(
            preferred,
            int(
                self.budget.strict_tokens
                * ratio
            ),
        )

        strict = min(
            strict,
            self.budget.effective_chunk_hard_tokens,
        )

        preferred = min(
            preferred,
            strict,
        )

        return _Limits(
            preferred=preferred,
            strict=strict,
            hard=(
                self.budget
                .effective_chunk_hard_tokens
            ),
        )


#--------------------ATOMIC BLOCKS--------------------

    def _handle_atomic(
        self,
        *,
        text: str,
        token_count: int,
        limits: _Limits,
    ) -> tuple[SemanticSplitFragment, ...]:
        if token_count <= limits.hard:
            return (
                SemanticSplitFragment(
                    text=text,
                    token_count=token_count,
                    paragraph_index=0,
                    oversized_atomic=(
                        token_count
                        > limits.strict
                    ),
                    split_reason=(
                        "atomic_above_strict"
                        if token_count > limits.strict
                        else None
                    ),
                ),
            )

        if (
            self.config
            .atomic_policy
            .quarantine_if_above_hard_limit
        ):
            return (
                SemanticSplitFragment(
                    text=text,
                    token_count=token_count,
                    paragraph_index=0,
                    split_reason="atomic_above_hard_limit",
                    oversized_atomic=True,
                    requires_quarantine=True,
                ),
            )

        raise SemanticSplitError(
            "Atomic block exceeds the effective embedding "
            f"hard limit: {token_count} > {limits.hard}"
        )


#--------------------UNSPLITTABLE BLOCKS--------------------

    def _handle_unsplittable(
        self,
        *,
        text: str,
        token_count: int,
        limits: _Limits,
    ) -> tuple[SemanticSplitFragment, ...]:
        if token_count <= limits.hard:
            return (
                SemanticSplitFragment(
                    text=text,
                    token_count=token_count,
                    paragraph_index=0,
                    split_reason="unsplittable_above_strict",
                ),
            )

        return (
            self._quarantine_fragment(
                text=text,
                token_count=token_count,
                paragraph_index=0,
                reason="unsplittable_above_hard_limit",
            ),
        )


#--------------------PARAGRAPH SPLITTING--------------------

    def _split_paragraphs(
        self,
        text: str,
        *,
        limits: _Limits,
    ) -> tuple[SemanticSplitFragment, ...]:
        paragraphs = tuple(
            paragraph
            for paragraph in (
                part.strip()
                for part in _PARAGRAPH_BOUNDARY.split(
                    text
                )
            )
            if paragraph
        )

        if not paragraphs:
            return ()

        counts = (
            self.tokenizer.count_many(
                paragraphs,
                add_special_tokens=(
                    self.config
                    .tokenization
                    .add_special_tokens
                ),
            )
            .counts
        )

        fragments: list[
            SemanticSplitFragment
        ] = []

        for paragraph_index, (
            paragraph,
            token_count,
        ) in enumerate(
            zip(
                paragraphs,
                counts,
            )
        ):
            if token_count <= limits.strict:
                fragments.append(
                    SemanticSplitFragment(
                        text=paragraph,
                        token_count=token_count,
                        paragraph_index=paragraph_index,
                        split_reason="paragraph_boundary",
                    )
                )
                continue

            fragments.extend(
                self._split_paragraph_sentences(
                    paragraph,
                    paragraph_index=paragraph_index,
                    limits=limits,
                )
            )

        return tuple(
            fragments
        )


#--------------------SENTENCE SPLITTING--------------------

    def _split_paragraph_sentences(
        self,
        paragraph: str,
        *,
        paragraph_index: int,
        limits: _Limits,
    ) -> tuple[SemanticSplitFragment, ...]:
        if not (
            self.config
            .sentence_splitting
            .enabled
        ):
            token_count = self.tokenizer.count(
                paragraph,
                add_special_tokens=(
                    self.config
                    .tokenization
                    .add_special_tokens
                ),
            )

            return (
                self._quarantine_if_required(
                    text=paragraph,
                    token_count=token_count,
                    paragraph_index=paragraph_index,
                    limits=limits,
                    reason="sentence_splitting_disabled",
                ),
            )

        sentence_texts = self._sentence_boundaries(
            paragraph
        )

        counts = (
            self.tokenizer.count_many(
                sentence_texts,
                add_special_tokens=(
                    self.config
                    .tokenization
                    .add_special_tokens
                ),
            )
            .counts
        )

        sentences = tuple(
            _Sentence(
                text=sentence,
                token_count=token_count,
            )
            for sentence, token_count in zip(
                sentence_texts,
                counts,
            )
        )

        return self._pack_sentences(
            sentences,
            paragraph_index=paragraph_index,
            limits=limits,
        )


    def _pack_sentences(
        self,
        sentences: Sequence[_Sentence],
        *,
        paragraph_index: int,
        limits: _Limits,
    ) -> tuple[SemanticSplitFragment, ...]:
        fragments: list[
            SemanticSplitFragment
        ] = []

        current: list[_Sentence] = []

        for sentence in sentences:
            if sentence.token_count > limits.hard:
                self._flush_sentence_buffer(
                    current,
                    fragments,
                    paragraph_index=paragraph_index,
                )

                current.clear()

                fragments.append(
                    self._quarantine_fragment(
                        text=sentence.text,
                        token_count=sentence.token_count,
                        paragraph_index=paragraph_index,
                        reason="sentence_above_hard_limit",
                        oversized_sentence=True,
                    )
                )
                continue

            if sentence.token_count > limits.strict:
                self._flush_sentence_buffer(
                    current,
                    fragments,
                    paragraph_index=paragraph_index,
                )

                current.clear()

                fragments.append(
                    SemanticSplitFragment(
                        text=sentence.text,
                        token_count=sentence.token_count,
                        paragraph_index=paragraph_index,
                        split_reason="sentence_above_strict",
                        oversized_sentence=True,
                    )
                )
                continue

            candidate = (
                current
                + [
                    sentence
                ]
            )

            candidate_text = self._join_sentences(
                candidate
            )

            candidate_count = self.tokenizer.count(
                candidate_text,
                add_special_tokens=(
                    self.config
                    .tokenization
                    .add_special_tokens
                ),
            )

            if (
                current
                and candidate_count > limits.preferred
            ):
                self._flush_sentence_buffer(
                    current,
                    fragments,
                    paragraph_index=paragraph_index,
                )

                current.clear()

            current.append(
                sentence
            )

        self._flush_sentence_buffer(
            current,
            fragments,
            paragraph_index=paragraph_index,
        )

        return tuple(
            fragments
        )


#--------------------BUFFER HELPERS--------------------

    def _flush_sentence_buffer(
        self,
        buffer: Sequence[_Sentence],
        output: list[SemanticSplitFragment],
        *,
        paragraph_index: int,
    ) -> None:
        if not buffer:
            return

        text = self._join_sentences(
            buffer
        )

        token_count = self.tokenizer.count(
            text,
            add_special_tokens=(
                self.config
                .tokenization
                .add_special_tokens
            ),
        )

        output.append(
            SemanticSplitFragment(
                text=text,
                token_count=token_count,
                paragraph_index=paragraph_index,
                split_reason="sentence_boundary",
            )
        )


    @staticmethod
    def _join_sentences(
        sentences: Sequence[_Sentence],
    ) -> str:
        return " ".join(
            sentence.text.strip()
            for sentence in sentences
            if sentence.text.strip()
        )


#--------------------BOUNDARY DETECTION--------------------

    def _sentence_boundaries(
        self,
        text: str,
    ) -> tuple[str, ...]:
        value = text.strip()

        if not value:
            return ()

        boundaries: list[str] = []
        start = 0
        index = 0
        length = len(
            value
        )

        while index < length:
            char = value[
                index
            ]

            if (
                char
                not in _SENTENCE_TERMINATORS
            ):
                index += 1
                continue

            if not self._is_sentence_end(
                value,
                index,
            ):
                index += 1
                continue

            end = index + 1

            while (
                end < length
                and value[end]
                in _SENTENCE_CLOSERS
            ):
                end += 1

            sentence = value[
                start:end
            ].strip()

            if sentence:
                boundaries.append(
                    sentence
                )

            while (
                end < length
                and value[end].isspace()
            ):
                end += 1

            start = end
            index = end

        tail = value[
            start:
        ].strip()

        if tail:
            boundaries.append(
                tail
            )

        return tuple(
            boundaries
        )


    def _is_sentence_end(
        self,
        text: str,
        index: int,
    ) -> bool:
        char = text[
            index
        ]

        if char in {
            "!",
            "?",
        }:
            return True

        if self._is_decimal_point(
            text,
            index,
        ):
            return False

        token = self._preceding_token(
            text,
            index,
        ).lower()

        if token in _COMMON_ABBREVIATIONS:
            return False

        next_index = index + 1

        while (
            next_index < len(
                text
            )
            and text[next_index]
            in _SENTENCE_CLOSERS
        ):
            next_index += 1

        if next_index >= len(
            text
        ):
            return True

        return text[
            next_index
        ].isspace()


    @staticmethod
    def _is_decimal_point(
        text: str,
        index: int,
    ) -> bool:
        if index <= 0:
            return False

        if (
            index + 1
            >= len(
                text
            )
        ):
            return False

        return (
            text[
                index - 1
            ].isdigit()
            and text[
                index + 1
            ].isdigit()
        )


    @staticmethod
    def _preceding_token(
        text: str,
        index: int,
    ) -> str:
        start = index

        while (
            start > 0
            and not text[
                start - 1
            ].isspace()
        ):
            start -= 1

        return text[
            start:index + 1
        ]


#--------------------QUARANTINE--------------------

    def _quarantine_if_required(
        self,
        *,
        text: str,
        token_count: int,
        paragraph_index: int,
        limits: _Limits,
        reason: str,
    ) -> SemanticSplitFragment:
        if token_count <= limits.hard:
            return SemanticSplitFragment(
                text=text,
                token_count=token_count,
                paragraph_index=paragraph_index,
                split_reason=reason,
            )

        return self._quarantine_fragment(
            text=text,
            token_count=token_count,
            paragraph_index=paragraph_index,
            reason=reason,
        )


    def _quarantine_fragment(
        self,
        *,
        text: str,
        token_count: int,
        paragraph_index: int,
        reason: str,
        oversized_sentence: bool = False,
    ) -> SemanticSplitFragment:
        if not (
            self.config
            .hard_limit_policy
            .quarantine
        ):
            raise SemanticSplitError(
                "Text exceeds the effective embedding "
                "hard limit and quarantine is disabled"
            )

        return SemanticSplitFragment(
            text=text,
            token_count=token_count,
            paragraph_index=paragraph_index,
            split_reason=reason,
            oversized_sentence=oversized_sentence,
            requires_quarantine=True,
        )


#--------------------POLICY HELPERS--------------------

    def _is_atomic(
        self,
        subtype: str,
    ) -> bool:
        return (
            self.config
            .atomic_policy
            .enabled
            and subtype
            in self.config
            .atomic_policy
            .subtypes
        )


    def _is_splittable(
        self,
        subtype: str,
    ) -> bool:
        return (
            self.config
            .splittable_policy
            .enabled
            and subtype
            in self.config
            .splittable_policy
            .subtypes
        )


#--------------------INPUT HELPERS--------------------

    @staticmethod
    def _validate_text(
        text: str,
    ) -> str:
        if not isinstance(
            text,
            str,
        ):
            raise TypeError(
                "Semantic splitter input must be a string"
            )

        return text.strip()


    @staticmethod
    def _normalize_subtype(
        subtype: str,
    ) -> str:
        if not isinstance(
            subtype,
            str,
        ):
            raise TypeError(
                "Chunk subtype must be a string"
            )

        normalized = (
            subtype
            .strip()
            .lower()
        )

        if not normalized:
            raise ValueError(
                "Chunk subtype cannot be empty"
            )

        return normalized


#--------------------RESULT BUILDING--------------------


    @staticmethod
    def _result(
        *,
        text: str,
        token_count: int,
        subtype: str,
        fragments: Sequence[
            SemanticSplitFragment
        ],
    ) -> SemanticSplitResult:
        return SemanticSplitResult(
            original_text=text,
            original_token_count=token_count,
            subtype=subtype,
            fragments=tuple(
                fragments
            ),
        )