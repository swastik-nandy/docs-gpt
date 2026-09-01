# rag/ingestion/data_transformation/chunking/tokenization/base.py

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Sequence


#--------------------ERRORS--------------------

class TokenizerError(RuntimeError):
    pass


class TokenizerUnavailableError(
    TokenizerError
):
    pass


class TokenizerEncodingError(
    TokenizerError
):
    pass


class TokenizerDecodingError(
    TokenizerError
):
    pass


#--------------------IDENTITY--------------------

@dataclass(frozen=True)
class TokenizerIdentity:
    backend: str
    tokenizer_id: str

    def __post_init__(
        self,
    ) -> None:
        if not self.backend.strip():
            raise ValueError(
                "Tokenizer backend cannot be empty"
            )

        if not self.tokenizer_id.strip():
            raise ValueError(
                "Tokenizer id cannot be empty"
            )


#--------------------TOKEN RESULTS--------------------

@dataclass(frozen=True)
class TokenizedText:
    text: str
    token_ids: tuple[int, ...]

    @property
    def token_count(
        self,
    ) -> int:
        return len(
            self.token_ids
        )


@dataclass(frozen=True)
class BatchTokenCount:
    counts: tuple[int, ...]

    def __post_init__(
        self,
    ) -> None:
        for count in self.counts:
            if isinstance(
                count,
                bool,
            ):
                raise TypeError(
                    "Token counts must be integers"
                )

            if not isinstance(
                count,
                int,
            ):
                raise TypeError(
                    "Token counts must be integers"
                )

            if count < 0:
                raise ValueError(
                    "Token counts cannot be negative"
                )

    @property
    def item_count(
        self,
    ) -> int:
        return len(
            self.counts
        )

    @property
    def total_tokens(
        self,
    ) -> int:
        return sum(
            self.counts
        )

    @property
    def maximum_tokens(
        self,
    ) -> int:
        if not self.counts:
            return 0

        return max(
            self.counts
        )

    @property
    def minimum_tokens(
        self,
    ) -> int:
        if not self.counts:
            return 0

        return min(
            self.counts
        )

    @property
    def average_tokens(
        self,
    ) -> float:
        if not self.counts:
            return 0.0

        return (
            self.total_tokens
            / self.item_count
        )


#--------------------TOKENIZER CONTRACT--------------------

class Tokenizer(ABC):
    @property
    @abstractmethod
    def identity(
        self,
    ) -> TokenizerIdentity:
        raise NotImplementedError


    @abstractmethod
    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> tuple[int, ...]:
        raise NotImplementedError


    @abstractmethod
    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        raise NotImplementedError


    def count(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> int:
        return len(
            self.encode(
                text,
                add_special_tokens=add_special_tokens,
            )
        )


    def tokenize(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> TokenizedText:
        normalized = self.normalize_text(
            text
        )

        token_ids = self.encode(
            normalized,
            add_special_tokens=add_special_tokens,
        )

        return TokenizedText(
            text=normalized,
            token_ids=token_ids,
        )


    def encode_many(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = False,
    ) -> tuple[tuple[int, ...], ...]:
        return tuple(
            self.encode(
                text,
                add_special_tokens=add_special_tokens,
            )
            for text in texts
        )


    def count_many(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = False,
    ) -> BatchTokenCount:
        encoded = self.encode_many(
            texts,
            add_special_tokens=add_special_tokens,
        )

        return BatchTokenCount(
            counts=tuple(
                len(
                    token_ids
                )
                for token_ids in encoded
            )
        )


#--------------------TEXT NORMALIZATION--------------------

    @staticmethod
    def normalize_text(
        text: str,
    ) -> str:
        if not isinstance(
            text,
            str,
        ):
            raise TypeError(
                "Tokenizer input must be a string"
            )

        if not text:
            return ""

        return text


#--------------------IDENTITY VALIDATION--------------------

def assert_tokenizer_identity(
    tokenizer: Tokenizer,
    *,
    expected_tokenizer_id: str | None = None,
    expected_backend: str | None = None,
) -> None:
    identity = tokenizer.identity

    if expected_tokenizer_id is not None:
        expected_id = (
            expected_tokenizer_id
            .strip()
        )

        if not expected_id:
            raise ValueError(
                "Expected tokenizer id cannot be empty"
            )

        if (
            identity.tokenizer_id
            != expected_id
        ):
            raise TokenizerUnavailableError(
                "Tokenizer identity mismatch: "
                f"expected {expected_id!r}, "
                f"got {identity.tokenizer_id!r}"
            )

    if expected_backend is not None:
        expected = (
            expected_backend
            .strip()
        )

        if not expected:
            raise ValueError(
                "Expected tokenizer backend cannot be empty"
            )

        if (
            identity.backend
            != expected
        ):
            raise TokenizerUnavailableError(
                "Tokenizer backend mismatch: "
                f"expected {expected!r}, "
                f"got {identity.backend!r}"
            )