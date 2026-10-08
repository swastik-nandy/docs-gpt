# rag/ingestion/data_transformation/chunking/tokenization/local_tokenizer.py

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from tokenizers import Tokenizer as HuggingFaceTokenizer

from .base import BatchTokenCount
from .base import Tokenizer
from .base import TokenizerDecodingError
from .base import TokenizerEncodingError
from .base import TokenizerIdentity
from .base import TokenizerUnavailableError


#--------------------LOCAL TOKENIZER--------------------

class LocalTokenizer(Tokenizer):
    def __init__(
        self,
        *,
        model_id: str,
        model_path: str | Path,
        revision: str | None = None,
    ) -> None:
        self._model_id = self._normalize_model_id(
            model_id
        )

        self._model_path = (
            Path(
                model_path
            )
            .expanduser()
            .resolve()
        )

        self._revision = (
            revision.strip()
            if isinstance(
                revision,
                str,
            )
            and revision.strip()
            else None
        )

        self._validate_model_path()

        self._tokenizer = (
            self._load_tokenizer()
        )

        self._identity = TokenizerIdentity(
            backend="tokenizers_local",
            tokenizer_id=self._build_tokenizer_id(),
        )


#--------------------IDENTITY--------------------

    @property
    def identity(
        self,
    ) -> TokenizerIdentity:
        return self._identity

    @property
    def model_id(
        self,
    ) -> str:
        return self._model_id

    @property
    def model_path(
        self,
    ) -> Path:
        return self._model_path

    @property
    def revision(
        self,
    ) -> str | None:
        return self._revision


#--------------------ENCODING--------------------

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> tuple[int, ...]:
        normalized = self.normalize_text(
            text
        )

        if not normalized:
            return ()

        try:
            encoding = self._tokenizer.encode(
                normalized,
                add_special_tokens=add_special_tokens,
            )
        except Exception as exc:
            raise TokenizerEncodingError(
                "Failed to encode text with local tokenizer "
                f"{self.identity.tokenizer_id!r}"
            ) from exc

        return self._normalize_token_ids(
            encoding.ids
        )


    def encode_many(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = False,
    ) -> tuple[tuple[int, ...], ...]:
        if not texts:
            return ()

        normalized_texts = tuple(
            self.normalize_text(
                text
            )
            for text in texts
        )

        try:
            encodings = self._tokenizer.encode_batch(
                list(
                    normalized_texts
                ),
                add_special_tokens=add_special_tokens,
            )
        except Exception as exc:
            raise TokenizerEncodingError(
                "Failed to batch encode text with local tokenizer "
                f"{self.identity.tokenizer_id!r}"
            ) from exc

        if len(
            encodings
        ) != len(
            normalized_texts
        ):
            raise TokenizerEncodingError(
                "Local tokenizer batch encoding returned "
                "an unexpected number of sequences"
            )

        return tuple(
            self._normalize_token_ids(
                encoding.ids
            )
            for encoding in encodings
        )


#--------------------COUNTING--------------------

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


#--------------------DECODING--------------------

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        normalized_ids = tuple(
            int(
                token_id
            )
            for token_id in token_ids
        )

        if not normalized_ids:
            return ""

        try:
            return self._tokenizer.decode(
                list(
                    normalized_ids
                ),
                skip_special_tokens=skip_special_tokens,
            )
        except Exception as exc:
            raise TokenizerDecodingError(
                "Failed to decode tokens with local tokenizer "
                f"{self.identity.tokenizer_id!r}"
            ) from exc


#--------------------LOADING--------------------

    def _load_tokenizer(
        self,
    ) -> HuggingFaceTokenizer:
        tokenizer_path = (
            self._model_path
            / "tokenizer.json"
        )

        try:
            return HuggingFaceTokenizer.from_file(
                tokenizer_path.as_posix()
            )
        except Exception as exc:
            raise TokenizerUnavailableError(
                "Unable to load local tokenizer for "
                f"{self._model_id!r} from "
                f"{tokenizer_path}. "
                "Runtime tokenizer downloads are disabled."
            ) from exc


#--------------------VALIDATION--------------------

    def _validate_model_path(
        self,
    ) -> None:
        if not self._model_path.is_dir():
            raise TokenizerUnavailableError(
                "Local tokenizer directory does not exist: "
                f"{self._model_path}"
            )

        tokenizer_path = (
            self._model_path
            / "tokenizer.json"
        )

        if not tokenizer_path.is_file():
            raise TokenizerUnavailableError(
                "Local tokenizer.json does not exist: "
                f"{tokenizer_path}"
            )


    @staticmethod
    def _normalize_model_id(
        model_id: str,
    ) -> str:
        if not isinstance(
            model_id,
            str,
        ):
            raise TypeError(
                "model_id must be a string"
            )

        normalized = model_id.strip()

        if not normalized:
            raise ValueError(
                "model_id cannot be empty"
            )

        return normalized


    @staticmethod
    def _normalize_token_ids(
        token_ids: Sequence[int],
    ) -> tuple[int, ...]:
        try:
            return tuple(
                int(
                    token_id
                )
                for token_id in token_ids
            )
        except Exception as exc:
            raise TokenizerEncodingError(
                "Tokenizer returned invalid token ids"
            ) from exc


#--------------------IDENTITY HELPERS--------------------

    def _build_tokenizer_id(
        self,
    ) -> str:
        if self._revision:
            return (
                f"{self._model_id}"
                f"@{self._revision}"
            )

        return self._model_id
