# rag/ingestion/data_transformation/chunking/tokenization/local_tokenizer.py

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from transformers import AutoTokenizer

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

        backend = (
            "transformers_fast_local"
            if bool(
                getattr(
                    self._tokenizer,
                    "is_fast",
                    False,
                )
            )
            else "transformers_local"
        )

        self._identity = TokenizerIdentity(
            backend=backend,
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
            token_ids = self._tokenizer.encode(
                normalized,
                add_special_tokens=add_special_tokens,
            )
        except Exception as exc:
            raise TokenizerEncodingError(
                "Failed to encode text with local tokenizer "
                f"{self.identity.tokenizer_id!r}"
            ) from exc

        return self._normalize_token_ids(
            token_ids
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
            encoded = self._tokenizer(
                list(
                    normalized_texts
                ),
                add_special_tokens=add_special_tokens,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
        except Exception as exc:
            raise TokenizerEncodingError(
                "Failed to batch encode text with local tokenizer "
                f"{self.identity.tokenizer_id!r}"
            ) from exc

        raw_input_ids = encoded.get(
            "input_ids"
        )

        if not isinstance(
            raw_input_ids,
            list,
        ):
            raise TokenizerEncodingError(
                "Local tokenizer batch encoding did not "
                "return input_ids"
            )

        if len(
            raw_input_ids
        ) != len(
            normalized_texts
        ):
            raise TokenizerEncodingError(
                "Local tokenizer batch encoding returned "
                "an unexpected number of sequences"
            )

        return tuple(
            self._normalize_token_ids(
                token_ids
            )
            for token_ids in raw_input_ids
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
                normalized_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )
        except Exception as exc:
            raise TokenizerDecodingError(
                "Failed to decode tokens with local tokenizer "
                f"{self.identity.tokenizer_id!r}"
            ) from exc


#--------------------LOADING--------------------

    def _load_tokenizer(
        self,
    ):
        try:
            return AutoTokenizer.from_pretrained(
                str(
                    self._model_path
                ),
                use_fast=True,
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as exc:
            raise TokenizerUnavailableError(
                "Unable to load local tokenizer for "
                f"{self._model_id!r} from "
                f"{self._model_path}. "
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