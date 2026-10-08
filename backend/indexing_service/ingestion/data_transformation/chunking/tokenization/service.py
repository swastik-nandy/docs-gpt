from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..config import get_chunking_config
from .base import BatchTokenCount
from .base import TokenizedText
from .base import Tokenizer
from .local_tokenizer import LocalTokenizer


# --------- MODEL CONTRACT ------------------------

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

BACKEND_ROOT = (
    Path(__file__)
    .resolve()
    .parents[4]
)

TOKENIZER_DIR = (
    BACKEND_ROOT
    / ".models"
    / "qwen3-embedding-0.6b"
)

TOKENIZER_PATH = (
    TOKENIZER_DIR
    / "tokenizer.json"
)


# --------- POLICY ------------------------

def _resolve_role(
    role: str | None,
) -> str:
    configured = (
        get_chunking_config()
        .tokenization
        .role
        if role is None
        else role
    )

    normalized = str(
        configured
    ).strip().lower()

    if normalized != "dense":
        raise ValueError(
            "Only the dense Qwen3 tokenizer "
            "is supported by this ingestion service"
        )

    return normalized


def _resolve_add_special_tokens(
    value: bool | None,
) -> bool:
    if value is not None:
        return value

    return (
        get_chunking_config()
        .tokenization
        .add_special_tokens
    )


# --------- TOKENIZER SINGLETON ------------------------

@lru_cache(maxsize=1)
def _load_tokenizer() -> Tokenizer:
    if not TOKENIZER_PATH.is_file():
        raise RuntimeError(
            "Qwen3 tokenizer is missing. Run "
            "scripts/download_qwen3_06B_tokenizer.py first. "
            f"Expected: {TOKENIZER_PATH}"
        )

    return LocalTokenizer(
        model_id=MODEL_ID,
        model_path=TOKENIZER_DIR,
    )


def get_tokenizer(
    *,
    role: str | None = None,
) -> Tokenizer:
    _resolve_role(
        role
    )

    return _load_tokenizer()


# --------- TOKEN COUNTING ------------------------

def count_tokens(
    text: str,
    *,
    role: str | None = None,
    add_special_tokens: bool | None = None,
) -> int:
    tokenizer = get_tokenizer(
        role=role
    )

    return tokenizer.count(
        text,
        add_special_tokens=(
            _resolve_add_special_tokens(
                add_special_tokens
            )
        ),
    )


def count_tokens_many(
    texts: list[str] | tuple[str, ...],
    *,
    role: str | None = None,
    add_special_tokens: bool | None = None,
) -> BatchTokenCount:
    tokenizer = get_tokenizer(
        role=role
    )

    return tokenizer.count_many(
        texts,
        add_special_tokens=(
            _resolve_add_special_tokens(
                add_special_tokens
            )
        ),
    )


# --------- TOKENIZATION ------------------------

def tokenize_text(
    text: str,
    *,
    role: str | None = None,
    add_special_tokens: bool | None = None,
) -> TokenizedText:
    tokenizer = get_tokenizer(
        role=role
    )

    return tokenizer.tokenize(
        text,
        add_special_tokens=(
            _resolve_add_special_tokens(
                add_special_tokens
            )
        ),
    )


def encode_tokens(
    text: str,
    *,
    role: str | None = None,
    add_special_tokens: bool | None = None,
) -> tuple[int, ...]:
    tokenizer = get_tokenizer(
        role=role
    )

    return tokenizer.encode(
        text,
        add_special_tokens=(
            _resolve_add_special_tokens(
                add_special_tokens
            )
        ),
    )


def encode_tokens_many(
    texts: list[str] | tuple[str, ...],
    *,
    role: str | None = None,
    add_special_tokens: bool | None = None,
) -> tuple[tuple[int, ...], ...]:
    tokenizer = get_tokenizer(
        role=role
    )

    return tokenizer.encode_many(
        texts,
        add_special_tokens=(
            _resolve_add_special_tokens(
                add_special_tokens
            )
        ),
    )


# --------- CACHE CONTROL ------------------------

def clear_tokenizer_cache() -> None:
    _load_tokenizer.cache_clear()
