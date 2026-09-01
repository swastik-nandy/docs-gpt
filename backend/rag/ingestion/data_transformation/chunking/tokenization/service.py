# rag/ingestion/data_transformation/chunking/tokenization/service.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import TypedDict

from ..config import get_chunking_config
from .base import BatchTokenCount
from .base import TokenizedText
from .base import Tokenizer
from .local_tokenizer import LocalTokenizer
from .registry import LocalModelAsset
from .registry import resolve_local_model_asset


#--------------------RUNTIME MODELS--------------------

@dataclass(frozen=True)
class TokenizerCacheKey:
    role: str
    model_id: str
    revision: str
    tokenizer_path: Path


@dataclass(frozen=True)
class TokenizerRuntime:
    asset: LocalModelAsset
    tokenizer: Tokenizer


class TokenizerRuntimeMetadata(TypedDict):
    role: str
    model_id: str
    revision: str
    model_path: str
    tokenizer_path: str
    offline_runtime: bool


#--------------------CACHE STATE--------------------

_TOKENIZER_CACHE: dict[
    TokenizerCacheKey,
    Tokenizer,
] = {}

_TOKENIZER_CACHE_LOCK = RLock()


#--------------------CACHE HELPERS--------------------

def _build_cache_key(
    asset: LocalModelAsset,
) -> TokenizerCacheKey:
    return TokenizerCacheKey(
        role=asset.role,
        model_id=asset.model_id,
        revision=asset.revision,
        tokenizer_path=asset.tokenizer_path,
    )


def _remove_stale_role_entries(
    role: str,
    *,
    keep: TokenizerCacheKey,
) -> None:
    stale_keys = [
        key
        for key in _TOKENIZER_CACHE
        if (
            key.role == role
            and key != keep
        )
    ]

    for key in stale_keys:
        _TOKENIZER_CACHE.pop(
            key,
            None,
        )


def _get_or_load_tokenizer(
    asset: LocalModelAsset,
) -> Tokenizer:
    key = _build_cache_key(
        asset
    )

    with _TOKENIZER_CACHE_LOCK:
        cached = _TOKENIZER_CACHE.get(
            key
        )

        if cached is not None:
            return cached

        _remove_stale_role_entries(
            asset.role,
            keep=key,
        )

        tokenizer = LocalTokenizer(
            model_id=asset.model_id,
            revision=asset.revision,
            model_path=asset.tokenizer_path,
        )

        _TOKENIZER_CACHE[
            key
        ] = tokenizer

        return tokenizer


#--------------------POLICY RESOLUTION--------------------

def _resolve_role(
    role: str | None,
) -> str:
    if role is None:
        role = (
            get_chunking_config()
            .tokenization
            .role
        )

    normalized = str(
        role
    ).strip().lower()

    if normalized not in {
        "dense",
        "sparse",
    }:
        raise ValueError(
            "Tokenizer role must be "
            "'dense' or 'sparse'"
        )

    return normalized


def _resolve_add_special_tokens(
    add_special_tokens: bool | None,
) -> bool:
    if add_special_tokens is not None:
        return add_special_tokens

    return (
        get_chunking_config()
        .tokenization
        .add_special_tokens
    )


#--------------------RUNTIME RESOLUTION--------------------

def get_tokenizer_runtime(
    *,
    role: str | None = None,
    models_root: str | Path | None = None,
) -> TokenizerRuntime:
    normalized_role = _resolve_role(
        role
    )

    asset = resolve_local_model_asset(
        role=normalized_role,
        models_root=models_root,
    )

    tokenizer = _get_or_load_tokenizer(
        asset
    )

    return TokenizerRuntime(
        asset=asset,
        tokenizer=tokenizer,
    )


def get_tokenizer(
    *,
    role: str | None = None,
    models_root: str | Path | None = None,
) -> Tokenizer:
    return get_tokenizer_runtime(
        role=role,
        models_root=models_root,
    ).tokenizer


#--------------------TOKEN COUNTING--------------------

def count_tokens(
    text: str,
    *,
    role: str | None = None,
    add_special_tokens: bool | None = None,
) -> int:
    tokenizer = get_tokenizer(
        role=role
    )

    include_special_tokens = (
        _resolve_add_special_tokens(
            add_special_tokens
        )
    )

    return tokenizer.count(
        text,
        add_special_tokens=include_special_tokens,
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

    include_special_tokens = (
        _resolve_add_special_tokens(
            add_special_tokens
        )
    )

    return tokenizer.count_many(
        texts,
        add_special_tokens=include_special_tokens,
    )


#--------------------TOKENIZATION--------------------

def tokenize_text(
    text: str,
    *,
    role: str | None = None,
    add_special_tokens: bool | None = None,
) -> TokenizedText:
    tokenizer = get_tokenizer(
        role=role
    )

    include_special_tokens = (
        _resolve_add_special_tokens(
            add_special_tokens
        )
    )

    return tokenizer.tokenize(
        text,
        add_special_tokens=include_special_tokens,
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

    include_special_tokens = (
        _resolve_add_special_tokens(
            add_special_tokens
        )
    )

    return tokenizer.encode(
        text,
        add_special_tokens=include_special_tokens,
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

    include_special_tokens = (
        _resolve_add_special_tokens(
            add_special_tokens
        )
    )

    return tokenizer.encode_many(
        texts,
        add_special_tokens=include_special_tokens,
    )


#--------------------DIAGNOSTICS--------------------

def tokenizer_runtime_metadata(
    *,
    role: str | None = None,
) -> TokenizerRuntimeMetadata:
    runtime = get_tokenizer_runtime(
        role=role
    )

    asset = runtime.asset

    return TokenizerRuntimeMetadata(
        role=asset.role,
        model_id=asset.model_id,
        revision=asset.revision,
        model_path=str(
            asset.model_path
        ),
        tokenizer_path=str(
            asset.tokenizer_path
        ),
        offline_runtime=asset.offline_runtime,
    )


#--------------------CACHE CONTROL--------------------

def clear_tokenizer_cache() -> None:
    with _TOKENIZER_CACHE_LOCK:
        _TOKENIZER_CACHE.clear()