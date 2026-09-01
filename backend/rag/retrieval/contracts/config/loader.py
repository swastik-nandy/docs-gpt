from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import rtoml

from rag.retrieval.contracts.config.models import RetrievalContractConfig
from rag.retrieval.contracts.config.parser import parse_retrieval_contract_config


DEFAULT_CONTRACT_CONFIG_PATH = Path(__file__).with_name("config.toml")


class RetrievalContractConfigLoadError(ValueError):
    """Raised when retrieval contract config cannot be loaded from disk."""


def load_retrieval_contract_config(
    config_path: str | Path | None = None,
) -> RetrievalContractConfig:
    path = Path(config_path) if config_path is not None else DEFAULT_CONTRACT_CONFIG_PATH
    raw = _load_toml(path)
    return parse_retrieval_contract_config(raw)


def get_retrieval_contract_config() -> RetrievalContractConfig:
    return _load_cached_default_config(str(DEFAULT_CONTRACT_CONFIG_PATH))


def clear_retrieval_contract_config_cache() -> None:
    _load_cached_default_config.cache_clear()


@lru_cache(maxsize=1)
def _load_cached_default_config(path_value: str) -> RetrievalContractConfig:
    return load_retrieval_contract_config(Path(path_value))


def _load_toml(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise RetrievalContractConfigLoadError(
            f"Retrieval contract config file does not exist: {path}"
        )

    if not path.is_file():
        raise RetrievalContractConfigLoadError(
            f"Retrieval contract config path is not a file: {path}"
        )

    try:
        data = rtoml.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RetrievalContractConfigLoadError(
            f"Failed to read or parse retrieval contract TOML config: {path}"
        ) from exc

    if not isinstance(data, Mapping):
        raise RetrievalContractConfigLoadError(
            "Retrieval contract config root must be a TOML table."
        )

    return data