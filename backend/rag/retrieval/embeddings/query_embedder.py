from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from rag.retrieval.contracts.embeddings import QueryEmbedding, SparseQueryVector
from rag.retrieval.contracts.query import NormalizedQuery
from rag.retrieval.gpu_runtime import (
    is_cuda_device,
    move_model_to_device,
    offload_model_to_cpu,
    retrieval_gpu_lease,
)


Payload = dict[str, Any]

_DOTENV_LOADED_PATHS: set[Path] = set()


@dataclass(frozen=True)
class QueryEmbeddingProviderConfig:
    backend: str = "flagembedding"
    model_name: str = ""
    dense_model_name: str = ""
    sparse_model_name: str = ""
    embedding_dimension: int | None = None
    use_fp16: bool = True
    device: str | None = None
    batch_size: int = 1
    max_length: int = 8192
    return_colbert_vecs: bool = False
    offload_after_query: bool = True

    @classmethod
    def from_env(
        cls,
        *,
        env_file: str | Path | None = None,
        load_dotenv: bool = True,
        override_env: bool = False,
    ) -> QueryEmbeddingProviderConfig:
        if load_dotenv:
            load_env_file(
                env_file=env_file,
                override=override_env,
            )

        model_name = _required_env("EMBEDDING_MODEL")
        sparse_model_name = _optional_env("SPARSE_EMBEDDING_MODEL", model_name)

        config = cls(
            backend=_optional_env("EMBEDDING_BACKEND", "flagembedding"),
            model_name=model_name,
            dense_model_name=_optional_env("DENSE_EMBEDDING_MODEL", model_name),
            sparse_model_name=sparse_model_name,
            embedding_dimension=_optional_env_int_or_none("EMBEDDING_VECTOR_SIZE"),
            use_fp16=_optional_env_bool("EMBEDDING_USE_FP16", True),
            device=_optional_env_or_none("EMBEDDING_DEVICE"),
            batch_size=_optional_env_int("EMBEDDING_BATCH_SIZE", 1),
            max_length=_optional_env_int("EMBEDDING_MAX_LENGTH", 8192),
            return_colbert_vecs=_optional_env_bool(
                "EMBEDDING_RETURN_COLBERT_VECS",
                False,
            ),
            offload_after_query=_optional_env_bool(
                "EMBEDDING_OFFLOAD_AFTER_QUERY",
                True,
            ),
        )
        config.validate()

        return config

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> QueryEmbeddingProviderConfig:
        model_name = _required_mapping_str(data, "model_name")
        sparse_model_name = _optional_mapping_str(
            data,
            "sparse_model_name",
            default=model_name,
        )

        config = cls(
            backend=_optional_mapping_str(data, "backend", default="flagembedding"),
            model_name=model_name,
            dense_model_name=_optional_mapping_str(
                data,
                "dense_model_name",
                default=model_name,
            ),
            sparse_model_name=sparse_model_name,
            embedding_dimension=_optional_mapping_int_or_none(
                data,
                "embedding_dimension",
                fallback_key="vector_size",
            ),
            use_fp16=_optional_mapping_bool(data, "use_fp16", default=True),
            device=_optional_mapping_str_or_none(data, "device"),
            batch_size=_optional_mapping_int(data, "batch_size", default=1),
            max_length=_optional_mapping_int(data, "max_length", default=8192),
            return_colbert_vecs=_optional_mapping_bool(
                data,
                "return_colbert_vecs",
                default=False,
            ),
            offload_after_query=_optional_mapping_bool(
                data,
                "offload_after_query",
                default=True,
            ),
        )
        config.validate()

        return config

    def validate(self) -> None:
        if not self.backend.strip():
            raise ValueError("backend cannot be empty.")

        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty.")

        if not self.dense_model_name.strip():
            raise ValueError("dense_model_name cannot be empty.")

        if not self.sparse_model_name.strip():
            raise ValueError("sparse_model_name cannot be empty.")

        if self.embedding_dimension is not None and self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be greater than zero.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        if self.max_length <= 0:
            raise ValueError("max_length must be greater than zero.")

        if not isinstance(self.use_fp16, bool):
            raise ValueError("use_fp16 must be boolean.")

        if not isinstance(self.return_colbert_vecs, bool):
            raise ValueError("return_colbert_vecs must be boolean.")

        if not isinstance(self.offload_after_query, bool):
            raise ValueError("offload_after_query must be boolean.")


@dataclass(frozen=True)
class RawQueryEmbedding:
    query: str
    dense_vector: list[float] | None
    sparse_indices: tuple[int, ...] = field(default_factory=tuple)
    sparse_values: tuple[float, ...] = field(default_factory=tuple)
    model_name: str = ""
    dense_model_name: str = ""
    sparse_model_name: str = ""
    embedding_dimension: int | None = None
    metadata: Payload = field(default_factory=dict)

    @property
    def has_dense(self) -> bool:
        return bool(self.dense_vector)

    @property
    def has_sparse(self) -> bool:
        return bool(self.sparse_indices) and bool(self.sparse_values)

    def validate(self) -> None:
        if not self.query.strip():
            raise ValueError("query cannot be empty.")

        if self.dense_vector is None and not self.has_sparse:
            raise ValueError("raw query embedding must contain dense or sparse data.")

        if self.dense_vector is not None:
            if not self.dense_vector:
                raise ValueError("dense_vector cannot be empty when provided.")

            for index, value in enumerate(self.dense_vector):
                if not isinstance(value, float):
                    raise ValueError(f"dense_vector[{index}] must be float.")

        if len(self.sparse_indices) != len(self.sparse_values):
            raise ValueError("sparse_indices and sparse_values must have the same length.")

        for index, token_id in enumerate(self.sparse_indices):
            if token_id < 0:
                raise ValueError(f"sparse_indices[{index}] cannot be negative.")

        for index, value in enumerate(self.sparse_values):
            if not isinstance(value, float):
                raise ValueError(f"sparse_values[{index}] must be float.")


class QueryEmbeddingProvider(Protocol):
    model_name: str
    dense_model_name: str
    sparse_model_name: str
    embedding_dimension: int | None

    def embed_query(self, query: str) -> RawQueryEmbedding:
        raise NotImplementedError


class FlagEmbeddingQueryProvider:
    """
    Retrieval-owned query embedding provider.

    Config is loaded from .env/env/TOML. The provider does not hardcode model names.
    """

    def __init__(
        self,
        config: QueryEmbeddingProviderConfig | None = None,
    ) -> None:
        self.config = config or QueryEmbeddingProviderConfig.from_env()
        self.config.validate()

        if self.config.backend != "flagembedding":
            raise ValueError(
                f"Unsupported embedding backend for FlagEmbeddingQueryProvider: "
                f"{self.config.backend!r}"
            )

        self._model: Any | None = None

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def dense_model_name(self) -> str:
        return self.config.dense_model_name

    @property
    def sparse_model_name(self) -> str:
        return self.config.sparse_model_name

    @property
    def embedding_dimension(self) -> int | None:
        return self.config.embedding_dimension

    def embed_query(self, query: str) -> RawQueryEmbedding:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query cannot be empty.")

        requested_device = (
            self.config.device
            or "cpu"
        )

        with retrieval_gpu_lease(
            requested_device
        ) as active_device:
            model = self._load_model(
                device=active_device
            )

            move_model_to_device(
                model,
                active_device,
            )

            try:
                output = model.encode(
                    [cleaned_query],
                    batch_size=self.config.batch_size,
                    max_length=self.config.max_length,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=(
                        self.config.return_colbert_vecs
                    ),
                )

            finally:
                if (
                    self.config.offload_after_query
                    and is_cuda_device(active_device)
                ):
                    offload_model_to_cpu(
                        model,
                        active_device,
                    )

        dense_vector = _first_dense_vector(output.get("dense_vecs"))
        sparse_indices, sparse_values = _first_sparse_vector(
            output.get("lexical_weights")
        )

        result = RawQueryEmbedding(
            query=cleaned_query,
            dense_vector=dense_vector,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            model_name=self.config.model_name,
            dense_model_name=self.config.dense_model_name,
            sparse_model_name=self.config.sparse_model_name,
            embedding_dimension=self.config.embedding_dimension,
            metadata={
                "backend": self.config.backend,
                "provider": "FlagEmbedding",
                "use_fp16": self.config.use_fp16,
                "device": self.config.device,
                "batch_size": self.config.batch_size,
                "max_length": self.config.max_length,
                "return_colbert_vecs": self.config.return_colbert_vecs,
                "offload_after_query": self.config.offload_after_query,
            },
        )
        result.validate()

        return result

    def _load_model(
        self,
        *,
        device: str | None = None,
    ) -> Any:
        if self._model is not None:
            return self._model

        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding is required for the flagembedding query embedding backend."
            ) from exc

        kwargs: Payload = {
            "use_fp16": self.config.use_fp16,
        }

        resolved_device = (
            device
            or self.config.device
        )

        if resolved_device:
            kwargs["device"] = resolved_device

        self._model = BGEM3FlagModel(
            self.config.model_name,
            **kwargs,
        )

        return self._model


@dataclass(frozen=True)
class RetrievalQueryEmbedderConfig:
    expected_dense_dimension: int | None = None
    run_sync_in_thread: bool = True
    require_dense: bool = True
    require_sparse: bool = False

    @classmethod
    def from_provider_config(
        cls,
        provider_config: QueryEmbeddingProviderConfig,
    ) -> RetrievalQueryEmbedderConfig:
        return cls(
            expected_dense_dimension=provider_config.embedding_dimension,
        )

    def validate(self) -> None:
        if self.expected_dense_dimension is not None and self.expected_dense_dimension <= 0:
            raise ValueError("expected_dense_dimension must be greater than zero.")

        if not isinstance(self.run_sync_in_thread, bool):
            raise ValueError("run_sync_in_thread must be boolean.")

        if not isinstance(self.require_dense, bool):
            raise ValueError("require_dense must be boolean.")

        if not isinstance(self.require_sparse, bool):
            raise ValueError("require_sparse must be boolean.")


class RetrievalQueryEmbedder:
    """
    Graph-facing query embedder.

    Converts NormalizedQuery into the retrieval QueryEmbedding contract.
    """

    def __init__(
        self,
        provider: QueryEmbeddingProvider | None = None,
        *,
        config: RetrievalQueryEmbedderConfig | None = None,
    ) -> None:
        self.provider = provider or FlagEmbeddingQueryProvider()
        self.config = config or RetrievalQueryEmbedderConfig(
            expected_dense_dimension=self.provider.embedding_dimension,
        )
        self.config.validate()

    async def embed(self, normalized_query: NormalizedQuery) -> QueryEmbedding:
        normalized_query.validate()

        query = normalized_query.normalized_query.strip()
        if not query:
            raise ValueError("normalized_query.normalized_query cannot be empty.")

        if self.config.run_sync_in_thread:
            raw = await asyncio.to_thread(self.provider.embed_query, query)
        else:
            raw = self.provider.embed_query(query)

        raw.validate()

        if self.config.require_dense and not raw.has_dense:
            raise ValueError("query embedding provider did not return a dense vector.")

        if self.config.require_sparse and not raw.has_sparse:
            raise ValueError("query embedding provider did not return a sparse vector.")

        sparse_vector = None
        if raw.has_sparse:
            sparse_vector = SparseQueryVector(
                indices=raw.sparse_indices,
                values=raw.sparse_values,
            )
            sparse_vector.validate()

        embedding = QueryEmbedding(
            query=raw.query,
            dense_vector=raw.dense_vector,
            sparse_vector=sparse_vector,
            embedding_model=raw.model_name,
            dense_model=raw.dense_model_name,
            sparse_model=raw.sparse_model_name,
            embedding_dimension=raw.embedding_dimension,
            metadata={
                **raw.metadata,
                "original_query": normalized_query.original_query,
                "intent": normalized_query.intent.value,
                "defects": [defect.value for defect in normalized_query.defects],
                "rewritten_query_count": len(normalized_query.rewritten_queries),
                "search_query_count": len(normalized_query.search_queries),
            },
        )

        embedding.validate(
            expected_dense_dimension=self.config.expected_dense_dimension,
        )

        return embedding


def load_env_file(
    *,
    env_file: str | Path | None = None,
    override: bool = False,
) -> Path | None:
    path = _resolve_env_file(env_file)
    if path is None:
        return None

    if path in _DOTENV_LOADED_PATHS and not override:
        return path

    values = _parse_env_file(path)

    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value

    _DOTENV_LOADED_PATHS.add(path)

    return path


def _resolve_env_file(env_file: str | Path | None) -> Path | None:
    if env_file is not None:
        path = Path(env_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f".env file not found: {path}")
        return path

    for directory in (Path.cwd(), *Path.cwd().parents):
        candidate = directory / ".env"
        if candidate.exists():
            return candidate.resolve()

    return None


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line.removeprefix("export ").strip()

        if "=" not in line:
            raise ValueError(f"Invalid .env line {line_number} in {path}: missing '='.")

        key, value = line.split("=", 1)
        key = key.strip()
        value = _clean_env_value(value.strip())

        if not key:
            raise ValueError(f"Invalid .env line {line_number} in {path}: empty key.")

        values[key] = value

    return values


def _clean_env_value(value: str) -> str:
    if not value:
        return ""

    if value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]

    comment_index = value.find(" #")
    if comment_index >= 0:
        return value[:comment_index].strip()

    return value


def _first_dense_vector(value: Any) -> list[float] | None:
    if value is None:
        return None

    if hasattr(value, "tolist"):
        value = value.tolist()

    if not isinstance(value, list):
        return None

    if not value:
        return None

    first = value[0]

    if hasattr(first, "tolist"):
        first = first.tolist()

    if not isinstance(first, list):
        first = value

    if not first:
        return None

    return [float(item) for item in first]


def _first_sparse_vector(value: Any) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if value is None:
        return (), ()

    if isinstance(value, list):
        if not value:
            return (), ()
        value = value[0]

    if not isinstance(value, dict):
        return (), ()

    pairs: list[tuple[int, float]] = []

    for raw_token_id, raw_weight in value.items():
        token_id = int(raw_token_id)
        weight = float(raw_weight)

        if weight == 0.0:
            continue

        pairs.append((token_id, weight))

    pairs.sort(key=lambda item: item[0])

    return (
        tuple(token_id for token_id, _ in pairs),
        tuple(weight for _, weight in pairs),
    )


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ValueError(f"Missing required environment variable: {name}")
    return value.strip()


def _optional_env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _optional_env_or_none(name: str) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _optional_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _optional_env_int_or_none(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return int(value)


def _optional_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True

    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(f"{name} must be a boolean-like value.")


def _required_mapping_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"Missing required query embedding config key: {key}")
    return str(value).strip()


def _optional_mapping_str(
    data: Mapping[str, Any],
    key: str,
    *,
    default: str,
) -> str:
    value = data.get(key)
    if value is None or not str(value).strip():
        return default
    return str(value).strip()


def _optional_mapping_str_or_none(
    data: Mapping[str, Any],
    key: str,
) -> str | None:
    value = data.get(key)
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _optional_mapping_int(
    data: Mapping[str, Any],
    key: str,
    *,
    default: int,
) -> int:
    value = data.get(key)
    if value is None:
        return default
    return int(value)


def _optional_mapping_int_or_none(
    data: Mapping[str, Any],
    key: str,
    *,
    fallback_key: str | None = None,
) -> int | None:
    value = data.get(key)

    if value is None and fallback_key is not None:
        value = data.get(fallback_key)

    if value is None:
        return None

    return int(value)


def _optional_mapping_bool(
    data: Mapping[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    value = data.get(key)
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True

    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(f"{key} must be a boolean-like value.")
