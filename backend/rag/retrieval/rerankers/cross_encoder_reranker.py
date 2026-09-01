from __future__ import annotations

import asyncio
import math
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from rag.retrieval.config.runtime import (
    RetrievalRuntimeConfig,
    load_retrieval_runtime_config,
)
from rag.retrieval.contracts.candidates import RerankedCandidate, RetrievalCandidate
from rag.retrieval.contracts.query import NormalizedQuery
from rag.retrieval.contracts.request import RetrievalRequest
from rag.retrieval.gpu_runtime import (
    is_cuda_device,
    offload_model_to_cpu,
    retrieval_gpu_lease,
)
from rag.retrieval.rerankers.score_reranker import ScoreReranker, ScoreRerankerConfig


Payload = dict[str, Any]
_DOTENV_LOADED_PATHS: set[Path] = set()


class CrossEncoderModel(Protocol):
    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
    ) -> Sequence[float]:
        raise NotImplementedError


@dataclass(frozen=True)
class CrossEncoderRerankerConfig:
    enabled: bool = True
    backend: str = "sentence_transformers"
    model_env: str = "RERANKER_MODEL"
    model_name: str | None = None
    max_candidates: int = 50
    batch_size: int = 8
    normalize_scores: bool = False
    fallback_to_score: bool = True
    run_sync_in_thread: bool = True

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: RetrievalRuntimeConfig,
    ) -> CrossEncoderRerankerConfig:
        config = cls(
            enabled=runtime_config.cross_encoder_reranker.enabled,
            backend=runtime_config.cross_encoder_reranker.backend,
            model_env=runtime_config.cross_encoder_reranker.model_env,
            model_name=runtime_config.cross_encoder_reranker.model_name,
            max_candidates=runtime_config.cross_encoder_reranker.max_candidates,
            batch_size=runtime_config.cross_encoder_reranker.batch_size,
            normalize_scores=runtime_config.cross_encoder_reranker.normalize_scores,
            fallback_to_score=runtime_config.cross_encoder_reranker.fallback_to_score,
        )
        config.validate()

        return config

    def resolved_model_name(self) -> str:
        if self.model_name and self.model_name.strip():
            return self.model_name.strip()

        load_env_file()

        value = os.getenv(self.model_env)
        if value is None or not value.strip():
            raise ValueError(f"Missing reranker model env variable: {self.model_env}")

        return value.strip()

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be boolean.")

        if not self.backend.strip():
            raise ValueError("backend cannot be empty.")

        if self.model_name is None and not self.model_env.strip():
            raise ValueError("model_env cannot be empty when model_name is not set.")

        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be greater than zero.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        if not isinstance(self.normalize_scores, bool):
            raise ValueError("normalize_scores must be boolean.")

        if not isinstance(self.fallback_to_score, bool):
            raise ValueError("fallback_to_score must be boolean.")

        if not isinstance(self.run_sync_in_thread, bool):
            raise ValueError("run_sync_in_thread must be boolean.")


class SentenceTransformersCrossEncoderProvider:
    def __init__(
        self,
        *,
        model_name: str,
    ) -> None:
        load_env_file()

        self.model_name = model_name
        self.device = (
            os.getenv(
                "RERANKER_DEVICE",
                "cuda:0",
            )
            .strip()
            or "cpu"
        )
        self.offload_after_query = _env_bool(
            "RERANKER_OFFLOAD_AFTER_QUERY",
            True,
        )
        self._model: Any | None = None

    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int,
    ) -> Sequence[float]:
        if not pairs:
            return ()

        with retrieval_gpu_lease(
            self.device
        ) as active_device:
            model = self._load_model()

            try:
                scores = _predict_cross_encoder(
                    model=model,
                    pairs=pairs,
                    batch_size=batch_size,
                    device=active_device,
                )

            finally:
                if (
                    self.offload_after_query
                    and is_cuda_device(active_device)
                ):
                    _set_legacy_cross_encoder_device(
                        model,
                        "cpu",
                    )
                    offload_model_to_cpu(
                        model,
                        active_device,
                    )

        if hasattr(scores, "tolist"):
            scores = scores.tolist()

        return tuple(float(score) for score in scores)

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the sentence_transformers "
                "cross encoder reranker backend."
            ) from exc

        self._model = CrossEncoder(
            self.model_name,
            device="cpu",
        )

        return self._model


def _predict_cross_encoder(
    *,
    model: Any,
    pairs: Sequence[tuple[str, str]],
    batch_size: int,
    device: str,
) -> Sequence[float]:
    import inspect

    predict = model.predict

    parameters = inspect.signature(
        predict
    ).parameters

    if "device" in parameters:
        return predict(
            list(pairs),
            batch_size=batch_size,
            device=device,
        )

    _set_legacy_cross_encoder_device(
        model,
        device,
    )

    return predict(
        list(pairs),
        batch_size=batch_size,
    )


def _set_legacy_cross_encoder_device(
    model: Any,
    device: str,
) -> None:
    if not hasattr(
        model,
        "_target_device",
    ):
        return

    try:
        import torch

        model._target_device = torch.device(
            device
        )

    except Exception:
        model._target_device = device


def _env_bool(
    name: str,
    default: bool,
) -> bool:
    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    normalized = value.strip().lower()

    if normalized in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }:
        return True

    if normalized in {
        "0",
        "false",
        "no",
        "n",
        "off",
    }:
        return False

    raise ValueError(
        f"{name} must be a boolean-like value."
    )


class CrossEncoderReranker:
    """
    Real cross-encoder reranker.

    It scores query-document pairs after dense/sparse/fusion retrieval.
    Model selection is controlled by TOML + env:
    - TOML chooses backend/model_env/behavior
    - env supplies RERANKER_MODEL by default
    """

    def __init__(
        self,
        config: CrossEncoderRerankerConfig | None = None,
        *,
        runtime_config: RetrievalRuntimeConfig | None = None,
        provider: CrossEncoderModel | None = None,
    ) -> None:
        resolved_runtime = runtime_config or load_retrieval_runtime_config()
        self.config = config or CrossEncoderRerankerConfig.from_runtime_config(
            resolved_runtime,
        )
        self.config.validate()

        self.model_name = self.config.resolved_model_name()

        print(
            "[retrieval.reranker] active=cross_encoder "
            f"backend={self.config.backend} "
            f"model_env={self.config.model_env} "
            f"model={self.model_name}",
            file=sys.stderr,
        )

        if provider is not None:
            self.provider = provider
        elif self.config.backend == "sentence_transformers":
            self.provider = SentenceTransformersCrossEncoderProvider(
                model_name=self.model_name,
            )
        else:
            raise ValueError(f"Unsupported cross encoder backend: {self.config.backend}")

    async def rerank(
        self,
        *,
        candidates: Sequence[RetrievalCandidate],
        request: RetrievalRequest,
        normalized_query: NormalizedQuery,
    ) -> Sequence[RerankedCandidate]:
        request.validate()
        normalized_query.validate()

        for candidate in candidates:
            candidate.validate()

        if not self.config.enabled:
            return ()

        selected = tuple(candidates[: self.config.max_candidates])
        if not selected:
            return ()

        query = normalized_query.normalized_query
        pairs = tuple((query, _base_candidate(candidate).text) for candidate in selected)

        try:
            if self.config.run_sync_in_thread:
                scores = await asyncio.to_thread(
                    self.provider.predict,
                    pairs,
                    batch_size=self.config.batch_size,
                )
            else:
                scores = self.provider.predict(
                    pairs,
                    batch_size=self.config.batch_size,
                )
        except Exception:
            if not self.config.fallback_to_score:
                raise

            fallback = ScoreReranker(
                config=ScoreRerankerConfig(
                    enabled=True,
                    max_candidates=self.config.max_candidates,
                    normalize_scores=self.config.normalize_scores,
                )
            )
            return await fallback.rerank(
                candidates=candidates,
                request=request,
                normalized_query=normalized_query,
            )

        score_values = tuple(float(score) for score in scores)

        reranked_pairs = sorted(
            zip(selected, score_values, strict=False),
            key=lambda item: (
                -item[1],
                _candidate_identity(item[0]),
            ),
        )

        normalized_scores = _normalize_scores(
            tuple(score for _, score in reranked_pairs),
            normalize=self.config.normalize_scores,
        )

        reranked = tuple(
            _build_reranked_candidate(
                candidate=_base_candidate(candidate),
                rank=index + 1,
                rerank_score=normalized_scores[index],
                original_score=_candidate_score(candidate),
                raw_cross_encoder_score=float(raw_score),
                model_name=self.model_name,
                backend=self.config.backend,
            )
            for index, (candidate, raw_score) in enumerate(reranked_pairs)
        )

        for item in reranked:
            item.validate()

        return reranked


def _build_reranked_candidate(
    *,
    candidate: RetrievalCandidate,
    rank: int,
    rerank_score: float,
    original_score: float,
    raw_cross_encoder_score: float,
    model_name: str,
    backend: str,
) -> RerankedCandidate:
    return RerankedCandidate(
        candidate=_base_candidate(candidate),
        rerank_score=rerank_score,
        rank=rank,
        reranker="cross_encoder",
        original_score=original_score,
        explanation=None,
        metadata={
            "reranker": "cross_encoder",
            "backend": backend,
            "model_name": model_name,
            "original_score": original_score,
            "raw_cross_encoder_score": raw_cross_encoder_score,
        },
    )



def _base_candidate(candidate_like: CandidateLike) -> RetrievalCandidate:
    nested = getattr(candidate_like, "candidate", None)

    if nested is not None and hasattr(nested, "text"):
        return nested

    return candidate_like  # type: ignore[return-value]


def _candidate_score(candidate_like: CandidateLike) -> float:
    fused_score = getattr(candidate_like, "fused_score", None)

    if fused_score is not None:
        return float(fused_score)

    return float(getattr(candidate_like, "score", 0.0))


def _validate_candidate_like(candidate_like: CandidateLike) -> None:
    validate = getattr(candidate_like, "validate", None)
    if callable(validate):
        validate()

    base = _base_candidate(candidate_like)
    base.validate()


def _normalize_scores(
    scores: Sequence[float],
    *,
    normalize: bool,
) -> tuple[float, ...]:
    values = tuple(
        float(score)
        for score in scores
    )

    if not normalize:
        return values

    if not values:
        return ()

    return tuple(
        _sigmoid_score(score)
        for score in values
    )


def _sigmoid_score(
    score: float,
) -> float:
    if score >= 0.0:
        exponent = math.exp(
            -score
        )

        return 1.0 / (
            1.0 + exponent
        )

    exponent = math.exp(
        score
    )

    return exponent / (
        1.0 + exponent
    )


def _candidate_identity(candidate: RetrievalCandidate) -> str:
    for attr_name in (
        "chunk_id",
        "point_id",
        "source_id",
        "parser_doc_id",
        "section_id",
    ):
        value = _clean_optional(getattr(candidate, attr_name, None))
        if value:
            return value

    return _clean_optional(getattr(candidate, "text", None)) or str(id(candidate))


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


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None

    cleaned = str(value).strip()
    return cleaned or None
