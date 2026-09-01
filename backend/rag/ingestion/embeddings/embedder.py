from __future__ import annotations

import contextlib
import io
import logging
import math
from typing import Any, List, Mapping, Optional, Sequence

import torch
from FlagEmbedding import BGEM3FlagModel

from rag.ingestion.data_transformation.chunking.models import Chunk
from rag.ingestion.embeddings.config import (
    EmbeddingConfig,
    get_embedding_pipeline_config,
)
from rag.ingestion.embeddings.types import (
    EmbeddingBatch,
    SparseVectorData,
    Vector,
)


LOGGER = logging.getLogger(__name__)


class Embedder:
    """
    BGE-M3 embedder using FlagEmbedding.

    One local model produces:
      - dense_vecs
      - lexical_weights sparse vectors

    ColBERT vectors are intentionally ignored for this phase.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        self.config = config or get_embedding_pipeline_config().embedding
        self.config.validate()

        self.model_name = self.config.model_name
        self.sparse_model_name = self.config.sparse_model_name
        self.device = self._resolve_device(self.config.device)
        self._vector_size = self.config.dense_vector_size

        self.model = self._load_model()

        LOGGER.info(
            "Loaded BGE-M3 embedder | model=%s device=%s dim=%s fp16=%s embedding_batch_size=%s sparse=%s",
            self.model_name,
            self.device,
            self._vector_size,
            self._use_fp16(),
            self.config.batch_size,
            self.config.enable_sparse,
        )

    @property
    def vector_size(self) -> int:
        return self._vector_size

    @property
    def normalized(self) -> bool:
        return self.config.normalize_embeddings

    def embed_documents(self, chunks: Sequence[Chunk]) -> EmbeddingBatch:
        if not chunks:
            return EmbeddingBatch(
                dense_vectors=[],
                sparse_vectors=[],
                model_name=self.model_name,
                dense_model_name=self.model_name,
                sparse_model_name=self.sparse_model_name,
                vector_size=self.vector_size,
                normalized=self.normalized,
            )

        texts = [
            self._with_prefix(
                self._format_passage(chunk),
                self.config.document_prefix,
            )
            for chunk in chunks
        ]

        dense_vectors, sparse_vectors = self._encode_texts(
            texts,
            batch_size=self.config.batch_size,
        )

        if len(dense_vectors) != len(chunks):
            raise RuntimeError(
                f"Dense embedding count mismatch: chunks={len(chunks)}, vectors={len(dense_vectors)}."
            )

        if self.config.enable_sparse and len(sparse_vectors) != len(chunks):
            raise RuntimeError(
                f"Sparse embedding count mismatch: chunks={len(chunks)}, vectors={len(sparse_vectors)}."
            )

        return EmbeddingBatch(
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors,
            model_name=self.model_name,
            dense_model_name=self.model_name,
            sparse_model_name=self.sparse_model_name,
            vector_size=self.vector_size,
            normalized=self.normalized,
        )

    def embed_chunks(self, chunks: Sequence[Chunk]) -> List[Vector]:
        """
        Backward-compatible dense-only helper.
        """
        return self.embed_documents(chunks).dense_vectors

    def embed_query(self, query: str) -> Vector:
        """
        Backward-compatible dense-only query helper.
        """
        cleaned = self._clean_text(query)
        if not cleaned:
            return []

        dense_vectors, _ = self._encode_texts(
            [self._with_prefix(cleaned, self.config.query_prefix)],
            batch_size=1,
        )

        return dense_vectors[0] if dense_vectors else []

    def embed_query_full(self, query: str) -> tuple[Vector, SparseVectorData | None]:
        """
        Dense + sparse query embedding for hybrid retrieval.
        """
        cleaned = self._clean_text(query)
        if not cleaned:
            return [], None

        dense_vectors, sparse_vectors = self._encode_texts(
            [self._with_prefix(cleaned, self.config.query_prefix)],
            batch_size=1,
        )

        dense = dense_vectors[0] if dense_vectors else []
        sparse = sparse_vectors[0] if sparse_vectors else None

        return dense, sparse

    def _load_model(self) -> BGEM3FlagModel:
        try:
            return BGEM3FlagModel(
                self.model_name,
                use_fp16=self._use_fp16(),
                devices=self.device,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load BGEM3FlagModel {self.model_name!r} on device {self.device!r}."
            ) from exc

    def _use_fp16(self) -> bool:
        return bool(self.config.use_fp16 and self.device.startswith("cuda"))

    def _resolve_device(self, configured_device: Optional[str]) -> str:
        device = (configured_device or "auto").strip().lower()

        if device == "auto":
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                LOGGER.info("Using CUDA for BGE-M3 embeddings | gpu=%s", gpu_name)
                return "cuda:0"

            LOGGER.info("CUDA unavailable. Using CPU for BGE-M3 embeddings.")
            return "cpu"

        if device == "cuda":
            device = "cuda:0"

        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"embedding.device={configured_device!r} was requested, but CUDA is unavailable."
                )

            gpu_name = torch.cuda.get_device_name(0)
            LOGGER.info(
                "Using configured CUDA for BGE-M3 embeddings | device=%s gpu=%s",
                device,
                gpu_name,
            )
            return device

        if device == "cpu":
            LOGGER.info("Using CPU for BGE-M3 embeddings.")
            return device

        raise ValueError(
            f"Invalid embedding.device={configured_device!r}. "
            "Use auto, cpu, cuda, or cuda:0."
        )

    def _encode_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> tuple[List[Vector], List[SparseVectorData]]:
        clean_texts = [text.strip() for text in texts if text and text.strip()]

        if not clean_texts:
            return [], []

        try:
            output = self._encode_with_suppressed_progress(
                clean_texts,
                batch_size=batch_size,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to generate BGE-M3 embeddings using {self.model_name!r}."
            ) from exc

        dense_vectors = self._dense_to_vectors(output.get("dense_vecs"))

        if self.config.normalize_embeddings:
            dense_vectors = [
                self._normalize_vector(vector)
                for vector in dense_vectors
            ]

        sparse_vectors: List[SparseVectorData] = []

        if self.config.enable_sparse:
            sparse_vectors = self._sparse_to_vectors(
                output.get("lexical_weights")
            )

        return dense_vectors, sparse_vectors

    def _encode_with_suppressed_progress(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
    ) -> Mapping[str, Any]:
        """
        FlagEmbedding prints tqdm progress bars internally.

        Do not pass show_progress_bar=False here. Some FlagEmbedding versions
        forward unknown kwargs into tokenizer.pad(...), causing:
            TypeError: PreTrainedTokenizerBase.pad() got an unexpected keyword argument 'show_progress_bar'
        """

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            output = self.model.encode(
                texts,
                batch_size=batch_size,
                max_length=self.config.max_length,
                return_dense=True,
                return_sparse=self.config.enable_sparse,
                return_colbert_vecs=self.config.return_colbert_vecs,
            )

        if not isinstance(output, Mapping):
            raise RuntimeError(
                f"Unexpected BGE-M3 output type: {type(output).__name__}"
            )

        return output

    def _dense_to_vectors(self, value: Any) -> List[Vector]:
        if value is None:
            return []

        if hasattr(value, "tolist"):
            value = value.tolist()

        if not isinstance(value, list):
            return []

        if value and not isinstance(value[0], list):
            value = [value]

        vectors: List[Vector] = []

        for vector in value:
            vectors.append([float(item) for item in vector])

        return vectors

    def _sparse_to_vectors(self, value: Any) -> List[SparseVectorData]:
        if value is None:
            return []

        if not isinstance(value, Sequence) or isinstance(
            value,
            (str, bytes, bytearray),
        ):
            return []

        result: List[SparseVectorData] = []

        for item in value:
            if not isinstance(item, Mapping):
                result.append(
                    SparseVectorData(
                        indices=[],
                        values=[],
                    )
                )
                continue

            pairs: list[tuple[int, float]] = []

            for raw_index, raw_weight in item.items():
                try:
                    index = int(raw_index)
                    weight = float(raw_weight)
                except (TypeError, ValueError):
                    continue

                if index < 0:
                    continue

                if not math.isfinite(weight):
                    continue

                if abs(weight) < self.config.sparse_min_abs_value:
                    continue

                pairs.append((index, weight))

            pairs.sort(key=lambda pair: abs(pair[1]), reverse=True)
            pairs = pairs[: self.config.sparse_max_nonzero]
            pairs.sort(key=lambda pair: pair[0])

            result.append(
                SparseVectorData(
                    indices=[index for index, _ in pairs],
                    values=[weight for _, weight in pairs],
                )
            )

        return result

    def _normalize_vector(self, vector: Vector) -> Vector:
        norm = math.sqrt(sum(value * value for value in vector))

        if norm <= 0:
            return vector

        return [float(value / norm) for value in vector]

    def _format_passage(self, chunk: Chunk) -> str:
        metadata = self._safe_mapping(getattr(chunk, "metadata", None))

        parts: List[str] = []

        heading = self._clean_text(getattr(chunk, "heading", None))
        context = self._clean_text(getattr(chunk, "context", None))
        text = self._clean_text(getattr(chunk, "text", None))

        if heading:
            parts.append(heading)

        if context and context != heading:
            parts.append(context)

        heading_path = self._extract_path(metadata)

        if heading_path:
            rendered_path = " > ".join(heading_path)

            if rendered_path and rendered_path not in parts:
                parts.append(rendered_path)

        if text:
            parts.append(text)

        formatted = "\n".join(parts).strip()

        if not formatted:
            chunk_id = getattr(chunk, "chunk_id", None) or metadata.get("chunk_id")
            raise ValueError(f"Cannot embed empty chunk. chunk_id={chunk_id!r}")

        return formatted

    def _extract_path(self, metadata: Mapping[str, Any]) -> List[str]:
        raw_path = metadata.get("heading_path") or metadata.get("path")

        if raw_path is None:
            return []

        if isinstance(raw_path, str):
            cleaned = self._clean_text(raw_path)
            return [cleaned] if cleaned else []

        if isinstance(raw_path, Sequence) and not isinstance(
            raw_path,
            (bytes, bytearray, str),
        ):
            path: List[str] = []

            for item in raw_path:
                cleaned = self._clean_text(item)

                if cleaned:
                    path.append(cleaned)

            return path

        cleaned = self._clean_text(raw_path)
        return [cleaned] if cleaned else []

    def _with_prefix(self, text: str, prefix: str) -> str:
        cleaned_text = text.strip()
        cleaned_prefix = prefix or ""

        if not cleaned_prefix:
            return cleaned_text

        return f"{cleaned_prefix}{cleaned_text}"

    @staticmethod
    def _safe_mapping(value: Any) -> Mapping[str, Any]:
        return value if isinstance(value, Mapping) else {}

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""

        return " ".join(str(value).split())