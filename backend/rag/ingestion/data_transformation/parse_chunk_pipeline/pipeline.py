# rag/ingestion/data_transformation/parse_chunk_pipeline/pipeline.py

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.ingestion.data_transformation.chunking.models import Chunk
from rag.ingestion.data_transformation.chunking.parser_result_adapter import chunk_parse_result
from rag.ingestion.data_transformation.chunking.config import ChunkingConfig
from rag.ingestion.data_transformation.parser.adapters.markdown.execution import execute_markdown_parse


#--------------------TIMINGS--------------------

@dataclass(frozen=True)
class ParseChunkTimings:
    parse_seconds: float
    chunk_seconds: float
    total_seconds: float

    def to_dict(
        self,
    ) -> dict[str, float]:
        return {
            "parse_seconds": round(
                self.parse_seconds,
                6,
            ),
            "chunk_seconds": round(
                self.chunk_seconds,
                6,
            ),
            "total_seconds": round(
                self.total_seconds,
                6,
            ),
        }


#--------------------RESULT--------------------

@dataclass(frozen=True)
class ParsedChunkedResult:
    source_path: str
    parse_result: Any
    chunks: list[Chunk]
    timings: ParseChunkTimings

    @property
    def doc_id(
        self,
    ) -> str:
        return self._parse_result_text(
            "doc_id"
        )

    @property
    def parser_family(
        self,
    ) -> str:
        return self._parse_result_text(
            "parser_family"
        )

    @property
    def parser_variant(
        self,
    ) -> str:
        return self._parse_result_text(
            "parser_variant"
        )

    @property
    def parse_strategy(
        self,
    ) -> str:
        return self._parse_result_text(
            "parse_strategy"
        )

    @property
    def unit_count(
        self,
    ) -> int:
        value = getattr(
            self.parse_result,
            "unit_count",
            0,
        )

        if isinstance(
            value,
            bool,
        ):
            return 0

        try:
            count = int(
                value
                or 0
            )
        except (
            TypeError,
            ValueError,
        ):
            return 0

        return max(
            0,
            count,
        )

    @property
    def chunk_count(
        self,
    ) -> int:
        return len(
            self.chunks
        )

    @property
    def total_chunk_tokens(
        self,
    ) -> int:
        return sum(
            chunk.tokens
            for chunk in self.chunks
        )

    def summary(
        self,
    ) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "doc_id": self.doc_id,
            "parser_family": self.parser_family,
            "parser_variant": self.parser_variant,
            "parse_strategy": self.parse_strategy,
            "unit_count": self.unit_count,
            "chunk_count": self.chunk_count,
            "total_chunk_tokens": self.total_chunk_tokens,
            "timings": self.timings.to_dict(),
        }

    def _parse_result_text(
        self,
        field_name: str,
    ) -> str:
        value = getattr(
            self.parse_result,
            field_name,
            "",
        )

        return str(
            value
            or ""
        ).strip()


#--------------------PIPELINE--------------------

def run_parse_chunk_pipeline(
    source_path: str | Path,
    *,
    enable_overlap: bool = False,
    policy_context: Any | None = None,
    chunking_config: ChunkingConfig | None = None,
) -> ParsedChunkedResult:
    path = _resolve_source_path(
        source_path
    )

    total_started = (
        time.perf_counter()
    )

    parse_started = (
        time.perf_counter()
    )

    parse_result = execute_markdown_parse(
        str(
            path
        ),
        policy_context=policy_context,
    )

    parse_seconds = (
        time.perf_counter()
        - parse_started
    )

    chunk_started = (
        time.perf_counter()
    )

    chunks = chunk_parse_result(
        parse_result,
        source=str(
            path
        ),
        enable_overlap=enable_overlap,
        config=chunking_config,
    )

    chunk_seconds = (
        time.perf_counter()
        - chunk_started
    )

    total_seconds = (
        time.perf_counter()
        - total_started
    )

    return ParsedChunkedResult(
        source_path=str(
            path
        ),
        parse_result=parse_result,
        chunks=chunks,
        timings=ParseChunkTimings(
            parse_seconds=parse_seconds,
            chunk_seconds=chunk_seconds,
            total_seconds=total_seconds,
        ),
    )


#--------------------PATH VALIDATION--------------------

def _resolve_source_path(
    source_path: str | Path,
) -> Path:
    if isinstance(
        source_path,
        str,
    ):
        if not source_path.strip():
            raise ValueError(
                "source_path cannot be empty"
            )

    elif not isinstance(
        source_path,
        Path,
    ):
        raise TypeError(
            "source_path must be a string or Path"
        )

    path = (
        Path(
            source_path
        )
        .expanduser()
        .resolve()
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Source document does not exist: {path}"
        )

    return path