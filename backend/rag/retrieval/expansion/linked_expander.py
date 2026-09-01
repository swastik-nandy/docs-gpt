from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from rag.retrieval.config.runtime import (
    RetrievalRuntimeConfig,
    load_retrieval_runtime_config,
)
from rag.retrieval.contracts.candidates import (
    CandidateOrigin,
    FusedCandidate,
    RetrievalCandidate,
)
from rag.retrieval.contracts.query import NormalizedQuery
from rag.retrieval.contracts.request import RetrievalRequest
from rag.retrieval.vector_store import QdrantCandidateMapper
from rag.retrieval.vector_store.qdrant_client import (
    QdrantClientFactory,
    QdrantConnectionConfig,
)


Payload = dict[str, Any]
CandidateLike = RetrievalCandidate | FusedCandidate


@dataclass(frozen=True)
class LinkedStructureExpanderConfig:
    collection_name: str
    enabled: bool = True
    max_seed_candidates: int = 10
    max_links_per_seed: int = 5
    max_chunks_per_link: int = 6
    deduplicate: bool = True
    with_payload: bool = True
    with_vectors: bool = False

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: RetrievalRuntimeConfig,
    ) -> LinkedStructureExpanderConfig:
        config = cls(
            collection_name=runtime_config.qdrant.collection_name,
            enabled=runtime_config.linked_expansion.enabled,
            max_seed_candidates=runtime_config.linked_expansion.max_seed_candidates,
            max_links_per_seed=runtime_config.linked_expansion.max_links_per_seed,
            max_chunks_per_link=runtime_config.linked_expansion.max_chunks_per_link,
            deduplicate=runtime_config.linked_expansion.deduplicate,
            with_payload=runtime_config.linked_expansion.with_payload,
            with_vectors=runtime_config.linked_expansion.with_vectors,
        )
        config.validate()

        return config

    def validate(self) -> None:
        if not self.collection_name.strip():
            raise ValueError("collection_name cannot be empty.")
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be boolean.")
        if self.max_seed_candidates <= 0:
            raise ValueError("max_seed_candidates must be greater than zero.")
        if self.max_links_per_seed <= 0:
            raise ValueError("max_links_per_seed must be greater than zero.")
        if self.max_chunks_per_link <= 0:
            raise ValueError("max_chunks_per_link must be greater than zero.")
        if not isinstance(self.deduplicate, bool):
            raise ValueError("deduplicate must be boolean.")
        if not isinstance(self.with_payload, bool):
            raise ValueError("with_payload must be boolean.")
        if not isinstance(self.with_vectors, bool):
            raise ValueError("with_vectors must be boolean.")


class LinkedStructureExpander:
    """
    Expands retrieved signpost chunks into linked target document chunks.

    The expander reads link metadata from retrieved payloads and tries to fetch
    chunks from matching linked source identifiers within the same tenant.
    """

    def __init__(
        self,
        *,
        client: AsyncQdrantClient | None = None,
        connection_config: QdrantConnectionConfig | None = None,
        runtime_config: RetrievalRuntimeConfig | None = None,
        config: LinkedStructureExpanderConfig | None = None,
        candidate_mapper: QdrantCandidateMapper | None = None,
    ) -> None:
        self.runtime_config = runtime_config or load_retrieval_runtime_config()
        self.runtime_config.validate()

        self.connection_config = connection_config or QdrantConnectionConfig.from_env()
        self.config = config or LinkedStructureExpanderConfig.from_runtime_config(
            self.runtime_config,
        )
        self.config.validate()

        self.client = client or QdrantClientFactory(
            self.connection_config,
        ).async_client()
        self.candidate_mapper = candidate_mapper or QdrantCandidateMapper()

    async def expand(
        self,
        *,
        candidates: Sequence[CandidateLike],
        request: RetrievalRequest,
        normalized_query: NormalizedQuery,
    ) -> Sequence[RetrievalCandidate]:
        request.validate()
        normalized_query.validate()

        for candidate in candidates:
            _validate_candidate_like(candidate)

        if not self.config.enabled:
            return ()

        seen = {
            _candidate_identity(candidate)
            for candidate in candidates
        }

        expanded: list[RetrievalCandidate] = []
        rank = 1

        for seed in candidates[: self.config.max_seed_candidates]:
            payload = _candidate_payload(seed)
            links = tuple(_extract_links(payload))[: self.config.max_links_per_seed]

            for href in links:
                source_ids = _source_id_variants_from_href(href)
                if not source_ids:
                    continue

                points = await self._fetch_by_source_ids(
                    tenant_id=request.tenant.tenant_id,
                    source_ids=source_ids,
                )

                for point in points:
                    candidate = self.candidate_mapper.from_scored_point(
                        point,
                        origin=_expanded_origin(),
                        rank=rank,
                    )

                    identity = _candidate_identity(candidate)
                    if self.config.deduplicate and identity in seen:
                        continue

                    seen.add(identity)
                    _attach_expansion_metadata(
                        candidate,
                        href=href,
                        source_ids=source_ids,
                        seed_identity=_candidate_identity(seed),
                    )
                    expanded.append(candidate)
                    rank += 1

        return tuple(expanded)

    async def _fetch_by_source_ids(
        self,
        *,
        tenant_id: str,
        source_ids: Sequence[str],
    ) -> tuple[Any, ...]:
        should = [
            models.FieldCondition(
                key="source_id",
                match=models.MatchValue(value=source_id),
            )
            for source_id in source_ids
        ]

        if not should:
            return ()

        scroll_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id),
                )
            ],
            should=should,
        )

        points, _ = await self.client.scroll(
            collection_name=self.config.collection_name,
            scroll_filter=scroll_filter,
            limit=self.config.max_chunks_per_link,
            with_payload=self.config.with_payload,
            with_vectors=self.config.with_vectors,
        )

        return tuple(points)


def _extract_links(value: Any) -> tuple[str, ...]:
    links: list[str] = []
    seen: set[str] = set()

    def walk(item: Any) -> None:
        if isinstance(item, Mapping):
            href = item.get("href")
            if isinstance(href, str):
                cleaned = href.strip()
                if cleaned and cleaned not in seen:
                    seen.add(cleaned)
                    links.append(cleaned)

            for nested in item.values():
                walk(nested)

        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for nested in item:
                walk(nested)

    walk(value)
    return tuple(links)


def _source_id_variants_from_href(href: str) -> tuple[str, ...]:
    stem = _href_stem(href)
    if not stem:
        return ()

    variants = [
        stem,
        stem.replace("-", "_"),
        stem.replace("_", "-"),
    ]

    cleaned: list[str] = []
    seen: set[str] = set()

    for variant in variants:
        value = variant.strip()
        if value and value not in seen:
            seen.add(value)
            cleaned.append(value)

    return tuple(cleaned)


def _href_stem(href: str) -> str | None:
    parsed = urlparse(href.strip())
    path = parsed.path or href.strip()
    name = path.rstrip("/").split("/")[-1].strip()

    if not name:
        return None

    if name.endswith(".md"):
        name = name[:-3]

    return name or None


def _candidate_payload(candidate_like: CandidateLike) -> Payload:
    for item in (candidate_like, _base_candidate(candidate_like)):
        payload = getattr(item, "payload", None)
        if isinstance(payload, Mapping):
            return dict(payload)

        metadata = getattr(item, "metadata", {}) or {}
        if isinstance(metadata, Mapping):
            nested_payload = metadata.get("payload")
            if isinstance(nested_payload, Mapping):
                return dict(nested_payload)

            if metadata:
                return dict(metadata)

    return {}


def _attach_expansion_metadata(
    candidate: RetrievalCandidate,
    *,
    href: str,
    source_ids: Sequence[str],
    seed_identity: str,
) -> None:
    metadata = getattr(candidate, "metadata", None)
    if not isinstance(metadata, dict):
        return

    metadata["expansion"] = {
        "type": "linked_document",
        "href": href,
        "source_id_variants": tuple(source_ids),
        "seed_identity": seed_identity,
    }


def _base_candidate(candidate_like: CandidateLike) -> RetrievalCandidate:
    nested = getattr(candidate_like, "candidate", None)

    if nested is not None and hasattr(nested, "text"):
        return nested

    return candidate_like  # type: ignore[return-value]


def _validate_candidate_like(candidate_like: CandidateLike) -> None:
    validate = getattr(candidate_like, "validate", None)
    if callable(validate):
        validate()

    _base_candidate(candidate_like).validate()


def _candidate_identity(candidate_like: CandidateLike) -> str:
    candidate = _base_candidate(candidate_like)

    for attr_name in (
        "chunk_id",
        "point_id",
        "source_id",
        "parser_doc_id",
        "section_id",
        "text_sha256",
    ):
        value = _clean_optional(getattr(candidate, attr_name, None))
        if value:
            return value

    ref = getattr(candidate, "ref", None)
    if ref is not None:
        for attr_name in (
            "chunk_id",
            "point_id",
            "source_id",
            "parser_doc_id",
            "section_id",
            "text_sha256",
        ):
            value = _clean_optional(getattr(ref, attr_name, None))
            if value:
                return value

    return _clean_optional(getattr(candidate, "text", None)) or str(id(candidate))


def _expanded_origin() -> CandidateOrigin:
    for name in (
        "STRUCTURE_EXPANDED",
        "EXPANDED",
        "RELATED",
        "RRF",
        "DENSE",
    ):
        if hasattr(CandidateOrigin, name):
            return getattr(CandidateOrigin, name)

    return CandidateOrigin.DENSE


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None

    cleaned = str(value).strip()
    return cleaned or None
