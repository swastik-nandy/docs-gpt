from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import rtoml

from rag.retrieval.config.runtime_extensions import (
    RetrievalRuntimeExtensions,
    parse_retrieval_runtime_extensions,
)


Payload = dict[str, Any]


DEFAULT_RETRIEVAL_CONFIG_PATH = Path(__file__).with_name("config.toml")


@dataclass(frozen=True)
class QdrantVectorStoreRuntimeConfig:
    collection_name: str
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"

    def validate(self) -> None:
        if not self.collection_name.strip():
            raise ValueError("vector_store.qdrant.collection_name cannot be empty.")
        if not self.dense_vector_name.strip():
            raise ValueError("vector_store.qdrant.dense_vector_name cannot be empty.")
        if not self.sparse_vector_name.strip():
            raise ValueError("vector_store.qdrant.sparse_vector_name cannot be empty.")


@dataclass(frozen=True)
class DenseRetrieverRuntimeConfig:
    limit_multiplier: int = 1
    score_threshold: float | None = None
    with_payload: bool = True
    with_vectors: bool = False

    def validate(self) -> None:
        if self.limit_multiplier <= 0:
            raise ValueError("retrievers.dense.limit_multiplier must be greater than zero.")
        if self.score_threshold is not None and self.score_threshold < 0:
            raise ValueError("retrievers.dense.score_threshold cannot be negative.")
        if not isinstance(self.with_payload, bool):
            raise ValueError("retrievers.dense.with_payload must be boolean.")
        if not isinstance(self.with_vectors, bool):
            raise ValueError("retrievers.dense.with_vectors must be boolean.")


@dataclass(frozen=True)
class SparseRetrieverRuntimeConfig:
    limit_multiplier: int = 1
    score_threshold: float | None = None
    with_payload: bool = True
    with_vectors: bool = False

    def validate(self) -> None:
        if self.limit_multiplier <= 0:
            raise ValueError("retrievers.sparse.limit_multiplier must be greater than zero.")
        if self.score_threshold is not None and self.score_threshold < 0:
            raise ValueError("retrievers.sparse.score_threshold cannot be negative.")
        if not isinstance(self.with_payload, bool):
            raise ValueError("retrievers.sparse.with_payload must be boolean.")
        if not isinstance(self.with_vectors, bool):
            raise ValueError("retrievers.sparse.with_vectors must be boolean.")


@dataclass(frozen=True)
class RrfFusionRuntimeConfig:
    k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    limit_multiplier: int = 1

    def validate(self) -> None:
        if self.k <= 0:
            raise ValueError("fusion.rrf.k must be greater than zero.")
        if self.dense_weight < 0:
            raise ValueError("fusion.rrf.dense_weight cannot be negative.")
        if self.sparse_weight < 0:
            raise ValueError("fusion.rrf.sparse_weight cannot be negative.")
        if self.limit_multiplier <= 0:
            raise ValueError("fusion.rrf.limit_multiplier must be greater than zero.")


@dataclass(frozen=True)
class LinkedExpansionRuntimeConfig:
    enabled: bool = True
    max_seed_candidates: int = 10
    max_links_per_seed: int = 5
    max_chunks_per_link: int = 6
    deduplicate: bool = True
    with_payload: bool = True
    with_vectors: bool = False

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("expansion.linked.enabled must be boolean.")
        if self.max_seed_candidates <= 0:
            raise ValueError("expansion.linked.max_seed_candidates must be greater than zero.")
        if self.max_links_per_seed <= 0:
            raise ValueError("expansion.linked.max_links_per_seed must be greater than zero.")
        if self.max_chunks_per_link <= 0:
            raise ValueError("expansion.linked.max_chunks_per_link must be greater than zero.")
        if not isinstance(self.deduplicate, bool):
            raise ValueError("expansion.linked.deduplicate must be boolean.")
        if not isinstance(self.with_payload, bool):
            raise ValueError("expansion.linked.with_payload must be boolean.")
        if not isinstance(self.with_vectors, bool):
            raise ValueError("expansion.linked.with_vectors must be boolean.")


@dataclass(frozen=True)
class RerankerRuntimeConfig:
    active: str = "cross_encoder"
    fallback: str = "score"

    def validate(self) -> None:
        allowed = {"cross_encoder", "score"}
        if self.active not in allowed:
            raise ValueError(
                f"reranker.active must be one of {sorted(allowed)}. Got: {self.active}"
            )
        if self.fallback not in allowed:
            raise ValueError(
                f"reranker.fallback must be one of {sorted(allowed)}. Got: {self.fallback}"
            )


@dataclass(frozen=True)
class ScoreRerankerRuntimeConfig:
    enabled: bool = True
    max_candidates: int = 50
    normalize_scores: bool = False

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("reranker.score.enabled must be boolean.")
        if self.max_candidates <= 0:
            raise ValueError("reranker.score.max_candidates must be greater than zero.")
        if not isinstance(self.normalize_scores, bool):
            raise ValueError("reranker.score.normalize_scores must be boolean.")


@dataclass(frozen=True)
class CrossEncoderRerankerRuntimeConfig:
    enabled: bool = True
    backend: str = "sentence_transformers"
    model_env: str = "RERANKER_MODEL"
    model_name: str | None = None
    max_candidates: int = 50
    batch_size: int = 8
    normalize_scores: bool = False
    fallback_to_score: bool = True

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("reranker.cross_encoder.enabled must be boolean.")
        if not self.backend.strip():
            raise ValueError("reranker.cross_encoder.backend cannot be empty.")
        if self.model_name is None and not self.model_env.strip():
            raise ValueError(
                "reranker.cross_encoder.model_env cannot be empty when model_name is not set."
            )
        if self.max_candidates <= 0:
            raise ValueError("reranker.cross_encoder.max_candidates must be greater than zero.")
        if self.batch_size <= 0:
            raise ValueError("reranker.cross_encoder.batch_size must be greater than zero.")
        if not isinstance(self.normalize_scores, bool):
            raise ValueError("reranker.cross_encoder.normalize_scores must be boolean.")
        if not isinstance(self.fallback_to_score, bool):
            raise ValueError("reranker.cross_encoder.fallback_to_score must be boolean.")


@dataclass(frozen=True)
class ContextPackerRuntimeConfig:
    max_blocks: int = 8
    min_rerank_score: float | None = 0.0
    deduplicate: bool = True
    allow_empty: bool = True

    def validate(self) -> None:
        if self.max_blocks <= 0:
            raise ValueError("context_packer.default.max_blocks must be greater than zero.")
        if self.min_rerank_score is not None and self.min_rerank_score < 0:
            raise ValueError("context_packer.default.min_rerank_score cannot be negative.")
        if not isinstance(self.deduplicate, bool):
            raise ValueError("context_packer.default.deduplicate must be boolean.")
        if not isinstance(self.allow_empty, bool):
            raise ValueError("context_packer.default.allow_empty must be boolean.")




@dataclass(frozen=True)
class AdaptiveConfidenceRuntimeConfig:
    score_gap_weight: float = 0.34
    exact_coverage_weight: float = 0.26
    source_coverage_weight: float = 0.18
    candidate_presence_weight: float = 0.12
    entropy_weight: float = 0.10

    high_entropy_threshold: float = 0.82
    high_entropy_penalty: float = 0.18
    low_corpus_affinity_penalty: float = 0.18
    no_candidate_penalty: float = 0.45

    def validate(self) -> None:
        total_weight = (
            self.score_gap_weight
            + self.exact_coverage_weight
            + self.source_coverage_weight
            + self.candidate_presence_weight
            + self.entropy_weight
        )

        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"adaptive.confidence.{name} cannot be negative.")

        if total_weight <= 0:
            raise ValueError("adaptive.confidence weights must sum to more than zero.")


@dataclass(frozen=True)
class AdaptiveCostRuntimeConfig:
    memory_lookup_ms: float = 12.0
    sparse_search_ms: float = 45.0
    dense_search_ms: float = 95.0
    fuse_ms: float = 4.0

    score_rerank_base_ms: float = 3.0
    score_rerank_per_candidate_ms: float = 0.08

    ce_rerank_base_ms: float = 90.0
    ce_rerank_per_candidate_ms: float = 8.0

    linked_expand_base_ms: float = 35.0
    linked_expand_per_seed_ms: float = 20.0
    linked_expand_per_link_hint_ms: float = 8.0

    escalate_k_ms: float = 30.0
    escalate_dense_ms: float = 95.0
    relax_scope_ms: float = 95.0

    verify_evidence_base_ms: float = 4.0
    verify_evidence_per_candidate_ms: float = 0.2

    pack_context_base_ms: float = 5.0
    pack_context_per_candidate_ms: float = 0.1

    stop_ms: float = 0.0
    fallback_no_answer_ms: float = 0.0
    min_action_cost_ms: float = 1.0

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"adaptive.cost.{name} cannot be negative.")


@dataclass(frozen=True)
class AdaptiveGainRuntimeConfig:
    sparse_base_gain: float = 0.72
    dense_base_gain: float = 0.68
    memory_base_gain: float = 0.70

    fuse_base_gain: float = 0.42
    score_rerank_base_gain: float = 0.18
    ce_rerank_base_gain: float = 0.52
    linked_expand_base_gain: float = 0.35

    relax_scope_base_gain: float = 0.82
    relax_scope_ineligible_gain: float = 0.02
    relax_scope_verification_failure_bonus: float = 0.12

    verify_evidence_base_gain: float = 0.28
    pack_context_base_gain: float = 0.95

    high_confidence_stop_threshold: float = 0.78
    min_confidence_to_pack: float = 0.55
    medium_confidence_threshold: float = 0.55

    strong_score_gap_threshold: float = 0.22
    flat_score_gap_threshold: float = 0.08
    entropy_high_threshold: float = 0.82

    min_gain: float = 0.0
    max_gain: float = 1.0

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if not 0 <= value <= 1:
                raise ValueError(f"adaptive.gain.{name} must be between 0 and 1.")

        if self.min_gain > self.max_gain:
            raise ValueError("adaptive.gain.min_gain must be <= adaptive.gain.max_gain.")


@dataclass(frozen=True)
class AdaptivePolicyRuntimeConfig:
    max_evaluations: int = 64
    max_hops: int = 6
    min_action_score: float = 0.001
    high_confidence_stop_threshold: float = 0.78
    min_confidence_to_pack: float = 0.55
    allow_repeat_actions: bool = False
    require_verification_before_pack: bool = True
    rerank_after_failed_verification: bool = True
    relax_scope_after_failed_verification: bool = True
    prefer_stop_after_context_packed: bool = True

    @property
    def allow_pack_without_verification(self) -> bool:
        return not self.require_verification_before_pack

    def validate(self) -> None:
        if self.max_evaluations <= 0:
            raise ValueError("adaptive.policy.max_evaluations must be greater than zero.")

        if self.max_hops <= 0:
            raise ValueError("adaptive.policy.max_hops must be greater than zero.")

        if self.min_action_score < 0:
            raise ValueError("adaptive.policy.min_action_score cannot be negative.")

        if not 0 <= self.high_confidence_stop_threshold <= 1:
            raise ValueError(
                "adaptive.policy.high_confidence_stop_threshold must be between 0 and 1."
            )

        if not 0 <= self.min_confidence_to_pack <= 1:
            raise ValueError(
                "adaptive.policy.min_confidence_to_pack must be between 0 and 1."
            )

        if not isinstance(self.allow_repeat_actions, bool):
            raise ValueError("adaptive.policy.allow_repeat_actions must be boolean.")

        if not isinstance(self.require_verification_before_pack, bool):
            raise ValueError(
                "adaptive.policy.require_verification_before_pack "
                "must be boolean."
            )

        if not isinstance(self.rerank_after_failed_verification, bool):
            raise ValueError(
                "adaptive.policy.rerank_after_failed_verification "
                "must be boolean."
            )

        if not isinstance(self.relax_scope_after_failed_verification, bool):
            raise ValueError(
                "adaptive.policy.relax_scope_after_failed_verification "
                "must be boolean."
            )

        if not isinstance(self.prefer_stop_after_context_packed, bool):
            raise ValueError(
                "adaptive.policy.prefer_stop_after_context_packed must be boolean."
            )


@dataclass(frozen=True)
class RetrievalRuntimeConfig:
    qdrant: QdrantVectorStoreRuntimeConfig
    dense: DenseRetrieverRuntimeConfig
    sparse: SparseRetrieverRuntimeConfig
    rrf: RrfFusionRuntimeConfig
    linked_expansion: LinkedExpansionRuntimeConfig
    reranker: RerankerRuntimeConfig
    score_reranker: ScoreRerankerRuntimeConfig
    cross_encoder_reranker: CrossEncoderRerankerRuntimeConfig
    context_packer: ContextPackerRuntimeConfig
    adaptive_cost: AdaptiveCostRuntimeConfig
    adaptive_gain: AdaptiveGainRuntimeConfig
    adaptive_policy: AdaptivePolicyRuntimeConfig
    adaptive_confidence: AdaptiveConfidenceRuntimeConfig
    extensions: RetrievalRuntimeExtensions

    def validate(self) -> None:
        self.qdrant.validate()
        self.dense.validate()
        self.sparse.validate()
        self.rrf.validate()
        self.linked_expansion.validate()
        self.reranker.validate()
        self.score_reranker.validate()
        self.cross_encoder_reranker.validate()
        self.context_packer.validate()
        self.adaptive_cost.validate()
        self.adaptive_gain.validate()
        self.adaptive_policy.validate()
        self.adaptive_confidence.validate()
        self.extensions.validate()


@lru_cache(maxsize=8)
def load_retrieval_runtime_config(
    path: str | Path | None = None,
) -> RetrievalRuntimeConfig:
    config_path = Path(path) if path is not None else DEFAULT_RETRIEVAL_CONFIG_PATH
    config_path = config_path.expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Retrieval config TOML not found: {config_path}")

    raw = rtoml.loads(config_path.read_text(encoding="utf-8"))
    config = parse_retrieval_runtime_config(raw)
    config.validate()

    return config


def parse_retrieval_runtime_config(raw: Mapping[str, Any]) -> RetrievalRuntimeConfig:
    vector_store = _mapping(raw.get("vector_store"), "vector_store")
    qdrant = _mapping(vector_store.get("qdrant"), "vector_store.qdrant")

    retrievers = _mapping(raw.get("retrievers"), "retrievers")
    dense = _mapping(retrievers.get("dense"), "retrievers.dense")
    sparse = _mapping(retrievers.get("sparse"), "retrievers.sparse")

    fusion = _mapping(raw.get("fusion"), "fusion")
    rrf = _mapping(fusion.get("rrf"), "fusion.rrf")

    expansion = _mapping(raw.get("expansion"), "expansion")
    linked_expansion = _mapping(expansion.get("linked"), "expansion.linked")

    reranker = _mapping(raw.get("reranker"), "reranker")
    score_reranker = _mapping(reranker.get("score"), "reranker.score")
    cross_encoder_reranker = _mapping(
        reranker.get("cross_encoder"),
        "reranker.cross_encoder",
    )

    context_packer = _mapping(raw.get("context_packer"), "context_packer")
    default_context_packer = _mapping(
        context_packer.get("default"),
        "context_packer.default",
    )

    adaptive = _mapping(raw.get("adaptive", {}), "adaptive")
    adaptive_cost = _mapping(adaptive.get("cost", {}), "adaptive.cost")
    adaptive_gain = _mapping(adaptive.get("gain", {}), "adaptive.gain")
    adaptive_policy = _mapping(adaptive.get("policy", {}), "adaptive.policy")
    adaptive_confidence = _mapping(adaptive.get("confidence", {}), "adaptive.confidence")

    return RetrievalRuntimeConfig(
        qdrant=QdrantVectorStoreRuntimeConfig(
            collection_name=_required_str(
                qdrant,
                "collection_name",
                section="vector_store.qdrant",
            ),
            dense_vector_name=_optional_str(qdrant, "dense_vector_name", default="dense"),
            sparse_vector_name=_optional_str(qdrant, "sparse_vector_name", default="sparse"),
        ),
        dense=DenseRetrieverRuntimeConfig(
            limit_multiplier=_optional_int(dense, "limit_multiplier", default=1),
            score_threshold=_optional_float_or_none(dense, "score_threshold"),
            with_payload=_optional_bool(dense, "with_payload", default=True),
            with_vectors=_optional_bool(dense, "with_vectors", default=False),
        ),
        sparse=SparseRetrieverRuntimeConfig(
            limit_multiplier=_optional_int(sparse, "limit_multiplier", default=1),
            score_threshold=_optional_float_or_none(sparse, "score_threshold"),
            with_payload=_optional_bool(sparse, "with_payload", default=True),
            with_vectors=_optional_bool(sparse, "with_vectors", default=False),
        ),
        rrf=RrfFusionRuntimeConfig(
            k=_optional_int(rrf, "k", default=60),
            dense_weight=_optional_float(rrf, "dense_weight", default=1.0),
            sparse_weight=_optional_float(rrf, "sparse_weight", default=1.0),
            limit_multiplier=_optional_int(rrf, "limit_multiplier", default=1),
        ),
        linked_expansion=LinkedExpansionRuntimeConfig(
            enabled=_optional_bool(linked_expansion, "enabled", default=True),
            max_seed_candidates=_optional_int(
                linked_expansion,
                "max_seed_candidates",
                default=10,
            ),
            max_links_per_seed=_optional_int(
                linked_expansion,
                "max_links_per_seed",
                default=5,
            ),
            max_chunks_per_link=_optional_int(
                linked_expansion,
                "max_chunks_per_link",
                default=6,
            ),
            deduplicate=_optional_bool(
                linked_expansion,
                "deduplicate",
                default=True,
            ),
            with_payload=_optional_bool(
                linked_expansion,
                "with_payload",
                default=True,
            ),
            with_vectors=_optional_bool(
                linked_expansion,
                "with_vectors",
                default=False,
            ),
        ),
        reranker=RerankerRuntimeConfig(
            active=_optional_str(reranker, "active", default="cross_encoder"),
            fallback=_optional_str(reranker, "fallback", default="score"),
        ),
        score_reranker=ScoreRerankerRuntimeConfig(
            enabled=_optional_bool(score_reranker, "enabled", default=True),
            max_candidates=_optional_int(score_reranker, "max_candidates", default=50),
            normalize_scores=_optional_bool(
                score_reranker,
                "normalize_scores",
                default=False,
            ),
        ),
        cross_encoder_reranker=CrossEncoderRerankerRuntimeConfig(
            enabled=_optional_bool(cross_encoder_reranker, "enabled", default=True),
            backend=_optional_str(
                cross_encoder_reranker,
                "backend",
                default="sentence_transformers",
            ),
            model_env=_optional_str(
                cross_encoder_reranker,
                "model_env",
                default="RERANKER_MODEL",
            ),
            model_name=_optional_str_or_none(cross_encoder_reranker, "model_name"),
            max_candidates=_optional_int(
                cross_encoder_reranker,
                "max_candidates",
                default=50,
            ),
            batch_size=_optional_int(cross_encoder_reranker, "batch_size", default=8),
            normalize_scores=_optional_bool(
                cross_encoder_reranker,
                "normalize_scores",
                default=False,
            ),
            fallback_to_score=_optional_bool(
                cross_encoder_reranker,
                "fallback_to_score",
                default=True,
            ),
        ),
        context_packer=ContextPackerRuntimeConfig(
            max_blocks=_optional_int(default_context_packer, "max_blocks", default=8),
            min_rerank_score=_optional_float_or_none(
                default_context_packer,
                "min_rerank_score",
            ),
            deduplicate=_optional_bool(
                default_context_packer,
                "deduplicate",
                default=True,
            ),
            allow_empty=_optional_bool(
                default_context_packer,
                "allow_empty",
                default=True,
            ),
        ),
        adaptive_cost=AdaptiveCostRuntimeConfig(
            memory_lookup_ms=_optional_float(adaptive_cost, "memory_lookup_ms", default=12.0),
            sparse_search_ms=_optional_float(adaptive_cost, "sparse_search_ms", default=45.0),
            dense_search_ms=_optional_float(adaptive_cost, "dense_search_ms", default=95.0),
            fuse_ms=_optional_float(adaptive_cost, "fuse_ms", default=4.0),
            score_rerank_base_ms=_optional_float(adaptive_cost, "score_rerank_base_ms", default=3.0),
            score_rerank_per_candidate_ms=_optional_float(adaptive_cost, "score_rerank_per_candidate_ms", default=0.08),
            ce_rerank_base_ms=_optional_float(adaptive_cost, "ce_rerank_base_ms", default=90.0),
            ce_rerank_per_candidate_ms=_optional_float(adaptive_cost, "ce_rerank_per_candidate_ms", default=8.0),
            linked_expand_base_ms=_optional_float(adaptive_cost, "linked_expand_base_ms", default=35.0),
            linked_expand_per_seed_ms=_optional_float(adaptive_cost, "linked_expand_per_seed_ms", default=20.0),
            linked_expand_per_link_hint_ms=_optional_float(adaptive_cost, "linked_expand_per_link_hint_ms", default=8.0),
            escalate_k_ms=_optional_float(adaptive_cost, "escalate_k_ms", default=30.0),
            escalate_dense_ms=_optional_float(adaptive_cost, "escalate_dense_ms", default=95.0),
            relax_scope_ms=_optional_float(adaptive_cost, "relax_scope_ms", default=95.0),
            verify_evidence_base_ms=_optional_float(adaptive_cost, "verify_evidence_base_ms", default=4.0),
            verify_evidence_per_candidate_ms=_optional_float(adaptive_cost, "verify_evidence_per_candidate_ms", default=0.2),
            pack_context_base_ms=_optional_float(adaptive_cost, "pack_context_base_ms", default=5.0),
            pack_context_per_candidate_ms=_optional_float(adaptive_cost, "pack_context_per_candidate_ms", default=0.1),
            stop_ms=_optional_float(adaptive_cost, "stop_ms", default=0.0),
            fallback_no_answer_ms=_optional_float(adaptive_cost, "fallback_no_answer_ms", default=0.0),
            min_action_cost_ms=_optional_float(adaptive_cost, "min_action_cost_ms", default=1.0),
        ),
        adaptive_gain=AdaptiveGainRuntimeConfig(
            sparse_base_gain=_optional_float(adaptive_gain, "sparse_base_gain", default=0.72),
            dense_base_gain=_optional_float(adaptive_gain, "dense_base_gain", default=0.68),
            memory_base_gain=_optional_float(adaptive_gain, "memory_base_gain", default=0.70),
            fuse_base_gain=_optional_float(adaptive_gain, "fuse_base_gain", default=0.42),
            score_rerank_base_gain=_optional_float(adaptive_gain, "score_rerank_base_gain", default=0.18),
            ce_rerank_base_gain=_optional_float(adaptive_gain, "ce_rerank_base_gain", default=0.52),
            linked_expand_base_gain=_optional_float(adaptive_gain, "linked_expand_base_gain", default=0.35),
            relax_scope_base_gain=_optional_float(
                adaptive_gain,
                "relax_scope_base_gain",
                default=0.82,
            ),
            relax_scope_ineligible_gain=_optional_float(
                adaptive_gain,
                "relax_scope_ineligible_gain",
                default=0.02,
            ),
            relax_scope_verification_failure_bonus=_optional_float(
                adaptive_gain,
                "relax_scope_verification_failure_bonus",
                default=0.12,
            ),
            verify_evidence_base_gain=_optional_float(adaptive_gain, "verify_evidence_base_gain", default=0.28),
            pack_context_base_gain=_optional_float(adaptive_gain, "pack_context_base_gain", default=0.95),
            high_confidence_stop_threshold=_optional_float(adaptive_gain, "high_confidence_stop_threshold", default=0.78),
            medium_confidence_threshold=_optional_float(adaptive_gain, "medium_confidence_threshold", default=0.55),
            strong_score_gap_threshold=_optional_float(adaptive_gain, "strong_score_gap_threshold", default=0.22),
            flat_score_gap_threshold=_optional_float(adaptive_gain, "flat_score_gap_threshold", default=0.08),
            entropy_high_threshold=_optional_float(adaptive_gain, "entropy_high_threshold", default=0.82),
            min_gain=_optional_float(adaptive_gain, "min_gain", default=0.0),
            max_gain=_optional_float(adaptive_gain, "max_gain", default=1.0),
        ),
        adaptive_policy=AdaptivePolicyRuntimeConfig(
            max_evaluations=_optional_int(adaptive_policy, "max_evaluations", default=64),
            max_hops=_optional_int(adaptive_policy, "max_hops", default=6),
            min_action_score=_optional_float(adaptive_policy, "min_action_score", default=0.001),
            high_confidence_stop_threshold=_optional_float(adaptive_policy, "high_confidence_stop_threshold", default=0.78),
            min_confidence_to_pack=_optional_float(adaptive_policy, "min_confidence_to_pack", default=0.55),
            allow_repeat_actions=_optional_bool(
                adaptive_policy,
                "allow_repeat_actions",
                default=False,
            ),
            require_verification_before_pack=_optional_bool(
                adaptive_policy,
                "require_verification_before_pack",
                default=not _optional_bool(
                    adaptive_policy,
                    "allow_pack_without_verification",
                    default=False,
                ),
            ),
            rerank_after_failed_verification=_optional_bool(
                adaptive_policy,
                "rerank_after_failed_verification",
                default=True,
            ),
            relax_scope_after_failed_verification=_optional_bool(
                adaptive_policy,
                "relax_scope_after_failed_verification",
                default=True,
            ),
            prefer_stop_after_context_packed=_optional_bool(
                adaptive_policy,
                "prefer_stop_after_context_packed",
                default=True,
            ),
        ),
        adaptive_confidence=AdaptiveConfidenceRuntimeConfig(
            score_gap_weight=_optional_float(adaptive_confidence, "score_gap_weight", default=0.34),
            exact_coverage_weight=_optional_float(adaptive_confidence, "exact_coverage_weight", default=0.26),
            source_coverage_weight=_optional_float(adaptive_confidence, "source_coverage_weight", default=0.18),
            candidate_presence_weight=_optional_float(adaptive_confidence, "candidate_presence_weight", default=0.12),
            entropy_weight=_optional_float(adaptive_confidence, "entropy_weight", default=0.10),
            high_entropy_threshold=_optional_float(
                adaptive_confidence,
                "high_entropy_threshold",
                default=0.82,
            ),
            high_entropy_penalty=_optional_float(adaptive_confidence, "high_entropy_penalty", default=0.18),
            low_corpus_affinity_penalty=_optional_float(adaptive_confidence, "low_corpus_affinity_penalty", default=0.18),
            no_candidate_penalty=_optional_float(adaptive_confidence, "no_candidate_penalty", default=0.45),
        ),
        extensions=parse_retrieval_runtime_extensions(raw),
    )


def _mapping(value: Any, section: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{section} must be a TOML table.")
    return value


def _required_str(data: Mapping[str, Any], key: str, *, section: str) -> str:
    value = data.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"{section}.{key} is required.")
    return str(value).strip()


def _optional_str(data: Mapping[str, Any], key: str, *, default: str) -> str:
    value = data.get(key)
    if value is None or not str(value).strip():
        return default
    return str(value).strip()


def _optional_str_or_none(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _optional_int(data: Mapping[str, Any], key: str, *, default: int) -> int:
    value = data.get(key)
    if value is None:
        return default
    return int(value)


def _optional_float(data: Mapping[str, Any], key: str, *, default: float) -> float:
    value = data.get(key)
    if value is None:
        return default
    return float(value)


def _optional_float_or_none(data: Mapping[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    return float(value)


def _optional_bool(data: Mapping[str, Any], key: str, *, default: bool) -> bool:
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

    raise ValueError(f"{key} must be boolean-like.")
