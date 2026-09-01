from __future__ import annotations

from rag.retrieval.adaptive.actions import (
    ACTION_PRECONDITIONS,
    RetrievalAction,
    RetrievalActionDecision,
    RetrievalActionEvaluation,
    RetrievalActionPreconditions,
    RetrievalActionRecord,
    RetrievalActionStatus,
    action_preconditions,
    action_produces_candidates,
    action_requires_embedding,
    completed_actions,
    is_terminal_action,
    non_terminal_actions,
)
from rag.retrieval.adaptive.confidence import (
    ConfidenceEstimator,
    StaticConfidenceConfig,
    StaticConfidenceEstimator,
    apply_confidence_estimate,
    estimate_exact_coverage,
    estimate_source_coverage,
)
from rag.retrieval.adaptive.cost import (
    ActionCostEstimate,
    ActionCostModel,
    StaticActionCostConfig,
    StaticActionCostModel,
)
from rag.retrieval.adaptive.gain import (
    ActionGainEstimate,
    ActionGainModel,
    StaticActionGainConfig,
    StaticActionGainModel,
)
from rag.retrieval.adaptive.policy import (
    AdaptivePolicyConfig,
    GreedyAdaptivePolicy,
    explain_decision,
    validate_policy_sample,
)
from rag.retrieval.adaptive.signals import (
    CorpusAffinityProvider,
    DefaultQuerySignalExtractor,
    QuerySignalExtractorConfig,
    QuerySignalTerms,
    SetCorpusAffinityProvider,
    clean_unique_strings,
    estimate_lexical_specificity,
    extract_query_signal_terms,
    validate_query_signal_extraction_sample,
)
from rag.retrieval.adaptive.state import (
    AdaptiveRetrievalState,
    QuerySignals,
    RetrievalConfidence,
    RetrievalStateSnapshot,
    candidate_identity,
    top_identity_set_changed,
)

__all__ = [
    "build_adaptive_retrieval_graph",
    "AdaptiveRetrievalGraphRuntimeConfig",
    "AdaptiveRetrievalGraphRuntime",
    "adaptive_result_to_graph_output",
    "estimate_source_coverage",
    "estimate_exact_coverage",
    "apply_confidence_estimate",
    "StaticConfidenceEstimator",
    "StaticConfidenceConfig",
    "ConfidenceEstimator",
    "ACTION_PRECONDITIONS",
    "ActionCostEstimate",
    "ActionCostModel",
    "ActionGainEstimate",
    "ActionGainModel",
    "AdaptivePolicyConfig",
    "AdaptiveRetrievalState",
    "CorpusAffinityProvider",
    "DefaultQuerySignalExtractor",
    "GreedyAdaptivePolicy",
    "QuerySignalExtractorConfig",
    "QuerySignalTerms",
    "QuerySignals",
    "RetrievalAction",
    "RetrievalActionDecision",
    "RetrievalActionEvaluation",
    "RetrievalActionPreconditions",
    "RetrievalActionRecord",
    "RetrievalActionStatus",
    "RetrievalConfidence",
    "RetrievalStateSnapshot",
    "SetCorpusAffinityProvider",
    "StaticActionCostConfig",
    "StaticActionCostModel",
    "StaticActionGainConfig",
    "StaticActionGainModel",
    "action_preconditions",
    "action_produces_candidates",
    "action_requires_embedding",
    "candidate_identity",
    "clean_unique_strings",
    "completed_actions",
    "estimate_lexical_specificity",
    "explain_decision",
    "extract_query_signal_terms",
    "is_terminal_action",
    "non_terminal_actions",
    "top_identity_set_changed",
    "validate_policy_sample",
    "validate_query_signal_extraction_sample",
]

from rag.retrieval.adaptive.output import adaptive_result_to_graph_output

from rag.retrieval.adaptive.graph_runtime import (
    AdaptiveRetrievalGraphRuntime,
    AdaptiveRetrievalGraphRuntimeConfig,
    build_adaptive_retrieval_graph,
)
