from __future__ import annotations

from enum import Enum


class RetrievalMode(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID_RRF = "hybrid_rrf"
    STRUCTURE_FIRST = "structure_first"
    CORRECTIVE = "corrective"
    AGENTIC = "agentic"


class QueryIntent(str, Enum):
    UNKNOWN = "unknown"
    CONCEPTUAL = "conceptual"
    HOW_TO = "how_to"
    API_REFERENCE = "api_reference"
    CONFIGURATION = "configuration"
    TROUBLESHOOTING = "troubleshooting"
    CODE_EXAMPLE = "code_example"
    COMPARISON = "comparison"
    NAVIGATION = "navigation"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"
    IMPLEMENTATION = "implementation"
    CHANGE_REQUEST = "change_request"
    FOLLOW_UP = "follow_up"


class QueryDefect(str, Enum):
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    AMBIGUOUS_REFERENT = "ambiguous_referent"
    CHAT_DEPENDENT = "chat_dependent"
    TYPO_NOISE = "typo_noise"
    MIXED_INTENTS = "mixed_intents"
    LOW_INFORMATION = "low_information"
    OVER_BROAD = "over_broad"
    UNDER_SPECIFIED_FILTERS = "under_specified_filters"


class CandidateOrigin(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    RRF = "rrf"
    STRUCTURE_EXPANSION = "structure_expansion"
    CROSS_ENCODER = "cross_encoder"
    CORRECTIVE_RETRY = "corrective_retry"
    MEMORY = "memory"
    LLAMAINDEX = "llamaindex"
    GRAPH_TOOL = "graph_tool"


class ConfidenceBand(str, Enum):
    UNKNOWN = "unknown"
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class MemoryScope(str, Enum):
    NONE = "none"
    SESSION = "session"
    USER = "user"
    TENANT = "tenant"


class QueryProcessingMode(str, Enum):
    RULE_BASED = "rule_based"
    LLM_ASSISTED = "llm_assisted"
    HYBRID = "hybrid"


class GenerationMode(str, Enum):
    RETRIEVAL_ONLY = "retrieval_only"
    GROUNDED_ANSWER = "grounded_answer"
    CHAT = "chat"
    TOOL_ASSISTED = "tool_assisted"