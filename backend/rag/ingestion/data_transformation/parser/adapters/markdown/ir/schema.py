# rag/ingestion/data_transformation/parser/adapters/markdown/ir/schema.py

from typing import Any, Dict, List, Literal, TypedDict


# ---------------- BASE TYPES ---------------- #

BaseType = Literal[
    "text",
    "code",
    "list",
    "table",
    "component",
]


# ---------------- FORM TYPES ---------------- #

FormType = Literal[
    "paragraph",
    "sentence",
    "item",
    "row",
    "block",
]


# ---------------- FUNCTION TYPES ---------------- #

FunctionType = Literal[
    "instruction",
    "explanation",
    "reference",
    "configuration",
    "example",
    "warning",
    "mixed",
]


# ---------------- RELATION TYPES ---------------- #

RelationType = Literal[
    "implements",
    "supports",
    "belongs_to",
    "follows",
    "references",
]


# ---------------- SIGNAL MAP ---------------- #

class SignalMap(TypedDict, total=False):
    is_command_like: bool
    starts_with_action: bool
    is_labeled_action: bool
    is_directive: bool
    has_flags: bool
    is_key_value: bool
    has_path: bool
    has_assignment: bool
    has_merge_conflict_marker: bool
    is_short: bool
    is_long: bool
    has_numbers: bool
    has_symbols: bool
    sentence_like: bool
    label: str


# ---------------- RELATION ---------------- #

class Relation(TypedDict, total=False):
    type: RelationType
    target_block_id: str
    source_block_id: str
    confidence: float
    reason: str


# ---------------- CORE IR BLOCK ---------------- #

class Block(TypedDict, total=False):
    # identity
    id: str
    doc_id: str
    section_id: str
    block_index: int

    # source position
    source_type: str
    source_index: int

    # semantic identity
    type: BaseType
    form: FormType
    function: FunctionType
    confidence: float

    # content
    text: str
    retrieval_text: str

    # structural signals
    signals: SignalMap

    # context
    heading: str
    context: str
    local_context: str

    # graph
    relationships: List[Relation]

    # metadata passthrough
    meta: Dict[str, Any]