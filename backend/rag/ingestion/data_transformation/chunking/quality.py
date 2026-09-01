from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rag.ingestion.data_transformation.chunking.config import get_chunking_config
from rag.ingestion.data_transformation.chunking.models import Chunk
from rag.ingestion.data_transformation.chunking.config import ChunkingConfig
from rag.ingestion.data_transformation.chunking.config import build_chunk_token_budget
from rag.ingestion.data_transformation.chunking.tokenization import count_tokens


#------------------------public api-------------------------

def evaluate_chunks_quality(
    chunks: Sequence[Chunk],
    *,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    if not chunks:
        return []

    active_config = (
        config
        if config is not None
        else get_chunking_config()
    )

    budget = build_chunk_token_budget(
        active_config
    )

    output: list[Chunk] = []

    for chunk in chunks:
        if not isinstance(chunk, Chunk):
            raise TypeError(
                "Quality evaluation requires Chunk objects"
            )

        output.append(
            _evaluate_one(
                chunk=chunk,
                config=active_config,
                budget=budget,
            )
        )

    return output


#------------------------evaluation-------------------------

def _evaluate_one(
    *,
    chunk: Chunk,
    config: ChunkingConfig,
    budget: Any,
) -> Chunk:
    retrieval_text = (
        chunk.text
        or ""
    ).strip()

    content_text = _content_text(
        chunk
    )

    retrieval_token_count = (
        count_tokens(
            retrieval_text,
            role=config.tokenization.role,
            add_special_tokens=(
                config.tokenization.add_special_tokens
            ),
        )
        if retrieval_text
        else 0
    )

    content_token_count = _content_token_count(
        chunk,
        content_text=content_text,
        config=config,
    )

    subtype = (
        chunk.subtype
        or _text(
            chunk.metadata.get("subtype")
        )
    ).strip().lower()

    atomic = subtype in (
        config.atomic_policy.subtypes
    )

    dependent_fragment = _is_dependent_fragment(
        text=content_text,
        subtype=subtype,
    )

    semantic_packed = bool(
        chunk.metadata.get("semantic_packed")
        or chunk.metadata.get("retrieval_packed")
    )

    complete_unit = (
        atomic
        or semantic_packed
        or _looks_complete_unit(content_text)
    )

    context_available = bool(
        (chunk.heading or "").strip()
        or (chunk.context or "").strip()
        or chunk.metadata.get("retrieval_context_applied")
    )

    hard_limit = _hard_limit(
        chunk,
        budget=budget,
    )

    within_hard_limit = (
        retrieval_token_count
        <= hard_limit
    )

    semantic_density = (
        content_token_count
        >= budget.minimum_tokens
        or complete_unit
    )

    checks = {
        "text_present": bool(content_text),
        "context_available": context_available,
        "retrieval_complete": not dependent_fragment,
        "within_hard_limit": within_hard_limit,
        "semantic_density": semantic_density,
    }

    passed_checks = sum(
        1
        for value in checks.values()
        if value
    )

    score = (
        passed_checks
        / len(checks)
    )

    issues = [
        key
        for key, value in checks.items()
        if not value
    ]

    diagnostics = {
        "token_count": retrieval_token_count,
        "retrieval_token_count": retrieval_token_count,
        "content_token_count": content_token_count,
        "minimum_tokens": budget.minimum_tokens,
        "preferred_tokens": budget.preferred_tokens,
        "strict_tokens": budget.strict_tokens,
        "effective_hard_tokens": hard_limit,
        "content_hard_tokens": budget.effective_chunk_hard_tokens,
        "model_max_input_tokens": budget.model_max_input_tokens,
        "below_minimum": content_token_count < budget.minimum_tokens,
        "above_preferred": content_token_count > budget.preferred_tokens,
        "above_strict": content_token_count > budget.strict_tokens,
        "above_hard": retrieval_token_count > hard_limit,
        "subtype": subtype,
        "atomic": atomic,
        "complete_unit": complete_unit,
        "dependent_fragment": dependent_fragment,
        "semantic_packed": semantic_packed,
        "retrieval_context_applied": bool(
            chunk.metadata.get("retrieval_context_applied")
        ),
        "checks": checks,
        "issues": issues,
    }

    metadata = dict(
        chunk.metadata
    )

    metadata["quality_score"] = round(
        score,
        6,
    )

    metadata["quality_diagnostics"] = diagnostics
    metadata["quality_issues"] = issues

    return chunk.copy_with(
        tokens=retrieval_token_count,
        metadata=metadata,
    )


#------------------------content resolution-------------------------

def _content_text(
    chunk: Chunk,
) -> str:
    text = (
        chunk.text
        or ""
    ).strip()

    prefix = chunk.metadata.get(
        "retrieval_prefix"
    )

    if not isinstance(prefix, str):
        return text

    normalized_prefix = prefix.strip()

    if not normalized_prefix:
        return text

    expected = f"{normalized_prefix}\n\n"

    if text.startswith(expected):
        return text[
            len(expected):
        ].strip()

    return text


def _content_token_count(
    chunk: Chunk,
    *,
    content_text: str,
    config: ChunkingConfig,
) -> int:
    value = chunk.metadata.get(
        "content_token_count"
    )

    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    ):
        return value

    return (
        count_tokens(
            content_text,
            role=config.tokenization.role,
            add_special_tokens=(
                config.tokenization.add_special_tokens
            ),
        )
        if content_text
        else 0
    )


def _hard_limit(
    chunk: Chunk,
    *,
    budget: Any,
) -> int:
    if chunk.metadata.get("retrieval_context_applied") is True:
        return budget.model_max_input_tokens

    return budget.effective_chunk_hard_tokens


#------------------------dependency signals-------------------------

def _is_dependent_fragment(
    *,
    text: str,
    subtype: str,
) -> bool:
    if subtype not in {
        "paragraph",
        "blockquote",
    }:
        return False

    if not text:
        return True

    if text.endswith(":"):
        return True

    normalized = " ".join(
        text.lower().split()
    )

    return normalized.endswith(
        (
            "as follows",
            "as follows.",
            "shown below",
            "shown below.",
            "looks like this",
            "looks like this.",
            "in the following example",
            "in the following example.",
            "in the following examples",
            "in the following examples.",
        )
    )


#------------------------completeness-------------------------

def _looks_complete_unit(
    text: str,
) -> bool:
    if not text:
        return False

    stripped = text.rstrip(
        "\"'”’)]}"
    )

    if stripped.endswith(
        (
            ".",
            "!",
            "?",
        )
    ):
        return True

    if "\n" in text:
        return True

    return False


#------------------------helpers-------------------------

def _text(
    value: Any,
) -> str:
    return str(
        value
        or ""
    ).strip()
