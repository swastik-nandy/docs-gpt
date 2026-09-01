from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Mapping
from typing import Any

from rag.retrieval.contracts.request import RetrievalRequest, SourceConstraints, TenantContext
from rag.retrieval.service import RetrievalService, RetrievalServiceConfig
from tenancy.dev import load_dev_identity_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect retrieval candidates and packed chunks without calling generation LLM.",
    )
    parser.add_argument("query")
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--conversation-id", default=None)

    parser.add_argument("--job-id", default=None)
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--parser-doc-id", default=None)
    parser.add_argument("--section-id", default=None)
    parser.add_argument("--ir-id", default=None)

    parser.add_argument(
        "--no-source-constraints",
        action="store_true",
        help="Do not apply job/source/parser/section/IR filters.",
    )
    parser.add_argument(
        "--no-job-filter",
        action="store_true",
        help="Keep other source constraints, but drop job_id.",
    )
    parser.add_argument(
        "--disable-stateful-resolution",
        action="store_true",
        help="Do not fetch conversation state or rewrite referential queries.",
    )

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=10)
    parser.add_argument("--text-chars", type=int, default=700)
    parser.add_argument("--pretty", action="store_true")
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    dev = load_dev_identity_config()

    tenant = TenantContext(
        tenant_id=args.tenant_id or dev.tenant_id,
        user_id=args.user_id or dev.user_id,
        session_id=args.session_id or dev.session_id,
    )

    if args.no_source_constraints:
        source_constraints = SourceConstraints()
    else:
        source_constraints = SourceConstraints(
            job_id=None if args.no_job_filter else (args.job_id or dev.job_id),
            source_id=args.source_id or dev.source_id,
            parser_doc_id=args.parser_doc_id or dev.parser_doc_id,
            section_id=args.section_id or dev.section_id,
            ir_id=args.ir_id or dev.ir_id,
        )

    request = RetrievalRequest.create(
        query=args.query,
        tenant=tenant,
        conversation_id=args.conversation_id or dev.conversation_id,
        source_constraints=source_constraints,
        top_k=args.top_k,
        metadata={
            "cli": "inspect_chunks",
            "no_generation_llm": True,
        },
    )

    service = RetrievalService(
        config=RetrievalServiceConfig(
            include_debug_metadata=True,
            enable_stateful_resolution=not args.disable_stateful_resolution,
        )
    )

    output = await service.retrieve(
        request,
        metadata={
            "source": "inspect_chunks",
            "no_generation_llm": True,
        },
    )

    payload = {
        "query": args.query,
        "tenant_id": tenant.tenant_id,
        "conversation_id": request.conversation_id,
        "source_constraints": {
            "job_id": source_constraints.job_id,
            "source_id": source_constraints.source_id,
            "parser_doc_id": source_constraints.parser_doc_id,
            "section_id": source_constraints.section_id,
            "ir_id": source_constraints.ir_id,
            "is_global": source_constraints.is_global,
        },
        "stateful_retrieval": (output.get("metadata") or {}).get("stateful_retrieval"),
        "retrieval_trace_id": (output.get("metadata") or {}).get("retrieval_trace_id"),
        "decision": _decision_payload(output.get("decision")),
        "action_summary": _action_summary(output),
        "counts": {
            "reranked_candidates": len(output.get("reranked_candidates") or []),
            "context_blocks": len(output.get("context_blocks") or []),
            "warnings": len(output.get("warnings") or []),
            "errors": len(output.get("errors") or []),
        },
        "context_blocks": [
            _context_block_payload(block, text_chars=args.text_chars)
            for block in (output.get("context_blocks") or [])
        ],
        "candidates": [
            _candidate_payload(candidate, text_chars=args.text_chars)
            for candidate in (output.get("reranked_candidates") or [])[: max(0, args.candidate_limit)]
        ],
        "warnings": [repr(item) for item in output.get("warnings") or []],
        "errors": [repr(item) for item in output.get("errors") or []],
    }

    print(json.dumps(payload, indent=2 if args.pretty else None, default=str, sort_keys=True))


def _decision_payload(decision: Any) -> dict[str, Any]:
    if decision is None:
        return {}

    return {
        "is_sufficient": getattr(decision, "is_sufficient", None),
        "confidence": str(getattr(decision, "confidence", "")),
        "confidence_score": getattr(decision, "confidence_score", None),
        "reason": getattr(decision, "reason", None),
        "metadata": _mapping_or_empty(getattr(decision, "metadata", None)),
    }


def _action_summary(output: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = output.get("metadata") or {}
    actions = metadata.get("actions") or []
    result = []

    for action in actions:
        if not isinstance(action, Mapping):
            continue

        result.append(
            {
                "action": action.get("action"),
                "status": action.get("status"),
                "input_count": action.get("input_count"),
                "output_count": action.get("output_count"),
                "after_count": action.get("after_count"),
                "finalization": action.get("finalization"),
                "reason": action.get("reason"),
                "error": action.get("error"),
            }
        )

    return result


def _context_block_payload(block: Any, *, text_chars: int) -> dict[str, Any]:
    text = _text_from_object(block)

    return {
        "type": type(block).__name__,
        "block_id": _value(block, "block_id"),
        "source_id": _value(block, "source_id"),
        "chunk_id": _value(block, "chunk_id"),
        "heading": _value(block, "heading"),
        "score": _value(block, "score"),
        "text_preview": text[:text_chars],
        "text_len": len(text),
        "metadata": _mapping_or_empty(_value(block, "metadata")),
        "metadata_keys": sorted(_mapping_or_empty(_value(block, "metadata")).keys()),
    }


def _candidate_payload(candidate: Any, *, text_chars: int) -> dict[str, Any]:
    inner = getattr(candidate, "candidate", candidate)
    ref = getattr(inner, "ref", None)

    text = _text_from_candidate(candidate)
    payloads = _payloads(candidate)

    return {
        "type": type(candidate).__name__,
        "inner_type": type(inner).__name__,
        "rank": _value(candidate, "rank"),
        "rerank_score": _value(candidate, "rerank_score"),
        "score": _first_value(candidate, inner, "score"),
        "point_id": _first_value(candidate, inner, ref, "point_id"),
        "source_id": _first_value(candidate, inner, ref, "source_id"),
        "chunk_id": _first_value(candidate, inner, ref, "chunk_id"),
        "parser_doc_id": _first_value(candidate, inner, ref, "parser_doc_id"),
        "section_id": _first_value(candidate, inner, ref, "section_id"),
        "heading": (
            _first_value(candidate, inner, "heading")
            or _first_value(candidate, inner, "heading_path")
            or _payload_value(payloads, "heading")
            or _payload_value(payloads, "heading_path")
        ),
        "payload_keys": sorted({key for payload in payloads for key in payload.keys()}),
        "text_preview": text[:text_chars],
        "text_len": len(text),
        "has_text": bool(text.strip()),
    }


def _text_from_candidate(candidate: Any) -> str:
    objects = [
        candidate,
        getattr(candidate, "candidate", None),
        getattr(getattr(candidate, "candidate", None), "ref", None),
    ]

    for obj in objects:
        text = _text_from_object(obj)
        if text:
            return text

    for payload in _payloads(candidate):
        for key in (
            "text",
            "content",
            "chunk_text",
            "raw_text",
            "markdown",
            "body",
            "page_content",
            "document",
        ):
            value = payload.get(key)
            if value is not None and str(value).strip():
                return str(value)

    return ""


def _text_from_object(obj: Any) -> str:
    if obj is None:
        return ""

    for attr in (
        "text",
        "content",
        "chunk_text",
        "raw_text",
        "markdown",
        "body",
        "page_content",
        "document",
    ):
        value = _value(obj, attr)
        if value is not None and str(value).strip():
            return str(value)

    return ""


def _payloads(obj: Any) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []

    queue = [
        obj,
        getattr(obj, "candidate", None),
        getattr(getattr(obj, "candidate", None), "ref", None),
    ]

    for item in queue:
        if item is None:
            continue

        for attr in ("payload", "metadata"):
            value = _value(item, attr)
            if isinstance(value, Mapping):
                payloads.append(dict(value))

    return payloads


def _payload_value(payloads: list[dict[str, Any]], key: str) -> Any:
    for payload in payloads:
        value = payload.get(key)
        if value is not None and str(value).strip():
            return value

    return None


def _first_value(*objects_and_key: Any) -> Any:
    *objects, key = objects_and_key

    for obj in objects:
        value = _value(obj, key)
        if value is not None and str(value).strip():
            return value

    return None


def _value(obj: Any, key: str) -> Any:
    if obj is None:
        return None

    if isinstance(obj, Mapping):
        return obj.get(key)

    return getattr(obj, key, None)


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)

    return {}


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
