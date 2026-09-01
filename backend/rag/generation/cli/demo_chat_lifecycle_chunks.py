from __future__ import annotations

import dataclasses
import inspect
import itertools
import json
import os
import runpy
import traceback
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any


RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("rag/ingestion/artifacts/debug/query_lifecycle_chunks") / RUN_ID
EVENTS_FILE = RUN_DIR / "events.jsonl"
RETRIEVAL_DIR = RUN_DIR / "retrieval"
PROMPT_DIR = RUN_DIR / "prompts"

CALL_IDS = itertools.count(1)
PROMPT_IDS = itertools.count(1)


def _mkdirs() -> None:
    RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)


def _write_event(event: str, payload: dict[str, Any]) -> None:
    _mkdirs()
    row = {
        "ts": datetime.now().isoformat(timespec="milliseconds"),
        "event": event,
        "payload": payload,
    }
    with EVENTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _safe_jsonable(obj: Any, *, depth: int = 0, seen: set[int] | None = None) -> Any:
    if seen is None:
        seen = set()

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if depth > 10:
        return repr(obj)[:800]

    obj_id = id(obj)
    if obj_id in seen:
        return "<cycle>"

    if isinstance(obj, (bytes, bytearray)):
        return f"<{type(obj).__name__} len={len(obj)}>"

    seen.add(obj_id)

    try:
        if dataclasses.is_dataclass(obj):
            return {
                field.name: _safe_jsonable(getattr(obj, field.name), depth=depth + 1, seen=seen)
                for field in dataclasses.fields(obj)
            }

        if hasattr(obj, "model_dump") and callable(obj.model_dump):
            return _safe_jsonable(obj.model_dump(), depth=depth + 1, seen=seen)

        if isinstance(obj, Mapping):
            return {
                str(k): _safe_jsonable(v, depth=depth + 1, seen=seen)
                for k, v in obj.items()
            }

        if isinstance(obj, (list, tuple, set, frozenset)):
            return [
                _safe_jsonable(v, depth=depth + 1, seen=seen)
                for v in list(obj)[:200]
            ]

        if hasattr(obj, "__dict__"):
            return {
                str(k): _safe_jsonable(v, depth=depth + 1, seen=seen)
                for k, v in vars(obj).items()
                if not str(k).startswith("_")
            }

        return repr(obj)[:800]
    finally:
        seen.discard(obj_id)


def _find_key(obj: Any, names: tuple[str, ...], *, depth: int = 0) -> Any:
    if depth > 8:
        return None

    if isinstance(obj, Mapping):
        for name in names:
            if name in obj and obj[name] not in (None, "", [], {}):
                return obj[name]

        for value in obj.values():
            found = _find_key(value, names, depth=depth + 1)
            if found not in (None, "", [], {}):
                return found

    elif isinstance(obj, list):
        for item in obj[:40]:
            found = _find_key(item, names, depth=depth + 1)
            if found not in (None, "", [], {}):
                return found

    return None


def _find_text(obj: Any) -> str:
    preferred = _find_key(
        obj,
        (
            "retrieval_text",
            "text",
            "content",
            "page_content",
            "document_text",
            "chunk_text",
        ),
    )

    if isinstance(preferred, str) and preferred.strip():
        return preferred.strip()

    candidates: list[str] = []

    def walk(x: Any, depth: int = 0) -> None:
        if depth > 7:
            return

        if isinstance(x, str):
            stripped = " ".join(x.split())
            if len(stripped) >= 80 and "/" not in stripped[:25]:
                candidates.append(stripped)
            return

        if isinstance(x, Mapping):
            for key, value in x.items():
                if str(key).lower() in {"source", "source_path", "path", "file_path"}:
                    continue
                walk(value, depth + 1)
            return

        if isinstance(x, list):
            for item in x[:30]:
                walk(item, depth + 1)

    walk(obj)

    if not candidates:
        return ""

    candidates.sort(key=len, reverse=True)
    return candidates[0]


def _candidate_id(obj: Any) -> str | None:
    value = _find_key(
        obj,
        (
            "chunk_id",
            "block_id",
            "point_id",
            "id",
            "document_id",
        ),
    )
    return str(value) if value is not None else None


def _summarize_candidate(obj: Any, index: int) -> dict[str, Any]:
    data = _safe_jsonable(obj)

    score = _find_key(data, ("score", "rerank_score", "final_score"))
    rerank_score = _find_key(data, ("rerank_score", "cross_encoder_score", "ce_score"))
    original_score = _find_key(data, ("original_score", "retrieval_score", "vector_score"))
    rank = _find_key(data, ("rank",))

    text = _find_text(data)

    return {
        "index": index,
        "rank": rank if rank is not None else index,
        "score": score,
        "rerank_score": rerank_score,
        "original_score": original_score,
        "chunk_id": _candidate_id(data),
        "source_id": _find_key(data, ("source_id", "doc_id", "source_doc_id")),
        "heading": _find_key(data, ("heading", "title", "section_title")),
        "source": _find_key(data, ("source", "source_path", "path")),
        "function": _find_key(data, ("function", "block_function")),
        "text_preview": text[:1800],
    }


def _get_list(data: Mapping[str, Any], keys: tuple[str, ...]) -> list[Any]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return value

    found = _find_key(data, keys)
    if isinstance(found, list):
        return found

    return []


def _extract_query(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    for key in ("query", "user_message", "message"):
        if key in kwargs and kwargs[key]:
            return str(kwargs[key])

    for arg in args:
        if isinstance(arg, str):
            return arg
        value = getattr(arg, "query", None)
        if value:
            return str(value)
        if isinstance(arg, Mapping) and arg.get("query"):
            return str(arg["query"])

    return None


def _log_retrieval(call_id: int, query: str | None, result: Any) -> None:
    data = _safe_jsonable(result)

    if not isinstance(data, Mapping):
        data = {"raw_result": data}

    reranked = _get_list(data, ("reranked_candidates", "candidates", "top_reranked_preview"))
    context_blocks = _get_list(data, ("context_blocks", "packed_context", "context_blocks_preview"))

    top5 = [
        _summarize_candidate(candidate, i)
        for i, candidate in enumerate(reranked[:5], start=1)
    ]

    packed = [
        _summarize_candidate(block, i)
        for i, block in enumerate(context_blocks[:8], start=1)
    ]

    metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
    actions = metadata.get("actions") if isinstance(metadata.get("actions"), list) else []

    payload = {
        "call_id": call_id,
        "query": query,
        "decision": data.get("decision"),
        "normalized_query": data.get("normalized_query"),
        "counts": {
            "reranked_count": len(reranked),
            "context_block_count": len(context_blocks),
            "action_count": len(actions),
            "warning_count": len(data.get("warnings") or []),
            "error_count": len(data.get("errors") or []),
        },
        "metadata_probe": {
            "backend": metadata.get("backend"),
            "graph_hard_final_context_recovery": metadata.get("graph_hard_final_context_recovery"),
            "graph_final_context_recovery": metadata.get("graph_final_context_recovery"),
            "answerability_candidate_mix": metadata.get("answerability_candidate_mix"),
        },
        "actions": actions,
        "warnings": data.get("warnings"),
        "errors": data.get("errors"),
        "top5_reranked_chunks": top5,
        "packed_context_chunks": packed,
    }

    json_path = RETRIEVAL_DIR / f"{call_id:03d}_retrieval_debug.json"
    md_path = RETRIEVAL_DIR / f"{call_id:03d}_top5_chunks.md"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    md_lines = []
    md_lines.append(f"# Retrieval debug call {call_id}")
    md_lines.append("")
    md_lines.append(f"Query: `{query}`")
    md_lines.append("")
    md_lines.append("## Counts")
    md_lines.append("")
    md_lines.append("```json")
    md_lines.append(json.dumps(payload["counts"], indent=2, ensure_ascii=False))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## Decision")
    md_lines.append("")
    md_lines.append("```json")
    md_lines.append(json.dumps(payload["decision"], indent=2, ensure_ascii=False, default=str))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## Actions")
    md_lines.append("")
    for action in actions:
        md_lines.append(
            f"- {action.get('action')} | {action.get('status')} | "
            f"in={action.get('input_count')} | out={action.get('output_count')} | "
            f"err={action.get('error')}"
        )
    md_lines.append("")
    md_lines.append("## Top 5 reranked chunks")
    md_lines.append("")

    for item in top5:
        md_lines.append(f"### #{item['index']} | score={item.get('score')} | rerank={item.get('rerank_score')}")
        md_lines.append("")
        md_lines.append(f"- chunk_id: `{item.get('chunk_id')}`")
        md_lines.append(f"- source_id: `{item.get('source_id')}`")
        md_lines.append(f"- heading: `{item.get('heading')}`")
        md_lines.append(f"- source: `{item.get('source')}`")
        md_lines.append("")
        md_lines.append("```text")
        md_lines.append(item.get("text_preview") or "")
        md_lines.append("```")
        md_lines.append("")

    md_lines.append("## Packed context chunks")
    md_lines.append("")
    if packed:
        for item in packed:
            md_lines.append(f"### Packed #{item['index']}")
            md_lines.append("")
            md_lines.append(f"- chunk_id: `{item.get('chunk_id')}`")
            md_lines.append(f"- heading: `{item.get('heading')}`")
            md_lines.append("")
            md_lines.append("```text")
            md_lines.append(item.get("text_preview") or "")
            md_lines.append("```")
            md_lines.append("")
    else:
        md_lines.append("None")
        md_lines.append("")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    _write_event("retrieval.debug", payload)

    print("\n🧭 retrieval.top5")
    print(f"call_id={call_id} query={query!r}")
    print(f"reranked={len(reranked)} packed_context={len(context_blocks)} actions={len(actions)}")
    for item in top5:
        text = " ".join((item.get("text_preview") or "").split())
        print(
            f"#{item['index']} score={item.get('score')} "
            f"rerank={item.get('rerank_score')} "
            f"chunk={item.get('chunk_id')} "
            f"heading={item.get('heading')}"
        )
        print(f"   {text[:280]}")
    if not context_blocks:
        print("⚠️ packed_context=0")
    print(f"saved: {md_path}")


def _patch_retrieval_service() -> None:
    import rag.retrieval.service as service_module

    cls = service_module.RetrievalService
    original = cls.retrieve

    if getattr(original, "_lifecycle_chunks_patched", False):
        return

    if inspect.iscoroutinefunction(original):
        async def patched(self, *args, **kwargs):
            call_id = next(CALL_IDS)
            query = _extract_query(args, kwargs)
            _write_event("retrieval.start", {"call_id": call_id, "query": query})
            try:
                result = await original(self, *args, **kwargs)
            except Exception as exc:
                _write_event(
                    "retrieval.exception",
                    {
                        "call_id": call_id,
                        "query": query,
                        "error": str(exc),
                        "exception_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
            _log_retrieval(call_id, query, result)
            return result
    else:
        def patched(self, *args, **kwargs):
            call_id = next(CALL_IDS)
            query = _extract_query(args, kwargs)
            _write_event("retrieval.start", {"call_id": call_id, "query": query})
            try:
                result = original(self, *args, **kwargs)
            except Exception as exc:
                _write_event(
                    "retrieval.exception",
                    {
                        "call_id": call_id,
                        "query": query,
                        "error": str(exc),
                        "exception_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
            _log_retrieval(call_id, query, result)
            return result

    patched._lifecycle_chunks_patched = True
    cls.retrieve = patched


def _patch_prompt_rendering() -> None:
    try:
        import rag.generation.prompt.rag_prompt as rag_prompt
    except Exception:
        return

    original = getattr(rag_prompt, "build_rag_user_prompt", None)
    if original is None or getattr(original, "_lifecycle_chunks_patched", False):
        return

    def patched(state, *args, **kwargs):
        call_id = next(PROMPT_IDS)
        result = original(state, *args, **kwargs)

        state_json = _safe_jsonable(state)
        if not isinstance(state_json, Mapping):
            state_json = {"state": state_json}

        prompt_text = str(result)

        (PROMPT_DIR / f"{call_id:03d}_user_prompt.txt").write_text(prompt_text, encoding="utf-8")
        (PROMPT_DIR / f"{call_id:03d}_state_before_prompt.json").write_text(
            json.dumps(state_json, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        context_bundle = state_json.get("context_bundle") or {}
        docs = []
        if isinstance(context_bundle, Mapping):
            docs = (
                context_bundle.get("retrieved_documents")
                or context_bundle.get("documents")
                or context_bundle.get("context_blocks")
                or []
            )

        _write_event(
            "prompt.user.debug",
            {
                "call_id": call_id,
                "current_user_message": state_json.get("current_user_message"),
                "retrieved_documents_count": len(docs) if isinstance(docs, list) else None,
                "prompt_char_count": len(prompt_text),
                "prompt_preview": prompt_text[:2500],
            },
        )

        print("\n🧭 prompt.user")
        print(f"call_id={call_id}")
        print(f"current_user_message={state_json.get('current_user_message')!r}")
        print(f"retrieved_documents_count={len(docs) if isinstance(docs, list) else None}")
        print(f"saved: {PROMPT_DIR / f'{call_id:03d}_user_prompt.txt'}")

        return result

    patched._lifecycle_chunks_patched = True
    rag_prompt.build_rag_user_prompt = patched


def main() -> None:
    _mkdirs()

    print("🧭 lifecycle_chunks.start")
    print(f"run_id={RUN_ID}")
    print(f"run_dir={RUN_DIR}")
    print("This runs the normal demo_chat continuously, but prints top 5 retrieved chunks per query.")
    print()

    _patch_retrieval_service()
    _patch_prompt_rendering()

    demo_module = os.getenv("RAG_DEMO_CHAT_MODULE", "rag.generation.cli.demo_chat")
    _write_event("lifecycle.start", {"run_id": RUN_ID, "run_dir": str(RUN_DIR), "demo_module": demo_module})

    try:
        runpy.run_module(demo_module, run_name="__main__")
    finally:
        _write_event("lifecycle.end", {"run_id": RUN_ID, "run_dir": str(RUN_DIR)})
        print("\n🧭 lifecycle_chunks.end")
        print(f"run_dir={RUN_DIR}")
        print(f"events_file={EVENTS_FILE}")


if __name__ == "__main__":
    main()
