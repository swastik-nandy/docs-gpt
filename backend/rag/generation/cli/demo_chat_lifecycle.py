from __future__ import annotations

import asyncio
import importlib
import json
import os
import runpy
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


LOG_DIR = Path(
    os.getenv(
        "RAG_LIFECYCLE_LOG_DIR",
        "rag/ingestion/artifacts/debug/query_lifecycle",
    )
)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = LOG_DIR / RUN_ID
EVENTS_FILE = RUN_DIR / "events.jsonl"
PROMPTS_DIR = RUN_DIR / "prompts"
RETRIEVAL_DIR = RUN_DIR / "retrieval"

CALL_COUNTERS = {
    "retrieval": 0,
    "system_prompt": 0,
    "user_prompt": 0,
}


def _ensure_dirs() -> None:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def _safe_filename(text: str, *, fallback: str = "item") -> str:
    keep = []
    for char in text[:90]:
        if char.isalnum() or char in {"-", "_", "."}:
            keep.append(char)
        elif char.isspace():
            keep.append("_")
    value = "".join(keep).strip("._-")
    return value or fallback


def _to_plain(value: Any, *, depth: int = 0, max_depth: int = 8) -> Any:
    if depth > max_depth:
        return repr(value)

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        try:
            return _to_plain(asdict(value), depth=depth + 1, max_depth=max_depth)
        except Exception:
            return repr(value)

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)

            # Avoid dumping huge embeddings/vectors into logs.
            if key_text.lower() in {
                "embedding",
                "dense_vector",
                "sparse_vector",
                "vector",
                "vectors",
                "values",
                "indices",
            }:
                out[key_text] = _summarize_vectorish(item)
                continue

            out[key_text] = _to_plain(item, depth=depth + 1, max_depth=max_depth)
        return out

    if isinstance(value, (list, tuple, set)):
        items = list(value)

        if _looks_vectorish(items):
            return _summarize_vectorish(items)

        return [
            _to_plain(item, depth=depth + 1, max_depth=max_depth)
            for item in items
        ]

    if hasattr(value, "to_dict"):
        try:
            return _to_plain(value.to_dict(), depth=depth + 1, max_depth=max_depth)
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return {
                str(k): _to_plain(v, depth=depth + 1, max_depth=max_depth)
                for k, v in vars(value).items()
                if not str(k).startswith("_")
            }
        except Exception:
            pass

    return repr(value)


def _looks_vectorish(items: list[Any]) -> bool:
    if len(items) < 32:
        return False

    sample = items[:32]
    return all(isinstance(x, (int, float)) for x in sample)


def _summarize_vectorish(value: Any) -> dict[str, Any]:
    try:
        if isinstance(value, dict):
            return {
                "type": "dict/vectorish",
                "keys": list(value.keys())[:20],
                "key_count": len(value),
            }

        items = list(value)
        preview = items[:5]
        return {
            "type": "vector/list",
            "length": len(items),
            "preview": preview,
        }
    except Exception:
        return {
            "type": type(value).__name__,
            "repr": repr(value)[:500],
        }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _to_plain(value),
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _event(event: str, payload: dict[str, Any]) -> None:
    _ensure_dirs()

    row = {
        "ts": _now(),
        "event": event,
        "payload": _to_plain(payload),
    }

    with EVENTS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    print("\n" + "🧭 " + event)
    print(json.dumps(row["payload"], indent=2, ensure_ascii=False, default=str)[:5000])


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def _short(text: Any, n: int = 900) -> str:
    value = str(text or "")
    value = value.replace("\r", "")
    return value if len(value) <= n else value[:n] + "\n...<truncated>..."


def _summarize_context_bundle(context_bundle: Any) -> dict[str, Any]:
    bundle = context_bundle if isinstance(context_bundle, dict) else {}

    docs = bundle.get("retrieved_documents") or []
    memory = bundle.get("retrieved_memory") or []
    provenance = bundle.get("provenance_map") or {}

    return {
        "retrieved_documents_count": len(docs),
        "retrieved_memory_count": len(memory),
        "provenance_count": len(provenance) if isinstance(provenance, dict) else None,
        "conversation_summary_present": bool(bundle.get("conversation_summary")),
        "retrieved_documents_preview": [
            _summarize_context_item(item, index=i)
            for i, item in enumerate(docs[:12], start=1)
        ],
        "retrieved_memory_preview": [
            _summarize_context_item(item, index=i)
            for i, item in enumerate(memory[:8], start=1)
        ],
    }


def _summarize_context_item(item: Any, *, index: int) -> dict[str, Any]:
    if is_dataclass(item):
        try:
            item = asdict(item)
        except Exception:
            pass

    if not isinstance(item, dict) and hasattr(item, "__dict__"):
        try:
            item = vars(item)
        except Exception:
            pass

    if not isinstance(item, dict):
        return {
            "index": index,
            "type": type(item).__name__,
            "repr": repr(item)[:700],
        }

    return {
        "index": index,
        "context_id": item.get("context_id"),
        "block_id": item.get("block_id"),
        "chunk_id": item.get("chunk_id"),
        "source_id": item.get("source_id"),
        "source": item.get("source"),
        "heading": item.get("heading") or item.get("title"),
        "score": item.get("score"),
        "text_preview": _short(item.get("text"), 700),
    }


def _summarize_retrieval_output(output: Any) -> dict[str, Any]:
    output_plain = _to_plain(output)

    if not isinstance(output_plain, dict):
        return {
            "output_type": type(output).__name__,
            "output": output_plain,
        }

    context_blocks = output_plain.get("context_blocks") or []
    reranked = output_plain.get("reranked_candidates") or []
    trace = output_plain.get("trace") or []
    warnings = output_plain.get("warnings") or []
    errors = output_plain.get("errors") or []

    return {
        "keys": sorted(output_plain.keys()),
        "normalized_query": output_plain.get("normalized_query"),
        "decision": output_plain.get("decision"),
        "metadata": output_plain.get("metadata"),
        "trace_count": len(trace),
        "warning_count": len(warnings),
        "error_count": len(errors),
        "reranked_count": len(reranked),
        "context_block_count": len(context_blocks),
        "warnings": warnings,
        "errors": errors,
        "trace": trace,
        "top_reranked_preview": [
            _summarize_candidate(item, index=i)
            for i, item in enumerate(reranked[:20], start=1)
        ],
        "context_blocks_preview": [
            _summarize_context_block(item, index=i)
            for i, item in enumerate(context_blocks[:20], start=1)
        ],
    }


def _summarize_candidate(item: Any, *, index: int) -> dict[str, Any]:
    item = _to_plain(item)

    if not isinstance(item, dict):
        return {
            "index": index,
            "repr": repr(item)[:700],
        }

    candidate = item.get("candidate") or item
    ref = candidate.get("ref") if isinstance(candidate, dict) else {}

    payload = candidate.get("payload") if isinstance(candidate, dict) else {}
    payload = payload if isinstance(payload, dict) else {}

    return {
        "index": index,
        "rank": item.get("rank"),
        "score": item.get("score"),
        "rerank_score": item.get("rerank_score"),
        "original_score": item.get("original_score"),
        "confidence": item.get("confidence"),
        "chunk_id": (
            candidate.get("chunk_id")
            or candidate.get("id")
            or payload.get("chunk_id")
            or payload.get("id")
            or payload.get("ir_id")
        ),
        "point_id": (ref or {}).get("point_id") if isinstance(ref, dict) else None,
        "source_id": (ref or {}).get("source_id") if isinstance(ref, dict) else None,
        "heading": candidate.get("heading") or payload.get("heading") or payload.get("context"),
        "source": candidate.get("source") or payload.get("source"),
        "text_preview": _short(
            candidate.get("text")
            or candidate.get("retrieval_text")
            or payload.get("text")
            or payload.get("retrieval_text"),
            900,
        ),
    }


def _summarize_context_block(block: Any, *, index: int) -> dict[str, Any]:
    block = _to_plain(block)

    if not isinstance(block, dict):
        return {
            "index": index,
            "repr": repr(block)[:700],
        }

    return {
        "index": index,
        "context_id": block.get("context_id"),
        "block_id": block.get("block_id"),
        "chunk_id": block.get("chunk_id"),
        "source_id": block.get("source_id"),
        "source": block.get("source"),
        "heading": block.get("heading") or block.get("title"),
        "score": block.get("score"),
        "text_preview": _short(block.get("text"), 1000),
        "metadata": block.get("metadata"),
    }


def _patch_retrieval_service() -> None:
    try:
        service_mod = importlib.import_module("rag.retrieval.service")
        RetrievalService = getattr(service_mod, "RetrievalService")
        original_retrieve = RetrievalService.retrieve
    except Exception as exc:
        _event(
            "patch.retrieval_service.failed",
            {
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return

    async def logged_retrieve(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        CALL_COUNTERS["retrieval"] += 1
        call_id = CALL_COUNTERS["retrieval"]

        request_plain = _to_plain(request)

        _event(
            "retrieval.request",
            {
                "call_id": call_id,
                "service_class": type(self).__name__,
                "request": request_plain,
                "metadata_arg": _to_plain(kwargs.get("metadata")),
            },
        )

        _write_json(RETRIEVAL_DIR / f"{call_id:03d}_request.json", request)

        started = time.perf_counter()

        try:
            output = await original_retrieve(self, request, *args, **kwargs)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

            summary = _summarize_retrieval_output(output)

            _event(
                "retrieval.output",
                {
                    "call_id": call_id,
                    "elapsed_ms": elapsed_ms,
                    **summary,
                },
            )

            _write_json(RETRIEVAL_DIR / f"{call_id:03d}_output_full.json", output)
            _write_json(RETRIEVAL_DIR / f"{call_id:03d}_output_summary.json", summary)

            return output

        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

            _event(
                "retrieval.exception",
                {
                    "call_id": call_id,
                    "elapsed_ms": elapsed_ms,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )

            raise

    RetrievalService.retrieve = logged_retrieve

    _event(
        "patch.retrieval_service.ok",
        {
            "class": "rag.retrieval.service.RetrievalService",
            "method": "retrieve",
        },
    )


def _patch_prompt_builder() -> None:
    try:
        rag_prompt_mod = importlib.import_module("rag.generation.prompt.rag_prompt")
    except Exception as exc:
        _event(
            "patch.prompt_builder.failed",
            {
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return

    original_system = getattr(rag_prompt_mod, "get_rag_system_prompt", None)
    original_user = getattr(rag_prompt_mod, "build_rag_user_prompt", None)

    if original_system is not None:
        def logged_system_prompt(*args: Any, **kwargs: Any) -> str:
            CALL_COUNTERS["system_prompt"] += 1
            call_id = CALL_COUNTERS["system_prompt"]

            prompt = original_system(*args, **kwargs)

            _event(
                "prompt.system",
                {
                    "call_id": call_id,
                    "char_count": len(prompt),
                    "preview": _short(prompt, 2000),
                },
            )

            _write_text(PROMPTS_DIR / f"{call_id:03d}_system_prompt.txt", prompt)

            return prompt

        rag_prompt_mod.get_rag_system_prompt = logged_system_prompt

    if original_user is not None:
        def logged_user_prompt(state: Any, *args: Any, **kwargs: Any) -> str:
            CALL_COUNTERS["user_prompt"] += 1
            call_id = CALL_COUNTERS["user_prompt"]

            state_plain = _to_plain(state)
            context_bundle = {}

            if isinstance(state, dict):
                context_bundle = state.get("context_bundle") or {}

            _event(
                "prompt.user.state_before_render",
                {
                    "call_id": call_id,
                    "current_user_message": (
                        state.get("current_user_message")
                        if isinstance(state, dict)
                        else None
                    ),
                    "turn_plan": (
                        state.get("turn_plan")
                        if isinstance(state, dict)
                        else None
                    ),
                    "context_bundle_summary": _summarize_context_bundle(context_bundle),
                    "state_keys": sorted(state.keys()) if isinstance(state, dict) else None,
                },
            )

            _write_json(PROMPTS_DIR / f"{call_id:03d}_state_before_user_prompt.json", state_plain)

            prompt = original_user(state, *args, **kwargs)

            _event(
                "prompt.user.rendered",
                {
                    "call_id": call_id,
                    "char_count": len(prompt),
                    "contains_retrieved_documents_section": "retrieved" in prompt.lower(),
                    "preview": _short(prompt, 4000),
                },
            )

            _write_text(PROMPTS_DIR / f"{call_id:03d}_user_prompt.txt", prompt)

            return prompt

        rag_prompt_mod.build_rag_user_prompt = logged_user_prompt

    _event(
        "patch.prompt_builder.ok",
        {
            "module": "rag.generation.prompt.rag_prompt",
            "patched": {
                "get_rag_system_prompt": original_system is not None,
                "build_rag_user_prompt": original_user is not None,
            },
        },
    )


def _patch_google_genai() -> None:
    """
    Best-effort provider call logging.

    This does not replace your provider layer. It only wraps google-genai if your
    existing demo_chat imports and calls it in-process.
    """

    try:
        from google import genai  # type: ignore
    except Exception:
        _event(
            "patch.google_genai.skipped",
            {
                "reason": "google.genai import failed or not installed",
            },
        )
        return

    try:
        original_client = genai.Client
    except Exception as exc:
        _event(
            "patch.google_genai.failed",
            {
                "error": repr(exc),
            },
        )
        return

    class LoggedClient(original_client):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _event(
                "generation.client.created",
                {
                    "provider": "google_genai",
                    "client_class": "google.genai.Client",
                    "kwargs_keys": sorted(kwargs.keys()),
                },
            )
            super().__init__(*args, **kwargs)

            try:
                original_generate_content = self.models.generate_content

                def logged_generate_content(*g_args: Any, **g_kwargs: Any) -> Any:
                    model = g_kwargs.get("model")
                    contents = g_kwargs.get("contents")

                    _event(
                        "generation.request",
                        {
                            "provider": "google_genai",
                            "model": model,
                            "contents_preview": _short(contents, 5000),
                            "kwargs_keys": sorted(g_kwargs.keys()),
                        },
                    )

                    started = time.perf_counter()

                    try:
                        response = original_generate_content(*g_args, **g_kwargs)
                        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

                        response_text = getattr(response, "text", None)

                        _event(
                            "generation.response",
                            {
                                "provider": "google_genai",
                                "model": model,
                                "elapsed_ms": elapsed_ms,
                                "response_type": type(response).__name__,
                                "text_preview": _short(response_text, 4000),
                                "raw_preview": _short(_to_plain(response), 4000),
                            },
                        )

                        return response

                    except Exception as exc:
                        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

                        _event(
                            "generation.exception",
                            {
                                "provider": "google_genai",
                                "model": model,
                                "elapsed_ms": elapsed_ms,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                            },
                        )
                        raise

                self.models.generate_content = logged_generate_content

            except Exception as exc:
                _event(
                    "patch.google_genai.instance_method.failed",
                    {
                        "error": repr(exc),
                    },
                )

    genai.Client = LoggedClient

    _event(
        "patch.google_genai.ok",
        {
            "wrapped": "google.genai.Client.models.generate_content",
        },
    )


def _resolve_demo_module() -> str:
    explicit = os.getenv("RAG_DEMO_CHAT_MODULE")
    if explicit:
        return explicit

    candidates = [
        "rag.generation.cli.demo_chat",
        "rag.generation.demo_chat",
    ]

    for module_name in candidates:
        try:
            importlib.util.find_spec(module_name)
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                return module_name
        except Exception:
            continue

    raise SystemExit(
        "Could not find demo_chat module. Set RAG_DEMO_CHAT_MODULE, for example:\n"
        "export RAG_DEMO_CHAT_MODULE='rag.generation.cli.demo_chat'"
    )


def main() -> None:
    _ensure_dirs()

    _event(
        "lifecycle.start",
        {
            "run_id": RUN_ID,
            "run_dir": str(RUN_DIR),
            "argv": sys.argv,
            "env": {
                "RAG_DEMO_CHAT_MODULE": os.getenv("RAG_DEMO_CHAT_MODULE"),
                "QDRANT_COLLECTION": os.getenv("QDRANT_COLLECTION"),
                "QDRANT_MEMORY_COLLECTION": os.getenv("QDRANT_MEMORY_COLLECTION"),
                "GENERATION_PROVIDER": os.getenv("GENERATION_PROVIDER"),
                "GENERATION_MODEL": os.getenv("GENERATION_MODEL"),
                "GOOGLE_GENAI_MODEL": os.getenv("GOOGLE_GENAI_MODEL"),
                "GEMINI_MODEL": os.getenv("GEMINI_MODEL"),
                "RERANKER_MODEL": os.getenv("RERANKER_MODEL"),
                "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL"),
            },
        },
    )

    _patch_retrieval_service()
    _patch_prompt_builder()
    _patch_google_genai()

    demo_module = _resolve_demo_module()

    _event(
        "demo_chat.launch",
        {
            "module": demo_module,
            "note": "Running existing demo_chat with lifecycle monkeypatches installed.",
        },
    )

    # Make argv look like the real demo_chat module was invoked directly.
    sys.argv[0] = demo_module.replace(".", "/") + ".py"

    try:
        runpy.run_module(demo_module, run_name="__main__")
    finally:
        _event(
            "lifecycle.end",
            {
                "run_id": RUN_ID,
                "run_dir": str(RUN_DIR),
                "events_file": str(EVENTS_FILE),
            },
        )


if __name__ == "__main__":
    main()
