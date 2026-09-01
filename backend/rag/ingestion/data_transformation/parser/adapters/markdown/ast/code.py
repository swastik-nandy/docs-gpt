# rag/ingestion/data_transformation/parser/adapters/markdown/ast/code.py

from __future__ import annotations

from typing import Any, Dict, Optional

from rag.ingestion.data_transformation.parser.adapters.markdown.ast.utils import compact_dict, normalize_language


def parse_fence(token: Any) -> Optional[Dict[str, Any]]:
    """
    Parse fenced Markdown code blocks.

    Handles:
        ```bash
        npm run dev
        ```

    Output:
        {
            "type": "code",
            "language": "bash",
            "text": "npm run dev"
        }
    """

    if token is None:
        return None

    code = getattr(token, "content", "") or ""
    code = code.rstrip("\n")

    if not code.strip():
        return None

    language = normalize_language(getattr(token, "info", "") or "")

    return compact_dict({
        "type": "code",
        "language": language,
        "text": code,
    })


def parse_code_block(token: Any) -> Optional[Dict[str, Any]]:
    """
    Parse indented Markdown code blocks.

    Output:
        {
            "type": "code",
            "language": "",
            "text": "..."
        }
    """

    if token is None:
        return None

    code = getattr(token, "content", "") or ""
    code = code.rstrip("\n")

    if not code.strip():
        return None

    return compact_dict({
        "type": "code",
        "language": "",
        "text": code,
    })