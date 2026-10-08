# rag/ingestion/data_transformation/parser/adapters/markdown/enrichment/tables/enricher.py

from __future__ import annotations

import re
from typing import Any, Dict, List


def _normalize_header(text: Any, index: int = 0) -> str:
    """
    Normalize header keys safely and generically.

    Falls back to column_N when header text is empty.
    """

    key = re.sub(r"\W+", "_", str(text or "").strip().lower()).strip("_")
    return key or f"column_{index + 1}"


def _clean_value(value: Any) -> str:
    """
    Normalize a cell value without changing meaning.
    """

    text = str(value) if value is not None else ""
    return re.sub(r"\s+", " ", text.strip())


def _row_has_content(row: Dict[str, str]) -> bool:
    return any(str(value).strip() for value in row.values())


def _row_width(row: Any) -> int:
    """
    Return structural width of a table row.
    """

    if isinstance(row, dict):
        return len(row)

    if isinstance(row, list):
        return len(row)

    return 0


def _derive_headers_from_rows(rows: List[Any]) -> List[str]:
    """
    Derive fallback headers from the widest row.
    """

    widest = max((_row_width(row) for row in rows or []), default=0)
    return [f"column_{index + 1}" for index in range(widest)]


def _normalize_rows(rows: List[Any], headers: List[str]) -> List[Dict[str, str]]:
    """
    Normalize rows into dictionaries keyed by normalized headers.

    Supports:
    - dict rows from parser/table_parser.py
    - list rows from raw AST tables
    """

    clean_rows: List[Dict[str, str]] = []

    for row in rows or []:
        normalized_row: Dict[str, str] = {}

        if isinstance(row, dict):
            for header in headers:
                normalized_row[header] = _clean_value(row.get(header, ""))
            clean_rows.append(normalized_row)
            continue

        if isinstance(row, list):
            for index, header in enumerate(headers):
                value = row[index] if index < len(row) else ""
                normalized_row[header] = _clean_value(value)
            clean_rows.append(normalized_row)
            continue

    return clean_rows


def _is_generic_column_headers(headers: List[str]) -> bool:
    """
    Detect fallback column_N headers produced for headerless tables.
    """

    if not headers:
        return False

    expected = [f"column_{index + 1}" for index in range(len(headers))]
    return list(headers) == expected


def _row_to_pair_text(headers: List[str], row: Dict[str, str]) -> str:
    """
    Render two-column headerless rows as key-value statements.
    """

    left = _clean_value(row.get(headers[0], "")) if len(headers) > 0 else ""
    right = _clean_value(row.get(headers[1], "")) if len(headers) > 1 else ""

    if left and right:
        return f"{left}: {right}"

    return left or right


def _row_to_text(headers: List[str], row: Dict[str, str]) -> str:
    """
    Convert one row into generic retrieval-friendly text.

    Example:
        option: --port | description: Specify port
    """

    if len(headers) == 2 and _is_generic_column_headers(headers):
        return _row_to_pair_text(headers, row)

    parts: List[str] = []

    for header in headers:
        value = _clean_value(row.get(header, ""))

        if not value:
            continue

        pretty_header = header.replace("_", " ").strip()
        parts.append(f"{pretty_header}: {value}")

    return " | ".join(parts).strip()

def _table_to_text(headers: List[str], rows: List[Dict[str, str]]) -> List[str]:
    """
    Convert table rows into flattened retrieval lines.
    """

    lines: List[str] = []

    for row in rows or []:
        if not _row_has_content(row):
            continue

        line = _row_to_text(headers, row)

        if line:
            lines.append(line)

    return lines


def _build_table_profile(
    headers: List[str],
    rows: List[Dict[str, str]],
    anomalies: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Generic structural signals about the table.
    """

    row_count = len(rows)
    non_empty_rows = sum(1 for row in rows if _row_has_content(row))
    column_count = len(headers)

    filled_cells = 0
    total_cells = row_count * column_count if row_count and column_count else 0

    for row in rows:
        for header in headers:
            if _clean_value(row.get(header, "")):
                filled_cells += 1

    density = (filled_cells / total_cells) if total_cells else 0.0

    return {
        "row_count": row_count,
        "non_empty_row_count": non_empty_rows,
        "column_count": column_count,
        "filled_cell_count": filled_cells,
        "cell_density": round(density, 4),
        "has_anomalies": bool(anomalies),
        "anomaly_count": len(anomalies or []),
    }


def enrich_tables(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enhance parsed tables without assuming domain semantics.

    Preserves:
    - raw rows
    - normalized rows
    - headers
    - raw headers
    - table shape
    - anomalies
    - AST cell metadata such as links / inline code
    """

    new_content: List[Dict[str, Any]] = []

    for item in content or []:
        if not item:
            continue

        if item.get("type") != "table":
            new_content.append(item)
            continue

        raw_headers = item.get("headers", []) or []
        header_raw = item.get("header_raw", []) or raw_headers
        raw_rows = item.get("rows", []) or []
        raw_row_backup = item.get("raw_rows", []) or []
        anomalies = item.get("anomalies", []) or []
        shape = item.get("shape", {}) or {}
        cell_meta = item.get("cell_meta", []) or []

        headers = [
            _normalize_header(header, index=index)
            for index, header in enumerate(raw_headers)
        ]

        if not headers:
            headers = _derive_headers_from_rows(raw_rows)

        clean_rows = _normalize_rows(raw_rows, headers) if headers else []
        row_texts = _table_to_text(headers, clean_rows) if headers else []

        enriched = {
            **item,
            "type": "generic_table",
            "headers": headers,
            "header_raw": header_raw,
            "rows": clean_rows,
            "row_text": row_texts,
            "text": "\n".join(row_texts).strip(),
            "cell_meta": cell_meta,
            "headerless": bool(item.get("headerless")),
            "table_rendering": (
                "pair_rows"
                if len(headers) == 2 and _is_generic_column_headers(headers)
                else "header_value_rows"
            ),
            "table_profile": _build_table_profile(
                headers=headers,
                rows=clean_rows,
                anomalies=anomalies,
            ),
        }

        if raw_row_backup:
            enriched["raw_rows"] = raw_row_backup
        elif raw_rows:
            enriched["raw_rows"] = raw_rows

        if anomalies:
            enriched["anomalies"] = anomalies

        if shape:
            enriched["shape"] = shape

        new_content.append(enriched)

    return new_content