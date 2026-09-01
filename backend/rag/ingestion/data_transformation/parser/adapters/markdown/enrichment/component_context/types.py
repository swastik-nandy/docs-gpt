# rag/ingestion/data_transformation/parser/adapters/markdown/enrichment/component_context/types.py

from __future__ import annotations

from typing import Any, Dict, List


ComponentFrame = Dict[str, Any]
ComponentStack = List[ComponentFrame]