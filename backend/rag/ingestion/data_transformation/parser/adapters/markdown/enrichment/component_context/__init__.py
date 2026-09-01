# rag/ingestion/data_transformation/parser/adapters/markdown/enrichment/component_context/__init__.py

from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.component_context.blocks import (
    attach_to_content_block,
    build_self_closing_component_block,
    merge_into_meta,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.component_context.format import (
    current_component_meta,
    effective_local_context,
    format_component,
    format_props,
    format_stack,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.component_context.stack import (
    copy_stack,
    pop_component,
    push_component,
    stack_from_content_or_section,
)
from rag.ingestion.data_transformation.parser.adapters.markdown.enrichment.component_context.types import (
    ComponentFrame,
    ComponentStack,
)


__all__ = [
    "ComponentFrame",
    "ComponentStack",
    "attach_to_content_block",
    "build_self_closing_component_block",
    "copy_stack",
    "current_component_meta",
    "effective_local_context",
    "format_component",
    "format_props",
    "format_stack",
    "merge_into_meta",
    "pop_component",
    "push_component",
    "stack_from_content_or_section",
]