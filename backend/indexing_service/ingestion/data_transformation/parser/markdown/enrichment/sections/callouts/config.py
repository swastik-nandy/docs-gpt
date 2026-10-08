from ingestion.data_transformation.parser.markdown.config import get_markdown_config


# --------------- CONFIG --------------------------------

_CALLOUT_CONFIG = get_markdown_config().callouts

CALLOUT_KIND_MAP = _CALLOUT_CONFIG.kind_map
MAX_CALLOUT_PARAGRAPHS = _CALLOUT_CONFIG.max_plain_callout_paragraphs
