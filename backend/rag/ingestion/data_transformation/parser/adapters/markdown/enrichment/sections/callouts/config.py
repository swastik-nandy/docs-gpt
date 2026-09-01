from rag.ingestion.data_transformation.parser.config_loader import load_parser_heuristics_config


# --------------- CONFIG --------------------------------

_CALLOUT_CONFIG = load_parser_heuristics_config().markdown.callouts

CALLOUT_KIND_MAP = _CALLOUT_CONFIG.kind_map
MAX_CALLOUT_PARAGRAPHS = _CALLOUT_CONFIG.max_plain_callout_paragraphs
