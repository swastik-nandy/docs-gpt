from __future__ import annotations

from pathlib import Path

from rag.ingestion.data_transformation.parser.config_loader import load_parser_heuristics_config


def read_extension(file_path: str) -> str:
    """
    Read normalized file extension from a path.

    Example:
    - docs/intro.MDX -> .mdx
    - index.HTML -> .html
    """

    if not file_path:
        return ""

    return Path(file_path).suffix.lower()


def detect_file_type(file_path: str) -> str:
    """
    Detect enabled parser family for a file.

    Known-but-disabled families return unknown.
    Example:
    - html may be known in config
    - but it should not be routed unless enabled_families includes html
    """

    extension = read_extension(file_path)
    config = load_parser_heuristics_config()

    for family, extensions in config.file_types.known_extensions.items():
        if extension in extensions and family in config.file_types.enabled_families:
            return family

    return "unknown"


def supported_extensions() -> dict[str, set[str]]:
    """
    Return enabled extensions grouped by parser family.
    """

    config = load_parser_heuristics_config()

    return {
        family: set(extensions)
        for family, extensions in config.file_types.known_extensions.items()
        if family in config.file_types.enabled_families
    }


def known_extensions() -> dict[str, set[str]]:
    """
    Return all known extensions, including disabled parser families.
    """

    config = load_parser_heuristics_config()

    return {
        family: set(extensions)
        for family, extensions in config.file_types.known_extensions.items()
    }