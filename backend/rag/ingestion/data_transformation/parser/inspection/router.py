from typing import Any, Callable

from rag.ingestion.data_transformation.parser.inspection.detector import detect_file_type


ParserFn = Callable[[str], Any]
InspectorFn = Callable[[str], Any]


def _load_markdown_parser() -> ParserFn:
    """
    Lazy-load markdown parser entrypoint.

    Keeps router independent from markdown internals during module import.
    """

    from rag.ingestion.data_transformation.parser.adapters.markdown import parse_markdown

    return parse_markdown


def _load_markdown_inspector() -> InspectorFn:
    """
    Lazy-load markdown inspection entrypoint.

    Inspection decides whether the file should use:
    - normal markdown parse
    - full parse then chunk fanout
    - spine-fragmented parse
    - quarantine
    """

    from rag.ingestion.data_transformation.parser.inspection.markdown import (
        inspect_markdown_file,
    )

    return inspect_markdown_file


# def _load_html_parser() -> ParserFn:
#     """
#     Lazy-load html parser entrypoint.
#     """
#
#     from rag.ingestion.data_transformation.parser.inspection.html import parse_html
#
#     return parse_html


PARSER_LOADERS: dict[str, Callable[[], ParserFn]] = {
    "markdown": _load_markdown_parser,
    # "html": _load_html_parser,
}


INSPECTOR_LOADERS: dict[str, Callable[[], InspectorFn]] = {
    "markdown": _load_markdown_inspector,
}


def get_parser_for_type(file_type: str) -> ParserFn:
    """
    Resolve parser entrypoint from detected file type.
    """

    loader = PARSER_LOADERS.get(file_type)

    if loader is None:
        raise ValueError(f"Unsupported parser file type: {file_type!r}")

    return loader()


def get_inspector_for_type(file_type: str) -> InspectorFn:
    """
    Resolve inspector entrypoint from detected file type.
    """

    loader = INSPECTOR_LOADERS.get(file_type)

    if loader is None:
        raise ValueError(f"Unsupported inspection file type: {file_type!r}")

    return loader()


def get_parser_for_file(file_path: str) -> ParserFn:
    """
    Detect file type from path and resolve parser entrypoint.
    """

    file_type = detect_file_type(file_path)
    return get_parser_for_type(file_type)


def get_inspector_for_file(file_path: str) -> InspectorFn:
    """
    Detect file type from path and resolve inspector entrypoint.
    """

    file_type = detect_file_type(file_path)
    return get_inspector_for_type(file_type)


def inspect_by_file_type(file_path: str) -> Any:
    """
    Detect file type and run the correct pre-parse inspection.

    Use this before queue routing.
    """

    inspector = get_inspector_for_file(file_path)
    return inspector(file_path)


def parse_by_file_type(file_path: str) -> Any:
    """
    Detect file type and route to the correct parser family.

    This remains the direct parse path for already-approved normal documents.
    """

    parser = get_parser_for_file(file_path)
    return parser(file_path)