# rag/ingestion/data_transformation/parser/adapters/markdown/adapters/mdx_adapter.py

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from rag.ingestion.data_transformation.parser.config_loader import load_parser_heuristics_config


# -------------------------------------- MARKDOWN CODE PROTECTION --------------------------------------

FENCE_RE = re.compile(
    r"(```[\s\S]*?```|~~~[\s\S]*?~~~)",
    re.MULTILINE,
)

INLINE_CODE_RE = re.compile(
    r"(`+)(?!`)(.+?)(?<!`)\1",
    re.DOTALL,
)


# -------------------------------------- EXECUTION-LAYER PATTERNS --------------------------------------

IMPORT_RE = re.compile(
    r"""
    ^\s*
    import
    (?:
        \s+[\w*{}\s,.$]+?\s+from\s+["'][^"']+["']
        |
        \s+["'][^"']+["']
    )
    \s*;?\s*$
    """,
    re.MULTILINE | re.VERBOSE,
)

HTML_COMMENT_RE = re.compile(
    r"<!--[\s\S]*?-->",
    re.MULTILINE,
)

JSX_COMMENT_RE = re.compile(
    r"\{/\*[\s\S]*?\*/\}",
    re.MULTILINE,
)


# -------------------------------------- COMPONENT PATTERNS --------------------------------------
# Only uppercase JSX-style names are treated as MDX components.
# This avoids converting normal HTML tags like <div>, <code>, <kbd>, etc.

SELF_CLOSING_COMPONENT_RE = re.compile(
    r"<([A-Z][A-Za-z0-9_.]*)\b([^<>]*?)/>",
    re.MULTILINE | re.DOTALL,
)

OPEN_COMPONENT_RE = re.compile(
    r"<([A-Z][A-Za-z0-9_.]*)\b([^<>]*?)>",
    re.MULTILINE | re.DOTALL,
)

CLOSE_COMPONENT_RE = re.compile(
    r"</([A-Z][A-Za-z0-9_.]*)>",
    re.MULTILINE,
)

PROP_RE = re.compile(
    r"""
    (?P<key>[A-Za-z_:][A-Za-z0-9_:.-]*)
    \s*=\s*
    (?P<value>
        "[^"]*"
        |
        '[^']*'
        |
        \{[^{}]*\}
    )
    """,
    re.VERBOSE | re.DOTALL,
)

JSX_EXPRESSION_RE = re.compile(
    r"\{([^{}\n]+)\}",
    re.MULTILINE,
)


# -------------------------------------- MARKERS --------------------------------------

_MARKERS = load_parser_heuristics_config().markdown.mdx_markers

COMPONENT_OPEN_MARKER = _MARKERS.open
COMPONENT_CLOSE_MARKER = _MARKERS.close
COMPONENT_SELF_MARKER = _MARKERS.self


# -------------------------------------- CODE PROTECTION --------------------------------------

def _protect_markdown_code(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Temporarily replace fenced code blocks and inline code spans.

    This prevents MDX cleanup from corrupting examples like:
        `<Card />`
        `import x from "y"`
        ```jsx
        <Tabs />
        ```
    """

    store: Dict[str, str] = {}

    def repl(match: re.Match) -> str:
        key = f"@@MDX_CODE_{len(store)}@@"
        store[key] = match.group(0)
        return key

    protected = FENCE_RE.sub(repl, text or "")
    protected = INLINE_CODE_RE.sub(repl, protected)

    return protected, store


def _restore_markdown_code(text: str, store: Dict[str, str]) -> str:
    """
    Restore protected fenced and inline code exactly as they appeared.
    """

    restored = text or ""

    for key, value in store.items():
        restored = restored.replace(key, value)

    return restored


# -------------------------------------- BASIC CLEANUP --------------------------------------

def _remove_comments(text: str) -> str:
    """
    Remove HTML and JSX comments.

    Comments rarely help retrieval and often contain implementation noise.
    """

    value = HTML_COMMENT_RE.sub("", text or "")
    value = JSX_COMMENT_RE.sub("", value)

    return value


def _remove_imports(text: str) -> str:
    """
    Remove MDX import statements.

    Imports are execution/build-layer details, not documentation content.
    Inline and fenced code are protected before this runs.
    """

    return IMPORT_RE.sub("", text or "")


def _looks_like_export_statement(line: str) -> bool:
    """
    Detect MDX/ESM export statements conservatively.

    Avoid removing normal prose that merely starts with the word "export".
    """

    stripped = (line or "").strip()

    if not stripped.startswith("export "):
        return False

    patterns = (
        "export const ",
        "export let ",
        "export var ",
        "export function ",
        "export class ",
        "export default ",
        "export {",
        "export *",
    )

    return any(stripped.startswith(pattern) for pattern in patterns)


def _remove_export_blocks(text: str) -> str:
    """
    Remove common MDX export declarations.

    Handles:
    - export const x = ...
    - export const x = { ... }
    - export default ...
    - export { A, B }
    - export * from "..."
    """

    lines = (text or "").splitlines()
    output: List[str] = []

    skipping = False
    brace_depth = 0
    paren_depth = 0
    bracket_depth = 0

    for line in lines:
        stripped = line.strip()

        if not skipping and _looks_like_export_statement(stripped):
            brace_depth = line.count("{") - line.count("}")
            paren_depth = line.count("(") - line.count(")")
            bracket_depth = line.count("[") - line.count("]")

            multiline = (
                brace_depth > 0
                or paren_depth > 0
                or bracket_depth > 0
            )

            ends_now = stripped.endswith(";") or stripped.endswith("}")

            if multiline and not ends_now:
                skipping = True

            continue

        if skipping:
            brace_depth += line.count("{") - line.count("}")
            paren_depth += line.count("(") - line.count(")")
            bracket_depth += line.count("[") - line.count("]")

            if brace_depth <= 0 and paren_depth <= 0 and bracket_depth <= 0:
                skipping = False
                brace_depth = 0
                paren_depth = 0
                bracket_depth = 0

            continue

        output.append(line)

    return "\n".join(output)


def _remove_execution_layer(text: str) -> str:
    """
    Remove MDX execution-layer syntax while keeping documentation content.
    """

    value = _remove_comments(text)
    value = _remove_imports(value)
    value = _remove_export_blocks(value)

    return value


# -------------------------------------- PROP HANDLING --------------------------------------

def _clean_prop_value(value: str) -> str:
    """
    Convert simple MDX prop values into marker-safe strings.

    Kept:
    - "literal"
    - 'literal'
    - {"literal"}
    - {'literal'}
    - {simple.identifier}
    - {simple/path-like/value}
    - {123}

    Dropped:
    - object expressions
    - arrays
    - functions
    - map/filter calls
    - nested expressions
    """

    raw = (value or "").strip()

    if not raw:
        return ""

    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1].strip()

    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1].strip()

    if raw.startswith("{") and raw.endswith("}"):
        inner = raw[1:-1].strip()

        if not inner:
            return ""

        if inner.startswith('"') and inner.endswith('"'):
            return inner[1:-1].strip()

        if inner.startswith("'") and inner.endswith("'"):
            return inner[1:-1].strip()

        # Keep simple literals / identifiers only.
        if re.match(r"^[A-Za-z0-9_.$:/#@+\-\s]+$", inner):
            return inner.strip()

        return ""

    return raw


def _extract_props(raw_props: str) -> Dict[str, str]:
    """
    Extract simple props from a JSX component opening tag.

    Boolean props like:
        <Callout important />

    are intentionally ignored for now because they do not have explicit values.
    """

    props: Dict[str, str] = {}

    for match in PROP_RE.finditer(raw_props or ""):
        key = (match.group("key") or "").strip()
        value = _clean_prop_value(match.group("value") or "")

        if key and value:
            props[key] = value

    return props


def _serialize_props(props: Dict[str, str]) -> str:
    """
    Serialize props into marker-safe quoted attributes.

    Format:
        key="value" key2="value2"
    """

    parts: List[str] = []

    for key, value in (props or {}).items():
        key = str(key or "").strip()

        if not key:
            continue

        safe_value = str(value or "")
        safe_value = safe_value.replace('"', "'")
        safe_value = safe_value.replace("\n", " ")
        safe_value = re.sub(r"\s+", " ", safe_value).strip()

        if not safe_value:
            continue

        parts.append(f'{key}="{safe_value}"')

    return " ".join(parts)


# -------------------------------------- COMPONENT MARKERS --------------------------------------

def _component_open_marker(name: str, props: Dict[str, str]) -> str:
    prop_text = _serialize_props(props)

    if prop_text:
        return f'\n\n{COMPONENT_OPEN_MARKER} name="{name}" {prop_text}\n\n'

    return f'\n\n{COMPONENT_OPEN_MARKER} name="{name}"\n\n'


def _component_close_marker(name: str) -> str:
    return f'\n\n{COMPONENT_CLOSE_MARKER} name="{name}"\n\n'


def _component_self_marker(name: str, props: Dict[str, str]) -> str:
    prop_text = _serialize_props(props)

    if prop_text:
        return f'\n\n{COMPONENT_SELF_MARKER} name="{name}" {prop_text}\n\n'

    return f'\n\n{COMPONENT_SELF_MARKER} name="{name}"\n\n'


def _replace_self_closing_component(match: re.Match) -> str:
    name = (match.group(1) or "").strip()
    raw_props = match.group(2) or ""
    props = _extract_props(raw_props)

    return _component_self_marker(name, props)


def _replace_open_component(match: re.Match) -> str:
    name = (match.group(1) or "").strip()
    raw_props = match.group(2) or ""
    props = _extract_props(raw_props)

    return _component_open_marker(name, props)


def _replace_close_component(match: re.Match) -> str:
    name = (match.group(1) or "").strip()

    return _component_close_marker(name)


def _normalize_components(text: str) -> str:
    """
    Convert MDX component syntax into explicit Markdown-safe markers.

    This preserves component boundaries without depending on specific
    component names like Tabs, Callout, Steps, Cards, etc.
    """

    value = text or ""

    # Self-closing first so <Component /> does not become open-only.
    value = SELF_CLOSING_COMPONENT_RE.sub(_replace_self_closing_component, value)
    value = OPEN_COMPONENT_RE.sub(_replace_open_component, value)
    value = CLOSE_COMPONENT_RE.sub(_replace_close_component, value)

    return value


# -------------------------------------- JSX EXPRESSION CLEANUP --------------------------------------

def _normalize_jsx_expressions(text: str) -> str:
    """
    Normalize simple inline JSX expressions.

    We do not evaluate JavaScript.

    Kept:
    - {"literal"}
    - {'literal'}
    - {simple.identifier}
    - {simple/path-like/value}

    Removed:
    - complex JS expressions
    - object/array/function expressions
    """

    def repl(match: re.Match) -> str:
        inner = (match.group(1) or "").strip()

        if not inner:
            return ""

        if inner.startswith('"') and inner.endswith('"'):
            return inner[1:-1]

        if inner.startswith("'") and inner.endswith("'"):
            return inner[1:-1]

        if re.match(r"^[A-Za-z0-9_.$:/#@+\-]+$", inner):
            return inner

        return ""

    return JSX_EXPRESSION_RE.sub(repl, text or "")


# -------------------------------------- SPACING --------------------------------------

def _cleanup_spacing(text: str) -> str:
    """
    Normalize excessive blank lines while preserving Markdown block separation.
    """

    lines = (text or "").splitlines()

    cleaned: List[str] = []
    blank_count = 0

    for line in lines:
        stripped = line.rstrip()

        if not stripped:
            blank_count += 1

            if blank_count <= 2:
                cleaned.append("")

            continue

        blank_count = 0
        cleaned.append(stripped)

    return "\n".join(cleaned).strip() + "\n"


# -------------------------------------- PUBLIC API --------------------------------------

def normalize_mdx(text: str) -> str:
    """
    Convert MDX into parser-safe Markdown-like text.

    This function is intentionally domain-agnostic.

    It does not know about:
    - framework docs
    - changelogs
    - release notes
    - product-specific components
    - company-specific components

    It only preserves:
    - Markdown content
    - fenced code blocks
    - inline code spans
    - component boundaries
    - component names
    - simple component props

    Output includes marker lines configured in parser/config/heuristics.toml.

    The AST parser later recognizes these markers as component events.
    """

    if not text:
        return ""

    protected, code_store = _protect_markdown_code(text)

    protected = _remove_execution_layer(protected)
    protected = _normalize_components(protected)
    protected = _normalize_jsx_expressions(protected)

    restored = _restore_markdown_code(protected, code_store)
    restored = _cleanup_spacing(restored)

    return restored