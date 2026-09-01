from __future__ import annotations

from typing import Any, Dict, List

from rag.ingestion.data_transformation.parser.adapters.markdown.ast.utils import (
    clean_single_line,
    clean_text,
    compact_dict,
    get_attr,
)


# --------------- RESULT --------------------------------

def _empty_inline_result() -> Dict[str, Any]:
    return {
        "text": "",
        "links": [],
        "images": [],
        "inline_code": [],
        "html_inline": [],
        "strong_text": [],
    }


# --------------- HELPERS --------------------------------

def _append_visible_text(
    parts: List[str],
    link_stack: List[Dict[str, Any]],
    strong_stack: List[List[str]],
    value: Any,
) -> None:
    text = str(value or "")

    if not text:
        return

    parts.append(text)

    if link_stack:
        link_stack[-1].setdefault("text_parts", []).append(text)

    if strong_stack:
        strong_stack[-1].append(text)


def _close_active_link(link_stack: List[Dict[str, Any]]) -> Dict[str, str] | None:
    if not link_stack:
        return None

    link = link_stack.pop()

    text = clean_single_line("".join(link.pop("text_parts", [])))
    href = clean_single_line(link.get("href", ""))
    title = clean_single_line(link.get("title", ""))

    if not text and not href:
        return None

    return compact_dict({
        "text": text,
        "href": href,
        "title": title,
    })


def _close_active_strong(
    strong_stack: List[List[str]],
) -> str:
    if not strong_stack:
        return ""

    value = clean_single_line("".join(strong_stack.pop()))

    return value


# --------------- INLINE RENDERER --------------------------------

def render_inline(token: Any) -> Dict[str, Any]:
    """
    Render markdown-it inline token into visible text plus metadata.

    Domain-agnostic preservation:
    - links
    - images
    - inline code
    - inline HTML
    - strong/bold visible text
    """

    result = _empty_inline_result()

    if token is None:
        return result

    children = getattr(token, "children", None) or []

    if not children:
        text = clean_text(getattr(token, "content", "") or "")
        result["text"] = text
        return result

    parts: List[str] = []
    links: List[Dict[str, str]] = []
    images: List[Dict[str, str]] = []
    inline_code: List[str] = []
    html_inline: List[str] = []
    strong_text: List[str] = []

    link_stack: List[Dict[str, Any]] = []
    strong_stack: List[List[str]] = []

    for child in children:
        child_type = getattr(child, "type", "")

        # --------------- PLAIN TEXT --------------------------------

        if child_type == "text":
            _append_visible_text(
                parts=parts,
                link_stack=link_stack,
                strong_stack=strong_stack,
                value=getattr(child, "content", ""),
            )
            continue

        # --------------- BREAKS --------------------------------

        if child_type == "softbreak":
            _append_visible_text(parts, link_stack, strong_stack, " ")
            continue

        if child_type == "hardbreak":
            _append_visible_text(parts, link_stack, strong_stack, "\n")
            continue

        # --------------- STRONG --------------------------------

        if child_type == "strong_open":
            strong_stack.append([])
            continue

        if child_type == "strong_close":
            value = _close_active_strong(strong_stack)

            if value:
                strong_text.append(value)

            continue

        # --------------- INLINE CODE --------------------------------

        if child_type == "code_inline":
            code = str(getattr(child, "content", "") or "").strip()

            if code:
                inline_code.append(code)
                _append_visible_text(
                    parts=parts,
                    link_stack=link_stack,
                    strong_stack=strong_stack,
                    value=f"`{code}`",
                )

            continue

        # --------------- INLINE HTML --------------------------------

        if child_type == "html_inline":
            html = str(getattr(child, "content", "") or "").strip()

            if html:
                html_inline.append(html)
                _append_visible_text(
                    parts=parts,
                    link_stack=link_stack,
                    strong_stack=strong_stack,
                    value=html,
                )

            continue

        # --------------- LINKS --------------------------------

        if child_type == "link_open":
            link_stack.append({
                "href": get_attr(child, "href"),
                "title": get_attr(child, "title"),
                "text_parts": [],
            })
            continue

        if child_type == "link_close":
            link = _close_active_link(link_stack)

            if link:
                links.append(link)

            continue

        # --------------- IMAGES --------------------------------

        if child_type == "image":
            alt = clean_single_line(
                getattr(child, "content", "") or get_attr(child, "alt")
            )
            src = clean_single_line(get_attr(child, "src"))
            title = clean_single_line(get_attr(child, "title"))

            image = compact_dict({
                "alt": alt,
                "src": src,
                "title": title,
            })

            if image:
                images.append(image)

            if alt:
                _append_visible_text(
                    parts=parts,
                    link_stack=link_stack,
                    strong_stack=strong_stack,
                    value=alt,
                )

            continue

        # --------------- FORMAT WRAPPERS --------------------------------

        if child_type.endswith("_open") or child_type.endswith("_close"):
            continue

        # --------------- FALLBACK --------------------------------

        fallback = getattr(child, "content", "") or ""

        if fallback:
            _append_visible_text(
                parts=parts,
                link_stack=link_stack,
                strong_stack=strong_stack,
                value=fallback,
            )

    while link_stack:
        link = _close_active_link(link_stack)

        if link:
            links.append(link)

    while strong_stack:
        value = _close_active_strong(strong_stack)

        if value:
            strong_text.append(value)

    result["text"] = clean_text("".join(parts))
    result["links"] = links
    result["images"] = images
    result["inline_code"] = inline_code
    result["html_inline"] = html_inline
    result["strong_text"] = strong_text

    return result


# --------------- CONVENIENCE --------------------------------

def render_inline_text(token: Any) -> str:
    return render_inline(token).get("text", "")


def has_inline_metadata(inline_result: Dict[str, Any]) -> bool:
    if not inline_result:
        return False

    return any(
        inline_result.get(key)
        for key in (
            "links",
            "images",
            "inline_code",
            "html_inline",
            "strong_text",
        )
    )