# -------------------------------------- MARKDOWN CLEANER -------------------------------------------

import re


def clean_markdown(text: str) -> str:
    """
    Safely clean Markdown without destroying document structure.

    KEEP:
    - headings
    - tables
    - lists
    - nested lists
    - fenced code blocks
    - indented code blocks
    - links
    - inline HTML such as <kbd>, <code>, <br>
    - original indentation where it matters

    CLEAN:
    - line endings
    - UTF-8 BOM
    - trailing whitespace
    - excessive blank lines

    NOTE:
    This cleaner should not remove semantic content.
    Parser and enrichment layers should decide meaning later.
    """

    if not text:
        return ""

    # ---------------- NORMALIZE LINE ENDINGS ----------------

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # ---------------- REMOVE UTF-8 BOM ----------------

    text = text.lstrip("\ufeff")

    # ---------------- REMOVE TRAILING WHITESPACE ONLY ----------------
    # Do not strip leading whitespace.
    # Leading spaces are important for nested lists and indented code blocks.

    lines = text.split("\n")
    cleaned_lines = [line.rstrip() for line in lines]

    text = "\n".join(cleaned_lines)

    # ---------------- NORMALIZE EXCESSIVE BLANK LINES ----------------
    # Keep up to 2 blank lines so Markdown block separation survives.

    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text.strip()