from __future__ import annotations


class Document:
    """
    Legacy parser output container.

    Keeps final text + structured enriched sections together.

    Note:
    This is the compatibility path. Newer execution code should prefer
    MarkdownParsedUnit from markdown/execution/schema.py.
    """

    def __init__(self, text: str, metadata: dict):
        self.text = text
        self.metadata = metadata

    def __repr__(self) -> str:
        sections = len(self.metadata.get("ast", []))
        doc_type = self.metadata.get("doc_type", "unknown")
        return f"<Document type={doc_type} sections={sections}>"