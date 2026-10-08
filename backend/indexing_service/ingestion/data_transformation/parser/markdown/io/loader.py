#--------------------------------------MARKDOWN LOADER-------------------------------------------

def load_markdown(file_path: str) -> str:
    """
    Load raw markdown content from file.

    IMPORTANT:
    - Do NOT clean aggressively here
    - Do NOT strip structure (headings, tables, lists)
    - Just read and return as-is (with minimal normalization)
    """

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    except Exception as e:
        raise ValueError(f"Markdown load failed for {file_path} | Error: {e}")

    if not text or not text.strip():
        raise ValueError(f"Empty markdown file: {file_path}")

    # Minimal normalization (safe)
    
    text = text.replace("\r\n", "\n")   # Windows → Unix
    text = text.replace("\r", "\n")

    return text.strip()