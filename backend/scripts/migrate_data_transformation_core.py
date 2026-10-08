from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


# --------- CONFIG ------------------------

ARCHIVE_NAME = "data_transformation(20261004-220650).zip"

ARCHIVE_PREFIX = (
    "backend/rag/ingestion/data_transformation/"
)

KEEP_ROOTS = (
    "parser/",
    "chunking/",
)

BACKEND_ROOT = Path.cwd().resolve()

TARGET_ROOT = (
    BACKEND_ROOT
    / "ingestion"
    / "data_transformation"
)


# --------- FIND ARCHIVE ------------------------

def find_archive() -> Path:
    candidates = [
        BACKEND_ROOT / ARCHIVE_NAME,
        Path.home() / "Downloads" / ARCHIVE_NAME,
        Path.home() / "Desktop" / ARCHIVE_NAME,
        Path.home() / "Documents" / ARCHIVE_NAME,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise SystemExit(
        f"Could not find {ARCHIVE_NAME}.\n\n"
        "Put the zip inside backend/ and run again."
    )


# --------- EXTRACT CORE ------------------------

def extract_core(
    archive: Path,
) -> None:
    if TARGET_ROOT.exists():
        shutil.rmtree(
            TARGET_ROOT
        )

    TARGET_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    extracted = 0

    with zipfile.ZipFile(
        archive
    ) as zf:
        for info in zf.infolist():
            name = info.filename

            if not name.startswith(
                ARCHIVE_PREFIX
            ):
                continue

            relative = name[
                len(ARCHIVE_PREFIX):
            ]

            if not relative.startswith(
                KEEP_ROOTS
            ):
                continue

            relative_path = Path(
                relative
            )

            if (
                info.is_dir()
                or "__pycache__"
                in relative_path.parts
            ):
                continue

            if relative_path.suffix not in {
                ".py",
                ".toml",
            }:
                continue

            destination = (
                TARGET_ROOT
                / relative_path
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            destination.write_bytes(
                zf.read(
                    info
                )
            )

            extracted += 1

    if extracted == 0:
        raise SystemExit(
            "Archive contained no parser/chunking "
            "source files."
        )

    ingestion_root = (
        BACKEND_ROOT
        / "ingestion"
    )

    ingestion_root.mkdir(
        exist_ok=True
    )

    (
        ingestion_root
        / "__init__.py"
    ).touch()

    (
        TARGET_ROOT
        / "__init__.py"
    ).touch()

    print(
        f"Extracted {extracted} source/config files."
    )


# --------- PACKAGE IMPORTS ------------------------

def rewrite_internal_imports() -> None:
    for path in TARGET_ROOT.rglob(
        "*.py"
    ):
        text = path.read_text(
            encoding="utf-8"
        )

        text = text.replace(
            "rag.ingestion.data_transformation",
            "ingestion.data_transformation",
        )

        path.write_text(
            text,
            encoding="utf-8",
        )


# --------- CHUNKING CONFIG ------------------------

def patch_chunking_config() -> None:
    path = (
        TARGET_ROOT
        / "chunking"
        / "config.py"
    )

    text = path.read_text(
        encoding="utf-8"
    )

    text = text.replace(
        "from rag.ingestion.embeddings.config "
        "import get_embedding_pipeline_config\n\n",
        "",
        1,
    )

    old = """class TokenizationConfig:
    enabled: bool
    role: str
    add_special_tokens: bool
    require_local_assets: bool
"""

    new = """class TokenizationConfig:
    enabled: bool
    role: str
    add_special_tokens: bool
    model_max_input_tokens: int
    require_local_assets: bool
"""

    if old not in text:
        raise SystemExit(
            "Could not patch TokenizationConfig."
        )

    text = text.replace(
        old,
        new,
        1,
    )

    old = """        strings=(
            "role",
        ),
    )

    semantic = _build(
"""

    new = """        ints=(
            "model_max_input_tokens",
        ),
        strings=(
            "role",
        ),
    )

    semantic = _build(
"""

    if old not in text:
        raise SystemExit(
            "Could not patch tokenization config loader."
        )

    text = text.replace(
        old,
        new,
        1,
    )

    start = text.find(
        "def resolve_model_max_input_tokens(\n) -> int:"
    )

    end = text.find(
        "\n\n#--------------------TOKEN BUDGET--------------------",
        start,
    )

    if start < 0 or end < 0:
        raise SystemExit(
            "Could not locate model token-limit resolver."
        )

    replacement = """def resolve_model_max_input_tokens(
    config: ChunkingConfig,
) -> int:
    max_input_tokens = (
        config
        .tokenization
        .model_max_input_tokens
    )

    if (
        isinstance(
            max_input_tokens,
            bool,
        )
        or not isinstance(
            max_input_tokens,
            int,
        )
    ):
        raise ChunkingConfigError(
            "tokenization.model_max_input_tokens "
            "must be integer"
        )

    if max_input_tokens <= 0:
        raise ChunkingConfigError(
            "tokenization.model_max_input_tokens "
            "must be positive"
        )

    return max_input_tokens
"""

    text = (
        text[:start]
        + replacement
        + text[end:]
    )

    text = text.replace(
        "resolve_model_max_input_tokens()",
        "resolve_model_max_input_tokens(config)",
    )

    text = text.replace(
        'if hard.limit_source != "embedding_config":',
        'if hard.limit_source != "tokenizer_contract":',
        1,
    )

    text = text.replace(
        '"\'embedding_config\'"',
        '"\'tokenizer_contract\'"',
        1,
    )

    path.write_text(
        text,
        encoding="utf-8",
    )

    # --------- TOML ------------------------

    toml_path = (
        TARGET_ROOT
        / "chunking"
        / "config.toml"
    )

    toml = toml_path.read_text(
        encoding="utf-8"
    )

    toml = toml.replace(
        "# Model identity, tokenizer identity and "
        "the actual model max input length\n"
        "# come from rag.ingestion.embeddings "
        "configuration at runtime.\n",
        "# Exact tokenizer contract for Cloudflare's hosted\n"
        "# @cf/qwen/qwen3-embedding-0.6b model.\n",
        1,
    )

    if (
        "model_max_input_tokens"
        not in toml
    ):
        toml = toml.replace(
            "add_special_tokens = false\n",
            (
                "add_special_tokens = false\n"
                "model_max_input_tokens = 8192\n"
            ),
            1,
        )

    toml = toml.replace(
        'limit_source = "embedding_config"',
        'limit_source = "tokenizer_contract"',
        1,
    )

    toml_path.write_text(
        toml,
        encoding="utf-8",
    )


# --------- TOKENIZER RUNTIME ------------------------

def replace_tokenizer_runtime() -> None:
    tokenization_root = (
        TARGET_ROOT
        / "chunking"
        / "tokenization"
    )

    (
        tokenization_root
        / "registry.py"
    ).unlink(
        missing_ok=True
    )

    service_path = (
        tokenization_root
        / "service.py"
    )

    service_path.write_text(
        '''from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..config import get_chunking_config
from .base import BatchTokenCount
from .base import TokenizedText
from .base import Tokenizer
from .local_tokenizer import LocalTokenizer


# --------- TOKENIZER CONTRACT ------------------------

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

BACKEND_ROOT = (
    Path(__file__)
    .resolve()
    .parents[4]
)

TOKENIZER_DIR = (
    BACKEND_ROOT
    / ".models"
    / "qwen3-embedding-0.6b"
)


# --------- POLICY ------------------------

def _resolve_role(
    role: str | None,
) -> str:
    configured = (
        get_chunking_config()
        .tokenization
        .role
        if role is None
        else role
    )

    normalized = str(
        configured
    ).strip().lower()

    if normalized != "dense":
        raise ValueError(
            "Only the dense Qwen3 tokenizer "
            "is available in this deployment"
        )

    return normalized


def _resolve_add_special_tokens(
    add_special_tokens: bool | None,
) -> bool:
    if add_special_tokens is not None:
        return add_special_tokens

    return (
        get_chunking_config()
        .tokenization
        .add_special_tokens
    )


# --------- TOKENIZER SINGLETON ------------------------

@lru_cache(maxsize=1)
def _load_tokenizer() -> Tokenizer:
    tokenizer_path = (
        TOKENIZER_DIR
        / "tokenizer.json"
    )

    if not tokenizer_path.is_file():
        raise RuntimeError(
            "Qwen3 tokenizer is missing. Run "
            "scripts/download_qwen3_06B_tokenizer.py first. "
            f"Expected: {tokenizer_path}"
        )

    return LocalTokenizer(
        model_id=MODEL_ID,
        model_path=TOKENIZER_DIR,
    )


def get_tokenizer(
    *,
    role: str | None = None,
) -> Tokenizer:
    _resolve_role(
        role
    )

    return _load_tokenizer()


# --------- TOKEN COUNTING ------------------------

def count_tokens(
    text: str,
    *,
    role: str | None = None,
    add_special_tokens: bool | None = None,
) -> int:
    tokenizer = get_tokenizer(
        role=role
    )

    return tokenizer.count(
        text,
        add_special_tokens=(
            _resolve_add_special_tokens(
                add_special_tokens
            )
        ),
    )


def count_tokens_many(
    texts: list[str] | tuple[str, ...],
    *,
    role: str | None = None,
    add_special_tokens: bool | None = None,
) -> BatchTokenCount:
    tokenizer = get_tokenizer(
        role=role
    )

    return tokenizer.count_many(
        texts,
        add_special_tokens=(
            _resolve_add_special_tokens(
                add_special_tokens
            )
        ),
    )


# --------- TOKENIZATION ------------------------

def tokenize_text(
    text: str,
    *,
    role: str | None = None,
    add_special_tokens: bool | None = None,
) -> TokenizedText:
    tokenizer = get_tokenizer(
        role=role
    )

    return tokenizer.tokenize(
        text,
        add_special_tokens=(
            _resolve_add_special_tokens(
                add_special_tokens
            )
        ),
    )


def encode_tokens(
    text: str,
    *,
    role: str | None = None,
    add_special_tokens: bool | None = None,
) -> tuple[int, ...]:
    tokenizer = get_tokenizer(
        role=role
    )

    return tokenizer.encode(
        text,
        add_special_tokens=(
            _resolve_add_special_tokens(
                add_special_tokens
            )
        ),
    )


def encode_tokens_many(
    texts: list[str] | tuple[str, ...],
    *,
    role: str | None = None,
    add_special_tokens: bool | None = None,
) -> tuple[tuple[int, ...], ...]:
    tokenizer = get_tokenizer(
        role=role
    )

    return tokenizer.encode_many(
        texts,
        add_special_tokens=(
            _resolve_add_special_tokens(
                add_special_tokens
            )
        ),
    )


# --------- CACHE ------------------------

def clear_tokenizer_cache() -> None:
    _load_tokenizer.cache_clear()
''',
        encoding="utf-8",
    )

    init_path = (
        tokenization_root
        / "__init__.py"
    )

    init_path.write_text(
        '''from .base import BatchTokenCount
from .base import TokenizedText
from .base import Tokenizer
from .base import TokenizerDecodingError
from .base import TokenizerEncodingError
from .base import TokenizerError
from .base import TokenizerIdentity
from .base import TokenizerUnavailableError
from .base import assert_tokenizer_identity
from .local_tokenizer import LocalTokenizer
from .service import clear_tokenizer_cache
from .service import count_tokens
from .service import count_tokens_many
from .service import encode_tokens
from .service import encode_tokens_many
from .service import get_tokenizer
from .service import tokenize_text


__all__ = [
    "BatchTokenCount",
    "LocalTokenizer",
    "TokenizedText",
    "Tokenizer",
    "TokenizerDecodingError",
    "TokenizerEncodingError",
    "TokenizerError",
    "TokenizerIdentity",
    "TokenizerUnavailableError",
    "assert_tokenizer_identity",
    "clear_tokenizer_cache",
    "count_tokens",
    "count_tokens_many",
    "encode_tokens",
    "encode_tokens_many",
    "get_tokenizer",
    "tokenize_text",
]
''',
        encoding="utf-8",
    )


# --------- VERIFY ------------------------

def verify() -> None:
    required = [
        (
            TARGET_ROOT
            / "parser"
            / "markdown"
            / "parser.py"
        ),
        (
            TARGET_ROOT
            / "parser"
            / "markdown"
            / "schema.py"
        ),
        (
            TARGET_ROOT
            / "chunking"
            / "chunker.py"
        ),
        (
            TARGET_ROOT
            / "chunking"
            / "config.py"
        ),
        (
            TARGET_ROOT
            / "chunking"
            / "config.toml"
        ),
        (
            TARGET_ROOT
            / "chunking"
            / "tokenization"
            / "service.py"
        ),
    ]

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]

    if missing:
        raise SystemExit(
            "Migration incomplete. Missing:\n  "
            + "\n  ".join(
                missing
            )
        )

    stale = []

    for path in TARGET_ROOT.rglob(
        "*.py"
    ):
        text = path.read_text(
            encoding="utf-8"
        )

        if "rag.ingestion" in text:
            stale.append(
                str(path)
            )

    if stale:
        raise SystemExit(
            "Old rag.ingestion imports remain:\n  "
            + "\n  ".join(
                stale
            )
        )

    print()
    print("Core migration verified.")
    print(f"Target: {TARGET_ROOT}")


# --------- MAIN ------------------------

def main() -> None:
    archive = find_archive()

    print(
        f"Using archive: {archive}"
    )

    extract_core(
        archive
    )

    rewrite_internal_imports()

    patch_chunking_config()

    replace_tokenizer_runtime()

    verify()

    print()
    print("Migration complete.")


if __name__ == "__main__":
    main()
