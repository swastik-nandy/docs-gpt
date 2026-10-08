#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


# --------- MODEL CONFIG ------------------------

REPO_ID = "Qwen/Qwen3-Embedding-0.6B"

BACKEND_ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_DIR = (
    BACKEND_ROOT
    / ".models"
    / "qwen3-embedding-0.6b"
)

TOKENIZER_PATH = TOKENIZER_DIR / "tokenizer.json"


# --------- CLI ------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download only the tokenizer.json required for "
            "Qwen3-Embedding-0.6B chunk token counting."
        )
    )

    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face revision to resolve and download.",
    )

    return parser.parse_args()


# --------- HELPERS ------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


# --------- MAIN ------------------------

def main() -> int:
    args = parse_args()

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        raise SystemExit(
            "\nMissing dependency: huggingface_hub\n\n"
            "Run with:\n"
            "  uv run --with huggingface-hub "
            "python scripts/download_qwen3_06B_tokenizer.py\n"
        )

    TOKENIZER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    api = HfApi()

    print(f"Repository : {REPO_ID}")
    print(f"Revision   : {args.revision}")
    print()

    print("Resolving immutable Hugging Face revision...")

    info = api.model_info(
        REPO_ID,
        revision=args.revision,
    )

    resolved_revision = str(info.sha or "").strip()

    if not resolved_revision:
        raise RuntimeError(
            "Unable to resolve Hugging Face revision."
        )

    print(f"Pinned SHA : {resolved_revision}")
    print()
    print("Downloading tokenizer.json only...")

    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="tokenizer.json",
        revision=resolved_revision,
        local_dir=TOKENIZER_DIR,
    )

    downloaded_path = Path(downloaded_path)

    if not downloaded_path.is_file():
        raise RuntimeError(
            f"Tokenizer was not downloaded: {downloaded_path}"
        )

    if downloaded_path.stat().st_size <= 0:
        raise RuntimeError(
            "Downloaded tokenizer.json is empty."
        )

    digest = sha256_file(downloaded_path)

    print()
    print("Tokenizer provisioned successfully.")
    print(f"Path       : {downloaded_path}")
    print(
        f"Size       : "
        f"{downloaded_path.stat().st_size / (1024 * 1024):.2f} MiB"
    )
    print(f"SHA-256    : {digest}")
    print(f"Revision   : {resolved_revision}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
