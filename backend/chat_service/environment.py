from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


# --------- PATHS ------------------------

SERVICE_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

BACKEND_ROOT = (
    SERVICE_ROOT
    .parent
)


# --------- ENVIRONMENT ------------------------

def load_environment() -> None:
    """
    OS environment variables always win.

    Local .env files are development fallbacks only.
    """

    candidates = (
        SERVICE_ROOT / ".env",
        BACKEND_ROOT / ".env",
    )

    for path in candidates:
        if path.is_file():
            load_dotenv(
                dotenv_path=path,
                override=False,
            )

            break
