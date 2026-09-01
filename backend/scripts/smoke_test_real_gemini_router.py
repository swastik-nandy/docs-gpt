from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.generation.llm.config import ModelRegistry
from rag.generation.llm.router import ModelRouter
from rag.generation.llm.schema import ChatMessage, ModelRequest


def load_local_env(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


async def main() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    load_local_env(backend_root / ".env")

    registry_path = backend_root / "rag" / "generation" / "config" / "models.toml"

    registry = ModelRegistry.from_toml(registry_path)
    router = ModelRouter.from_registry(registry)

    try:
        response = await router.generate(
            ModelRequest(
                role="generation_primary",
                system="You are a smoke test responder. Reply briefly.",
                messages=[
                    ChatMessage(
                        role="user",
                        content="Say: generation router is working.",
                    )
                ],
                temperature=0.0,
                max_output_tokens=128,
            )
        )

        print("provider:", response.provider)
        print("model:", response.model)
        print("finish_reason:", response.raw_finish_reason)
        print("text:", response.text)
        print("Real Gemini router smoke test passed.")
    finally:
        await router.close()


if __name__ == "__main__":
    asyncio.run(main())
