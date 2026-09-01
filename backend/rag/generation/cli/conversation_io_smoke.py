from __future__ import annotations

import argparse
import asyncio
import json

from rag.conversation.repository import PostgresConversationRepository
from rag.generation.conversation_io import (
    build_generation_state_from_turn,
    persist_assistant_turn_from_generation_state,
)
from tenancy.dev import load_dev_identity_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test generation conversation state I/O without LLM.",
    )
    parser.add_argument("--conversation-id", required=True)
    parser.add_argument("--turn-id", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--persist-fake-answer", action="store_true")
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    dev = load_dev_identity_config()

    repo = await PostgresConversationRepository.connect()

    state = await build_generation_state_from_turn(
        repository=repo,
        conversation_id=args.conversation_id,
        tenant_id=dev.tenant_id,
        user_id=dev.user_id,
        user_turn_id=args.turn_id,
        current_user_message=args.message,
    )

    payload = {
        "tenant_id": state["tenant_id"],
        "user_id": state["user_id"],
        "conversation_id": state["conversation_id"],
        "turn_id": state["turn_id"],
        "current_user_message": state["current_user_message"],
        "recent_turn_count": len(state.get("recent_turns", [])),
        "conversation_summary": state.get("conversation_summary"),
    }

    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.persist_fake_answer:
        state["final_answer"] = "Fake assistant answer for generation I/O smoke test."
        state["generation_metadata"] = {
            "provider": "fake",
            "model": "fake-model",
            "role": "generation_primary",
        }
        state["grounding_result"] = {
            "passed": True,
            "score": 1.0,
            "reasons": ["smoke_test"],
        }

        assistant_turn = await persist_assistant_turn_from_generation_state(
            repository=repo,
            state=state,
            metadata={"smoke_test": True},
        )

        print("assistant_turn_id:", assistant_turn.turn_id)
        print("assistant_turn_index:", assistant_turn.turn_index)

    await repo.close()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
