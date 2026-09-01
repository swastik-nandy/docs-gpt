from __future__ import annotations

import asyncio

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from app.core.settings import Settings
from app.core.settings import get_settings
from app.core.tenancy import resolve_tenant_context
from app.schemas.chat import CreateChatMessageRequest
from app.schemas.chat import CreateChatMessageResponse
from app.schemas.chat import CreateConversationResponse
from app.services.chat_service import ChatService
from app.services.chat_service import ConversationBusyError
from app.services.redis_event_bus import RedisEventBus


router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


#------------------------DEPENDENCIES-------------------------

def get_chat_service(
    request: Request,
) -> ChatService:
    service = getattr(
        request.app.state,
        "chat_service",
        None,
    )

    if service is None:
        raise RuntimeError(
            "ChatService is not initialized."
        )

    return service


#------------------------CONVERSATIONS-------------------------

@router.post(
    "/conversations",
    response_model=CreateConversationResponse,
    status_code=201,
)
async def create_conversation(
    settings: Settings = Depends(
        get_settings
    ),
    service: ChatService = Depends(
        get_chat_service
    ),
) -> CreateConversationResponse:
    tenant = resolve_tenant_context(
        requested_tenant_id=None,
        settings=settings,
    )

    conversation = (
        await service.create_conversation(
            tenant_id=tenant.tenant_id,
            user_id=settings.default_user_id,
        )
    )

    return CreateConversationResponse(
        conversation_id=(
            conversation.conversation_id
        ),
        status="ready",
    )


#------------------------MESSAGES-------------------------

@router.post(
    "/messages",
    response_model=CreateChatMessageResponse,
    status_code=202,
)
async def create_message(
    payload: CreateChatMessageRequest,
    settings: Settings = Depends(
        get_settings
    ),
    service: ChatService = Depends(
        get_chat_service
    ),
) -> CreateChatMessageResponse:
    tenant = resolve_tenant_context(
        requested_tenant_id=None,
        settings=settings,
    )

    try:
        submission = (
            await service.submit_message(
                tenant_id=tenant.tenant_id,
                user_id=(
                    settings.default_user_id
                ),
                conversation_id=(
                    payload.conversation_id
                ),
                message=payload.message,
            )
        )

    except ConversationBusyError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(
                exc
            ),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(
                exc
            ),
        ) from exc

    return CreateChatMessageResponse(
        conversation_id=(
            submission.conversation_id
        ),
        user_turn_id=(
            submission.user_turn_id
        ),
        status="processing",
    )



#------------------------WEBSOCKET-------------------------

@router.websocket("/ws/{conversation_id}")
async def chat_websocket(
    websocket: WebSocket,
    conversation_id: str,
    settings: Settings = Depends(
        get_settings
    ),
) -> None:
    tenant = resolve_tenant_context(
        requested_tenant_id=None,
        settings=settings,
    )

    event_bus = getattr(
        websocket.app.state,
        "redis_event_bus",
        None,
    )

    await websocket.accept()

    if not isinstance(
        event_bus,
        RedisEventBus,
    ):
        await websocket.close(
            code=1011,
            reason=(
                "Redis event bus is unavailable."
            ),
        )
        return

    await websocket.send_json(
        {
            "event_type": "connection.ready",
            "conversation_id": conversation_id,
            "turn_id": None,
            "text": "",
            "payload": {
                "status": "ready",
            },
        }
    )

    relay_task = asyncio.create_task(
        _relay_chat_events(
            websocket=websocket,
            event_bus=event_bus,
            tenant_id=tenant.tenant_id,
            conversation_id=conversation_id,
        ),
        name=(
            "chat-ws-relay:"
            f"{conversation_id}"
        ),
    )

    disconnect_task = asyncio.create_task(
        _wait_for_websocket_disconnect(
            websocket
        ),
        name=(
            "chat-ws-disconnect:"
            f"{conversation_id}"
        ),
    )

    tasks = {
        relay_task,
        disconnect_task,
    }

    try:
        done, _ = await asyncio.wait(
            tasks,
            return_when=(
                asyncio.FIRST_COMPLETED
            ),
        )

        if relay_task in done:
            exception = (
                relay_task.exception()
            )

            if exception is not None:
                await _close_websocket_safely(
                    websocket,
                    code=1011,
                    reason=(
                        "Chat event stream failed."
                    ),
                )

    finally:
        for task in tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )


#------------------------WEBSOCKET RELAY-------------------------

async def _relay_chat_events(
    *,
    websocket: WebSocket,
    event_bus: RedisEventBus,
    tenant_id: str,
    conversation_id: str,
) -> None:
    try:
        async for event in (
            event_bus.subscribe_chat(
                tenant_id=tenant_id,
                conversation_id=conversation_id,
            )
        ):
            await websocket.send_json(
                event
            )

    except WebSocketDisconnect:
        return


async def _wait_for_websocket_disconnect(
    websocket: WebSocket,
) -> None:
    while True:
        message = await websocket.receive()

        if (
            message.get("type")
            == "websocket.disconnect"
        ):
            return


async def _close_websocket_safely(
    websocket: WebSocket,
    *,
    code: int,
    reason: str,
) -> None:
    try:
        await websocket.close(
            code=code,
            reason=reason,
        )

    except Exception:
        pass
