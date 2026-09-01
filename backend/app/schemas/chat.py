from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import Field


#------------------------CONVERSATION-------------------------

class CreateConversationResponse(BaseModel):
    conversation_id: str
    status: Literal["ready"]


#------------------------MESSAGE-------------------------

class CreateChatMessageRequest(BaseModel):
    conversation_id: str = Field(
        min_length=1,
        max_length=160,
    )

    message: str = Field(
        min_length=1,
        max_length=20000,
    )


class CreateChatMessageResponse(BaseModel):
    conversation_id: str
    user_turn_id: str
    status: Literal["processing"]
