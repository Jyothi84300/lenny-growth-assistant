from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class ChatRequest(BaseModel):
    session_id: UUID
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ChatMessageResponse(BaseModel):
    id: UUID
    role: str
    content: str

    model_config = ConfigDict(from_attributes=True)


class ChatResponse(BaseModel):
    session_id: UUID
    user_message: ChatMessageResponse
    assistant_message: ChatMessageResponse
    grounded: bool = False
    citations: list[str] = Field(default_factory=list)
