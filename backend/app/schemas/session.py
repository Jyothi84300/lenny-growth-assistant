from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class SessionCreateResponse(BaseModel):
    id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SessionListResponse(SessionCreateResponse):
    pass


class SessionMessageResponse(BaseModel):
    id: UUID
    session_id: UUID
    role: str
    content: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
