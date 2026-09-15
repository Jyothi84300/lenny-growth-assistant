from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.db.dependencies import get_db
from backend.app.db.models import Message, Session as SessionModel
from backend.app.schemas.session import (
    SessionCreateResponse,
    SessionListResponse,
    SessionMessageResponse,
)


router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionCreateResponse, status_code=status.HTTP_201_CREATED)
def create_session(db: Session = Depends(get_db)) -> SessionModel:
    session = SessionModel()

    try:
        db.add(session)
        db.commit()
        db.refresh(session)
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to create session.",
        ) from exc

    return session


@router.get("", response_model=list[SessionListResponse])
def list_sessions(db: Session = Depends(get_db)) -> list[SessionModel]:
    statement = select(SessionModel).order_by(SessionModel.updated_at.desc())
    return list(db.scalars(statement))


@router.get("/{session_id}/messages", response_model=list[SessionMessageResponse])
def list_session_messages(
    session_id: UUID, db: Session = Depends(get_db)
) -> list[Message]:
    if db.get(SessionModel, session_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
        )

    statement = (
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
    )
    return list(db.scalars(statement))
