from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.db.dependencies import get_db
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.services.chat_service import ChatService, ChatSessionNotFoundError


router = APIRouter(prefix="/chat", tags=["chat"])
chat_service = ChatService()


@router.post("", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
def create_chat_response(
    request: ChatRequest, db: Session = Depends(get_db)
) -> ChatResponse:
    try:
        user_message, assistant_message = chat_service.create_placeholder_response(
            db, request.session_id, request.message
        )
    except ChatSessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found.",
        ) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to save chat messages.",
        ) from exc

    return ChatResponse(
        session_id=request.session_id,
        user_message=user_message,
        assistant_message=assistant_message,
    )
