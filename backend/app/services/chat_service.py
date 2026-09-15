from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.db.models import Message, Session as SessionModel


PLACEHOLDER_RESPONSE = (
    "Placeholder response: the LLM and RAG layers have not been implemented yet."
)


class ChatSessionNotFoundError(Exception):
    pass


class ChatService:
    def create_placeholder_response(
        self, db: Session, session_id: UUID, content: str
    ) -> tuple[Message, Message]:
        try:
            session = db.get(SessionModel, session_id)
            if session is None:
                raise ChatSessionNotFoundError

            # This ordered, session-filtered conversation will become LLM context later.
            list(
                db.scalars(
                    select(Message)
                    .where(Message.session_id == session.id)
                    .order_by(Message.created_at.asc())
                )
            )

            user_message = Message(
                session_id=session.id,
                role="user",
                content=content,
            )
            assistant_message = Message(
                session_id=session.id,
                role="assistant",
                content=PLACEHOLDER_RESPONSE,
            )
            db.add(user_message)
            db.add(assistant_message)
            db.commit()
            db.refresh(user_message)
            db.refresh(assistant_message)
        except SQLAlchemyError:
            db.rollback()
            raise

        return user_message, assistant_message
