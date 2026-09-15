from datetime import UTC, datetime
import unittest
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from backend.app.db.dependencies import get_db
from backend.app.db.models import Message, Session as SessionModel
from backend.app.main import app
from backend.app.services.chat_service import PLACEHOLDER_RESPONSE


class FakeDatabase:
    def __init__(self, fail_commit: bool = False) -> None:
        self.sessions: dict[UUID, SessionModel] = {}
        self.messages: list[Message] = []
        self.fail_commit = fail_commit
        self.rollback_called = False

    def add(self, item: SessionModel | Message) -> None:
        now = datetime.now(UTC)
        if isinstance(item, SessionModel):
            item.id = uuid4()
            item.created_at = now
            item.updated_at = now
            self.sessions[item.id] = item
            return

        item.id = uuid4()
        item.created_at = now
        self.messages.append(item)

    def commit(self) -> None:
        if self.fail_commit:
            raise SQLAlchemyError("database unavailable")

    def refresh(self, item: SessionModel | Message) -> None:
        pass

    def rollback(self) -> None:
        self.rollback_called = True

    def get(self, model: type[SessionModel], session_id: UUID) -> SessionModel | None:
        return self.sessions.get(session_id)

    def scalars(self, statement):
        session_id = statement._where_criteria[0].right.value
        return iter(
            sorted(
                (message for message in self.messages if message.session_id == session_id),
                key=lambda message: message.created_at,
            )
        )

    def close(self) -> None:
        pass


class ChatApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database = FakeDatabase()
        self.session = SessionModel()
        self.database.add(self.session)
        app.dependency_overrides[get_db] = lambda: self.database
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        app.dependency_overrides.clear()

    def test_chat_persists_user_and_placeholder_messages(self) -> None:
        other_session = SessionModel()
        self.database.add(other_session)
        other_message = Message(
            id=uuid4(),
            session_id=other_session.id,
            role="user",
            content="Other session message",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        self.database.messages.append(other_message)

        response = self.client.post(
            "/api/chat",
            json={"session_id": str(self.session.id), "message": "  Help me grow.  "},
        )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["session_id"], str(self.session.id))
        self.assertEqual(payload["user_message"]["content"], "Help me grow.")
        self.assertEqual(payload["assistant_message"]["content"], PLACEHOLDER_RESPONSE)
        self.assertFalse(payload["grounded"])
        self.assertEqual(payload["citations"], [])

        persisted = [
            message for message in self.database.messages if message.session_id == self.session.id
        ]
        self.assertEqual([message.role for message in persisted], ["user", "assistant"])
        self.assertEqual([message.content for message in persisted], ["Help me grow.", PLACEHOLDER_RESPONSE])

    def test_chat_returns_404_for_unknown_session(self) -> None:
        response = self.client.post(
            "/api/chat", json={"session_id": str(uuid4()), "message": "Hello"}
        )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Session not found.")

    def test_chat_rejects_an_empty_message(self) -> None:
        response = self.client.post(
            "/api/chat", json={"session_id": str(self.session.id), "message": "   "}
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.database.messages, [])

    def test_chat_rolls_back_database_error(self) -> None:
        database = FakeDatabase(fail_commit=True)
        session = SessionModel()
        database.add(session)
        app.dependency_overrides[get_db] = lambda: database

        response = self.client.post(
            "/api/chat", json={"session_id": str(session.id), "message": "Hello"}
        )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Unable to save chat messages.")
        self.assertTrue(database.rollback_called)
