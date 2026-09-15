from datetime import UTC, datetime
import unittest
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

from backend.app.db.dependencies import get_db
from backend.app.db.models import Message, Session as SessionModel
from backend.app.main import app


class FakeDatabase:
    def __init__(self) -> None:
        self.sessions: dict[UUID, SessionModel] = {}
        self.messages: list[Message] = []
        self.rollback_called = False

    def add(self, session: SessionModel) -> None:
        now = datetime.now(UTC)
        session.id = uuid4()
        session.created_at = now
        session.updated_at = now
        self.sessions[session.id] = session

    def commit(self) -> None:
        pass

    def refresh(self, session: SessionModel) -> None:
        pass

    def rollback(self) -> None:
        self.rollback_called = True

    def get(self, model: type[SessionModel], session_id: UUID) -> SessionModel | None:
        return self.sessions.get(session_id)

    def scalars(self, statement):
        description = str(statement)
        if "FROM sessions" in description:
            return iter(sorted(self.sessions.values(), key=lambda item: item.updated_at, reverse=True))
        return iter(sorted(self.messages, key=lambda item: item.created_at))

    def close(self) -> None:
        pass


class SessionApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database = FakeDatabase()
        app.dependency_overrides[get_db] = lambda: self.database
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        app.dependency_overrides.clear()

    def test_session_endpoints(self) -> None:
        created = self.client.post("/api/sessions")
        self.assertEqual(created.status_code, 201)
        payload = created.json()
        session_id = UUID(payload["id"])
        self.assertTrue(payload["created_at"])
        self.assertTrue(payload["updated_at"])

        listed = self.client.get("/api/sessions")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual([item["id"] for item in listed.json()], [str(session_id)])

        later = Message(
            id=uuid4(),
            session_id=session_id,
            role="assistant",
            content="Later message",
            created_at=datetime(2026, 1, 2, tzinfo=UTC),
        )
        earlier = Message(
            id=uuid4(),
            session_id=session_id,
            role="user",
            content="Earlier message",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        self.database.messages.extend([later, earlier])

        messages = self.client.get(f"/api/sessions/{session_id}/messages")
        self.assertEqual(messages.status_code, 200)
        self.assertEqual(
            [message["id"] for message in messages.json()],
            [str(earlier.id), str(later.id)],
        )

        missing = self.client.get(f"/api/sessions/{uuid4()}/messages")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["detail"], "Session not found.")

    def test_create_session_rolls_back_database_error(self) -> None:
        class FailingDatabase(FakeDatabase):
            def commit(self) -> None:
                raise SQLAlchemyError("database unavailable")

        database = FailingDatabase()
        app.dependency_overrides[get_db] = lambda: database

        response = self.client.post("/api/sessions")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Unable to create session.")
        self.assertTrue(database.rollback_called)
