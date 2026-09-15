from collections.abc import Generator

from sqlalchemy.orm import Session

from backend.app.db.database import SessionLocal


def get_db() -> Generator[Session, None, None]:
    """Provide a database session for the duration of a request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
