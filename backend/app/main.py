from fastapi import FastAPI
from sqlalchemy import text

from backend.app.api.chat import router as chat_router
from backend.app.api.sessions import router as sessions_router
from backend.app.db.database import engine


app = FastAPI(
    title="Lenny Growth Assistant API",
    version="0.1.0",
)

app.include_router(sessions_router, prefix="/api")
app.include_router(chat_router, prefix="/api")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/health/db")
def database_health_check():
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        result.scalar_one()

    return {"status": "ok", "database": "connected"}
