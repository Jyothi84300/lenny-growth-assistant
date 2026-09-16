"""Vector retrieval for the transcript knowledge base only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.db.models import Chunk, Document
from backend.app.rag.embeddings import EmbeddingService
from backend.app.rag.ingestion import EmbeddingDimensionError, validate_embedding_dimension


DEFAULT_TOP_K = 5
DEFAULT_MAX_COSINE_DISTANCE = 0.7


class RetrievalEmbeddingError(RuntimeError):
    pass


class RetrievalDatabaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: UUID
    document_id: UUID
    content: str
    cosine_distance: float
    document_title: str
    source_url: str | None
    source_type: str
    document_metadata: dict[str, Any]
    chunk_metadata: dict[str, Any]


class TranscriptRetrievalService:
    """Retrieve transcript chunks by cosine distance from a query embedding."""

    def __init__(
        self,
        db: Session,
        embedding_service: EmbeddingService,
        default_top_k: int = DEFAULT_TOP_K,
        default_max_cosine_distance: float | None = DEFAULT_MAX_COSINE_DISTANCE,
    ) -> None:
        self.db = db
        self.embedding_service = embedding_service
        self.default_top_k = default_top_k
        self.default_max_cosine_distance = default_max_cosine_distance

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        max_cosine_distance: float | None = None,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("Retrieval query cannot be empty.")

        limit = self.default_top_k if top_k is None else top_k
        if limit <= 0:
            raise ValueError("top_k must be greater than zero.")

        threshold = (
            self.default_max_cosine_distance
            if max_cosine_distance is None
            else max_cosine_distance
        )
        if threshold is not None and not 0 <= threshold <= 2:
            raise ValueError("max_cosine_distance must be between 0 and 2.")

        try:
            embedding = self.embedding_service.embed(query)
            validate_embedding_dimension(embedding)
        except EmbeddingDimensionError:
            raise
        except Exception as exc:
            raise RetrievalEmbeddingError(f"Could not embed retrieval query: {exc}") from exc

        cosine_distance = Chunk.embedding.cosine_distance(embedding).label("cosine_distance")
        statement = (
            select(Chunk, Document, cosine_distance)
            .join(Document, Chunk.document_id == Document.id)
            .order_by(cosine_distance.asc())
            .limit(limit)
        )
        if threshold is not None:
            statement = statement.where(cosine_distance <= threshold)

        try:
            rows = list(self.db.execute(statement))
        except SQLAlchemyError as exc:
            raise RetrievalDatabaseError("Could not retrieve transcript chunks.") from exc

        results = [
            RetrievalResult(
                chunk_id=chunk.id,
                document_id=document.id,
                content=chunk.content,
                cosine_distance=float(distance),
                document_title=document.title,
                source_url=document.source_url,
                source_type=document.source_type,
                document_metadata=document.metadata_json,
                chunk_metadata=chunk.metadata_json,
            )
            for chunk, document, distance in rows
        ]
        return sorted(results, key=lambda result: result.cosine_distance)[:limit]
