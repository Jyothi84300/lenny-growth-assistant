from uuid import uuid4
import unittest

from sqlalchemy.exc import SQLAlchemyError

from backend.app.db.models import Chunk, Document
from backend.app.rag.ingestion import EMBEDDING_DIMENSION, EmbeddingDimensionError
from backend.app.rag.retrieval import (
    RetrievalDatabaseError,
    TranscriptRetrievalService,
)


class FakeEmbeddingService:
    def __init__(self, embedding: list[float] | None = None) -> None:
        self.embedding = embedding or [0.1] * EMBEDDING_DIMENSION
        self.queries: list[str] = []

    def embed(self, query: str) -> list[float]:
        self.queries.append(query)
        return self.embedding


class FakeDatabase:
    def __init__(self, rows: list[tuple[Chunk, Document, float]]) -> None:
        self.rows = rows
        self.statement = None

    def execute(self, statement):
        self.statement = statement
        return self.rows


def make_row(content: str, distance: float, title: str = "Episode") -> tuple[Chunk, Document, float]:
    document = Document(
        id=uuid4(),
        title=title,
        source_url="https://example.com/episode",
        source_type="lenny_podcast_transcript",
        metadata_json={"guest": "Ada", "video_id": "abc123"},
    )
    chunk = Chunk(
        id=uuid4(),
        document_id=document.id,
        content=content,
        embedding=[0.1] * EMBEDDING_DIMENSION,
        metadata_json={"chunk_index": 2, "char_start": 100},
    )
    return chunk, document, distance


class RetrievalTests(unittest.TestCase):
    def test_embeds_query_and_orders_results_by_cosine_distance(self) -> None:
        embedding_service = FakeEmbeddingService()
        database = FakeDatabase(
            [make_row("Less relevant", 0.5), make_row("Most relevant", 0.1)]
        )

        results = TranscriptRetrievalService(database, embedding_service).retrieve(
            "How should a product team experiment?"
        )

        self.assertEqual(embedding_service.queries, ["How should a product team experiment?"])
        self.assertEqual([result.content for result in results], ["Most relevant", "Less relevant"])
        self.assertEqual(results[0].cosine_distance, 0.1)

    def test_respects_top_k_and_propagates_source_metadata(self) -> None:
        database = FakeDatabase(
            [make_row("One", 0.1, "Relevant episode"), make_row("Two", 0.2), make_row("Three", 0.3)]
        )

        results = TranscriptRetrievalService(database, FakeEmbeddingService()).retrieve(
            "product strategy", top_k=2
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].document_title, "Relevant episode")
        self.assertEqual(results[0].source_url, "https://example.com/episode")
        self.assertEqual(results[0].document_metadata["guest"], "Ada")
        self.assertEqual(results[0].chunk_metadata["chunk_index"], 2)
        self.assertIn("ORDER BY", str(database.statement))

    def test_returns_empty_list_when_no_chunks_match(self) -> None:
        results = TranscriptRetrievalService(FakeDatabase([]), FakeEmbeddingService()).retrieve(
            "unrepresented topic", max_cosine_distance=0.1
        )

        self.assertEqual(results, [])

    def test_rejects_empty_query_and_invalid_embedding_dimension(self) -> None:
        service = TranscriptRetrievalService(FakeDatabase([]), FakeEmbeddingService())
        with self.assertRaises(ValueError):
            service.retrieve("   ")

        bad_embedding_service = FakeEmbeddingService([0.1] * (EMBEDDING_DIMENSION - 1))
        with self.assertRaises(EmbeddingDimensionError):
            TranscriptRetrievalService(FakeDatabase([]), bad_embedding_service).retrieve("growth")

    def test_wraps_database_errors(self) -> None:
        class FailingDatabase(FakeDatabase):
            def execute(self, statement):
                raise SQLAlchemyError("database unavailable")

        with self.assertRaises(RetrievalDatabaseError):
            TranscriptRetrievalService(FailingDatabase([]), FakeEmbeddingService()).retrieve("growth")
