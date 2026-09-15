from datetime import UTC, datetime
from pathlib import Path
import unittest
from uuid import UUID, uuid4

from backend.app.db.models import Chunk, Document
from backend.app.rag.ingestion import (
    EMBEDDING_DIMENSION,
    EmbeddingDimensionError,
    TranscriptIngestor,
    chunk_transcript,
    parse_transcript_file,
    transcript_metadata,
    validate_embedding_dimension,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "transcripts"
FIXTURE_PATH = FIXTURE_ROOT / "episodes" / "ada-lovelace" / "transcript.md"
MALFORMED_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "malformed_transcripts"


class FakeEmbeddingService:
    def embed(self, text: str) -> list[float]:
        return [0.25] * EMBEDDING_DIMENSION


class FakeDatabase:
    def __init__(self) -> None:
        self.documents_by_source_id: dict[str, Document] = {}
        self.chunks: list[Chunk] = []
        self.rollback_called = False

    def add(self, item: Document | Chunk) -> None:
        now = datetime.now(UTC)
        if isinstance(item, Document):
            item.id = uuid4()
            item.created_at = now
            self.documents_by_source_id[item.metadata_json["source_id"]] = item
        else:
            item.id = uuid4()
            item.created_at = now
            self.chunks.append(item)

    def flush(self) -> None:
        pass

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        self.rollback_called = True


class TestIngestor(TranscriptIngestor):
    def find_existing_document(self, source_id: str) -> Document | None:
        return self.db.documents_by_source_id.get(source_id)


class IngestionTests(unittest.TestCase):
    def test_parses_frontmatter_and_transcript_content(self) -> None:
        parsed = parse_transcript_file(FIXTURE_PATH)

        self.assertEqual(parsed.metadata["guest"], "Ada Lovelace")
        self.assertEqual(parsed.metadata["video_id"], "fixture123")
        self.assertTrue(parsed.content.startswith("Lenny: Welcome"))

    def test_preserves_document_metadata(self) -> None:
        parsed = parse_transcript_file(FIXTURE_PATH)
        title, source_url, metadata = transcript_metadata(
            parsed, Path("episodes/ada-lovelace/transcript.md")
        )

        self.assertEqual(title, "Designing resilient product systems")
        self.assertEqual(source_url, "https://www.youtube.com/watch?v=fixture123")
        self.assertEqual(metadata["publish_date"], "2026-09-01")
        self.assertEqual(metadata["keywords"], ["product strategy", "experimentation"])
        self.assertTrue(metadata["source_id"].startswith("lenny-podcast:"))

    def test_chunking_uses_overlap_and_source_offsets(self) -> None:
        content = ("Speaker A: First point.\n\nSpeaker B: Second point.\n\n" * 4)
        chunks = chunk_transcript(content, chunk_size=60, overlap=15)

        self.assertGreater(len(chunks), 1)
        self.assertLess(chunks[1].char_start, chunks[0].char_end)
        self.assertEqual(chunks[0].content, content[chunks[0].char_start : chunks[0].char_end])
        self.assertEqual(chunks[1].content, content[chunks[1].char_start : chunks[1].char_end])

    def test_embedding_dimension_validation(self) -> None:
        validate_embedding_dimension([0.0] * EMBEDDING_DIMENSION)
        with self.assertRaises(EmbeddingDimensionError):
            validate_embedding_dimension([0.0] * (EMBEDDING_DIMENSION - 1))

    def test_ingestion_is_idempotent(self) -> None:
        database = FakeDatabase()
        ingestor = TestIngestor(database, FakeEmbeddingService(), chunk_size=100, overlap=20)

        first = ingestor.ingest_directory(FIXTURE_ROOT)
        second = ingestor.ingest_directory(FIXTURE_ROOT)

        self.assertEqual(first.documents_created, 1)
        self.assertGreater(first.chunks_created, 0)
        self.assertEqual(second.documents_created, 0)
        self.assertEqual(second.documents_skipped, 1)
        self.assertEqual(len(database.documents_by_source_id), 1)
        self.assertEqual(len(database.chunks), first.chunks_created)

    def test_malformed_transcript_is_reported_without_creating_a_document(self) -> None:
        database = FakeDatabase()
        summary = TestIngestor(
            database, FakeEmbeddingService()
        ).ingest_directory(MALFORMED_FIXTURE_ROOT)

        self.assertEqual(summary.errors, 1)
        self.assertEqual(summary.documents_created, 0)
        self.assertEqual(database.documents_by_source_id, {})
