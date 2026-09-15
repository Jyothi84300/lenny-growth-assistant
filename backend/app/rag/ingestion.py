"""Local ingestion for ChatPRD-style Lenny's Podcast transcript directories."""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.database import SessionLocal
from backend.app.db.models import Chunk, Document
from backend.app.rag.embeddings import EmbeddingService


LOGGER = logging.getLogger(__name__)
SOURCE_TYPE = "lenny_podcast_transcript"
EMBEDDING_DIMENSION = 1024
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 250
FRONTMATTER_PATTERN = re.compile(r"\A\ufeff?---\r?\n(.*?)\r?\n---(?:\r?\n|\Z)", re.DOTALL)


class TranscriptParseError(ValueError):
    pass


class EmbeddingDimensionError(ValueError):
    pass


class EmbeddingGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ParsedTranscript:
    metadata: dict[str, Any]
    content: str


@dataclass(frozen=True)
class TranscriptChunk:
    content: str
    char_start: int
    char_end: int


@dataclass
class IngestionSummary:
    transcripts_discovered: int = 0
    documents_created: int = 0
    documents_skipped: int = 0
    chunks_created: int = 0
    embedding_failures: int = 0
    errors: int = 0


def parse_transcript_file(path: Path) -> ParsedTranscript:
    """Read a transcript markdown file with required YAML frontmatter."""
    raw_content = path.read_text(encoding="utf-8")
    match = FRONTMATTER_PATTERN.match(raw_content)
    if match is None:
        raise TranscriptParseError(f"{path}: expected YAML frontmatter delimited by '---'.")

    try:
        metadata = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise TranscriptParseError(f"{path}: invalid YAML frontmatter: {exc}") from exc

    if not isinstance(metadata, dict):
        raise TranscriptParseError(f"{path}: YAML frontmatter must be a mapping.")

    content = raw_content[match.end() :]
    if not content.strip():
        raise TranscriptParseError(f"{path}: transcript content is empty.")

    return ParsedTranscript(metadata=metadata, content=content)


def discover_transcript_files(source_dir: Path) -> list[Path]:
    episodes_dir = source_dir / "episodes"
    if not episodes_dir.is_dir():
        raise ValueError(f"Expected an episodes directory at: {episodes_dir}")
    return sorted(episodes_dir.glob("*/transcript.md"))


def _best_end_boundary(content: str, start: int, hard_end: int) -> int:
    for pattern in (r"\n[ \t]*\n", r"\n", r"[.!?][\"')\]]?\s+", r"\s+"):
        matches = list(re.finditer(pattern, content[start:hard_end]))
        if matches:
            return start + matches[-1].end()
    return hard_end


def _best_next_start(content: str, current_start: int, end: int, overlap: int) -> int:
    target = end - overlap
    if target <= current_start:
        return end

    for pattern in (r"\n[ \t]*\n", r"\n", r"\s+"):
        matches = list(re.finditer(pattern, content[current_start:target]))
        if matches:
            return current_start + matches[-1].end()
    return target


def chunk_transcript(
    content: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TranscriptChunk]:
    """Chunk text on natural transcript boundaries, retaining source character offsets."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least zero and smaller than chunk_size.")

    chunks: list[TranscriptChunk] = []
    start = 0
    content_length = len(content)
    while start < content_length:
        hard_end = min(start + chunk_size, content_length)
        end = content_length if hard_end == content_length else _best_end_boundary(content, start, hard_end)
        if end <= start:
            end = hard_end

        chunks.append(
            TranscriptChunk(
                content=content[start:end],
                char_start=start,
                char_end=end,
            )
        )
        if end == content_length:
            break
        start = _best_next_start(content, start, end, overlap)

    return chunks


def validate_embedding_dimension(embedding: list[float]) -> None:
    if len(embedding) != EMBEDDING_DIMENSION:
        raise EmbeddingDimensionError(
            f"Expected a {EMBEDDING_DIMENSION}-dimension embedding, got {len(embedding)}."
        )


def _json_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def transcript_metadata(
    parsed: ParsedTranscript, relative_path: Path
) -> tuple[str, str | None, dict[str, Any]]:
    metadata_keys = (
        "guest",
        "title",
        "youtube_url",
        "video_id",
        "publish_date",
        "description",
        "duration",
        "duration_seconds",
        "channel",
        "keywords",
    )
    metadata = {
        key: _json_value(parsed.metadata[key])
        for key in metadata_keys
        if key in parsed.metadata and parsed.metadata[key] is not None
    }
    source_path = relative_path.as_posix()
    source_identity = str(metadata.get("video_id") or source_path)
    source_id = "lenny-podcast:" + hashlib.sha256(source_identity.encode("utf-8")).hexdigest()
    metadata.update({"source_id": source_id, "source_path": source_path})

    title = str(metadata.get("title") or metadata.get("guest") or relative_path.parent.name)
    source_url = metadata.get("youtube_url")
    return title, str(source_url) if source_url else None, metadata


class TranscriptIngestor:
    def __init__(
        self,
        db: Session,
        embedding_service: EmbeddingService,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.db = db
        self.embedding_service = embedding_service
        self.chunk_size = chunk_size
        self.overlap = overlap

    def find_existing_document(self, source_id: str) -> Document | None:
        statement = select(Document).where(
            Document.metadata_json["source_id"].astext == source_id
        )
        return self.db.scalars(statement).first()

    def ingest_file(self, transcript_path: Path, source_dir: Path) -> int | None:
        parsed = parse_transcript_file(transcript_path)
        relative_path = transcript_path.relative_to(source_dir)
        title, source_url, metadata = transcript_metadata(parsed, relative_path)

        if self.find_existing_document(metadata["source_id"]) is not None:
            LOGGER.info("Skipping already ingested transcript: %s", relative_path)
            return None

        chunks = chunk_transcript(parsed.content, self.chunk_size, self.overlap)
        try:
            document = Document(
                title=title,
                source_url=source_url,
                source_type=SOURCE_TYPE,
                metadata_json=metadata,
            )
            self.db.add(document)
            self.db.flush()

            for index, chunk in enumerate(chunks):
                try:
                    embedding = self.embedding_service.embed(chunk.content)
                    validate_embedding_dimension(embedding)
                except EmbeddingDimensionError:
                    raise
                except Exception as exc:
                    raise EmbeddingGenerationError(
                        f"Could not generate embedding for chunk {index}: {exc}"
                    ) from exc
                self.db.add(
                    Chunk(
                        document_id=document.id,
                        content=chunk.content,
                        embedding=embedding,
                        metadata_json={
                            "chunk_index": index,
                            "char_start": chunk.char_start,
                            "char_end": chunk.char_end,
                            "source_path": metadata["source_path"],
                        },
                    )
                )

            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

        LOGGER.info("Ingested %s with %d chunks.", relative_path, len(chunks))
        return len(chunks)

    def ingest_directory(self, source_dir: Path) -> IngestionSummary:
        transcript_files = discover_transcript_files(source_dir)
        summary = IngestionSummary(transcripts_discovered=len(transcript_files))
        LOGGER.info("Discovered %d transcript(s).", summary.transcripts_discovered)

        for transcript_path in transcript_files:
            try:
                chunk_count = self.ingest_file(transcript_path, source_dir)
            except (EmbeddingDimensionError, EmbeddingGenerationError) as exc:
                summary.embedding_failures += 1
                summary.errors += 1
                LOGGER.error("Embedding failed for %s: %s", transcript_path, exc)
            except Exception as exc:
                summary.errors += 1
                LOGGER.error("Could not ingest %s: %s", transcript_path, exc)
            else:
                if chunk_count is None:
                    summary.documents_skipped += 1
                else:
                    summary.documents_created += 1
                    summary.chunks_created += chunk_count

        LOGGER.info(
            "Ingestion complete: discovered=%d created=%d skipped=%d chunks=%d "
            "embedding_failures=%d errors=%d",
            summary.transcripts_discovered,
            summary.documents_created,
            summary.documents_skipped,
            summary.chunks_created,
            summary.embedding_failures,
            summary.errors,
        )
        return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest local Lenny Podcast transcripts.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    db = SessionLocal()
    try:
        summary = TranscriptIngestor(
            db,
            EmbeddingService(),
            chunk_size=arguments.chunk_size,
            overlap=arguments.overlap,
        ).ingest_directory(arguments.source_dir)
    finally:
        db.close()

    return 1 if summary.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
