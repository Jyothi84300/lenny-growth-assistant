from backend.app.db.database import SessionLocal
from backend.app.db.models import Document, Chunk
from backend.app.rag.embeddings import EmbeddingService


def main() -> None:
    db = SessionLocal()

    try:
        text = (
            "Product-market fit happens when a product solves a real problem "
            "for a specific group of users and those users consistently value it."
        )

        embedding_service = EmbeddingService()
        embedding = embedding_service.embed(text)

        document = Document(
            title="Test Product Growth Document",
            source_url="https://example.com/test",
            source_type="test",
            metadata_json={
                "purpose": "vector-storage-test",
            },
        )

        db.add(document)
        db.flush()

        chunk = Chunk(
            document_id=document.id,
            content=text,
            embedding=embedding,
            metadata_json={
                "chunk_index": 0,
            },
        )

        db.add(chunk)
        db.commit()

        print("Document ID:", document.id)
        print("Chunk ID:", chunk.id)
        print("Embedding dimensions:", len(embedding))
        print("Vector stored successfully.")

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    main()