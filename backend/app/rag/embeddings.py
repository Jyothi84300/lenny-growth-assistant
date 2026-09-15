import httpx


OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"


class EmbeddingService:
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = EMBEDDING_MODEL,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Cannot create an embedding for empty text.")

        response = httpx.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": text,
            },
            timeout=120.0,
        )

        response.raise_for_status()

        data = response.json()
        embeddings = data.get("embeddings")

        if not embeddings or not embeddings[0]:
            raise RuntimeError("Ollama returned an empty embedding.")

        return embeddings[0]