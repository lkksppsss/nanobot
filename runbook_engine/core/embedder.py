"""Embedder Protocol 定義向量化介面。

OllamaEmbedder / SentenceTransformerEmbedder 為 bundled 方便實作，依需求選用。
你也可以自行實作 Embedder Protocol，只需提供 async embed(text) -> list[float] 即可。

RunbookEngine 的 embedder 參數預設為 None（不做向量搜尋），
若需要 route() 向量搜尋功能，需在建立 engine 時明確傳入：
    engine = RunbookEngine(..., embedder=OllamaEmbedder())
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    """Calls Ollama /api/embeddings. No extra Python dependencies."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model

    async def embed(self, text: str) -> list[float]:
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": text},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]


class SentenceTransformerEmbedder:
    """Uses sentence_transformers locally. Model loaded once and cached."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    async def embed(self, text: str) -> list[float]:
        return self._get_model().encode(text, normalize_embeddings=True).tolist()
