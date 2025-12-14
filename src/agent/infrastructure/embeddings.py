"""Embedding providers for Agent layer - moved from src/infrastructure/embeddings/"""

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings


class OllamaEmbeddingProvider:
    """Ollama-based embedding provider (local, no API calls)."""

    def __init__(self, model: str = "mxbai-embed-large", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._embeddings = None

    def get_embeddings(self) -> Embeddings:
        if self._embeddings is None:
            self._embeddings = OllamaEmbeddings(model=self.model, base_url=self.base_url)
        return self._embeddings
