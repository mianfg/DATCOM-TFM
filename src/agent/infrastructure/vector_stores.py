"""Vector store providers for Agent layer - moved from src/infrastructure/vector_stores/"""

import uuid
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_qdrant import QdrantVectorStore as LangChainQdrant
from qdrant_client import QdrantClient

from src.agent.domain.protocols import EmbeddingProvider


class QdrantVectorStore:
    """
    Qdrant-based vector store implementation.

    Supports dynamic collection management - you can work with multiple
    collections (projects) independently.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        collection_name: str = "plant_docs",
        url: str = "http://localhost:6333",
        api_key: str | None = None,
    ):
        self.embedding_provider = embedding_provider
        self.embeddings = embedding_provider.get_embeddings()
        self.default_collection_name = collection_name
        self.url = url
        self.api_key = api_key
        self.client = QdrantClient(url=url, api_key=api_key)
        # Cache of vector stores per collection
        self._vector_stores: dict[str, LangChainQdrant] = {}

    def add_documents(self, documents: list[Document], collection_name: str | None = None) -> list[str]:
        """
        Add documents to a specific Qdrant collection.

        Args:
            documents: Documents to add
            collection_name: Collection to add to (uses default if not specified)
            
        Returns:
            List of Qdrant point IDs (UUIDs) for the added documents
        """
        if not documents:
            return []

        collection = collection_name or self.default_collection_name
        
        # Generate UUIDs for each document
        # These will be used as Qdrant point IDs
        point_ids = [str(uuid.uuid4()) for _ in documents]

        if collection not in self._vector_stores:
            # Create new vector store with IDs
            self._vector_stores[collection] = LangChainQdrant.from_documents(
                documents=documents,
                embedding=self.embeddings,
                url=self.url,
                api_key=self.api_key,
                collection_name=collection,
                ids=point_ids,  # Pass our generated IDs
            )
        else:
            # Add to existing vector store with IDs
            self._vector_stores[collection].add_documents(
                documents=documents,
                ids=point_ids  # Pass our generated IDs
            )
        
        return point_ids

    def get_retriever(self, collection_name: str | None = None, **kwargs) -> VectorStoreRetriever:
        """
        Get a retriever for a specific collection.

        Args:
            collection_name: Collection to search (uses default if not specified)
            **kwargs: Additional arguments (e.g., search_kwargs={"k": 5})

        Returns:
            Retriever for the specified collection
        """
        collection = collection_name or self.default_collection_name

        if collection not in self._vector_stores:
            # Create empty collection if it doesn't exist
            empty_doc = Document(page_content="No documents available", metadata={"source": "empty"})
            self._vector_stores[collection] = LangChainQdrant.from_documents(
                documents=[empty_doc],
                embedding=self.embeddings,
                url=self.url,
                api_key=self.api_key,
                collection_name=collection,
            )

        search_kwargs = kwargs.get("search_kwargs", {"k": 5})
        return self._vector_stores[collection].as_retriever(search_kwargs=search_kwargs)

    def clear(self, collection_name: str | None = None) -> None:
        """
        Clear a specific collection.

        Args:
            collection_name: Collection to clear (clears default if not specified)
        """
        collection = collection_name or self.default_collection_name

        try:
            self.client.delete_collection(collection_name=collection)
            if collection in self._vector_stores:
                del self._vector_stores[collection]
        except Exception:
            pass

    def list_collections(self) -> list[str]:
        """List all Qdrant collections."""
        try:
            collections = self.client.get_collections()
            return [col.name for col in collections.collections]
        except Exception:
            return []
