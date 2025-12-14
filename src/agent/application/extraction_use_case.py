"""Use case for rule extraction from documents."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.agent.application.extraction_workflow import RuleExtractionWorkflow
from src.agent.domain.protocols import DocumentLoader, LLMProvider, VectorStoreProvider


class RuleExtractionUseCase:
    """
    Provides low-level primitives for document processing.

    The API layer orchestrates these primitives to implement business logic
    (e.g., process all documents, process only new documents, etc.)
    """

    def __init__(
        self,
        document_loader: DocumentLoader,
        vector_store: VectorStoreProvider,
        llm_provider: LLMProvider,
        min_suggested_searches: int = 2,
        max_suggested_searches: int = 4,
        hard_limit_searches: int = 4,
    ):
        """
        Initialize use case with agent primitives.

        Args:
            document_loader: Document loading implementation
            vector_store: Vector store implementation
            llm_provider: LLM provider implementation
            min_suggested_searches: Minimum searches to suggest to LLM
            max_suggested_searches: Maximum searches to suggest to LLM
            hard_limit_searches: Hard limit on searches performed
        """
        self.document_loader = document_loader
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.min_suggested_searches = min_suggested_searches
        self.max_suggested_searches = max_suggested_searches
        self.hard_limit_searches = hard_limit_searches
        self._workflow: RuleExtractionWorkflow | None = None

    def load_and_chunk_documents(self, document_paths: list[str | Path]) -> list:
        """
        Load documents and split into chunks.

        This is a primitive operation - just loads and chunks,
        doesn't add to Qdrant or process.

        Args:
            document_paths: Paths to documents to load

        Returns:
            List of document chunks
        """
        logger.info(f"📄 Loading {len(document_paths)} documents...")
        chunks = self.document_loader.load_documents(document_paths)

        if not chunks:
            logger.warning("No chunks generated from documents")
            return []

        logger.info(f"✓ Generated {len(chunks)} chunks from {len(document_paths)} documents")
        return chunks

    def add_chunks_to_vector_store(self, chunks: list, collection_name: str | None = None) -> list[str]:
        """
        Add document chunks to a specific Qdrant collection.

        This embeds the chunks and stores them in the specified collection.
        Different projects can use different collections for isolation.

        Args:
            chunks: Document chunks to add
            collection_name: Qdrant collection name (uses default if not specified)

        Returns:
            List of Qdrant point IDs for the added chunks
        """
        if not chunks:
            logger.warning("No chunks to add to vector store")
            return []

        collection = collection_name or "default"
        logger.info(f"📊 Adding {len(chunks)} chunks to Qdrant collection '{collection}'...")
        point_ids = self.vector_store.add_documents(chunks, collection_name=collection_name)
        logger.info(f"✓ Successfully added {len(chunks)} chunks to collection '{collection}' with point IDs")
        return point_ids

    def get_workflow(self) -> RuleExtractionWorkflow:
        """
        Get the LangGraph workflow for processing a single chunk.

        The API layer uses this to process individual chunks,
        deciding which chunks to process based on business logic.

        Returns:
            Initialized RuleExtractionWorkflow
        """
        return self._ensure_workflow()

    def clear_vector_store(self, collection_name: str | None = None) -> None:
        """
        Clear all documents from a specific Qdrant collection.

        Utility method for development/testing.

        Args:
            collection_name: Collection to clear (clears default if not specified)
        """
        collection = collection_name or "default"
        logger.info(f"🗑️  Clearing Qdrant collection '{collection}'...")
        self.vector_store.clear(collection_name=collection_name)
        self._workflow = None
        logger.info(f"✓ Collection '{collection}' cleared")

    def list_collections(self) -> list[str]:
        """List all Qdrant collections."""
        return self.vector_store.list_collections()

    def _ensure_workflow(self) -> RuleExtractionWorkflow:
        """Ensure the workflow is initialized."""
        if self._workflow is None:
            self._workflow = RuleExtractionWorkflow(
                vector_store=self.vector_store,
                llm_provider=self.llm_provider,
                min_suggested_searches=self.min_suggested_searches,
                max_suggested_searches=self.max_suggested_searches,
                hard_limit_searches=self.hard_limit_searches,
            )
        return self._workflow
