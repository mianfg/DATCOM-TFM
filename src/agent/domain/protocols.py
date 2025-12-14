"""Domain protocols for Agent layer - consolidated from src/domain/protocols/"""

from pathlib import Path
from typing import Protocol, runtime_checkable, Any
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.language_models import BaseChatModel


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers to ensure interchangeability."""
    
    def get_embeddings(self) -> Embeddings:
        """Return the embeddings model."""
        ...


@runtime_checkable
class VectorStoreProvider(Protocol):
    """Protocol for vector store providers to ensure interchangeability."""
    
    def add_documents(self, documents: list[Document], collection_name: str | None = None) -> list[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: Documents to add
            collection_name: Optional collection name for multi-collection support
            
        Returns:
            List of Qdrant point IDs (UUIDs) for the added documents
        """
        ...
    
    def get_retriever(self, collection_name: str | None = None, **kwargs) -> VectorStoreRetriever:
        """
        Get a retriever for the vector store.
        
        Args:
            collection_name: Optional collection name for multi-collection support
            **kwargs: Additional arguments (e.g., search_kwargs)
        """
        ...
    
    def clear(self, collection_name: str | None = None) -> None:
        """
        Clear all documents from the vector store.
        
        Args:
            collection_name: Optional collection name to clear (clears default if None)
        """
        ...


@runtime_checkable
class DocumentLoader(Protocol):
    """Protocol for document loaders to ensure interchangeability."""
    
    def load_documents(self, document_paths: list[str | Path]) -> list[Document]:
        """
        Load and chunk documents.
        
        Args:
            document_paths: List of paths to documents
            
        Returns:
            List of chunked documents ready for embedding
        """
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers to ensure interchangeability."""
    
    def get_llm(self) -> BaseChatModel:
        """Return the LLM model."""
        ...

