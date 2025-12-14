"""Domain protocols for API layer to ensure interchangeability."""

from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable
from sqlalchemy.ext.asyncio import AsyncSession


@runtime_checkable
class FileStorageProtocol(Protocol):
    """Protocol for file storage implementations."""
    
    def save_file(self, file: BinaryIO, collection_id: int, filename: str) -> tuple[str, str, int]:
        """
        Save uploaded file to storage.
        
        Returns:
            Tuple of (file_path, content_hash, file_size)
        """
        ...
    
    def get_file_path(self, relative_path: str) -> Path:
        """Get absolute file path from relative path."""
        ...
    
    def delete_file(self, relative_path: str) -> None:
        """Delete a file from storage."""
        ...
    
    def delete_collection_files(self, collection_id: int) -> None:
        """Delete all files for a collection."""
        ...


@runtime_checkable
class DocumentLoaderProtocol(Protocol):
    """Protocol for document loaders (from agent layer)."""
    
    def load_documents(self, document_paths: list[str | Path]) -> list:
        """Load and chunk documents."""
        ...


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Protocol for vector stores (from agent layer)."""
    
    def add_documents(self, documents: list) -> None:
        """Add documents to the vector store."""
        ...
    
    def get_retriever(self, **kwargs):
        """Get a retriever for the vector store."""
        ...
    
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        ...

