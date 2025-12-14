"""Service for document management."""

from typing import BinaryIO

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.domain.models import QdrantStatus
from src.api.domain.protocols import FileStorageProtocol
from src.api.domain.schemas import DocumentPreview, DocumentResponse
from src.api.infrastructure.repositories import CollectionRepository, DocumentRepository


class DocumentService:
    """Service for managing documents."""

    def __init__(self, file_storage: FileStorageProtocol):
        """
        Initialize service.

        Args:
            file_storage: File storage implementation (adheres to FileStorageProtocol)
        """
        self.file_storage = file_storage

    async def upload_document(
        self,
        session: AsyncSession,
        collection_id: int,
        filename: str,
        file: BinaryIO,
        mime_type: str,
        metadata: dict | None = None,
    ) -> DocumentResponse:
        """Upload a document to a collection."""
        # Verify collection exists
        collection_repo = CollectionRepository(session)
        collection = await collection_repo.get_by_id(collection_id)
        if not collection:
            raise ValueError(f"Collection with ID {collection_id} not found")

        # Save file to storage
        file_path, content_hash, file_size = self.file_storage.save_file(file, collection_id, filename)

        # Create document record
        doc_repo = DocumentRepository(session)
        document = await doc_repo.create(
            collection_id=collection_id,
            filename=filename,
            file_path=file_path,
            content_hash=content_hash,
            file_size=file_size,
            mime_type=mime_type,
            metadata=metadata,
        )

        logger.info(f"✓ Uploaded document: {filename} (ID: {document.id})")

        return await self._to_response(doc_repo, document)

    async def get_document(self, session: AsyncSession, document_id: int) -> DocumentResponse:
        """Get a document by ID."""
        repo = DocumentRepository(session)
        document = await repo.get_by_id(document_id)

        if not document:
            raise ValueError(f"Document with ID {document_id} not found")

        return await self._to_response(repo, document)

    async def list_documents(self, session: AsyncSession, collection_id: int) -> list[DocumentPreview]:
        """List documents in a collection."""
        # Verify collection exists
        collection_repo = CollectionRepository(session)
        collection = await collection_repo.get_by_id(collection_id)
        if not collection:
            raise ValueError(f"Collection with ID {collection_id} not found")

        repo = DocumentRepository(session)
        documents = await repo.list_by_collection(collection_id)

        result = []
        for doc in documents:
            chunk_count = await repo.get_chunk_count(doc.id)
            result.append(
                DocumentPreview(
                    id=doc.id,
                    filename=doc.filename,
                    mime_type=doc.mime_type,
                    file_size=doc.file_size,
                    qdrant_status=doc.qdrant_status,
                    uploaded_at=doc.uploaded_at,
                    chunk_count=chunk_count,
                )
            )

        return result

    async def delete_document(self, session: AsyncSession, document_id: int) -> None:
        """Delete a document."""
        repo = DocumentRepository(session)
        document = await repo.get_by_id(document_id)

        if not document:
            raise ValueError(f"Document with ID {document_id} not found")

        # Delete file from storage
        self.file_storage.delete_file(document.file_path)

        # Delete from database (cascades to chunks and tasks)
        await repo.delete(document)
        logger.info(f"✓ Deleted document: {document.filename} (ID: {document_id})")

    async def update_qdrant_status(self, session: AsyncSession, document_id: int, status: QdrantStatus) -> None:
        """Update document's Qdrant upload status."""
        repo = DocumentRepository(session)
        await repo.update_qdrant_status(document_id, status)
        logger.debug(f"Updated Qdrant status for document {document_id}: {status}")

    async def _to_response(self, repo: DocumentRepository, document) -> DocumentResponse:
        """Convert document to response schema."""
        chunk_count = await repo.get_chunk_count(document.id)

        return DocumentResponse(
            id=document.id,
            collection_id=document.collection_id,
            filename=document.filename,
            file_path=document.file_path,
            content_hash=document.content_hash,
            file_size=document.file_size,
            mime_type=document.mime_type,
            qdrant_status=document.qdrant_status,
            metadata=document.metadata_,
            uploaded_at=document.uploaded_at,
            chunk_count=chunk_count,
        )
