"""Service for synchronizing documents to Qdrant."""

import hashlib
from pathlib import Path

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.domain.models import QdrantStatus
from src.api.domain.protocols import DocumentLoaderProtocol, FileStorageProtocol, VectorStoreProtocol
from src.api.infrastructure.repositories import (
    ChunkRepository,
    CollectionRepository,
    DocumentRepository,
)
from src.config import AppConfig


class QdrantSyncService:
    """Service for syncing documents to Qdrant."""

    def __init__(
        self,
        file_storage: FileStorageProtocol,
        document_loader: DocumentLoaderProtocol,
        vector_store_provider: VectorStoreProtocol,
    ):
        """
        Initialize service.

        Args:
            file_storage: File storage implementation (adheres to FileStorageProtocol)
            document_loader: Document loader from agent layer (adheres to DocumentLoaderProtocol)
            vector_store_provider: Vector store from agent layer (adheres to VectorStoreProtocol)
        """
        self.file_storage = file_storage
        self.document_loader = document_loader
        self.vector_store_provider = vector_store_provider
        
        # Load configuration
        self._config = AppConfig()
        self.chunk_preview_length = self._config.preview.chunk_preview_length

    async def sync_collection_to_qdrant(self, session: AsyncSession, collection_id: int) -> dict:
        """
        Sync all documents in a collection to Qdrant.

        Returns:
            dict with sync statistics
        """
        # Get collection
        collection_repo = CollectionRepository(session)
        collection = await collection_repo.get_by_id(collection_id)
        if not collection:
            raise ValueError(f"Collection with ID {collection_id} not found")

        # Get all documents that need syncing
        doc_repo = DocumentRepository(session)
        documents = await doc_repo.list_by_collection(collection_id)

        documents_to_sync = [
            d for d in documents if d.qdrant_status in [QdrantStatus.NOT_UPLOADED, QdrantStatus.FAILED]
        ]

        if not documents_to_sync:
            logger.info(f"All documents in collection {collection_id} are already synced to Qdrant")
            return {
                "total_documents": len(documents),
                "synced_documents": 0,
                "failed_documents": 0,
                "total_chunks": 0,
            }

        logger.info(f"Syncing {len(documents_to_sync)} documents to Qdrant...")

        synced_count = 0
        failed_count = 0
        total_chunks = 0

        chunk_repo = ChunkRepository(session)

        for document in documents_to_sync:
            try:
                # Update status to uploading
                await doc_repo.update_qdrant_status(document.id, QdrantStatus.UPLOADING)
                await session.commit()

                # Load and chunk document
                file_path = self.file_storage.get_file_path(document.file_path)
                chunks = await self._load_and_chunk_document(file_path, document)

                if not chunks:
                    logger.warning(f"No chunks generated for document {document.id}")
                    await doc_repo.update_qdrant_status(document.id, QdrantStatus.FAILED)
                    failed_count += 1
                    continue

                # Save chunks to database FIRST to get chunk IDs
                db_chunks = []
                for idx, chunk in enumerate(chunks):
                    content_preview = chunk.page_content[:self.chunk_preview_length]
                    content_hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()
                    chunk_size = len(chunk.page_content)

                    db_chunk = await chunk_repo.create(
                        document_id=document.id,
                        chunk_index=idx,
                        content_preview=content_preview,
                        content_hash=content_hash,
                        chunk_size=chunk_size,
                        qdrant_point_id=None,  # Will be updated after Qdrant upload
                    )
                    db_chunks.append(db_chunk)

                # Flush to get IDs before uploading to Qdrant
                await session.flush()

                # Now add chunk_id to metadata for Qdrant
                for idx, chunk in enumerate(chunks):
                    chunk.metadata["chunk_id"] = db_chunks[idx].id

                # Add chunks to Qdrant with complete metadata including chunk_id
                point_ids = await self._add_chunks_to_qdrant(collection.qdrant_collection_name, chunks)

                # Update chunks with Qdrant point IDs
                for idx, db_chunk in enumerate(db_chunks):
                    if idx < len(point_ids):
                        db_chunk.qdrant_point_id = point_ids[idx]

                # Update document status to uploaded
                await doc_repo.update_qdrant_status(document.id, QdrantStatus.UPLOADED)
                await session.commit()

                synced_count += 1
                total_chunks += len(chunks)
                logger.info(f"✓ Synced document {document.id} ({len(chunks)} chunks)")

            except Exception as e:
                logger.error(f"✗ Failed to sync document {document.id}: {e}")
                await doc_repo.update_qdrant_status(document.id, QdrantStatus.FAILED)
                failed_count += 1
                await session.commit()

        logger.info(
            f"✓ Qdrant sync complete: {synced_count} documents synced, "
            f"{failed_count} failed, {total_chunks} total chunks"
        )

        return {
            "total_documents": len(documents_to_sync),
            "synced_documents": synced_count,
            "failed_documents": failed_count,
            "total_chunks": total_chunks,
        }

    async def _load_and_chunk_document(self, file_path: Path, document) -> list:
        """Load and chunk a document using the agent loader."""
        # Use DocumentLoaderProtocol (agent layer primitive)
        chunks = self.document_loader.load_documents([file_path])

        # Add document metadata to chunks
        for chunk in chunks:
            chunk.metadata["document_id"] = document.id
            chunk.metadata["collection_id"] = document.collection_id
            chunk.metadata["filename"] = document.filename

        return chunks

    async def _add_chunks_to_qdrant(self, collection_name: str, chunks: list) -> list[str]:
        """
        Add chunks to Qdrant vector store.
        
        Returns:
            List of Qdrant point IDs for the added chunks
        """
        # Use VectorStoreProtocol (agent layer primitive)
        point_ids = self.vector_store_provider.add_documents(chunks, collection_name=collection_name)
        logger.debug(f"Added {len(chunks)} chunks to Qdrant collection '{collection_name}' with {len(point_ids)} point IDs")
        return point_ids
