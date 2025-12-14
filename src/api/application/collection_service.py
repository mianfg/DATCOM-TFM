"""Service for collection management."""

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.domain.models import Collection
from src.api.domain.protocols import FileStorageProtocol
from src.api.domain.schemas import CollectionCreate, CollectionResponse, CollectionUpdate
from src.api.infrastructure.repositories import CollectionRepository


class CollectionService:
    """Service for managing collections."""

    def __init__(self, file_storage: FileStorageProtocol):
        """
        Initialize service.

        Args:
            file_storage: File storage implementation (adheres to FileStorageProtocol)
        """
        self.file_storage = file_storage

    async def create_collection(self, session: AsyncSession, data: CollectionCreate) -> CollectionResponse:
        """Create a new collection."""
        repo = CollectionRepository(session)

        # Check if collection with this name already exists
        existing = await repo.get_by_name(data.name)
        if existing:
            raise ValueError(f"Collection with name '{data.name}' already exists")

        # Generate Qdrant collection name (lowercase, replace spaces with underscores)
        qdrant_name = f"collection_{data.name.lower().replace(' ', '_')}"

        # Create collection
        collection = await repo.create(
            name=data.name,
            description=data.description,
            qdrant_collection_name=qdrant_name,
        )

        logger.info(f"✓ Created collection: {collection.name} (ID: {collection.id})")

        return await self._to_response(repo, collection)

    async def get_collection(self, session: AsyncSession, collection_id: int) -> CollectionResponse:
        """Get a collection by ID."""
        repo = CollectionRepository(session)
        collection = await repo.get_by_id(collection_id)

        if not collection:
            raise ValueError(f"Collection with ID {collection_id} not found")

        return await self._to_response(repo, collection)

    async def list_collections(self, session: AsyncSession) -> list[CollectionResponse]:
        """List all collections."""
        repo = CollectionRepository(session)
        collections = await repo.list_all()
        return [await self._to_response(repo, c) for c in collections]

    async def update_collection(
        self, session: AsyncSession, collection_id: int, data: CollectionUpdate
    ) -> CollectionResponse:
        """Update a collection."""
        repo = CollectionRepository(session)
        collection = await repo.get_by_id(collection_id)

        if not collection:
            raise ValueError(f"Collection with ID {collection_id} not found")

        # Check name uniqueness if changing
        if data.name and data.name != collection.name:
            existing = await repo.get_by_name(data.name)
            if existing:
                raise ValueError(f"Collection with name '{data.name}' already exists")
            collection.name = data.name

        if data.description is not None:
            collection.description = data.description

        collection = await repo.update(collection)
        logger.info(f"✓ Updated collection: {collection.name} (ID: {collection.id})")

        return await self._to_response(repo, collection)

    async def delete_collection(self, session: AsyncSession, collection_id: int) -> None:
        """Delete a collection and all its files."""
        repo = CollectionRepository(session)
        collection = await repo.get_by_id(collection_id)

        if not collection:
            raise ValueError(f"Collection with ID {collection_id} not found")

        # Delete files from storage
        self.file_storage.delete_collection_files(collection_id)

        # Delete from database (cascades to documents, chunks, jobs, tasks)
        await repo.delete(collection)
        logger.info(f"✓ Deleted collection: {collection.name} (ID: {collection_id})")

    async def _to_response(self, repo: CollectionRepository, collection: Collection) -> CollectionResponse:
        """Convert collection to response schema."""
        document_count = await repo.get_document_count(collection.id)
        chunk_count = await repo.get_chunk_count(collection.id)

        return CollectionResponse(
            id=collection.id,
            name=collection.name,
            description=collection.description,
            qdrant_collection_name=collection.qdrant_collection_name,
            created_at=collection.created_at,
            updated_at=collection.updated_at,
            document_count=document_count,
            total_chunks=chunk_count,
        )
