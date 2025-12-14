"""API router for collection management."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.domain.schemas import CollectionCreate, CollectionUpdate, CollectionResponse
from src.api.application.collection_service import CollectionService
from src.api.infrastructure.database import get_db_session
from src.api.infrastructure.container import get_container

router = APIRouter(prefix="/collections", tags=["collections"])


def get_collection_service() -> CollectionService:
    """Dependency to get collection service."""
    container = get_container()
    return CollectionService(file_storage=container.file_storage())


@router.post("", response_model=CollectionResponse, status_code=201)
async def create_collection(
    data: CollectionCreate,
    session: AsyncSession = Depends(get_db_session),
    service: CollectionService = Depends(get_collection_service),
):
    """Create a new collection."""
    try:
        return await service.create_collection(session, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("", response_model=list[CollectionResponse])
async def list_collections(
    session: AsyncSession = Depends(get_db_session),
    service: CollectionService = Depends(get_collection_service),
):
    """List all collections."""
    return await service.list_collections(session)


@router.get("/{collection_id}", response_model=CollectionResponse)
async def get_collection(
    collection_id: int,
    session: AsyncSession = Depends(get_db_session),
    service: CollectionService = Depends(get_collection_service),
):
    """Get a collection by ID."""
    try:
        return await service.get_collection(session, collection_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.patch("/{collection_id}", response_model=CollectionResponse)
async def update_collection(
    collection_id: int,
    data: CollectionUpdate,
    session: AsyncSession = Depends(get_db_session),
    service: CollectionService = Depends(get_collection_service),
):
    """Update a collection."""
    try:
        return await service.update_collection(session, collection_id, data)
    except ValueError as e:
        raise HTTPException(status_code=404 if "not found" in str(e).lower() else 400, detail=str(e))


@router.delete("/{collection_id}", status_code=204)
async def delete_collection(
    collection_id: int,
    session: AsyncSession = Depends(get_db_session),
    service: CollectionService = Depends(get_collection_service),
):
    """Delete a collection."""
    try:
        await service.delete_collection(session, collection_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

