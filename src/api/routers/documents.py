"""API router for document management."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.domain.schemas import DocumentResponse, DocumentPreview
from src.api.application.document_service import DocumentService
from src.api.application.qdrant_sync_service import QdrantSyncService
from src.api.infrastructure.database import get_db_session
from src.api.infrastructure.container import get_container

router = APIRouter(prefix="/collections/{collection_id}/documents", tags=["documents"])


def get_document_service() -> DocumentService:
    """Dependency to get document service."""
    container = get_container()
    return DocumentService(file_storage=container.file_storage())


def get_qdrant_sync_service() -> QdrantSyncService:
    """Dependency to get Qdrant sync service."""
    from src.llm._dependency_injection import LLMContainer
    from src.config import AppConfig
    
    # Initialize LLM container
    config = AppConfig()
    llm_container = LLMContainer()
    llm_container.config.from_pydantic(config)
    
    container = get_container()
    return QdrantSyncService(
        file_storage=container.file_storage(),
        document_loader=llm_container.document_loader(),
        vector_store_provider=llm_container.vector_store(),
    )


@router.post("", response_model=DocumentResponse, status_code=201)
async def upload_document(
    collection_id: int,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db_session),
    service: DocumentService = Depends(get_document_service),
):
    """Upload a document to a collection."""
    try:
        return await service.upload_document(
            session=session,
            collection_id=collection_id,
            filename=file.filename,
            file=file.file,
            mime_type=file.content_type or "application/octet-stream",
        )
    except ValueError as e:
        raise HTTPException(status_code=400 if "not found" not in str(e).lower() else 404, detail=str(e))


@router.get("", response_model=list[DocumentPreview])
async def list_documents(
    collection_id: int,
    session: AsyncSession = Depends(get_db_session),
    service: DocumentService = Depends(get_document_service),
):
    """List documents in a collection."""
    try:
        return await service.list_documents(session, collection_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    collection_id: int,
    document_id: int,
    session: AsyncSession = Depends(get_db_session),
    service: DocumentService = Depends(get_document_service),
):
    """Get a document by ID."""
    try:
        document = await service.get_document(session, document_id)
        # Verify it belongs to the collection
        if document.collection_id != collection_id:
            raise HTTPException(status_code=404, detail="Document not found in this collection")
        return document
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    collection_id: int,
    document_id: int,
    session: AsyncSession = Depends(get_db_session),
    service: DocumentService = Depends(get_document_service),
):
    """Delete a document."""
    try:
        document = await service.get_document(session, document_id)
        # Verify it belongs to the collection
        if document.collection_id != collection_id:
            raise HTTPException(status_code=404, detail="Document not found in this collection")
        await service.delete_document(session, document_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/sync-to-qdrant", status_code=200)
async def sync_to_qdrant(
    collection_id: int,
    session: AsyncSession = Depends(get_db_session),
    service: QdrantSyncService = Depends(get_qdrant_sync_service),
):
    """Sync all documents in the collection to Qdrant."""
    try:
        result = await service.sync_collection_to_qdrant(session, collection_id)
        return {
            "message": "Documents synced to Qdrant",
            "stats": result,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

