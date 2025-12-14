"""API router for processing management."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.application.processing_service import ProcessingService
from src.api.domain.schemas import (
    ProcessingJobCreate,
    ProcessingJobCreateSelective,
    ProcessingJobResponse,
    ProcessingTaskDetail,
)
from src.api.infrastructure.container import get_container
from src.api.infrastructure.database import get_db_session

router = APIRouter(prefix="/processing", tags=["processing"])


def get_processing_service() -> ProcessingService:
    """Dependency to get processing service."""
    container = get_container()
    return ProcessingService(file_storage=container.file_storage())


@router.post("/jobs", response_model=ProcessingJobResponse, status_code=201)
async def create_job(
    data: ProcessingJobCreate,
    session: AsyncSession = Depends(get_db_session),
    service: ProcessingService = Depends(get_processing_service),
):
    """Create a new processing job for a collection (processes ALL chunks)."""
    try:
        return await service.create_job(session, data.collection_id, use_grounding=data.use_grounding)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/jobs/selective", response_model=ProcessingJobResponse, status_code=201)
async def create_selective_job(
    data: ProcessingJobCreateSelective,
    session: AsyncSession = Depends(get_db_session),
    service: ProcessingService = Depends(get_processing_service),
):
    """
    Create a selective processing job for specific documents or chunks.

    Allows processing only a subset of documents/chunks rather than the entire collection.

    Use cases:
    - Process only newly uploaded documents: `document_ids=[new_doc_ids]`
    - Process specific chunks: `chunk_ids=[chunk_ids]`
    - Process all (same as /jobs): omit both document_ids and chunk_ids
    """
    try:
        return await service.create_selective_job(
            session,
            data.collection_id,
            document_ids=data.document_ids,
            chunk_ids=data.chunk_ids,
            use_grounding=data.use_grounding,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/jobs/{job_id}", response_model=ProcessingJobResponse)
async def get_job(
    job_id: int,
    session: AsyncSession = Depends(get_db_session),
    service: ProcessingService = Depends(get_processing_service),
):
    """Get a processing job by ID."""
    try:
        return await service.get_job(session, job_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/collections/{collection_id}/jobs", response_model=list[ProcessingJobResponse])
async def list_jobs(
    collection_id: int,
    session: AsyncSession = Depends(get_db_session),
    service: ProcessingService = Depends(get_processing_service),
):
    """List all jobs for a collection."""
    return await service.list_jobs(session, collection_id)


@router.get("/jobs/{job_id}/tasks", response_model=list[ProcessingTaskDetail])
async def get_job_tasks(
    job_id: int,
    session: AsyncSession = Depends(get_db_session),
    service: ProcessingService = Depends(get_processing_service),
):
    """Get all tasks for a job."""
    try:
        return await service.get_job_tasks(session, job_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/tasks/{task_id}", response_model=ProcessingTaskDetail)
async def get_task(
    task_id: int,
    session: AsyncSession = Depends(get_db_session),
    service: ProcessingService = Depends(get_processing_service),
):
    """Get a task by ID."""
    try:
        return await service.get_task(session, task_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/jobs/{job_id}/start", status_code=202)
async def start_job(
    job_id: int,
    session: AsyncSession = Depends(get_db_session),
):
    """Start processing a job (runs tasks in background)."""
    try:
        # Import here to avoid circular dependencies
        # Start job execution in background
        import asyncio

        from src.api.application.job_executor import execute_job

        asyncio.create_task(execute_job(job_id))

        return {"message": f"Job {job_id} started", "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
