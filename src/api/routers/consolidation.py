"""API routes for rule consolidation."""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.application.consolidation_service import ConsolidationService
from src.api.domain.schemas import (
    ConsolidationJobCreate,
    ConsolidationJobDetailResponse,
    ConsolidationJobResponse,
)
from src.api.infrastructure.database import get_db
from src.config import AppConfig

# Load configuration
_app_config = AppConfig()

router = APIRouter(prefix="/consolidation", tags=["consolidation"])


@router.post("/collections/{collection_id}/consolidate", response_model=ConsolidationJobResponse)
async def consolidate_collection_rules(
    collection_id: int,
    confidence_threshold: float = _app_config.consolidation.default_confidence,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    session: AsyncSession = Depends(get_db),
):
    """
    Consolidate all active rules in a collection.

    Creates a background job that will:
    1. Load all active rules from the collection
    2. Use LLM to identify redundant, mergeable, and simplifiable rules
    3. Verify consolidated rules
    4. Save consolidated rules and mark originals as superseded
    
    Args:
        collection_id: Collection to consolidate rules from
        confidence_threshold: Minimum confidence (0.0-1.0) to apply consolidation
        background_tasks: FastAPI background tasks
        session: Database session
    
    Returns:
        Consolidation job with status PENDING
    """
    consolidation_service = ConsolidationService(session)

    # Create consolidation job
    job_data = ConsolidationJobCreate(
        collection_id=collection_id, processing_job_id=None, confidence_threshold=confidence_threshold
    )

    try:
        job = await consolidation_service.create_consolidation_job(job_data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Queue background task
    from src.api.application.consolidation_executor import execute_consolidation_job

    background_tasks.add_task(execute_consolidation_job, job.id)

    return job


@router.post("/jobs/{job_id}/consolidate", response_model=ConsolidationJobResponse)
async def consolidate_job_rules(
    job_id: int,
    confidence_threshold: float = _app_config.consolidation.default_confidence,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    session: AsyncSession = Depends(get_db),
):
    """
    Consolidate rules from a specific processing job.

    Creates a background job that consolidates only the rules extracted
    from the specified processing job.
    
    Args:
        job_id: Processing job ID to consolidate rules from
        confidence_threshold: Minimum confidence (0.0-1.0) to apply consolidation
        background_tasks: FastAPI background tasks
        session: Database session
    
    Returns:
        Consolidation job with status PENDING
    """
    consolidation_service = ConsolidationService(session)

    # Create consolidation job
    job_data = ConsolidationJobCreate(
        collection_id=None, processing_job_id=job_id, confidence_threshold=confidence_threshold
    )

    try:
        job = await consolidation_service.create_consolidation_job(job_data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Queue background task
    from src.api.application.consolidation_executor import execute_consolidation_job

    background_tasks.add_task(execute_consolidation_job, job.id)

    return job


@router.get("/jobs/{consolidation_job_id}", response_model=ConsolidationJobDetailResponse)
async def get_consolidation_job_detail(
    consolidation_job_id: int,
    session: AsyncSession = Depends(get_db),
):
    """
    Get detailed information about a consolidation job.
    
    Includes statistics and consolidation summary.
    
    Args:
        consolidation_job_id: Consolidation job ID
        session: Database session
    
    Returns:
        Detailed consolidation job information
    """
    consolidation_service = ConsolidationService(session)

    job = await consolidation_service.get_consolidation_job(consolidation_job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Consolidation job {consolidation_job_id} not found")

    return ConsolidationJobDetailResponse.model_validate(job)


@router.get("/collections/{collection_id}/jobs", response_model=list[ConsolidationJobResponse])
async def list_collection_consolidation_jobs(
    collection_id: int,
    session: AsyncSession = Depends(get_db),
):
    """
    List all consolidation jobs for a collection.
    
    Args:
        collection_id: Collection ID
        session: Database session
    
    Returns:
        List of consolidation jobs
    """
    consolidation_service = ConsolidationService(session)

    jobs = await consolidation_service.list_consolidation_jobs(collection_id=collection_id)
    return jobs


@router.get("/processing-jobs/{job_id}/consolidations", response_model=list[ConsolidationJobResponse])
async def list_processing_job_consolidations(
    job_id: int,
    session: AsyncSession = Depends(get_db),
):
    """
    List all consolidation jobs for a processing job.
    
    Args:
        job_id: Processing job ID
        session: Database session
    
    Returns:
        List of consolidation jobs
    """
    consolidation_service = ConsolidationService(session)

    jobs = await consolidation_service.list_consolidation_jobs(processing_job_id=job_id)
    return jobs

