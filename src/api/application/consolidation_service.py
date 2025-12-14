"""Service for rule consolidation operations."""

from datetime import datetime

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.domain.models import ProcessingStatus
from src.api.domain.schemas import ConsolidationJobCreate, ConsolidationJobResponse
from src.api.infrastructure.repositories import (
    CollectionRepository,
    ConsolidationJobRepository,
    ProcessingJobRepository,
    RuleRepository,
    SensorRepository,
)


class ConsolidationService:
    """Service for managing rule consolidation."""

    def __init__(self, session: AsyncSession):
        """Initialize service with database session."""
        self.session = session
        self.consolidation_job_repo = ConsolidationJobRepository(session)
        self.rule_repo = RuleRepository(session)
        self.sensor_repo = SensorRepository(session)
        self.collection_repo = CollectionRepository(session)
        self.processing_job_repo = ProcessingJobRepository(session)

    async def create_consolidation_job(self, data: ConsolidationJobCreate) -> ConsolidationJobResponse:
        """
        Create a new consolidation job.

        Args:
            data: Consolidation job creation data

        Returns:
            Created consolidation job response

        Raises:
            ValueError: If neither collection_id nor processing_job_id provided
            ValueError: If specified collection/job doesn't exist
        """
        # Validate input
        if not data.collection_id and not data.processing_job_id:
            raise ValueError("Either collection_id or processing_job_id must be provided")

        # Verify collection/job exists
        if data.collection_id:
            collection = await self.collection_repo.get_by_id(data.collection_id)
            if not collection:
                raise ValueError(f"Collection {data.collection_id} not found")

        if data.processing_job_id:
            processing_job = await self.processing_job_repo.get_by_id(data.processing_job_id)
            if not processing_job:
                raise ValueError(f"Processing job {data.processing_job_id} not found")

        # Create job
        job = await self.consolidation_job_repo.create(
            collection_id=data.collection_id,
            processing_job_id=data.processing_job_id,
            confidence_threshold=data.confidence_threshold,
        )

        await self.session.commit()

        logger.info(
            f"Created consolidation job {job.id} for "
            f"collection={data.collection_id}, job={data.processing_job_id}"
        )

        return ConsolidationJobResponse.model_validate(job)

    async def get_consolidation_job(self, job_id: int) -> ConsolidationJobResponse | None:
        """Get consolidation job by ID."""
        job = await self.consolidation_job_repo.get_by_id(job_id)
        if not job:
            return None

        return ConsolidationJobResponse.model_validate(job)

    async def list_consolidation_jobs(
        self, collection_id: int | None = None, processing_job_id: int | None = None
    ) -> list[ConsolidationJobResponse]:
        """
        List consolidation jobs.

        Args:
            collection_id: Filter by collection
            processing_job_id: Filter by processing job

        Returns:
            List of consolidation jobs
        """
        if collection_id:
            jobs = await self.consolidation_job_repo.list_by_collection(collection_id)
        elif processing_job_id:
            jobs = await self.consolidation_job_repo.list_by_processing_job(processing_job_id)
        else:
            # Could add list_all method to repository if needed
            raise ValueError("Either collection_id or processing_job_id must be provided")

        return [ConsolidationJobResponse.model_validate(job) for job in jobs]

    async def update_job_status(
        self,
        job_id: int,
        status: ProcessingStatus,
        error: str | None = None,
    ) -> None:
        """Update consolidation job status."""
        job = await self.consolidation_job_repo.get_by_id(job_id)
        if not job:
            raise ValueError(f"Consolidation job {job_id} not found")

        job.status = status
        if error:
            job.error = error

        if status == ProcessingStatus.RUNNING and not job.started_at:
            job.started_at = datetime.utcnow()
        elif status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
            job.completed_at = datetime.utcnow()

        await self.consolidation_job_repo.update(job)
        await self.session.commit()

    async def finalize_consolidation_job(
        self,
        job_id: int,
        input_count: int,
        output_count: int,
        superseded_count: int,
        remove_count: int,
        merge_count: int,
        simplify_count: int,
        summary: dict,
    ) -> None:
        """
        Finalize consolidation job with statistics.

        Args:
            job_id: Consolidation job ID
            input_count: Number of input rules
            output_count: Number of output rules
            superseded_count: Number of superseded rules
            remove_count: Number of removed rules
            merge_count: Number of merged rules
            simplify_count: Number of simplified rules
            summary: Detailed consolidation summary
        """
        job = await self.consolidation_job_repo.get_by_id(job_id)
        if not job:
            raise ValueError(f"Consolidation job {job_id} not found")

        job.input_rules_count = input_count
        job.output_rules_count = output_count
        job.rules_removed = remove_count
        job.rules_merged = merge_count
        job.rules_simplified = simplify_count
        job.consolidation_summary = summary
        job.status = ProcessingStatus.COMPLETED
        job.completed_at = datetime.utcnow()

        await self.consolidation_job_repo.update(job)
        await self.session.commit()

        logger.info(
            f"Finalized consolidation job {job_id}: "
            f"{input_count} → {output_count} rules ({superseded_count} superseded)"
        )

