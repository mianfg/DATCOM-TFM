"""Service for processing management - integrates API with LLM layer."""

from datetime import datetime
from typing import Any

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.domain.models import ProcessingStatus
from src.api.domain.protocols import FileStorageProtocol
from src.api.domain.schemas import ProcessingJobResponse, ProcessingTaskDetail, ProcessingTaskResponse
from src.api.infrastructure.repositories import (
    ChunkRepository,
    CollectionRepository,
    ProcessingJobRepository,
    ProcessingTaskRepository,
)


class ProcessingService:
    """Service for managing processing jobs and tasks."""

    def __init__(self, file_storage: FileStorageProtocol):
        """
        Initialize service.

        Args:
            file_storage: File storage implementation (adheres to FileStorageProtocol)
        """
        self.file_storage = file_storage

    async def create_job(self, session: AsyncSession, collection_id: int, use_grounding: bool = True) -> ProcessingJobResponse:
        """Create a new processing job for a collection."""
        # Verify collection exists
        collection_repo = CollectionRepository(session)
        collection = await collection_repo.get_by_id(collection_id)
        if not collection:
            raise ValueError(f"Collection with ID {collection_id} not found")

        # Get all chunks for this collection
        chunk_repo = ChunkRepository(session)
        chunks = await chunk_repo.list_by_collection(collection_id)

        if not chunks:
            raise ValueError(f"No chunks found for collection {collection_id}. Upload and process documents first.")

        # Create job
        job_repo = ProcessingJobRepository(session)
        job = await job_repo.create(collection_id=collection_id, use_grounding=use_grounding)
        job.total_chunks = len(chunks)
        await job_repo.update(job)

        # Create tasks for each chunk
        task_repo = ProcessingTaskRepository(session)
        for chunk in chunks:
            await task_repo.create(job_id=job.id, chunk_id=chunk.id)

        await session.commit()

        logger.info(f"✓ Created processing job {job.id} with {len(chunks)} tasks for collection {collection_id}")

        return await self._job_to_response(job)

    async def create_selective_job(
        self,
        session: AsyncSession,
        collection_id: int,
        document_ids: list[int] | None = None,
        chunk_ids: list[int] | None = None,
        use_grounding: bool = True,
    ) -> ProcessingJobResponse:
        """
        Create a processing job for specific documents or chunks.

        This allows you to process only a subset of documents/chunks rather than
        the entire collection. Useful for:
        - Processing only newly uploaded documents
        - Reprocessing specific documents after corrections
        - Testing with a small subset

        Args:
            session: Database session
            collection_id: Collection ID
            document_ids: Optional list of specific document IDs to process
            chunk_ids: Optional list of specific chunk IDs to process
            use_grounding: Whether to use web grounding (default: True)

        Returns:
            Created job with selected tasks

        Raises:
            ValueError: If collection not found or no chunks match criteria
        """
        # Verify collection exists
        collection_repo = CollectionRepository(session)
        collection = await collection_repo.get_by_id(collection_id)

        if not collection:
            raise ValueError(f"Collection with ID {collection_id} not found")

        # Get chunks based on criteria
        chunk_repo = ChunkRepository(session)

        if chunk_ids:
            # Get specific chunks by IDs
            chunks = await chunk_repo.get_by_ids(chunk_ids)
            logger.info(f"Selected {len(chunks)} specific chunks")
        elif document_ids:
            # Get all chunks from specific documents
            chunks = await chunk_repo.get_by_documents(document_ids)
            logger.info(f"Selected {len(chunks)} chunks from {len(document_ids)} documents")
        else:
            # No filter - get all chunks (same as create_job)
            chunks = await chunk_repo.list_by_collection(collection_id)
            logger.info(f"Selected all {len(chunks)} chunks from collection")

        if not chunks:
            raise ValueError(
                f"No chunks found matching criteria for collection {collection_id}. "
                "Upload and process documents first, or check your filters."
            )

        # Create job
        job_repo = ProcessingJobRepository(session)
        job = await job_repo.create(collection_id=collection_id, use_grounding=use_grounding)
        job.total_chunks = len(chunks)
        await job_repo.update(job)

        # Create tasks for selected chunks only
        task_repo = ProcessingTaskRepository(session)
        for chunk in chunks:
            await task_repo.create(job_id=job.id, chunk_id=chunk.id)

        await session.commit()

        log_msg = f"✓ Created selective processing job {job.id} with {len(chunks)} tasks"
        if document_ids:
            log_msg += f" (from {len(document_ids)} documents)"
        elif chunk_ids:
            log_msg += f" (specific {len(chunk_ids)} chunks)"
        logger.info(log_msg)

        return await self._job_to_response(job)

    async def get_job(self, session: AsyncSession, job_id: int) -> ProcessingJobResponse:
        """Get a processing job by ID."""
        repo = ProcessingJobRepository(session)
        job = await repo.get_by_id(job_id)

        if not job:
            raise ValueError(f"Processing job with ID {job_id} not found")

        return await self._job_to_response(job)

    async def list_jobs(self, session: AsyncSession, collection_id: int) -> list[ProcessingJobResponse]:
        """List all jobs for a collection."""
        repo = ProcessingJobRepository(session)
        jobs = await repo.list_by_collection(collection_id)
        return [await self._job_to_response(job) for job in jobs]

    async def get_job_tasks(self, session: AsyncSession, job_id: int) -> list[ProcessingTaskDetail]:
        """Get all tasks for a job."""
        # Verify job exists
        job_repo = ProcessingJobRepository(session)
        job = await job_repo.get_by_id(job_id)
        if not job:
            raise ValueError(f"Processing job with ID {job_id} not found")

        task_repo = ProcessingTaskRepository(session)
        tasks = await task_repo.list_by_job(job_id)

        return [await self._task_to_detail(task) for task in tasks]

    async def get_task(self, session: AsyncSession, task_id: int) -> ProcessingTaskDetail:
        """Get a task by ID."""
        repo = ProcessingTaskRepository(session)
        task = await repo.get_by_id(task_id)

        if not task:
            raise ValueError(f"Processing task with ID {task_id} not found")

        return await self._task_to_detail(task)

    async def start_task(self, session: AsyncSession, task_id: int, langgraph_thread_id: str) -> ProcessingTaskResponse:
        """Mark a task as started."""
        repo = ProcessingTaskRepository(session)
        task = await repo.get_by_id(task_id)

        if not task:
            raise ValueError(f"Processing task with ID {task_id} not found")

        task.status = ProcessingStatus.RUNNING
        task.started_at = datetime.utcnow()
        task.langgraph_thread_id = langgraph_thread_id

        await repo.update(task)
        await session.commit()

        logger.info(f"✓ Started task {task_id} with thread {langgraph_thread_id}")

        return await self._task_to_response(task)

    async def complete_task(
        self,
        session: AsyncSession,
        task_id: int,
        result: dict[str, Any],
    ) -> ProcessingTaskResponse:
        """Mark a task as completed."""
        task_repo = ProcessingTaskRepository(session)
        task = await task_repo.get_by_id(task_id)

        if not task:
            raise ValueError(f"Processing task with ID {task_id} not found")

        task.status = ProcessingStatus.COMPLETED
        task.completed_at = datetime.utcnow()
        task.result = result

        await task_repo.update(task)

        # Update job progress
        await self._update_job_progress(session, task.job_id)

        await session.commit()

        logger.info(f"✓ Completed task {task_id}")

        return await self._task_to_response(task)

    async def fail_task(
        self,
        session: AsyncSession,
        task_id: int,
        error: str,
    ) -> ProcessingTaskResponse:
        """Mark a task as failed."""
        task_repo = ProcessingTaskRepository(session)
        task = await task_repo.get_by_id(task_id)

        if not task:
            raise ValueError(f"Processing task with ID {task_id} not found")

        task.status = ProcessingStatus.FAILED
        task.completed_at = datetime.utcnow()
        task.error = error

        await task_repo.update(task)

        # Update job progress
        await self._update_job_progress(session, task.job_id)

        await session.commit()

        logger.error(f"✗ Failed task {task_id}: {error}")

        return await self._task_to_response(task)

    async def _update_job_progress(self, session: AsyncSession, job_id: int) -> None:
        """Update job progress based on task statuses."""
        job_repo = ProcessingJobRepository(session)
        task_repo = ProcessingTaskRepository(session)

        job = await job_repo.get_by_id(job_id)
        if not job:
            return

        tasks = await task_repo.list_by_job(job_id)

        completed = sum(1 for t in tasks if t.status == ProcessingStatus.COMPLETED)
        failed = sum(1 for t in tasks if t.status == ProcessingStatus.FAILED)
        running = sum(1 for t in tasks if t.status == ProcessingStatus.RUNNING)

        job.completed_chunks = completed
        job.failed_chunks = failed

        # Update job status
        if job.status == ProcessingStatus.PENDING and running > 0:
            job.status = ProcessingStatus.RUNNING
            job.started_at = datetime.utcnow()

        if completed + failed == job.total_chunks:
            if failed > 0 and completed == 0:
                job.status = ProcessingStatus.FAILED
            else:
                job.status = ProcessingStatus.COMPLETED
            job.completed_at = datetime.utcnow()

        await job_repo.update(job)

    async def _job_to_response(self, job) -> ProcessingJobResponse:
        """Convert job to response schema."""
        progress = 0.0
        if job.total_chunks > 0:
            progress = ((job.completed_chunks + job.failed_chunks) / job.total_chunks) * 100

        return ProcessingJobResponse(
            id=job.id,
            collection_id=job.collection_id,
            status=job.status,
            use_grounding=job.use_grounding,
            total_chunks=job.total_chunks,
            completed_chunks=job.completed_chunks,
            failed_chunks=job.failed_chunks,
            error=job.error,
            started_at=job.started_at,
            completed_at=job.completed_at,
            created_at=job.created_at,
            progress_percentage=progress,
        )

    async def _task_to_response(self, task) -> ProcessingTaskResponse:
        """Convert task to response schema."""
        from src.api.domain.schemas import ProcessingTaskResponse

        return ProcessingTaskResponse(
            id=task.id,
            job_id=task.job_id,
            chunk_id=task.chunk_id,
            status=task.status,
            langgraph_thread_id=task.langgraph_thread_id,
            result=task.result,
            error=task.error,
            started_at=task.started_at,
            completed_at=task.completed_at,
            created_at=task.created_at,
        )

    async def _task_to_detail(self, task) -> ProcessingTaskDetail:
        """Convert task to detailed response schema."""
        from src.api.domain.schemas import ChunkResponse, ProcessingTaskDetail

        chunk_response = ChunkResponse(
            id=task.chunk.id,
            document_id=task.chunk.document_id,
            chunk_index=task.chunk.chunk_index,
            qdrant_point_id=task.chunk.qdrant_point_id,
            content_preview=task.chunk.content_preview,
            content_hash=task.chunk.content_hash,
            chunk_size=task.chunk.chunk_size,
            created_at=task.chunk.created_at,
        )

        return ProcessingTaskDetail(
            id=task.id,
            job_id=task.job_id,
            chunk_id=task.chunk_id,
            status=task.status,
            langgraph_thread_id=task.langgraph_thread_id,
            result=task.result,
            error=task.error,
            started_at=task.started_at,
            completed_at=task.completed_at,
            created_at=task.created_at,
            chunk=chunk_response,
        )
