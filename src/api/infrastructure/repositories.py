"""Repositories for data access."""

from typing import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.api.domain.dtos import CreateRuleDTO
from src.api.domain.models import (
    Chunk,
    Collection,
    ConsolidationJob,
    Document,
    ProcessingJob,
    ProcessingStatus,
    ProcessingTask,
    QdrantStatus,
    Rule,
    RuleContextChunk,
    RuleGroundingSearch,
    RuleLifecycleStatus,
    Sensor,
    SensorParsingStatus,
    TimeParsingStatus,
    VerificationStatus,
)


class CollectionRepository:
    """Repository for Collection entities."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, name: str, description: str | None, qdrant_collection_name: str) -> Collection:
        """Create a new collection."""
        collection = Collection(
            name=name,
            description=description,
            qdrant_collection_name=qdrant_collection_name,
        )
        self.session.add(collection)
        await self.session.flush()
        return collection

    async def get_by_id(self, collection_id: int) -> Collection | None:
        """Get collection by ID."""
        result = await self.session.execute(select(Collection).where(Collection.id == collection_id))
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> Collection | None:
        """Get collection by name."""
        result = await self.session.execute(select(Collection).where(Collection.name == name))
        return result.scalar_one_or_none()

    async def list_all(self) -> Sequence[Collection]:
        """List all collections."""
        result = await self.session.execute(select(Collection).order_by(Collection.created_at.desc()))
        return result.scalars().all()

    async def update(self, collection: Collection) -> Collection:
        """Update a collection."""
        await self.session.flush()
        return collection

    async def delete(self, collection: Collection) -> None:
        """Delete a collection."""
        await self.session.delete(collection)
        await self.session.flush()

    async def get_document_count(self, collection_id: int) -> int:
        """Get document count for a collection."""
        result = await self.session.execute(
            select(func.count(Document.id)).where(Document.collection_id == collection_id)
        )
        return result.scalar_one()

    async def get_chunk_count(self, collection_id: int) -> int:
        """Get total chunk count for a collection."""
        result = await self.session.execute(
            select(func.count(Chunk.id))
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.collection_id == collection_id)
        )
        return result.scalar_one()


class DocumentRepository:
    """Repository for Document entities."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        collection_id: int,
        filename: str,
        file_path: str,
        content_hash: str,
        file_size: int,
        mime_type: str,
        metadata: dict | None = None,
    ) -> Document:
        """Create a new document."""
        document = Document(
            collection_id=collection_id,
            filename=filename,
            file_path=file_path,
            content_hash=content_hash,
            file_size=file_size,
            mime_type=mime_type,
            metadata_=metadata,
            qdrant_status=QdrantStatus.NOT_UPLOADED,
        )
        self.session.add(document)
        await self.session.flush()
        return document

    async def get_by_id(self, document_id: int) -> Document | None:
        """Get document by ID."""
        result = await self.session.execute(select(Document).where(Document.id == document_id))
        return result.scalar_one_or_none()

    async def list_by_collection(self, collection_id: int) -> Sequence[Document]:
        """List documents in a collection."""
        result = await self.session.execute(
            select(Document).where(Document.collection_id == collection_id).order_by(Document.uploaded_at.desc())
        )
        return result.scalars().all()

    async def update_qdrant_status(self, document_id: int, status: QdrantStatus) -> None:
        """Update document's Qdrant status."""
        result = await self.session.execute(select(Document).where(Document.id == document_id))
        document = result.scalar_one_or_none()
        if document:
            document.qdrant_status = status
            await self.session.flush()

    async def delete(self, document: Document) -> None:
        """Delete a document."""
        await self.session.delete(document)
        await self.session.flush()

    async def get_chunk_count(self, document_id: int) -> int:
        """Get chunk count for a document."""
        result = await self.session.execute(select(func.count(Chunk.id)).where(Chunk.document_id == document_id))
        return result.scalar_one()


class ChunkRepository:
    """Repository for Chunk entities."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        document_id: int,
        chunk_index: int,
        content_preview: str,
        content_hash: str,
        chunk_size: int,
        qdrant_point_id: str | None = None,
    ) -> Chunk:
        """Create a new chunk."""
        chunk = Chunk(
            document_id=document_id,
            chunk_index=chunk_index,
            content_preview=content_preview,
            content_hash=content_hash,
            chunk_size=chunk_size,
            qdrant_point_id=qdrant_point_id,
        )
        self.session.add(chunk)
        await self.session.flush()
        return chunk

    async def get_by_id(self, chunk_id: int) -> Chunk | None:
        """Get chunk by ID."""
        result = await self.session.execute(
            select(Chunk)
            .options(selectinload(Chunk.document))  # Eagerly load document relationship
            .where(Chunk.id == chunk_id)
        )
        return result.scalar_one_or_none()

    async def list_by_document(self, document_id: int) -> Sequence[Chunk]:
        """List chunks in a document."""
        result = await self.session.execute(
            select(Chunk)
            .options(selectinload(Chunk.document))  # Eagerly load document relationship
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        )
        return result.scalars().all()

    async def list_by_collection(self, collection_id: int) -> Sequence[Chunk]:
        """List all chunks in a collection."""
        result = await self.session.execute(
            select(Chunk)
            .options(selectinload(Chunk.document))  # Eagerly load document relationship
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.collection_id == collection_id)
            .order_by(Document.id, Chunk.chunk_index)
        )
        return result.scalars().all()

    async def get_by_ids(self, chunk_ids: list[int]) -> Sequence[Chunk]:
        """Get specific chunks by their IDs."""
        result = await self.session.execute(
            select(Chunk)
            .options(selectinload(Chunk.document))  # Eagerly load document relationship
            .where(Chunk.id.in_(chunk_ids))
            .order_by(Chunk.chunk_index)
        )
        return result.scalars().all()

    async def get_by_documents(self, document_ids: list[int]) -> Sequence[Chunk]:
        """Get all chunks from specific documents."""
        result = await self.session.execute(
            select(Chunk)
            .options(selectinload(Chunk.document))  # Eagerly load document relationship
            .where(Chunk.document_id.in_(document_ids))
            .order_by(Chunk.document_id, Chunk.chunk_index)
        )
        return result.scalars().all()

    async def list_by_collection_with_data(self, collection_id: int) -> list[dict]:
        """
        List all chunks in a collection with data extracted (session-safe).

        Returns plain dicts that can be used outside the session context.
        """
        chunks = await self.list_by_collection(collection_id)

        # Extract all data within session context
        return [
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "document_filename": chunk.document.filename,
                "chunk_index": chunk.chunk_index,
                "content_preview": chunk.content_preview,
                "content_hash": chunk.content_hash,
                "qdrant_point_id": chunk.qdrant_point_id,
                "preview_length": len(chunk.content_preview),
                "created_at": chunk.created_at,
            }
            for chunk in chunks
        ]


class ProcessingJobRepository:
    """Repository for ProcessingJob entities."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, collection_id: int, use_grounding: bool = True) -> ProcessingJob:
        """Create a new processing job."""
        job = ProcessingJob(
            collection_id=collection_id,
            status=ProcessingStatus.PENDING,
            use_grounding=use_grounding,
            total_chunks=0,
            completed_chunks=0,
            failed_chunks=0,
        )
        self.session.add(job)
        await self.session.flush()
        return job

    async def get_by_id(self, job_id: int) -> ProcessingJob | None:
        """Get job by ID."""
        result = await self.session.execute(select(ProcessingJob).where(ProcessingJob.id == job_id))
        return result.scalar_one_or_none()

    async def list_by_collection(self, collection_id: int) -> Sequence[ProcessingJob]:
        """List jobs for a collection."""
        result = await self.session.execute(
            select(ProcessingJob)
            .where(ProcessingJob.collection_id == collection_id)
            .order_by(ProcessingJob.created_at.desc())
        )
        return result.scalars().all()

    async def update(self, job: ProcessingJob) -> ProcessingJob:
        """Update a job."""
        await self.session.flush()
        return job


class ProcessingTaskRepository:
    """Repository for ProcessingTask entities."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, job_id: int, chunk_id: int) -> ProcessingTask:
        """Create a new processing task."""
        task = ProcessingTask(
            job_id=job_id,
            chunk_id=chunk_id,
            status=ProcessingStatus.PENDING,
        )
        self.session.add(task)
        await self.session.flush()
        return task

    async def get_by_id(self, task_id: int) -> ProcessingTask | None:
        """Get task by ID with chunk relationship."""
        result = await self.session.execute(
            select(ProcessingTask).options(selectinload(ProcessingTask.chunk)).where(ProcessingTask.id == task_id)
        )
        return result.scalar_one_or_none()

    async def list_by_job(self, job_id: int) -> Sequence[ProcessingTask]:
        """List tasks for a job."""
        result = await self.session.execute(
            select(ProcessingTask)
            .options(selectinload(ProcessingTask.chunk))
            .where(ProcessingTask.job_id == job_id)
            .order_by(ProcessingTask.id)
        )
        return result.scalars().all()

    async def update(self, task: ProcessingTask) -> ProcessingTask:
        """Update a task."""
        await self.session.flush()
        return task


class RuleRepository:
    """Repository for rule database operations."""

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session."""
        self.session = session

    async def create(
        self,
        task_id: int,
        collection_id: int,
        document_id: int,
        chunk_id: int,
        rule_name: str,
        rule_description: str,
        rule_reasoning: str,
        rule_source: str,
        rule_body_original: str,
        rule_body: str,
        sensor_parsing_status: str | SensorParsingStatus = SensorParsingStatus.OK,
        time_parsing_status: str | TimeParsingStatus = TimeParsingStatus.OK,
        verification_status: str | VerificationStatus = VerificationStatus.OK,
        rule_type: str | None = None,
        confidence: float | None = None,
        source_chunk_preview: str | None = None,
        extraction_metadata: dict | None = None,
        # Consolidation fields
        is_consolidated: bool = False,
        lifecycle_status: str | RuleLifecycleStatus = RuleLifecycleStatus.EXTRACTED,
        consolidated_from_ids: list[int] | None = None,
        consolidation_confidence: float | None = None,
        consolidation_reasoning: str | None = None,
        consolidation_job_id: int | None = None,
    ) -> Rule:
        """Create a new Python rule."""
        # Convert strings to enums if needed
        if isinstance(sensor_parsing_status, str):
            sensor_parsing_status = SensorParsingStatus(sensor_parsing_status.lower())
        if isinstance(time_parsing_status, str):
            time_parsing_status = TimeParsingStatus(time_parsing_status.lower())
        if isinstance(verification_status, str):
            verification_status = VerificationStatus(verification_status.lower())
        if isinstance(lifecycle_status, str):
            lifecycle_status = RuleLifecycleStatus(lifecycle_status.lower())

        rule = Rule(
            task_id=task_id,
            collection_id=collection_id,
            document_id=document_id,
            chunk_id=chunk_id,
            rule_name=rule_name,
            rule_description=rule_description,
            rule_reasoning=rule_reasoning,
            rule_source=rule_source,
            rule_body_original=rule_body_original,
            rule_body=rule_body,
            sensor_parsing_status=sensor_parsing_status,
            time_parsing_status=time_parsing_status,
            verification_status=verification_status,
            rule_type=rule_type,
            confidence=confidence,
            source_chunk_preview=source_chunk_preview,
            extraction_metadata=extraction_metadata,
            # Consolidation fields
            is_consolidated=is_consolidated,
            lifecycle_status=lifecycle_status,
            consolidated_from_ids=consolidated_from_ids or [],
            consolidation_confidence=consolidation_confidence,
            consolidation_reasoning=consolidation_reasoning,
            consolidation_job_id=consolidation_job_id,
        )
        self.session.add(rule)
        await self.session.flush()
        return rule

    async def create_from_dto(self, dto: CreateRuleDTO) -> Rule:
        """
        Create a new rule from a DTO (Data Transfer Object).
        
        This is the preferred method for creating rules as it provides better
        type safety and is easier to maintain than the 20+ parameter method.
        
        Args:
            dto: CreateRuleDTO with all rule data
            
        Returns:
            Created rule entity
            
        Example:
            >>> dto = CreateRuleDTO(
            ...     task_id=1,
            ...     collection_id=1,
            ...     document_id=1,
            ...     chunk_id=1,
            ...     rule_name="high_temp_alert",
            ...     rule_description="Alert on high temperature",
            ...     rule_reasoning="Safety critical",
            ...     rule_source="Section 5.2",
            ...     rule_body_original="def high_temp_alert...",
            ...     rule_body="def high_temp_alert...",
            ... )
            >>> rule = await rule_repo.create_from_dto(dto)
        """
        # Convert DTO to dict, handling enum conversions
        rule_data = dto.model_dump()
        
        # Convert string enums to enum instances if needed
        if isinstance(rule_data["sensor_parsing_status"], str):
            rule_data["sensor_parsing_status"] = SensorParsingStatus(
                rule_data["sensor_parsing_status"].lower()
            )
        if isinstance(rule_data["time_parsing_status"], str):
            rule_data["time_parsing_status"] = TimeParsingStatus(
                rule_data["time_parsing_status"].lower()
            )
        if isinstance(rule_data["verification_status"], str):
            rule_data["verification_status"] = VerificationStatus(
                rule_data["verification_status"].lower()
            )
        if isinstance(rule_data["lifecycle_status"], str):
            rule_data["lifecycle_status"] = RuleLifecycleStatus(
                rule_data["lifecycle_status"].lower()
            )
        
        # Create rule from data
        rule = Rule(**rule_data)
        self.session.add(rule)
        await self.session.flush()
        return rule

    async def get_by_id(self, rule_id: int) -> Rule | None:
        """Get rule by ID."""
        result = await self.session.execute(select(Rule).where(Rule.id == rule_id))
        return result.scalar_one_or_none()

    async def list_by_collection(self, collection_id: int, limit: int = 100, offset: int = 0) -> list[Rule]:
        """List rules for a collection."""
        result = await self.session.execute(
            select(Rule)
            .where(Rule.collection_id == collection_id)
            .order_by(Rule.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def list_by_document(self, document_id: int) -> list[Rule]:
        """List rules for a document."""
        result = await self.session.execute(
            select(Rule).where(Rule.document_id == document_id).order_by(Rule.created_at.desc())
        )
        return list(result.scalars().all())

    async def list_by_task(self, task_id: int) -> list[Rule]:
        """List rules for a task."""
        result = await self.session.execute(
            select(Rule).where(Rule.task_id == task_id).order_by(Rule.created_at.desc())
        )
        return list(result.scalars().all())

    async def list_by_type(self, collection_id: int, rule_type: str) -> list[Rule]:
        """List rules by type for a collection."""
        result = await self.session.execute(
            select(Rule)
            .where(Rule.collection_id == collection_id, Rule.rule_type == rule_type)
            .order_by(Rule.created_at.desc())
        )
        return list(result.scalars().all())

    async def count_by_collection(self, collection_id: int) -> int:
        """Count rules in a collection."""
        result = await self.session.execute(
            select(func.count()).select_from(Rule).where(Rule.collection_id == collection_id)
        )
        return result.scalar() or 0

    async def get_stats_by_collection(self, collection_id: int) -> dict:
        """Get rule statistics for a collection."""
        # Total count
        total = await self.count_by_collection(collection_id)

        # Count by type
        type_result = await self.session.execute(
            select(Rule.rule_type, func.count()).where(Rule.collection_id == collection_id).group_by(Rule.rule_type)
        )
        rules_by_type = {row[0] or "unclassified": row[1] for row in type_result}

        # Count by document
        doc_result = await self.session.execute(
            select(Rule.document_id, func.count()).where(Rule.collection_id == collection_id).group_by(Rule.document_id)
        )
        rules_by_document = {str(row[0]): row[1] for row in doc_result}

        # Latest extraction
        latest_result = await self.session.execute(
            select(func.max(Rule.created_at)).where(Rule.collection_id == collection_id)
        )
        latest_extraction = latest_result.scalar()

        return {
            "total_rules": total,
            "rules_by_type": rules_by_type,
            "rules_by_document": rules_by_document,
            "latest_extraction": latest_extraction,
        }

    # Consolidation methods
    async def list_active_by_collection(self, collection_id: int) -> list[Rule]:
        """List active (non-superseded) rules for a collection."""
        result = await self.session.execute(
            select(Rule)
            .where(
                Rule.collection_id == collection_id,
                Rule.lifecycle_status != RuleLifecycleStatus.SUPERSEDED,
            )
            .order_by(Rule.created_at.desc())
        )
        return list(result.scalars().all())

    async def list_active_by_processing_job(self, job_id: int) -> list[Rule]:
        """List active rules from a specific processing job."""
        result = await self.session.execute(
            select(Rule)
            .join(ProcessingTask, Rule.task_id == ProcessingTask.id)
            .where(
                ProcessingTask.job_id == job_id,
                Rule.lifecycle_status != RuleLifecycleStatus.SUPERSEDED,
            )
            .order_by(Rule.created_at.desc())
        )
        return list(result.scalars().all())

    async def mark_as_superseded(self, rule_ids: list[int]) -> int:
        """Mark rules as superseded. Returns count of updated rules."""
        result = await self.session.execute(select(Rule).where(Rule.id.in_(rule_ids)))
        rules = result.scalars().all()

        for rule in rules:
            rule.lifecycle_status = RuleLifecycleStatus.SUPERSEDED

        await self.session.flush()
        return len(rules)

    async def get_by_ids(self, rule_ids: list[int]) -> list[Rule]:
        """Get multiple rules by their IDs."""
        result = await self.session.execute(select(Rule).where(Rule.id.in_(rule_ids)).order_by(Rule.id))
        return list(result.scalars().all())

    async def list_by_consolidation_job(self, consolidation_job_id: int) -> list[Rule]:
        """
        List rules created by a specific consolidation job.
        
        Used for idempotency checks to prevent duplicate consolidations.
        
        Args:
            consolidation_job_id: ID of the consolidation job
            
        Returns:
            List of rules with is_consolidated=True for this job
        """
        result = await self.session.execute(
            select(Rule)
            .where(
                Rule.consolidation_job_id == consolidation_job_id,
                Rule.is_consolidated == True,  # noqa: E712
            )
            .order_by(Rule.created_at)
        )
        return list(result.scalars().all())


class SensorRepository:
    """Repository for sensor database operations."""

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session."""
        self.session = session

    async def upsert(
        self,
        collection_id: int,
        sensor_id: str,
        name: str,
        description: str | None = None,
        unit: str | None = None,
        example: str | None = None,
    ) -> Sensor:
        """
        Upsert a sensor (update if exists, insert if new).

        Uses sensor_id as the unique identifier within a collection.
        """
        # Try to find existing sensor
        result = await self.session.execute(
            select(Sensor).where(Sensor.collection_id == collection_id, Sensor.sensor_id == sensor_id)
        )
        sensor = result.scalar_one_or_none()

        if sensor:
            # Update existing sensor
            sensor.name = name
            sensor.description = description
            sensor.unit = unit
            sensor.example = example
        else:
            # Create new sensor
            sensor = Sensor(
                collection_id=collection_id,
                sensor_id=sensor_id,
                name=name,
                description=description,
                unit=unit,
                example=example,
            )
            self.session.add(sensor)

        await self.session.flush()
        return sensor

    async def bulk_upsert(
        self,
        collection_id: int,
        sensors_data: list[dict],
    ) -> list[Sensor]:
        """
        Bulk upsert sensors.

        Args:
            collection_id: Collection ID
            sensors_data: List of dicts with sensor data
                          Each dict must have: sensor_id, name
                          Optional: description, unit, example

        Returns:
            List of upserted sensors
        """
        sensors = []
        for data in sensors_data:
            sensor = await self.upsert(
                collection_id=collection_id,
                sensor_id=data["sensor_id"],
                name=data["name"],
                description=data.get("description"),
                unit=data.get("unit"),
                example=data.get("example"),
            )
            sensors.append(sensor)

        return sensors

    async def get_by_id(self, sensor_id: int) -> Sensor | None:
        """Get sensor by database ID."""
        result = await self.session.execute(select(Sensor).where(Sensor.id == sensor_id))
        return result.scalar_one_or_none()

    async def get_by_sensor_id(self, collection_id: int, sensor_id: str) -> Sensor | None:
        """Get sensor by sensor_id within a collection."""
        result = await self.session.execute(
            select(Sensor).where(Sensor.collection_id == collection_id, Sensor.sensor_id == sensor_id)
        )
        return result.scalar_one_or_none()

    async def list_by_collection(self, collection_id: int) -> list[Sensor]:
        """List all sensors for a collection."""
        result = await self.session.execute(
            select(Sensor).where(Sensor.collection_id == collection_id).order_by(Sensor.sensor_id)
        )
        return list(result.scalars().all())

    async def delete(self, sensor: Sensor) -> None:
        """Delete a sensor."""
        await self.session.delete(sensor)
        await self.session.flush()

    async def delete_all_by_collection(self, collection_id: int) -> int:
        """Delete all sensors for a collection. Returns count of deleted sensors."""
        result = await self.session.execute(select(Sensor).where(Sensor.collection_id == collection_id))
        sensors = result.scalars().all()
        count = len(sensors)

        for sensor in sensors:
            await self.session.delete(sensor)

        await self.session.flush()
        return count

    async def count_by_collection(self, collection_id: int) -> int:
        """Count sensors in a collection."""
        result = await self.session.execute(
            select(func.count()).select_from(Sensor).where(Sensor.collection_id == collection_id)
        )
        return result.scalar() or 0


class RuleContextChunkRepository:
    """Repository for rule context chunk operations."""

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session."""
        self.session = session

    async def create(
        self, rule_id: int, chunk_id: int, relevance_score: float | None, rank: int
    ) -> RuleContextChunk:
        """Create a new rule context chunk entry."""
        context_chunk = RuleContextChunk(
            rule_id=rule_id, chunk_id=chunk_id, relevance_score=relevance_score, rank=rank
        )
        self.session.add(context_chunk)
        await self.session.flush()
        return context_chunk

    async def bulk_create(
        self, rule_id: int, chunks_data: list[dict]
    ) -> list[RuleContextChunk]:
        """Bulk create rule context chunks."""
        context_chunks = []
        for data in chunks_data:
            context_chunk = RuleContextChunk(
                rule_id=rule_id,
                chunk_id=data["chunk_id"],
                relevance_score=data.get("relevance_score"),
                rank=data["rank"],
            )
            self.session.add(context_chunk)
            context_chunks.append(context_chunk)

        await self.session.flush()
        return context_chunks

    async def list_by_rule(self, rule_id: int) -> list[RuleContextChunk]:
        """Get all context chunks for a rule."""
        result = await self.session.execute(
            select(RuleContextChunk)
            .where(RuleContextChunk.rule_id == rule_id)
            .order_by(RuleContextChunk.rank)
            .options(selectinload(RuleContextChunk.chunk))
        )
        return list(result.scalars().all())

    async def delete_by_rule(self, rule_id: int) -> int:
        """Delete all context chunks for a rule."""
        result = await self.session.execute(
            select(RuleContextChunk).where(RuleContextChunk.rule_id == rule_id)
        )
        chunks = result.scalars().all()
        count = len(chunks)

        for chunk in chunks:
            await self.session.delete(chunk)

        await self.session.flush()
        return count


class RuleGroundingSearchRepository:
    """Repository for rule grounding search operations."""

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session."""
        self.session = session

    async def create(
        self, rule_id: int, search_query: str, search_results: dict, search_rank: int
    ) -> RuleGroundingSearch:
        """Create a new rule grounding search entry."""
        grounding_search = RuleGroundingSearch(
            rule_id=rule_id,
            search_query=search_query,
            search_results=search_results,
            search_rank=search_rank,
        )
        self.session.add(grounding_search)
        await self.session.flush()
        return grounding_search

    async def bulk_create(
        self, rule_id: int, searches_data: list[dict]
    ) -> list[RuleGroundingSearch]:
        """Bulk create rule grounding searches."""
        searches = []
        for data in searches_data:
            search = RuleGroundingSearch(
                rule_id=rule_id,
                search_query=data["search_query"],
                search_results=data["search_results"],
                search_rank=data["search_rank"],
            )
            self.session.add(search)
            searches.append(search)

        await self.session.flush()
        return searches

    async def list_by_rule(self, rule_id: int) -> list[RuleGroundingSearch]:
        """Get all grounding searches for a rule."""
        result = await self.session.execute(
            select(RuleGroundingSearch)
            .where(RuleGroundingSearch.rule_id == rule_id)
            .order_by(RuleGroundingSearch.search_rank)
        )
        return list(result.scalars().all())

    async def delete_by_rule(self, rule_id: int) -> int:
        """Delete all grounding searches for a rule."""
        result = await self.session.execute(
            select(RuleGroundingSearch).where(RuleGroundingSearch.rule_id == rule_id)
        )
        searches = result.scalars().all()
        count = len(searches)

        for search in searches:
            await self.session.delete(search)

        await self.session.flush()
        return count


class ConsolidationJobRepository:
    """Repository for consolidation job operations."""

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session."""
        self.session = session

    async def create(
        self,
        collection_id: int | None = None,
        processing_job_id: int | None = None,
        confidence_threshold: float = 0.7,
    ) -> ConsolidationJob:
        """Create a new consolidation job."""
        job = ConsolidationJob(
            collection_id=collection_id,
            processing_job_id=processing_job_id,
            confidence_threshold=confidence_threshold,
            status=ProcessingStatus.PENDING,
        )
        self.session.add(job)
        await self.session.flush()
        return job

    async def get_by_id(self, job_id: int) -> ConsolidationJob | None:
        """Get consolidation job by ID."""
        result = await self.session.execute(select(ConsolidationJob).where(ConsolidationJob.id == job_id))
        return result.scalar_one_or_none()

    async def list_by_collection(self, collection_id: int) -> Sequence[ConsolidationJob]:
        """List consolidation jobs for a collection."""
        result = await self.session.execute(
            select(ConsolidationJob)
            .where(ConsolidationJob.collection_id == collection_id)
            .order_by(ConsolidationJob.created_at.desc())
        )
        return result.scalars().all()

    async def list_by_processing_job(self, processing_job_id: int) -> Sequence[ConsolidationJob]:
        """List consolidation jobs for a processing job."""
        result = await self.session.execute(
            select(ConsolidationJob)
            .where(ConsolidationJob.processing_job_id == processing_job_id)
            .order_by(ConsolidationJob.created_at.desc())
        )
        return result.scalars().all()

    async def update(self, job: ConsolidationJob) -> ConsolidationJob:
        """Update a consolidation job."""
        await self.session.flush()
        return job
