"""Job executor for processing tasks with LangGraph integration."""

import asyncio
import uuid

from langfuse import propagate_attributes
from loguru import logger

from src.api.domain.dtos import CreateRuleDTO
from src.api.domain.models import ProcessingStatus, SensorParsingStatus, TimeParsingStatus, VerificationStatus
from src.api.infrastructure.container import get_container
from src.api.infrastructure.database import Database
from src.api.infrastructure.langfuse_client import get_langfuse_manager
from src.api.infrastructure.repositories import (
    ChunkRepository,
    CollectionRepository,
    DocumentRepository,
    ProcessingJobRepository,
    ProcessingTaskRepository,
    RuleContextChunkRepository,
    RuleGroundingSearchRepository,
    RuleRepository,
    SensorRepository,
)
from src.api.infrastructure.storage import FileStorage


async def execute_job(job_id: int):
    """
    Execute a processing job by running all its tasks in parallel.

    This function:
    1. Gets all tasks for the job
    2. Processes tasks in parallel (with concurrency limit)
    3. Updates task and job status in database
    """
    logger.info(f"🚀 Starting execution of job {job_id}")

    # Get database and services
    container = get_container()
    db = container.database()
    file_storage = container.file_storage()

    # Import agent components
    from src.agent.infrastructure.container import get_agent_container

    agent_container = get_agent_container()

    # Get processing configuration
    from src.config import AppConfig

    app_config = AppConfig()
    max_concurrent = app_config.processing.max_concurrent_tasks
    preview_length = app_config.preview.source_preview_length

    try:
        # Get job and tasks
        async with db.get_async_session() as session:
            job_repo = ProcessingJobRepository(session)
            task_repo = ProcessingTaskRepository(session)

            job = await job_repo.get_by_id(job_id)
            if not job:
                logger.error(f"Job {job_id} not found")
                return

            tasks = await task_repo.list_by_job(job_id)
            if not tasks:
                logger.warning(f"No tasks found for job {job_id}")
                return

            # Mark job as running
            job.status = ProcessingStatus.RUNNING
            await job_repo.update(job)
            await session.commit()

        logger.info(f"⚡ Processing {len(tasks)} tasks with max {max_concurrent} concurrent workers")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(task):
            """Process a single task with semaphore for concurrency control."""
            async with semaphore:
                try:
                    await _process_task(
                        job_id=job_id,
                        task=task,
                        db=db,
                        file_storage=file_storage,
                        agent_container=agent_container,
                        preview_length=preview_length,
                    )
                    return {"success": True, "task_id": task.id}
                except Exception as e:
                    logger.error(f"Task {task.id} failed: {e}")
                    return {"success": False, "task_id": task.id, "error": str(e)}

        # Process all tasks concurrently
        results = await asyncio.gather(
            *[process_with_semaphore(task) for task in tasks],
            return_exceptions=False,  # Let individual tasks handle their errors
        )

        # Count successes and failures
        successes = sum(1 for r in results if r.get("success"))
        failures = len(results) - successes

        logger.info(f"✓ Parallel processing complete: {successes} succeeded, {failures} failed")
        logger.info(f"✓ Job {job_id} execution completed")

    except Exception as e:
        logger.error(f"✗ Job {job_id} execution failed: {e}")
        logger.exception("Full error:")

        # Mark job as failed
        async with db.get_async_session() as session:
            job_repo = ProcessingJobRepository(session)
            job = await job_repo.get_by_id(job_id)
            if job:
                job.status = ProcessingStatus.FAILED
                job.error = str(e)
                await job_repo.update(job)
                await session.commit()


async def _process_task(
    job_id: int,
    task,
    db: Database,
    file_storage: FileStorage,
    agent_container,
    preview_length: int,
) -> None:
    """
    Process a single task.

    Args:
        job_id: ID of the processing job
        task: Task to process
        db: Database instance
        file_storage: File storage instance
        agent_container: Agent container for workflow access
        preview_length: Maximum length for source chunk preview
    """
    task_id = task.id
    chunk_id = task.chunk_id

    logger.info(f"📋 Processing task {task_id} (chunk {chunk_id})")

    try:
        # Generate unique thread ID for LangGraph
        thread_id = str(uuid.uuid4())

        # Mark task as running
        async with db.get_async_session() as session:
            task_repo = ProcessingTaskRepository(session)
            task = await task_repo.get_by_id(task_id)
            task.status = ProcessingStatus.RUNNING
            task.langgraph_thread_id = thread_id
            await task_repo.update(task)
            await session.commit()

        # Get chunk data, collection info, job configuration, and sensors
        async with db.get_async_session() as session:
            chunk_repo = ChunkRepository(session)
            doc_repo = DocumentRepository(session)
            collection_repo = CollectionRepository(session)
            job_repo = ProcessingJobRepository(session)
            sensor_repo = SensorRepository(session)

            chunk = await chunk_repo.get_by_id(chunk_id)
            if not chunk:
                raise ValueError(f"Chunk {chunk_id} not found")

            document = await doc_repo.get_by_id(chunk.document_id)
            if not document:
                raise ValueError(f"Document {chunk.document_id} not found")

            collection = await collection_repo.get_by_id(document.collection_id)
            if not collection:
                raise ValueError(f"Collection {document.collection_id} not found")

            # Get job to access use_grounding configuration
            job = await job_repo.get_by_id(job_id)
            if not job:
                raise ValueError(f"Job {job_id} not found")

            use_grounding = job.use_grounding

            # Get sensors for this collection
            sensors_orm = await sensor_repo.list_by_collection(document.collection_id)
            # Convert ORM objects to dicts for workflow
            sensors = [
                {
                    "id": s.id,
                    "sensor_id": s.sensor_id,
                    "name": s.name,
                    "description": s.description,
                    "unit": s.unit,
                    "example": s.example,
                }
                for s in sensors_orm
            ]
            logger.info(f"📡 Loaded {len(sensors)} sensors for collection {collection.name}")

        # Reconstruct the Document object for LangGraph
        # In a real implementation, you'd load the full chunk content from Qdrant
        # For now, we'll use the preview
        from langchain_core.documents import Document as LangChainDocument

        doc_chunk = LangChainDocument(
            page_content=chunk.content_preview,  # TODO: Load full content from Qdrant
            metadata={
                "chunk_id": chunk.id,
                "document_id": document.id,
                "collection_id": document.collection_id,
                "source": document.filename,
                "chunk_index": chunk.chunk_index,
            },
        )

        # Get workflow from agent container
        workflow = agent_container.rule_extraction_use_case()._ensure_workflow()

        # Execute workflow with optional Langfuse tracing
        result = await _execute_workflow_with_callbacks(
            workflow=workflow,
            chunk=doc_chunk,
            collection_name=collection.qdrant_collection_name,
            collection_id=document.collection_id,
            collection_display_name=collection.name,
            sensors=sensors,
            use_grounding=use_grounding,
            task_id=task_id,
            job_id=job_id,
        )

        # Save extracted rules to database
        await _save_extracted_rules(
            result=result,
            task_id=task_id,
            collection_id=document.collection_id,
            document_id=document.id,
            chunk_id=chunk_id,
            chunk_preview=chunk.content_preview,
            preview_length=preview_length,
        )

        # Mark task as completed
        async with db.get_async_session() as session:
            task_repo = ProcessingTaskRepository(session)
            task = await task_repo.get_by_id(task_id)
            task.status = ProcessingStatus.COMPLETED
            task.result = result
            await task_repo.update(task)
            await session.commit()

        logger.info(f"✓ Task {task_id} completed successfully")

    except Exception as e:
        logger.error(f"✗ Task {task_id} failed: {e}")

        # Mark task as failed
        async with db.get_async_session() as session:
            task_repo = ProcessingTaskRepository(session)
            task = await task_repo.get_by_id(task_id)
            task.status = ProcessingStatus.FAILED
            task.error = str(e)
            await task_repo.update(task)
            await session.commit()


async def _save_extracted_rules(
    result: dict,
    task_id: int,
    collection_id: int,
    document_id: int,
    chunk_id: int,
    chunk_preview: str,
    preview_length: int,
) -> None:
    """
    Save extracted Python rules from structured LLM output to database.

    The LLM returns rules as structured data via Pydantic models.
    We directly save each rule to the database with full traceability.

    Args:
        result: Workflow result containing extracted rules
        task_id: ID of the processing task
        collection_id: ID of the collection
        document_id: ID of the document
        chunk_id: ID of the chunk
        chunk_preview: Preview of the chunk content
        preview_length: Maximum length for source chunk preview
    """
    extracted_rules = result.get("extracted_rules", {})

    # Handle both dict format (from structured output) and error cases
    if isinstance(extracted_rules, str):
        logger.warning(f"Received string instead of structured rules for task {task_id}")
        return

    if "error" in extracted_rules:
        logger.error(f"Error in rule extraction for task {task_id}: {extracted_rules['error']}")
        return

    rules = extracted_rules.get("rules", [])
    if not rules:
        logger.warning(f"No extracted rules for task {task_id}")
        return

    # Get database container
    container = get_container()
    db = container.database()

    # Extract observability data from result
    context_chunks = result.get("context_chunks", [])
    grounding_searches = result.get("grounding_searches", [])

    async with db.get_async_session() as session:
        rule_repo = RuleRepository(session)
        context_chunk_repo = RuleContextChunkRepository(session)
        grounding_search_repo = RuleGroundingSearchRepository(session)

        # Save each rule (already structured from Pydantic model)
        for rule_data in rules:
            try:
                # Create rule using DTO
                rule_dto = CreateRuleDTO(
                    task_id=task_id,
                    collection_id=collection_id,
                    document_id=document_id,
                    chunk_id=chunk_id,
                    rule_name=rule_data["rule_name"],
                    rule_description=rule_data["rule_description"],
                    rule_reasoning=rule_data["rule_reasoning"],
                    rule_source=rule_data["rule_source"],
                    rule_body_original=rule_data.get("rule_body_original", rule_data["rule_body"]),
                    rule_body=rule_data["rule_body"],
                    sensor_parsing_status=rule_data.get("sensor_parsing_status", SensorParsingStatus.OK),
                    time_parsing_status=rule_data.get("time_parsing_status", TimeParsingStatus.OK),
                    verification_status=rule_data.get("verification_status", VerificationStatus.OK),
                    rule_type=rule_data.get("rule_type"),
                    source_chunk_preview=chunk_preview[:preview_length],
                    extraction_metadata={
                        "num_retrieved_docs": result.get("metadata", {}).get("num_retrieved_docs"),
                        "sources": result.get("metadata", {}).get("sources", []),
                        "collection_name": result.get("metadata", {}).get("collection_name"),
                    },
                )

                rule = await rule_repo.create_from_dto(rule_dto)

                # Save context chunks for this rule
                if context_chunks:
                    for context_chunk in context_chunks:
                        # chunk_id should be in metadata (added during Qdrant sync)
                        db_chunk_id = context_chunk.get("chunk_id")
                        if db_chunk_id:
                            await context_chunk_repo.create(
                                rule_id=rule.id,
                                chunk_id=db_chunk_id,
                                relevance_score=context_chunk.get("relevance_score"),
                                rank=context_chunk.get("rank", 0),
                            )
                        else:
                            logger.warning(
                                f"Context chunk missing chunk_id metadata - was Qdrant synced with new code? "
                                f"qdrant_point_id: {context_chunk.get('qdrant_point_id')}"
                            )

                # Save grounding searches for this rule
                if grounding_searches:
                    for grounding_search in grounding_searches:
                        await grounding_search_repo.create(
                            rule_id=rule.id,
                            search_query=grounding_search.get("search_query", ""),
                            search_results=grounding_search.get("search_results", {}),
                            search_rank=grounding_search.get("search_rank", 0),
                        )

            except Exception as e:
                logger.error(f"Failed to save rule {rule_data.get('rule_name', 'unknown')}: {e}")
                logger.exception("Full error:")

        await session.commit()
        logger.info(f"✓ Saved {len(rules)} Python rules with observability data for task {task_id}")


async def _execute_workflow_with_callbacks(
    workflow,
    chunk,
    collection_name: str,
    collection_id: int,
    collection_display_name: str,
    sensors: list[dict],
    use_grounding: bool,
    task_id: int,
    job_id: int,
):
    """Execute workflow with optional Langfuse tracing."""

    # Get Langfuse manager and create callback handler
    langfuse_manager = get_langfuse_manager()
    langfuse_client = langfuse_manager.get_client()
    langfuse_handler = langfuse_manager.get_callback_handler()

    # Execute workflow with or without Langfuse tracing
    if langfuse_handler and langfuse_client:
        logger.info(f"✅ Langfuse tracing enabled for task {task_id}")

        # Wrap execution with Langfuse span and propagate attributes
        with langfuse_client.start_as_current_span(name=f"Task-{task_id}-Job-{job_id}") as span:
            # Set trace metadata
            span.update_trace(
                user_id=f"job-{job_id}",
                input={
                    "collection_name": collection_name,
                    "collection_display_name": collection_display_name,
                    "task_id": task_id,
                    "job_id": job_id,
                    "sensors_count": len(sensors),
                    "use_grounding": use_grounding,
                },
            )

            # Propagate session_id, metadata, and tags to all child observations
            with propagate_attributes(
                session_id=f"job-{job_id}",
                metadata={
                    "job_id": str(job_id),
                    "task_id": str(task_id),
                    "collection_id": str(collection_id),
                    "collection_name": collection_display_name,
                    "sensors_count": str(len(sensors)),
                    "use_grounding": str(use_grounding),
                },
                tags=[
                    "rule-extraction",
                    f"job-{job_id}",
                    f"collection-{collection_id}",
                    f"collection:{collection_display_name}",
                ],
            ):
                result = await workflow.arun(
                    chunk=chunk,
                    collection_name=collection_name,
                    collection_id=collection_id,
                    sensors=sensors,
                    use_grounding=use_grounding,
                    config={
                        "callbacks": [langfuse_handler],
                        "run_name": f"Task-{task_id}-Job-{job_id}",
                    },
                )

            # Set trace output
            span.update_trace(output=result)
    else:
        # Run without callbacks (Langfuse disabled or failed to initialize)
        logger.info(f"Running task {task_id} without Langfuse tracing")
        result = await workflow.arun(
            chunk=chunk,
            collection_name=collection_name,
            collection_id=collection_id,
            sensors=sensors,
            use_grounding=use_grounding,
        )

    return result
