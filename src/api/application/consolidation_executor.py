"""Executor for running consolidation jobs in the background."""

from loguru import logger

from src.agent.infrastructure.container import get_agent_container
from src.api.application.consolidation_service import ConsolidationService
from src.api.domain.dtos import CreateRuleDTO
from src.api.domain.exceptions import (
    ConsolidationAlreadyExecutedError,
    ConsolidationJobNotFoundError,
    TransactionError,
)
from src.api.domain.models import (
    ProcessingStatus,
    RuleLifecycleStatus,
    SensorParsingStatus,
    TimeParsingStatus,
    VerificationStatus,
)
from src.api.infrastructure.container import get_container
from src.api.infrastructure.logging import LoggingContext
from src.api.infrastructure.repositories import (
    ConsolidationJobRepository,
    RuleRepository,
    SensorRepository,
)


async def execute_consolidation_job(consolidation_job_id: int):
    """
    Execute a consolidation job in the background.

    Uses a single database transaction for atomicity. All database operations
    (loading rules, marking superseded, creating consolidated rules) happen
    in one transaction that either fully commits or fully rolls back.

    Args:
        consolidation_job_id: ID of the consolidation job to execute
    """
    # Set up logging context for this consolidation job
    with LoggingContext(consolidation_job_id=consolidation_job_id, job_id=None):
        logger.info("🚀 Starting consolidation job")

        # Get dependencies
        container = get_container()
        db = container.database()

        # Single session for entire operation
        async with db.get_async_session() as session:
            try:
                # Initialize repositories and services
                consolidation_service = ConsolidationService(session)
                job_repo = ConsolidationJobRepository(session)
                rule_repo = RuleRepository(session)
                sensor_repo = SensorRepository(session)

                # 1. Check for idempotency - has this job already been executed?
                existing_consolidated_rules = await rule_repo.list_by_consolidation_job(consolidation_job_id)
                if existing_consolidated_rules:
                    raise ConsolidationAlreadyExecutedError(
                        job_id=consolidation_job_id, existing_rules_count=len(existing_consolidated_rules)
                    )

                # 2. Update job status to RUNNING
                await consolidation_service.update_job_status(consolidation_job_id, ProcessingStatus.RUNNING)
                await session.commit()  # Commit status so other systems can see progress

                # 3. Load job and rules
                job = await job_repo.get_by_id(consolidation_job_id)
                if not job:
                    raise ConsolidationJobNotFoundError(consolidation_job_id)

                # Get rules to consolidate
                if job.collection_id:
                    rules = await rule_repo.list_active_by_collection(job.collection_id)
                    sensors = await sensor_repo.list_by_collection(job.collection_id)
                    scope_desc = f"collection {job.collection_id}"
                elif job.processing_job_id:
                    rules = await rule_repo.list_active_by_processing_job(job.processing_job_id)
                    # Get collection_id from first rule
                    if rules:
                        sensors = await sensor_repo.list_by_collection(rules[0].collection_id)
                    else:
                        sensors = []
                    scope_desc = f"processing job {job.processing_job_id}"
                else:
                    raise ValueError("Job has neither collection_id nor processing_job_id")

                logger.info(f"Loaded {len(rules)} active rules and {len(sensors)} sensors from {scope_desc}")

                # Handle case with no rules
                if not rules:
                    logger.warning(f"No active rules to consolidate for {scope_desc}")
                    await consolidation_service.finalize_consolidation_job(
                        job_id=consolidation_job_id,
                        input_count=0,
                        output_count=0,
                        superseded_count=0,
                        remove_count=0,
                        merge_count=0,
                        simplify_count=0,
                        summary={"message": "No active rules to consolidate"},
                    )
                    await session.commit()
                    return

                # Convert rules to dicts for workflow
                rules_data = [
                    {
                        "id": rule.id,
                        "rule_name": rule.rule_name,
                        "rule_description": rule.rule_description,
                        "rule_reasoning": rule.rule_reasoning,
                        "rule_source": rule.rule_source,
                        "rule_body": rule.rule_body,
                        "rule_type": rule.rule_type,
                        "sensor_parsing_status": rule.sensor_parsing_status.value,
                        "time_parsing_status": rule.time_parsing_status.value,
                        "verification_status": rule.verification_status.value,
                    }
                    for rule in rules
                ]

                sensors_data = [
                    {
                        "sensor_id": sensor.sensor_id,
                        "name": sensor.name,
                        "description": sensor.description,
                        "unit": sensor.unit,
                    }
                    for sensor in sensors
                ]

                # 4. Run consolidation workflow (read-only, no DB writes)
                logger.info("Running consolidation workflow...")
                agent_container = get_agent_container()
                use_case = agent_container.rule_consolidation_use_case()
                workflow = use_case.get_workflow()

                # Run workflow asynchronously (doesn't block event loop)
                result = await workflow.arun(
                    rules=rules_data, sensors=sensors_data, confidence_threshold=job.confidence_threshold
                )

                logger.info("Consolidation workflow completed")

                # 5. Save all results atomically
                results = result.get("results", {})
                consolidated_rules = results.get("consolidated_rules", [])
                superseded_rule_ids = results.get("superseded_rule_ids", [])

                # Mark superseded rules
                if superseded_rule_ids:
                    await rule_repo.mark_as_superseded(superseded_rule_ids)
                    logger.info(f"Marked {len(superseded_rule_ids)} rules as superseded")

                # Save consolidated rules
                for consolidated_rule_data in consolidated_rules:
                    # Get first original rule for traceability (task_id, collection_id, document_id, chunk_id)
                    original_rule_ids = consolidated_rule_data.get("consolidated_from_ids", [])
                    if original_rule_ids:
                        original_rules = await rule_repo.get_by_ids(original_rule_ids)
                        if original_rules:
                            first_original = original_rules[0]

                            # Create consolidated rule using DTO
                            rule_dto = CreateRuleDTO(
                                # Traceability from original rule
                                task_id=first_original.task_id,
                                collection_id=first_original.collection_id,
                                document_id=first_original.document_id,
                                chunk_id=first_original.chunk_id,
                                # Rule content from consolidation
                                rule_name=consolidated_rule_data["rule_name"],
                                rule_description=consolidated_rule_data["rule_description"],
                                rule_reasoning=consolidated_rule_data["rule_reasoning"],
                                rule_source=consolidated_rule_data["rule_source"],
                                rule_body_original=consolidated_rule_data["rule_body"],
                                rule_body=consolidated_rule_data["rule_body"],
                                # Status fields
                                sensor_parsing_status=SensorParsingStatus(
                                    consolidated_rule_data["sensor_parsing_status"]
                                ),
                                time_parsing_status=TimeParsingStatus(consolidated_rule_data["time_parsing_status"]),
                                verification_status=VerificationStatus(consolidated_rule_data["verification_status"]),
                                # Optional metadata
                                rule_type=consolidated_rule_data.get("rule_type"),
                                source_chunk_preview=first_original.source_chunk_preview,
                                # Consolidation-specific fields
                                is_consolidated=True,
                                lifecycle_status=RuleLifecycleStatus.CONSOLIDATED,
                                consolidated_from_ids=consolidated_rule_data.get("consolidated_from_ids", []),
                                consolidation_confidence=consolidated_rule_data.get("consolidation_confidence"),
                                consolidation_reasoning=consolidated_rule_data.get("consolidation_reasoning"),
                                consolidation_job_id=consolidation_job_id,
                            )

                            await rule_repo.create_from_dto(rule_dto)

                            logger.debug(f"Saved consolidated rule: {consolidated_rule_data['rule_name']}")

                # Finalize job
                await consolidation_service.finalize_consolidation_job(
                    job_id=consolidation_job_id,
                    input_count=results.get("input_count", 0),
                    output_count=results.get("output_count", 0),
                    superseded_count=results.get("superseded_count", 0),
                    remove_count=results.get("remove_count", 0),
                    merge_count=results.get("merge_count", 0),
                    simplify_count=results.get("simplify_count", 0),
                    summary=results,
                )

                # COMMIT ALL CHANGES AT ONCE (atomic operation)
                await session.commit()

                logger.info(f"✓ Consolidation job {consolidation_job_id} completed successfully")

            except ConsolidationAlreadyExecutedError as e:
                # Not actually an error - job was already done
                logger.warning(f"⚠️  {e.message}")
                logger.debug(f"Details: {e.details}")

                # Ensure job status is COMPLETED
                job = await job_repo.get_by_id(consolidation_job_id)
                if job and job.status != ProcessingStatus.COMPLETED:
                    await consolidation_service.update_job_status(consolidation_job_id, ProcessingStatus.COMPLETED)
                    await session.commit()

            except ConsolidationJobNotFoundError as e:
                logger.error(f"✗ {e.message}")
                logger.debug(f"Details: {e.details}")

            except TransactionError as e:
                logger.error(f"✗ Database transaction failed for job {consolidation_job_id}: {e.message}")
                logger.debug(f"Details: {e.details}")
                logger.exception("Full error:")

                # Rollback happens automatically when exiting context
                # Update job status in a NEW separate session
                async with db.get_async_session() as error_session:
                    error_service = ConsolidationService(error_session)
                    await error_service.update_job_status(
                        consolidation_job_id, ProcessingStatus.FAILED, error=e.message
                    )
                    await error_session.commit()

            except Exception as e:
                logger.error(f"✗ Consolidation job {consolidation_job_id} failed with unexpected error: {e}")
                logger.exception("Full error:")

                # Rollback happens automatically when exiting context
                # Update job status in a NEW separate session
                async with db.get_async_session() as error_session:
                    error_service = ConsolidationService(error_session)
                    await error_service.update_job_status(
                        consolidation_job_id, ProcessingStatus.FAILED, error=f"Unexpected error: {str(e)}"
                    )
                    await error_session.commit()
