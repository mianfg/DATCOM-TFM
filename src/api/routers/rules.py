"""API routes for rule retrieval."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.domain.schemas import (
    RuleResponse,
    RuleDetailResponse,
    RuleStats,
    RuleContextChunkResponse,
    RuleGroundingSearchResponse,
)
from src.api.infrastructure.database import get_db
from src.api.infrastructure.repositories import (
    RuleRepository,
    DocumentRepository,
    ChunkRepository,
    RuleContextChunkRepository,
    RuleGroundingSearchRepository,
)
from src.config import AppConfig

# Load configuration
_app_config = AppConfig()

router = APIRouter(prefix="/rules", tags=["rules"])


@router.get("/collections/{collection_id}", response_model=list[RuleResponse])
async def list_rules_by_collection(
    collection_id: int,
    limit: int = Query(_app_config.query.default_limit, ge=1, le=_app_config.query.max_limit),
    offset: int = Query(_app_config.query.default_offset, ge=0),
    session: AsyncSession = Depends(get_db),
):
    """
    Get all rules for a collection.
    
    Returns rules with full traceability (task, document, chunk IDs).
    """
    rule_repo = RuleRepository(session)
    rules = await rule_repo.list_by_collection(collection_id, limit=limit, offset=offset)
    return rules


@router.get("/collections/{collection_id}/stats", response_model=RuleStats)
async def get_collection_rule_stats(
    collection_id: int,
    session: AsyncSession = Depends(get_db),
):
    """
    Get rule statistics for a collection.
    
    Returns counts by type, by document, and latest extraction date.
    """
    rule_repo = RuleRepository(session)
    stats = await rule_repo.get_stats_by_collection(collection_id)
    return stats


@router.get("/collections/{collection_id}/by-type/{rule_type}", response_model=list[RuleResponse])
async def list_rules_by_type(
    collection_id: int,
    rule_type: str,
    session: AsyncSession = Depends(get_db),
):
    """Get rules of a specific type (e.g., 'safety', 'operational')."""
    rule_repo = RuleRepository(session)
    rules = await rule_repo.list_by_type(collection_id, rule_type)
    return rules


@router.get("/documents/{document_id}", response_model=list[RuleResponse])
async def list_rules_by_document(
    document_id: int,
    session: AsyncSession = Depends(get_db),
):
    """Get all rules extracted from a specific document."""
    rule_repo = RuleRepository(session)
    rules = await rule_repo.list_by_document(document_id)
    return rules


@router.get("/tasks/{task_id}", response_model=list[RuleResponse])
async def list_rules_by_task(
    task_id: int,
    session: AsyncSession = Depends(get_db),
):
    """Get all rules extracted by a specific processing task."""
    rule_repo = RuleRepository(session)
    rules = await rule_repo.list_by_task(task_id)
    return rules


@router.get("/{rule_id}", response_model=RuleDetailResponse)
async def get_rule_detail(
    rule_id: int,
    session: AsyncSession = Depends(get_db),
):
    """
    Get detailed information about a specific rule.
    
    Includes related document filename, chunk index, context chunks, and grounding searches.
    """
    rule_repo = RuleRepository(session)
    doc_repo = DocumentRepository(session)
    chunk_repo = ChunkRepository(session)
    context_chunk_repo = RuleContextChunkRepository(session)
    grounding_search_repo = RuleGroundingSearchRepository(session)
    
    rule = await rule_repo.get_by_id(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")
    
    # Get related info
    document = await doc_repo.get_by_id(rule.document_id)
    chunk = await chunk_repo.get_by_id(rule.chunk_id)
    
    # Get observability data
    context_chunks = await context_chunk_repo.list_by_rule(rule_id)
    grounding_searches = await grounding_search_repo.list_by_rule(rule_id)
    
    # Build context chunk responses with additional details
    context_chunk_responses = []
    for ctx_chunk in context_chunks:
        chunk_detail = await chunk_repo.get_by_id(ctx_chunk.chunk_id)
        doc_detail = await doc_repo.get_by_id(chunk_detail.document_id) if chunk_detail else None
        
        context_chunk_responses.append(
            RuleContextChunkResponse(
                id=ctx_chunk.id,
                rule_id=ctx_chunk.rule_id,
                chunk_id=ctx_chunk.chunk_id,
                relevance_score=ctx_chunk.relevance_score,
                rank=ctx_chunk.rank,
                created_at=ctx_chunk.created_at,
                chunk_content_preview=chunk_detail.content_preview if chunk_detail else None,
                chunk_qdrant_point_id=chunk_detail.qdrant_point_id if chunk_detail else None,
                document_filename=doc_detail.filename if doc_detail else None,
            )
        )
    
    # Build response
    response = RuleDetailResponse(
        id=rule.id,
        task_id=rule.task_id,
        collection_id=rule.collection_id,
        document_id=rule.document_id,
        chunk_id=rule.chunk_id,
        rule_name=rule.rule_name,
        rule_description=rule.rule_description,
        rule_reasoning=rule.rule_reasoning,
        rule_source=rule.rule_source,
        rule_body_original=rule.rule_body_original,
        rule_body=rule.rule_body,
        sensor_parsing_status=rule.sensor_parsing_status,
        time_parsing_status=rule.time_parsing_status,
        verification_status=rule.verification_status,
        rule_type=rule.rule_type,
        confidence=rule.confidence,
        source_chunk_preview=rule.source_chunk_preview,
        extraction_metadata=rule.extraction_metadata,
        created_at=rule.created_at,
        document_filename=document.filename if document else None,
        chunk_index=chunk.chunk_index if chunk else None,
        context_chunks=context_chunk_responses,
        grounding_searches=[RuleGroundingSearchResponse.model_validate(gs) for gs in grounding_searches],
    )
    
    return response

