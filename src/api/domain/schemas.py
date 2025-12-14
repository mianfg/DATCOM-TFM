"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.api.domain.models import (
    ProcessingStatus,
    QdrantStatus,
    RuleLifecycleStatus,
    SensorParsingStatus,
    TimeParsingStatus,
    VerificationStatus,
)


# Collection Schemas
class CollectionCreate(BaseModel):
    """Schema for creating a collection."""

    name: str = Field(..., min_length=1, max_length=255, description="Collection name")
    description: str | None = Field(None, description="Collection description")


class CollectionUpdate(BaseModel):
    """Schema for updating a collection."""

    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None


class CollectionResponse(BaseModel):
    """Schema for collection response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None
    qdrant_collection_name: str
    created_at: datetime
    updated_at: datetime
    document_count: int = 0
    total_chunks: int = 0


# Document Schemas
class DocumentUpload(BaseModel):
    """Schema for document upload metadata."""

    filename: str
    mime_type: str
    file_size: int
    metadata: dict[str, Any] | None = None


class DocumentResponse(BaseModel):
    """Schema for document response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    collection_id: int
    filename: str
    file_path: str
    content_hash: str
    file_size: int
    mime_type: str
    qdrant_status: QdrantStatus
    metadata_: dict[str, Any] | None = Field(alias="metadata")
    uploaded_at: datetime
    chunk_count: int = 0


class DocumentPreview(BaseModel):
    """Schema for document preview."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    mime_type: str
    file_size: int
    qdrant_status: QdrantStatus
    uploaded_at: datetime
    chunk_count: int = 0


# Chunk Schemas
class ChunkResponse(BaseModel):
    """Schema for chunk response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    document_id: int
    chunk_index: int
    qdrant_point_id: str | None
    content_preview: str
    content_hash: str
    chunk_size: int
    created_at: datetime


# Processing Job Schemas
class ProcessingJobCreate(BaseModel):
    """Schema for creating a processing job."""

    collection_id: int
    use_grounding: bool = True  # Enable web grounding by default


class ProcessingJobCreateSelective(BaseModel):
    """Schema for creating a selective processing job (subset of documents/chunks)."""

    collection_id: int
    document_ids: list[int] | None = None
    chunk_ids: list[int] | None = None
    use_grounding: bool = True  # Enable web grounding by default


class ProcessingJobResponse(BaseModel):
    """Schema for processing job response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    collection_id: int
    status: ProcessingStatus
    use_grounding: bool
    total_chunks: int
    completed_chunks: int
    failed_chunks: int
    error: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    progress_percentage: float = 0.0


# Processing Task Schemas
class ProcessingTaskResponse(BaseModel):
    """Schema for processing task response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    job_id: int
    chunk_id: int
    status: ProcessingStatus
    langgraph_thread_id: str | None
    result: dict[str, Any] | None
    error: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime


class ProcessingTaskDetail(BaseModel):
    """Schema for detailed processing task response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    job_id: int
    chunk_id: int
    status: ProcessingStatus
    langgraph_thread_id: str | None
    result: dict[str, Any] | None
    error: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    chunk: ChunkResponse


# Rule Schemas
class RuleResponse(BaseModel):
    """Schema for rule response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    task_id: int
    collection_id: int
    document_id: int
    chunk_id: int
    rule_name: str
    rule_description: str
    rule_reasoning: str
    rule_source: str
    rule_body_original: str
    rule_body: str
    sensor_parsing_status: SensorParsingStatus
    time_parsing_status: TimeParsingStatus
    verification_status: VerificationStatus
    rule_type: str | None
    confidence: float | None
    source_chunk_preview: str | None
    extraction_metadata: dict[str, Any] | None
    lifecycle_status: RuleLifecycleStatus
    is_consolidated: bool
    consolidated_from_ids: list[int] | None = None
    consolidation_confidence: float | None = None
    consolidation_reasoning: str | None = None
    consolidation_job_id: int | None = None
    created_at: datetime


class RuleContextChunkResponse(BaseModel):
    """Schema for rule context chunk response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    rule_id: int
    chunk_id: int
    relevance_score: float | None
    rank: int
    created_at: datetime
    # Chunk details
    chunk_content_preview: str | None = None
    chunk_qdrant_point_id: str | None = None
    document_filename: str | None = None


class RuleGroundingSearchResponse(BaseModel):
    """Schema for rule grounding search response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    rule_id: int
    search_query: str
    search_results: dict[str, Any]
    search_rank: int
    created_at: datetime


class RuleDetailResponse(BaseModel):
    """Schema for detailed rule response with relationships."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    task_id: int
    collection_id: int
    document_id: int
    chunk_id: int
    rule_name: str
    rule_description: str
    rule_reasoning: str
    rule_source: str
    rule_body_original: str
    rule_body: str
    sensor_parsing_status: SensorParsingStatus
    time_parsing_status: TimeParsingStatus
    verification_status: VerificationStatus
    rule_type: str | None
    confidence: float | None
    source_chunk_preview: str | None
    extraction_metadata: dict[str, Any] | None
    lifecycle_status: RuleLifecycleStatus
    is_consolidated: bool
    consolidated_from_ids: list[int] | None = None
    consolidation_confidence: float | None = None
    consolidation_reasoning: str | None = None
    consolidation_job_id: int | None = None
    created_at: datetime
    # Related info
    document_filename: str | None = None
    chunk_index: int | None = None
    # Observability
    context_chunks: list[RuleContextChunkResponse] = []
    grounding_searches: list[RuleGroundingSearchResponse] = []


class RuleStats(BaseModel):
    """Schema for rule statistics."""

    total_rules: int
    rules_by_type: dict[str, int]
    rules_by_document: dict[str, int]
    latest_extraction: datetime | None


# Sensor Schemas
class SensorBase(BaseModel):
    """Base schema for sensor."""

    sensor_id: str  # e.g., "14A3003I"
    name: str
    description: str | None = None
    unit: str | None = None
    example: str | None = None


class SensorCreate(SensorBase):
    """Schema for creating/updating sensors."""

    pass


class SensorBulkUpsert(BaseModel):
    """Schema for bulk upserting sensors."""

    sensors: list[SensorCreate]


class SensorResponse(SensorBase):
    """Schema for sensor response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    collection_id: int
    created_at: datetime
    updated_at: datetime


class SensorListResponse(BaseModel):
    """Schema for list of sensors with count."""

    sensors: list[SensorResponse]
    total: int


# Consolidation Schemas
class ConsolidationJobCreate(BaseModel):
    """Schema for creating a consolidation job."""

    collection_id: int | None = Field(None, description="Consolidate all rules in this collection")
    processing_job_id: int | None = Field(None, description="Consolidate rules from this processing job")
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence to apply consolidation")


class ConsolidationJobResponse(BaseModel):
    """Schema for consolidation job response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    collection_id: int | None
    processing_job_id: int | None
    confidence_threshold: float
    input_rules_count: int
    output_rules_count: int
    rules_removed: int
    rules_merged: int
    rules_simplified: int
    status: ProcessingStatus
    error: str | None
    consolidation_summary: dict[str, Any] | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class ConsolidationJobDetailResponse(ConsolidationJobResponse):
    """Schema for detailed consolidation job response with related data."""

    # Can add before/after rule lists if needed
    pass
