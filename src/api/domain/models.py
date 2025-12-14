"""Domain models for API layer."""

from datetime import datetime
from enum import Enum
from typing import Any

import sqlalchemy
from sqlalchemy import JSON, BigInteger, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class ProcessingStatus(str, Enum):
    """Status of processing jobs and tasks."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QdrantStatus(str, Enum):
    """Status of document in Qdrant."""

    NOT_UPLOADED = "not_uploaded"
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    FAILED = "failed"


class SensorParsingStatus(str, Enum):
    """Status of sensor resolution in extracted rules."""

    NO_SENSORS = "no_sensors"  # No status.get() calls found in rule body
    SENSORS_NOT_FOUND = "sensors_not_found"  # Some sensors could not be resolved to IDs
    OK = "ok"  # All sensors successfully resolved


class TimeParsingStatus(str, Enum):
    """Status of time expression parsing in extracted rules."""

    OK = "ok"  # All time expressions parsed correctly
    PARSE_ERROR = "parse_error"  # Failed to parse time expression
    INVALID_STATISTIC = "invalid_statistic"  # Interval without statistic or invalid statistic value


class VerificationStatus(str, Enum):
    """Status of rule verification and validation."""

    OK = "ok"  # All validation checks passed
    SYNTAX_ERROR = "syntax_error"  # Python syntax error in rule body
    INVALID_SENSOR = "invalid_sensor"  # Sensor ID not in collection's sensor list
    INVALID_TIME = "invalid_time"  # Time expression cannot be parsed
    INVALID_STATISTIC = "invalid_statistic"  # Missing or invalid statistic for interval


class RuleLifecycleStatus(str, Enum):
    """Lifecycle status of a rule."""

    EXTRACTED = "extracted"  # Freshly extracted from documentation
    CONSOLIDATED = "consolidated"  # Result of consolidation process
    SUPERSEDED = "superseded"  # Original rule that was consolidated into another
    ACTIVE = "active"  # Manually approved/active


class Collection(Base):
    """Collection of documents."""

    __tablename__ = "collections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    qdrant_collection_name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    documents: Mapped[list["Document"]] = relationship(
        "Document", back_populates="collection", cascade="all, delete-orphan"
    )
    jobs: Mapped[list["ProcessingJob"]] = relationship(
        "ProcessingJob", back_populates="collection", cascade="all, delete-orphan"
    )
    rules: Mapped[list["Rule"]] = relationship("Rule", back_populates="collection", cascade="all, delete-orphan")
    sensors: Mapped[list["Sensor"]] = relationship("Sensor", back_populates="collection", cascade="all, delete-orphan")


class Document(Base):
    """Document in a collection."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    collection_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)  # Path on disk
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # SHA256 hash
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)  # Size in bytes
    mime_type: Mapped[str] = mapped_column(String(127), nullable=False)
    qdrant_status: Mapped[QdrantStatus] = mapped_column(
        SQLEnum(QdrantStatus, name="qdrant_status_enum"), default=QdrantStatus.NOT_UPLOADED, nullable=False, index=True
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True, name="metadata")  # Custom metadata
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    collection: Mapped["Collection"] = relationship("Collection", back_populates="documents")
    chunks: Mapped[list["Chunk"]] = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    rules: Mapped[list["Rule"]] = relationship("Rule", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    """Chunk of a document (stored in Qdrant)."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)  # Order within document
    qdrant_point_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)  # ID in Qdrant
    content_preview: Mapped[str] = mapped_column(Text, nullable=False)  # First 500 chars
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # SHA256 of full content
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False)  # Size in characters
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
    tasks: Mapped[list["ProcessingTask"]] = relationship(
        "ProcessingTask", back_populates="chunk", cascade="all, delete-orphan"
    )
    rules: Mapped[list["Rule"]] = relationship("Rule", back_populates="chunk", cascade="all, delete-orphan")


class ProcessingJob(Base):
    """Processing job for a collection."""

    __tablename__ = "processing_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    collection_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[ProcessingStatus] = mapped_column(
        SQLEnum(ProcessingStatus, name="processing_status_enum"),
        default=ProcessingStatus.PENDING,
        nullable=False,
        index=True,
    )
    use_grounding: Mapped[bool] = mapped_column(default=True, nullable=False)  # Enable web grounding by default
    total_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_chunks: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    collection: Mapped["Collection"] = relationship("Collection", back_populates="jobs")
    tasks: Mapped[list["ProcessingTask"]] = relationship(
        "ProcessingTask", back_populates="job", cascade="all, delete-orphan"
    )


class ProcessingTask(Base):
    """Processing task for a single chunk."""

    __tablename__ = "processing_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("processing_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[ProcessingStatus] = mapped_column(
        SQLEnum(ProcessingStatus, name="processing_status_enum"),
        default=ProcessingStatus.PENDING,
        nullable=False,
        index=True,
    )
    langgraph_thread_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)  # Extracted rules
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    job: Mapped["ProcessingJob"] = relationship("ProcessingJob", back_populates="tasks")
    chunk: Mapped["Chunk"] = relationship("Chunk", back_populates="tasks")
    rules: Mapped[list["Rule"]] = relationship("Rule", back_populates="task", cascade="all, delete-orphan")


class Rule(Base):
    """Extracted Python rule from document processing with full traceability."""

    __tablename__ = "rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Foreign keys for complete traceability
    task_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("processing_tasks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    collection_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Python rule schema
    rule_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)  # Function name in snake_case
    rule_description: Mapped[str] = mapped_column(Text, nullable=False)  # Human-readable description
    rule_reasoning: Mapped[str] = mapped_column(Text, nullable=False)  # Why this rule exists
    rule_source: Mapped[str] = mapped_column(Text, nullable=False)  # Where in docs it came from
    rule_body_original: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # Original rule body with natural language sensors
    rule_body: Mapped[str] = mapped_column(Text, nullable=False)  # Final rule body with resolved sensor IDs
    sensor_parsing_status: Mapped[SensorParsingStatus] = mapped_column(
        SQLEnum(SensorParsingStatus, name="sensor_parsing_status_enum"),
        default=SensorParsingStatus.OK,
        nullable=False,
        index=True,
    )  # Sensor resolution status
    time_parsing_status: Mapped[TimeParsingStatus] = mapped_column(
        SQLEnum(TimeParsingStatus, name="time_parsing_status_enum"),
        default=TimeParsingStatus.OK,
        nullable=False,
        index=True,
    )  # Time expression parsing status
    verification_status: Mapped[VerificationStatus] = mapped_column(
        SQLEnum(VerificationStatus, name="verification_status_enum"),
        default=VerificationStatus.OK,
        nullable=False,
        index=True,
    )  # Rule verification status
    rule_type: Mapped[str | None] = mapped_column(
        String(50), nullable=True, index=True
    )  # e.g., "safety", "operational", "maintenance"
    confidence: Mapped[float | None] = mapped_column(nullable=True)  # Optional confidence score

    # Context and metadata
    source_chunk_preview: Mapped[str | None] = mapped_column(Text, nullable=True)  # Preview of source chunk
    extraction_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )  # LLM metadata, sources used, etc.

    # Consolidation tracking
    lifecycle_status: Mapped[RuleLifecycleStatus] = mapped_column(
        SQLEnum(RuleLifecycleStatus, name="rule_lifecycle_status_enum"),
        default=RuleLifecycleStatus.EXTRACTED,
        nullable=False,
        index=True,
    )  # Current lifecycle stage
    is_consolidated: Mapped[bool] = mapped_column(
        default=False, nullable=False, index=True
    )  # True if this rule is a result of consolidation
    consolidated_from_ids: Mapped[list[int] | None] = mapped_column(
        JSON, nullable=True
    )  # List of rule IDs that were merged into this rule
    consolidation_confidence: Mapped[float | None] = mapped_column(nullable=True)  # LLM confidence score (0.0-1.0)
    consolidation_reasoning: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # Why these rules were consolidated
    consolidation_job_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("consolidation_jobs.id", ondelete="SET NULL"), nullable=True, index=True
    )  # Which consolidation job created this

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    task: Mapped["ProcessingTask"] = relationship("ProcessingTask", back_populates="rules")
    collection: Mapped["Collection"] = relationship("Collection", back_populates="rules")
    document: Mapped["Document"] = relationship("Document", back_populates="rules")
    chunk: Mapped["Chunk"] = relationship("Chunk", back_populates="rules")
    consolidation_job: Mapped["ConsolidationJob"] = relationship(
        "ConsolidationJob", back_populates="consolidated_rules"
    )
    context_chunks: Mapped[list["RuleContextChunk"]] = relationship(
        "RuleContextChunk", back_populates="rule", cascade="all, delete-orphan"
    )
    grounding_searches: Mapped[list["RuleGroundingSearch"]] = relationship(
        "RuleGroundingSearch", back_populates="rule", cascade="all, delete-orphan"
    )


class Sensor(Base):
    """Sensor associated with a collection."""

    __tablename__ = "sensors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    collection_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Sensor identification (from CSV or manual input)
    sensor_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)  # e.g., "14A3003I"
    name: Mapped[str] = mapped_column(String(500), nullable=False)  # e.g., "Propane/Propylene at Bottom"
    description: Mapped[str | None] = mapped_column(Text, nullable=True)  # Detailed description
    unit: Mapped[str | None] = mapped_column(String(100), nullable=True)  # e.g., "%", "kg/cm²"
    example: Mapped[str | None] = mapped_column(String(255), nullable=True)  # Example value

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    collection: Mapped["Collection"] = relationship("Collection", back_populates="sensors")

    # Unique constraint: sensor_id must be unique per collection
    __table_args__ = (sqlalchemy.UniqueConstraint("collection_id", "sensor_id", name="uq_collection_sensor_id"),)


class RuleContextChunk(Base):
    """Tracks which chunks were used as RAG context when extracting a rule."""

    __tablename__ = "rule_context_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rule_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("rules.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Context metadata
    relevance_score: Mapped[float | None] = mapped_column(nullable=True)  # Qdrant similarity score
    rank: Mapped[int] = mapped_column(nullable=False)  # Order in retrieval (1-based)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    rule: Mapped["Rule"] = relationship("Rule", back_populates="context_chunks")
    chunk: Mapped["Chunk"] = relationship("Chunk")


class RuleGroundingSearch(Base):
    """Tracks grounding searches performed when extracting a rule."""

    __tablename__ = "rule_grounding_searches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    rule_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("rules.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Search details
    search_query: Mapped[str] = mapped_column(Text, nullable=False)  # The query sent to Tavily
    search_results: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)  # Tavily response
    search_rank: Mapped[int] = mapped_column(nullable=False)  # Order of search (1-based)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    rule: Mapped["Rule"] = relationship("Rule", back_populates="grounding_searches")


class ConsolidationJob(Base):
    """Rule consolidation job for optimizing and merging rules."""

    __tablename__ = "consolidation_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Scope: Either collection-wide or job-specific
    collection_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("collections.id", ondelete="CASCADE"), nullable=True, index=True
    )
    processing_job_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("processing_jobs.id", ondelete="CASCADE"), nullable=True, index=True
    )

    # Configuration
    confidence_threshold: Mapped[float] = mapped_column(
        default=0.7, nullable=False
    )  # Minimum confidence to apply consolidation

    # Statistics
    input_rules_count: Mapped[int] = mapped_column(default=0, nullable=False)  # Rules before consolidation
    output_rules_count: Mapped[int] = mapped_column(default=0, nullable=False)  # Rules after consolidation
    rules_removed: Mapped[int] = mapped_column(default=0, nullable=False)  # Rules marked as superseded
    rules_merged: Mapped[int] = mapped_column(default=0, nullable=False)  # New consolidated rules created
    rules_simplified: Mapped[int] = mapped_column(default=0, nullable=False)  # Rules simplified

    # Status tracking
    status: Mapped[ProcessingStatus] = mapped_column(
        SQLEnum(ProcessingStatus, name="processing_status_enum"),
        default=ProcessingStatus.PENDING,
        nullable=False,
        index=True,
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)  # Error message if failed

    # Results
    consolidation_summary: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )  # Detailed summary of consolidations

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    collection: Mapped["Collection"] = relationship("Collection")
    processing_job: Mapped["ProcessingJob"] = relationship("ProcessingJob")
    consolidated_rules: Mapped[list["Rule"]] = relationship("Rule", back_populates="consolidation_job")
