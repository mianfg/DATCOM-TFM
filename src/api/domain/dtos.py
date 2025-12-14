"""Data Transfer Objects (DTOs) for the API layer."""

from pydantic import BaseModel, Field

from src.api.domain.models import (
    RuleLifecycleStatus,
    SensorParsingStatus,
    TimeParsingStatus,
    VerificationStatus,
)


class CreateRuleDTO(BaseModel):
    """
    Data Transfer Object for creating a new rule.
    
    This DTO encapsulates all the parameters needed to create a rule,
    making the API cleaner and easier to maintain than 20+ individual parameters.
    """

    # Required fields
    task_id: int = Field(description="ID of the processing task that extracted this rule")
    collection_id: int = Field(description="ID of the collection this rule belongs to")
    document_id: int = Field(description="ID of the source document")
    chunk_id: int = Field(description="ID of the specific chunk within the document")
    rule_name: str = Field(description="Name of the rule (snake_case function name)")
    rule_description: str = Field(description="Human-readable description of the rule")
    rule_reasoning: str = Field(description="Explanation of why this rule exists")
    rule_source: str = Field(description="Source section/page from documentation")
    rule_body_original: str = Field(description="Original rule body before sensor resolution")
    rule_body: str = Field(description="Final Python code for the rule")

    # Status fields (with defaults)
    sensor_parsing_status: SensorParsingStatus = Field(
        default=SensorParsingStatus.OK,
        description="Status of sensor parsing/resolution"
    )
    time_parsing_status: TimeParsingStatus = Field(
        default=TimeParsingStatus.OK,
        description="Status of time expression parsing"
    )
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.OK,
        description="Status of rule verification"
    )

    # Optional metadata fields
    rule_type: str | None = Field(default=None, description="Type/category of rule")
    confidence: float | None = Field(default=None, description="Confidence score from LLM")
    source_chunk_preview: str | None = Field(
        default=None,
        description="Preview of source chunk (first 500 chars)"
    )
    extraction_metadata: dict | None = Field(
        default=None,
        description="Additional metadata from extraction process"
    )

    # Consolidation fields (only set for consolidated rules)
    is_consolidated: bool = Field(
        default=False,
        description="Whether this rule was created via consolidation"
    )
    lifecycle_status: RuleLifecycleStatus = Field(
        default=RuleLifecycleStatus.EXTRACTED,
        description="Lifecycle stage of this rule"
    )
    consolidated_from_ids: list[int] = Field(
        default_factory=list,
        description="IDs of original rules that were consolidated into this one"
    )
    consolidation_confidence: float | None = Field(
        default=None,
        description="LLM confidence in the consolidation (0.0-1.0)"
    )
    consolidation_reasoning: str | None = Field(
        default=None,
        description="LLM explanation of why consolidation was performed"
    )
    consolidation_job_id: int | None = Field(
        default=None,
        description="ID of the consolidation job that created this rule"
    )

    class Config:
        """Pydantic configuration."""
        from_attributes = True  # Allow ORM mode


class UpdateRuleDTO(BaseModel):
    """
    Data Transfer Object for updating an existing rule.
    
    All fields are optional - only provided fields will be updated.
    """

    rule_name: str | None = None
    rule_description: str | None = None
    rule_reasoning: str | None = None
    rule_source: str | None = None
    rule_body: str | None = None
    rule_type: str | None = None
    confidence: float | None = None
    sensor_parsing_status: SensorParsingStatus | None = None
    time_parsing_status: TimeParsingStatus | None = None
    verification_status: VerificationStatus | None = None
    lifecycle_status: RuleLifecycleStatus | None = None

    class Config:
        """Pydantic configuration."""
        from_attributes = True

