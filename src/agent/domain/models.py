"""Domain models for Agent layer - moved from src/domain/models/agent_state.py"""

from enum import Enum, StrEnum
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field, field_validator


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


class ChunkModel(BaseModel):
    """
    Pydantic model for a document chunk.

    JSON-serializable alternative to LangChain's Document class.
    """

    page_content: str = Field(..., description="The text content of the chunk")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata about the chunk")

    @classmethod
    def from_document(cls, doc: Document) -> "ChunkModel":
        """Create ChunkModel from LangChain Document."""
        return cls(page_content=doc.page_content, metadata=doc.metadata)

    def to_document(self) -> Document:
        """Convert to LangChain Document."""
        return Document(page_content=self.page_content, metadata=self.metadata)

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dict for state."""
        return {"page_content": self.page_content, "metadata": self.metadata}


class AgentState(TypedDict):
    """
    State for the rule extraction agent.

    Uses TypedDict for LangGraph compatibility, but chunk field accepts
    dict format that matches ChunkModel for easy serialization.

    All fields are JSON-serializable for LangGraph Studio compatibility.

    Simplified single-chunk workflow:
    - Input: chunk (dict with 'page_content' and 'metadata')
    - Input: collection_name (which Qdrant collection to search)
    - Input: collection_id (for fetching sensors)
    - Input: sensors (list of available sensors for resolution)
    - Process: gather context → (optional) ground → extract rules → resolve sensors
    - Output: extracted_rules
    """

    # LangGraph Studio compatibility
    messages: Annotated[list[BaseMessage], add_messages]

    # Input: single chunk to process (JSON-serializable dict)
    # Format matches ChunkModel: {'page_content': str, 'metadata': dict}
    chunk: dict[str, Any]

    # Input: Qdrant collection name to search (enables project isolation)
    collection_name: str

    # Input: Collection ID for sensor lookup
    collection_id: int

    # Input: Available sensors for this collection
    sensors: list[dict[str, Any]]  # List of sensor dicts with id, sensor_id, name, unit, description, example

    # Configuration
    use_grounding: bool  # Whether to use web grounding stage

    # Processing state
    context: str
    grounding_info: str  # External knowledge from web search
    extracted_rules: str | dict[str, Any]  # Can be text or structured output

    # Observability: track context chunks and searches per chunk (before per-rule tracking)
    context_chunks: list[dict[str, Any]]  # Chunks retrieved from Qdrant with scores
    grounding_searches: list[dict[str, Any]]  # Grounding searches performed

    # Metadata
    metadata: dict[str, Any]


class PythonRule(BaseModel):
    """
    Schema for a single Python operational rule.

    This model defines the structure that the LLM must follow when extracting
    rules from industrial documentation.
    """

    rule_name: str = Field(..., description="Function name in snake_case (e.g., 'column_high_pressure_alert')")
    rule_description: str = Field(..., description="Brief human-readable description of what the rule does")
    rule_reasoning: str = Field(..., description="Explanation of why this rule exists and its importance")
    rule_source: str = Field(..., description="Section or location in the documentation where this rule came from")
    rule_body: str = Field(
        ..., description="Complete Python function code using status.get() with natural language time expressions"
    )
    rule_type: str | None = Field(
        None, description="Category of rule: 'safety', 'operational', 'maintenance', or 'optimization'"
    )
    sensor_parsing_status: str = Field(
        default=SensorParsingStatus.OK,
        description="Status of sensor resolution: 'no_sensors', 'sensors_not_found', or 'ok'",
    )
    time_parsing_status: str = Field(
        default=TimeParsingStatus.OK,
        description="Status of time expression parsing: 'ok', 'parse_error', or 'invalid_statistic'",
    )
    verification_status: str = Field(
        default=VerificationStatus.OK,
        description="Status of rule verification: 'ok', 'syntax_error', 'invalid_sensor', 'invalid_time', or 'invalid_statistic'",
    )

    @field_validator("rule_name")
    @classmethod
    def validate_snake_case(cls, v: str) -> str:
        """Ensure rule_name is in snake_case format."""
        if not v.replace("_", "").isalnum():
            raise ValueError(f"rule_name must be snake_case alphanumeric: {v}")
        if v != v.lower():
            raise ValueError(f"rule_name must be lowercase: {v}")
        return v

    @field_validator("rule_body")
    @classmethod
    def validate_function_body(cls, v: str) -> str:
        """Ensure rule_body contains a Python function definition."""
        if "def " not in v:
            raise ValueError("rule_body must contain a Python function definition")
        if "status.get(" not in v:
            raise ValueError("rule_body must use status.get() API")
        return v


class ExtractedRules(BaseModel):
    """
    Container for all rules extracted from a document chunk.

    This is the top-level schema that the LLM returns using structured output.
    """

    rules: list[PythonRule] = Field(
        default_factory=list, description="List of all operational rules extracted from the document chunk"
    )

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "rules": [
                    {
                        "rule_name": "column_high_pressure_alert",
                        "rule_description": "Alert when column pressure exceeds safety limit",
                        "rule_reasoning": "Critical alarm set at 15.5 kg/cm² to prevent overpressure scenarios",
                        "rule_source": "Section 7. Process Safety Considerations",
                        "rule_body": 'def column_high_pressure_alert(status) -> str:\n    current_pressure = status.get("column pressure", "current")\n    if current_pressure and current_pressure > 15.5:\n        return "column_high_pressure_alert"\n    return None',
                        "rule_type": "safety",
                    }
                ]
            }
        }


# =============================================================================
# Sensor Resolution Structured Output
# =============================================================================


class SensorMapping(BaseModel):
    """Encapsulates sensor resolution mapping."""

    sensor_description: str = Field(..., description="Sensor description in natural language.")
    sensor_id: str | None = Field(..., description="Sensor ID that corresponds to the description.")


class SensorMappings(BaseModel):
    """Container for all sensor mappings from a rule body."""

    mappings: list[SensorMapping] = Field(default_factory=list, description="List of sensor description to ID mappings")


# =============================================================================
# Time Parsing Structured Output
# =============================================================================


class TimeMapping(BaseModel):
    """Encapsulates time parsing mapping."""

    time_description: str = Field(..., description="Time expression description in natural language.")
    time_expression: str = Field(..., description="Time point or interval in custom grammar.")
    time_statistic: str | None = Field(None, description="Aggregate statistic, only for time intervals")


class TimeMappings(BaseModel):
    """Container for all time expression mappings."""

    mappings: list[TimeMapping] = Field(default_factory=list, description="List of time expression mappings")


# =============================================================================
# Rule Consolidation Structured Output
# =============================================================================


class RuleConsolidationActionType(StrEnum):
    """Type of consolidation action."""

    REMOVE = "remove"
    MERGE = "merge"
    SIMPLIFY = "simplify"


class RuleConsolidation(BaseModel):
    """Encapsulates a single rule consolidation action."""

    action_type: Literal["remove", "merge", "simplify"] = Field(..., description="Type of consolidation performed.")
    input_rule_ids: list[int] = Field(..., min_length=1, description="IDs of rules used in consolidation.")
    output_rule: dict[str, Any] | None = Field(None, description="Consolidated rule (null if action is 'remove').")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence coefficient in consolidated rule, in [0, 1]."
    )
    reasoning: str = Field(..., min_length=10, description="Reasoning behind consolidation.")


class RuleConsolidations(BaseModel):
    """Container for all rule consolidation actions."""

    consolidations: list[RuleConsolidation] = Field(
        default_factory=list, description="List of consolidation actions to perform"
    )
