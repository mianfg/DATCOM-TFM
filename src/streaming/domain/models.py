"""Domain models for streaming rule checker."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class RuleRequirement:
    """Represents a single requirement from a rule (e.g., one status.get() call)."""

    sensor_id: str
    time_expression: str  # e.g., "0", "5m", "5m:", "10h:2m"
    statistic: str | None = None  # e.g., "mean", "max", None for point queries


@dataclass
class RuleMetadata:
    """Metadata about a rule for execution."""

    rule_name: str
    rule_body: str
    rule_description: str
    requirements: list[RuleRequirement] = field(default_factory=list)
    compiled_func: Any = None  # Compiled Python function


@dataclass
class RuleEvent:
    """Represents a rule trigger event."""

    timestamp: datetime
    rule_name: str
    triggered: bool
    result: str | None
    sensor_values: dict[str, float] = field(default_factory=dict)
    explanation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame/storage."""
        return {
            "timestamp": self.timestamp,
            "rule_name": self.rule_name,
            "triggered": self.triggered,
            "result": self.result,
            "sensor_values": self.sensor_values,
            "explanation": self.explanation,
            **self.metadata,
        }


@dataclass
class CheckerConfig:
    """Configuration for RuleChecker."""

    granularity: str = "1min"  # Time resolution of input data
    skip_on_missing_data: bool = True  # Skip rule execution if data missing
    max_history_buffer_size: int = 1440  # Max buffer size (e.g., 24 hours at 1-min resolution)
    enable_caching: bool = True  # Cache status.get() results within same timestamp

