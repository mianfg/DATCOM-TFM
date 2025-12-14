"""Protocols (interfaces) for streaming components."""

from typing import Protocol, Any
import pandas as pd

from src.streaming.domain.models import RuleEvent, RuleMetadata


class StatusProvider(Protocol):
    """Interface for status.get() implementation."""

    def get(self, sensor_id: str, time_expr: str, statistic: str | None = None) -> float | None:
        """
        Get sensor value with time-based query.

        Args:
            sensor_id: Sensor identifier (e.g., "14TI0041")
            time_expr: Time expression (e.g., "0", "5m", "5m:", "10h:2m")
            statistic: Statistic to compute for intervals (e.g., "mean", "max")

        Returns:
            Sensor value or None if not available
        """
        ...

    def update(self, sensor_id: str, value: float | None) -> None:
        """Update buffers with new sensor value."""
        ...

    def get_snapshot(self) -> dict[str, float]:
        """Get current values of all sensors."""
        ...


class RuleLoader(Protocol):
    """Interface for loading rules from various sources."""

    async def load_rules(self, **kwargs) -> list[dict[str, Any]]:
        """
        Load rules from source.

        Returns:
            List of rule dictionaries with 'rule_name', 'rule_body', etc.
        """
        ...


class EventSink(Protocol):
    """Interface for storing/processing rule events."""

    def write_event(self, event: RuleEvent) -> None:
        """Write a single event."""
        ...

    def write_events(self, events: list[RuleEvent]) -> None:
        """Write multiple events."""
        ...

    def to_dataframe(self) -> pd.DataFrame:
        """Convert stored events to DataFrame."""
        ...

