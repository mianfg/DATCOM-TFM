"""Event sinks for storing rule trigger events."""

import pandas as pd
from loguru import logger

from src.streaming.domain.models import RuleEvent
from src.streaming.domain.protocols import EventSink


class InMemoryEventSink(EventSink):
    """Simple in-memory event storage for testing and small datasets."""

    def __init__(self):
        self.events: list[RuleEvent] = []

    def write_event(self, event: RuleEvent) -> None:
        """Write a single event."""
        self.events.append(event)

    def write_events(self, events: list[RuleEvent]) -> None:
        """Write multiple events."""
        self.events.extend(events)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert stored events to DataFrame."""
        if not self.events:
            return pd.DataFrame()

        return pd.DataFrame([event.to_dict() for event in self.events])

    def clear(self):
        """Clear all stored events."""
        self.events.clear()

    def __len__(self):
        return len(self.events)


class CSVEventSink(EventSink):
    """Event sink that writes to CSV file incrementally."""

    def __init__(self, filepath: str, mode: str = "w"):
        """
        Initialize CSV sink.

        Args:
            filepath: Path to CSV file
            mode: 'w' for overwrite, 'a' for append
        """
        self.filepath = filepath
        self.mode = mode
        self._buffer: list[RuleEvent] = []
        self._buffer_size = 100  # Write every N events
        self._header_written = mode == "a"

    def write_event(self, event: RuleEvent) -> None:
        """Buffer event and write when buffer full."""
        self._buffer.append(event)

        if len(self._buffer) >= self._buffer_size:
            self.flush()

    def write_events(self, events: list[RuleEvent]) -> None:
        """Write multiple events."""
        self._buffer.extend(events)

        if len(self._buffer) >= self._buffer_size:
            self.flush()

    def flush(self):
        """Write buffered events to CSV."""
        if not self._buffer:
            return

        df = pd.DataFrame([event.to_dict() for event in self._buffer])

        # Write with or without header
        df.to_csv(
            self.filepath,
            mode="a" if self._header_written else "w",
            header=not self._header_written,
            index=False,
        )

        self._header_written = True
        self._buffer.clear()
        logger.debug(f"Flushed {len(df)} events to {self.filepath}")

    def to_dataframe(self) -> pd.DataFrame:
        """Read all events from CSV."""
        try:
            return pd.read_csv(self.filepath)
        except FileNotFoundError:
            return pd.DataFrame()

    def __del__(self):
        """Ensure buffer is flushed on destruction."""
        self.flush()

