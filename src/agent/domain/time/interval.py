from typing import Self

from pydantic import BaseModel

from src.agent.domain.time.delta import TimeDelta


class TimeDeltaInterval(BaseModel):
    start: TimeDelta
    end: TimeDelta | None = None

    @classmethod
    def from_str(cls, delta_interval_str: str, strict: bool = False) -> Self:
        delta_interval_str = delta_interval_str.strip()

        if not delta_interval_str:
            raise ValueError("Empty time delta interval string")

        if ":" not in delta_interval_str:
            # Point in time: "5m" or "1h30m"
            start = TimeDelta(delta_interval_str, strict=strict)
            return cls(start=start, end=None)
        else:
            # Time interval: "5m:1m" or "1h30m:"
            start_str, end_str = delta_interval_str.split(":", 1)

            if not start_str.strip():
                raise ValueError("Start time cannot be empty")

            start = TimeDelta(start_str, strict=strict)
            end = TimeDelta(end_str, strict=strict) if end_str.strip() else TimeDelta("0")

            return cls(start=start, end=end)

    def is_interval(self) -> bool:
        return self.end is not None

    def is_point(self) -> bool:
        return self.end is None

    def range_delta(self) -> TimeDelta:
        if self.end is None:
            return TimeDelta("0")  # Zero duration for points

        return self.start - self.end

    def __str__(self) -> str:
        if self.end is None:
            return str(self.start)
        else:
            return f"{self.start}:{self.end}"
