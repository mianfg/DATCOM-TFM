from enum import StrEnum


class Statistic(StrEnum):
    """Statistical aggregation functions for time-based queries."""

    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    STD = "std"
    VARIANCE = "variance"

    @classmethod
    def get_available_statistics(cls) -> list[str]:
        """Return list of available statistics for LLM prompts."""
        return [stat.value for stat in cls]
