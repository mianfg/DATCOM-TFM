"""River-based implementation of StatusProvider with efficient rolling windows."""

import ast
from collections import defaultdict
from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger
from river import stats, utils

from src.agent.domain.time import TimeDeltaInterval
from src.streaming.domain.models import RuleRequirement
from src.streaming.domain.protocols import StatusProvider


class BufferedStatus(StatusProvider):
    """
    Implements status.get() API using River's efficient rolling statistics.

    Maintains rolling buffers for each (sensor, window, statistic) combination
    required by the rules. Uses River for O(1) statistic retrieval.
    """

    def __init__(self, requirements: list[RuleRequirement], granularity: pd.Timedelta):
        """
        Initialize buffers based on rule requirements.

        Args:
            requirements: List of all requirements from all rules
            granularity: Time resolution of input data (e.g., pd.Timedelta('1min'))
        """
        self.granularity = granularity
        self.current_timestamp: datetime | None = None

        # Current values
        self.current_values: dict[str, float | None] = {}

        # Rolling statistics: {sensor_id: {window_key: {stat: river_object}}}
        self.rolling_stats: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))

        # Historical buffers for point queries: {sensor_id: deque}
        self.history_buffers: dict[str, list] = defaultdict(list)
        self.max_history_size = 1440  # Default: 24 hours at 1-min resolution

        # Cache for computed values within a timestamp
        self._cache: dict[tuple, float | None] = {}

        # Initialize buffers based on requirements
        self._initialize_buffers(requirements)

        logger.info(f"Initialized BufferedStatus with {len(requirements)} requirements")

    def _initialize_buffers(self, requirements: list[RuleRequirement]):
        """Pre-allocate River rolling statistics based on rule requirements."""
        for req in requirements:
            try:
                interval = TimeDeltaInterval.from_str(req.time_expression)

                if interval.is_interval():
                    # Calculate window size in number of samples
                    window_seconds = interval.range_delta().delta.total_seconds()
                    window_size = int(window_seconds / self.granularity.total_seconds())

                    if window_size <= 0:
                        logger.warning(f"Invalid window size for {req.time_expression}")
                        continue

                    # Create River stat object
                    stat_obj = self._create_river_stat(req.statistic, window_size)

                    if stat_obj:
                        window_key = req.time_expression
                        self.rolling_stats[req.sensor_id][window_key][req.statistic] = stat_obj
                        logger.debug(
                            f"Initialized {req.statistic} buffer for {req.sensor_id} "
                            f"with window {req.time_expression} (size={window_size})"
                        )

            except Exception as e:
                logger.error(f"Failed to initialize buffer for requirement {req}: {e}")

    def _create_river_stat(self, statistic: str, window_size: int, **kwargs) -> Any:
        """Create appropriate River rolling statistic object using utils.Rolling wrapper."""
        try:
            # 1. Stats that use generic Rolling wrapper
            if statistic == "mean":
                return utils.Rolling(stats.Mean(), window_size=window_size)
            elif statistic == "variance":
                return utils.Rolling(stats.Var(ddof=kwargs.get("ddof", 1)), window_size=window_size)
            elif statistic == "std":
                # compute variance, take sqrt when retrieving value externally
                return utils.Rolling(stats.Var(ddof=kwargs.get("ddof", 1)), window_size=window_size)
            elif statistic == "sum":
                return utils.Rolling(stats.Sum(), window_size=window_size)

            # 2. Stats that have dedicated rolling‐classes in River
            elif statistic == "max":
                return stats.RollingMax(window_size=window_size)
            elif statistic == "min":
                return stats.RollingMin(window_size=window_size)
            elif statistic == "abs_max":
                return stats.RollingAbsMax(window_size=window_size)
            elif statistic == "quantile":
                q = kwargs.get("q", 0.5)
                return stats.RollingQuantile(q=q, window_size=window_size)
            elif statistic == "iqr":
                q_inf = kwargs.get("q_inf", 0.25)
                q_sup = kwargs.get("q_sup", 0.75)
                return stats.RollingIQR(q_inf=q_inf, q_sup=q_sup, window_size=window_size)
            elif statistic == "mode":
                return stats.RollingMode(window_size=window_size)
            # … add additional dedicated rolling‐stat classes as available

        except Exception as e:
            logger.error(f"Failed to create River stat {statistic}: {e}")
            return None

    def update(self, sensor_id: str, value: float | None) -> None:
        """
        Update all buffers for a sensor with new value.

        This is called once per sensor per timestamp. River handles
        the efficient rolling window updates internally.
        """
        # Update current value
        self.current_values[sensor_id] = value

        if value is None:
            return  # Don't update rolling stats with None

        # Update all rolling stats for this sensor
        for window_stats in self.rolling_stats[sensor_id].values():
            for stat_obj in window_stats.values():
                stat_obj.update(value)

        # Update history buffer for point queries
        self.history_buffers[sensor_id].append(value)
        if len(self.history_buffers[sensor_id]) > self.max_history_size:
            self.history_buffers[sensor_id].pop(0)

    def set_timestamp(self, timestamp: datetime):
        """Set current timestamp and clear cache."""
        self.current_timestamp = timestamp
        self._cache.clear()

    def get(self, sensor_id: str, time_expr: str, statistic: str | None = None) -> float | None:
        """
        Get sensor value with time-based query.

        Uses cache to avoid redundant calculations for same query within timestamp.
        """
        # Check cache first
        cache_key = (sensor_id, time_expr, statistic)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            interval = TimeDeltaInterval.from_str(time_expr)

            if interval.is_point():
                # Point query: get single historical value
                result = self._get_point_value(sensor_id, interval)
            else:
                # Interval query: get statistic from rolling buffer
                result = self._get_interval_stat(sensor_id, time_expr, statistic)

            # Cache result
            self._cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"Error in status.get({sensor_id}, {time_expr}, {statistic}): {e}")
            return None

    def _get_point_value(self, sensor_id: str, interval: TimeDeltaInterval) -> float | None:
        """Get value at specific point in time."""
        if interval.start.delta.total_seconds() == 0:
            # Current value
            return self.current_values.get(sensor_id)

        # Historical value - look back in buffer
        steps_back = int(interval.start.delta.total_seconds() / self.granularity.total_seconds())

        history = self.history_buffers.get(sensor_id, [])
        if len(history) > steps_back:
            return history[-(steps_back + 1)]

        return None  # Not enough history

    def _get_interval_stat(self, sensor_id: str, time_expr: str, statistic: str | None) -> float | None:
        """Get statistic from River rolling buffer - O(1) operation!"""
        if not statistic:
            logger.warning(f"Interval query {time_expr} requires statistic")
            return None

        # Get the pre-computed River stat object
        window_stats = self.rolling_stats.get(sensor_id, {}).get(time_expr, {})
        stat_obj = window_stats.get(statistic)

        if not stat_obj:
            logger.warning(
                f"No rolling buffer found for {sensor_id}, {time_expr}, {statistic}. "
                "Was this requirement specified during initialization?"
            )
            return None

        # Get current value - O(1) thanks to River!
        try:
            value = stat_obj.get()

            # Special handling for std (take sqrt of variance)
            if statistic == "std" and value is not None:
                return value**0.5

            return value
        except Exception as e:
            logger.error(f"Error getting stat from River: {e}")
            return None

    def get_snapshot(self) -> dict[str, float]:
        """Get current values of all sensors."""
        return {k: v for k, v in self.current_values.items() if v is not None}


def extract_requirements_from_rules(rules: list[dict]) -> list[RuleRequirement]:
    """
    Extract all status.get() requirements from rules using AST parsing.

    Args:
        rules: List of rule dicts with 'rule_body' field

    Returns:
        List of unique requirements across all rules
    """
    requirements = []

    for rule in rules:
        rule_body = rule.get("rule_body", "")
        rule_name = rule.get("rule_name", "unknown")

        try:
            tree = ast.parse(rule_body)
            extractor = StatusCallExtractor()
            extractor.visit(tree)

            for sensor_id, time_expr, statistic in extractor.calls:
                if sensor_id and time_expr:
                    requirements.append(
                        RuleRequirement(sensor_id=sensor_id, time_expression=time_expr, statistic=statistic)
                    )

        except Exception as e:
            logger.error(f"Failed to extract requirements from rule {rule_name}: {e}")

    # Remove duplicates
    unique_reqs = list({(r.sensor_id, r.time_expression, r.statistic): r for r in requirements}.values())

    logger.info(f"Extracted {len(unique_reqs)} unique requirements from {len(rules)} rules")
    return unique_reqs


class StatusCallExtractor(ast.NodeVisitor):
    """AST visitor to extract status.get() calls."""

    def __init__(self):
        self.calls = []

    def visit_Call(self, node):
        """Visit Call nodes and extract status.get() calls."""
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "status"
        ):
            # Extract arguments
            sensor_id = None
            time_expr = None
            statistic = None

            if len(node.args) >= 1 and isinstance(node.args[0], ast.Constant):
                sensor_id = node.args[0].value

            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                time_expr = node.args[1].value

            if len(node.args) >= 3 and isinstance(node.args[2], ast.Constant):
                statistic = node.args[2].value

            self.calls.append((sensor_id, time_expr, statistic))

        self.generic_visit(node)
