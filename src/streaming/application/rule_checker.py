"""Main RuleChecker application service for streaming rule evaluation."""

from typing import Any

import pandas as pd
from loguru import logger

from src.streaming.domain.models import CheckerConfig, RuleEvent, RuleMetadata
from src.streaming.domain.protocols import EventSink, StatusProvider
from src.streaming.infrastructure.buffered_status import (
    BufferedStatus,
    extract_requirements_from_rules,
)
from src.streaming.infrastructure.event_sink import InMemoryEventSink


class RuleChecker:
    """
    Efficient single-pass rule checker for time-series sensor data.

    Uses River for O(1) rolling statistics and maintains efficient buffers
    for each sensor/window combination required by rules.
    """

    def __init__(
        self,
        rules: list[dict[str, Any]],
        config: CheckerConfig | None = None,
        event_sink: EventSink | None = None,
    ):
        """
        Initialize rule checker.

        Args:
            rules: List of rule dicts with 'rule_name' and 'rule_body'
            config: Configuration for checker behavior
            event_sink: Where to store events (defaults to in-memory)
        """
        self.config = config or CheckerConfig()
        self.event_sink = event_sink or InMemoryEventSink()

        # Parse granularity
        self.granularity = pd.Timedelta(self.config.granularity)

        # Compile rules
        self.rules = self._compile_rules(rules)

        # Extract requirements and initialize buffers
        requirements = extract_requirements_from_rules(rules)
        self.status: BufferedStatus = BufferedStatus(requirements, self.granularity)

        logger.info(
            f"Initialized RuleChecker with {len(self.rules)} rules, "
            f"granularity={self.config.granularity}"
        )

    def _compile_rules(self, rules: list[dict]) -> list[RuleMetadata]:
        """Compile rule functions for execution."""
        compiled_rules = []

        for rule in rules:
            try:
                rule_body = rule["rule_body"]
                rule_name = rule["rule_name"]

                # Compile the function
                namespace = {}
                exec(rule_body, namespace)

                # Extract the function (assumes function name matches rule_name)
                if rule_name in namespace:
                    func = namespace[rule_name]
                else:
                    # Try to find any function in namespace
                    funcs = [v for v in namespace.values() if callable(v)]
                    if funcs:
                        func = funcs[0]
                    else:
                        logger.error(f"No function found in rule {rule_name}")
                        continue

                metadata = RuleMetadata(
                    rule_name=rule_name,
                    rule_body=rule_body,
                    rule_description=rule.get("rule_description", ""),
                    compiled_func=func,
                )

                compiled_rules.append(metadata)
                logger.debug(f"Compiled rule: {rule_name}")

            except Exception as e:
                logger.error(f"Failed to compile rule {rule.get('rule_name', 'unknown')}: {e}")

        return compiled_rules

    def check_dataframe(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        Single-pass check through DataFrame.

        Args:
            df: Wide-format DataFrame with timestamp index and sensor columns
            verbose: Log progress every N rows

        Returns:
            Events DataFrame with rule triggers
        """
        logger.info(f"Starting single-pass check on {len(df)} rows...")

        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not DatetimeIndex, attempting conversion")
            df.index = pd.to_datetime(df.index)

        row_count = 0
        event_count = 0

        # Single pass through data
        for timestamp, row in df.iterrows():
            # Update timestamp and clear cache
            self.status.set_timestamp(timestamp)

            # Update all buffers with new values - O(n_sensors)
            for sensor_id, value in row.items():
                if pd.notna(value):
                    self.status.update(sensor_id, float(value))
                else:
                    self.status.update(sensor_id, None)

            # Execute all rules - O(n_rules)
            for rule_meta in self.rules:
                try:
                    # Call rule function with status object
                    result = rule_meta.compiled_func(self.status)

                    if result is not None:
                        # Rule triggered!
                        event = RuleEvent(
                            timestamp=timestamp,
                            rule_name=rule_meta.rule_name,
                            triggered=True,
                            result=result,
                            sensor_values=self.status.get_snapshot(),
                            explanation=self._explain_trigger(rule_meta, result),
                            metadata={"rule_description": rule_meta.rule_description},
                        )

                        self.event_sink.write_event(event)
                        event_count += 1

                        if verbose:
                            logger.info(f"  🔔 {rule_meta.rule_name} triggered at {timestamp}")

                except Exception as e:
                    if self.config.skip_on_missing_data:
                        # Silently skip - likely missing data
                        pass
                    else:
                        logger.error(
                            f"Error executing rule {rule_meta.rule_name} at {timestamp}: {e}"
                        )

            row_count += 1
            if verbose and row_count % 100 == 0:
                logger.info(f"Processed {row_count} rows, found {event_count} events")

        logger.info(
            f"✓ Completed check: {row_count} rows processed, {event_count} events detected"
        )

        return self.event_sink.to_dataframe()

    def _explain_trigger(self, rule_meta: RuleMetadata, result: str) -> str:
        """Generate human-readable explanation for why rule triggered."""
        # Simple explanation - could be enhanced
        return f"Rule '{rule_meta.rule_name}' returned: {result}"

    def get_statistics(self) -> dict:
        """Get statistics about checker state."""
        return {
            "num_rules": len(self.rules),
            "granularity": str(self.granularity),
            "num_sensors_tracked": len(self.status.current_values),
            "total_events": len(self.event_sink) if hasattr(self.event_sink, "__len__") else None,
        }


async def create_checker_from_db(
    collection_id: int, config: CheckerConfig | None = None, event_sink: EventSink | None = None
) -> RuleChecker:
    """
    Convenience function to create RuleChecker from database rules.

    Args:
        collection_id: Collection to load rules from
        config: Checker configuration
        event_sink: Where to store events

    Returns:
        Initialized RuleChecker
    """
    from src.streaming.infrastructure.rule_loader import DatabaseRuleLoader

    loader = DatabaseRuleLoader()
    rules = await loader.load_rules(collection_id=collection_id)

    return RuleChecker(rules=rules, config=config, event_sink=event_sink)

