"""Rule loaders for loading rules from various sources."""

from typing import Any

from loguru import logger

from src.api.domain.models import VerificationStatus
from src.api.infrastructure.container import get_container
from src.api.infrastructure.repositories import RuleRepository
from src.streaming.domain.protocols import RuleLoader


class DatabaseRuleLoader(RuleLoader):
    """Load verified rules from the database."""

    async def load_rules(
        self, collection_id: int | None = None, task_id: int | None = None, **kwargs
    ) -> list[dict[str, Any]]:
        """
        Load verified rules from database.

        Args:
            collection_id: Load all rules from a collection
            task_id: Load rules from a specific task
            **kwargs: Additional filters

        Returns:
            List of rule dictionaries with 'rule_name', 'rule_body', etc.
        """
        container = get_container()
        db = container.database()

        async with db.get_async_session() as session:
            rule_repo = RuleRepository(session)

            # Fetch rules based on criteria
            if task_id:
                rules = await rule_repo.list_by_task(task_id)
            elif collection_id:
                rules = await rule_repo.list_by_collection(collection_id, limit=1000)
            else:
                raise ValueError("Must provide either collection_id or task_id")

            # Filter for verified rules only
            verified_rules = [
                {
                    "rule_name": r.rule_name,
                    "rule_body": r.rule_body,
                    "rule_description": r.rule_description,
                    "rule_reasoning": r.rule_reasoning,
                    "rule_source": r.rule_source,
                    "rule_type": r.rule_type,
                    "sensor_parsing_status": r.sensor_parsing_status.value,
                    "time_parsing_status": r.time_parsing_status.value,
                    "verification_status": r.verification_status.value,
                }
                for r in rules
                if r.verification_status == VerificationStatus.OK
            ]

            logger.info(
                f"Loaded {len(verified_rules)} verified rules "
                f"(out of {len(rules)} total) from database"
            )

            return verified_rules


class DictRuleLoader(RuleLoader):
    """Simple loader that returns pre-provided rules."""

    def __init__(self, rules: list[dict[str, Any]]):
        """Initialize with rule list."""
        self.rules = rules

    async def load_rules(self, **kwargs) -> list[dict[str, Any]]:
        """Return the provided rules."""
        logger.info(f"Loaded {len(self.rules)} rules from dict")
        return self.rules

