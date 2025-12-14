"""LangGraph workflow for rule consolidation and optimization."""

import json
from typing import Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph, add_messages
from loguru import logger
from pydantic import ValidationError
from typing_extensions import Annotated

from src.agent.application.consolidation_prompts import (
    consolidation_json_fallback,
    consolidation_prompt,
)
from src.agent.domain.exceptions import (
    LLMInvocationError,
    LLMResponseParsingError,
    LLMValidationError,
)
from src.agent.domain.models import RuleConsolidations
from src.agent.domain.protocols import LLMProvider
from src.agent.domain.time import Statistic
from src.config import AppConfig

# Load configuration
_app_config = AppConfig()

# Configuration constants (can be overridden at runtime)
MAX_RULES_PER_BATCH = _app_config.consolidation.max_rules_per_batch
DEFAULT_CONFIDENCE_THRESHOLD = _app_config.consolidation.default_confidence
MIN_CONFIDENCE = _app_config.consolidation.min_confidence


class ConsolidationState(TypedDict):
    """State for rule consolidation workflow."""

    # LangGraph Studio compatibility
    messages: Annotated[list[BaseMessage], add_messages]

    # Input
    rules: list[dict[str, Any]]  # Rules to consolidate
    sensors: list[dict[str, Any]]  # Available sensors
    confidence_threshold: float  # Minimum confidence to apply changes

    # Processing state
    rule_batches: list[list[dict[str, Any]]]  # Rules grouped into batches
    consolidation_actions: list[dict[str, Any]]  # Actions suggested by LLM
    verified_consolidated_rules: list[dict[str, Any]]  # Rules that passed verification
    failed_rules: list[dict[str, Any]]  # Rules that failed verification

    # Output
    results: dict[str, Any]  # Final consolidation results
    metadata: dict[str, Any]  # Metadata about the consolidation


class RuleConsolidationWorkflow:
    """
    LangGraph workflow for consolidating and optimizing extracted rules.

    Flow: analyze_rules → consolidate_with_llm → verify_consolidated → finalize
    """

    def __init__(self, llm_provider: LLMProvider):
        """
        Initialize the consolidation workflow.

        Args:
            llm_provider: LLM provider for consolidation logic
        """
        self.llm = llm_provider.get_llm()

        # Try to create structured output LLM for consolidations
        self.structured_llm = None
        try:
            self.structured_llm = self.llm.with_structured_output(RuleConsolidations)
            logger.info("✓ Structured output supported by LLM provider for consolidation")
        except Exception as e:
            logger.warning(f"Structured output not supported by LLM provider: {e}")
            logger.info("Will use fallback JSON parsing for consolidation")

        self.graph = self._build_graph()

    def _extract_sensors_from_rule_body(self, rule_body: str) -> set[str]:
        """
        Extract sensor IDs from rule body using AST parsing.

        Args:
            rule_body: Python code of the rule

        Returns:
            Set of sensor IDs found in status.get() calls
        """
        import ast

        try:
            tree = ast.parse(rule_body)
            from src.agent.application.extraction_workflow import StatusCallExtractor

            extractor = StatusCallExtractor()
            extractor.visit(tree)

            # Extract sensor IDs from calls
            sensor_ids = {call[0] for call in extractor.calls if call[0]}
            return sensor_ids
        except Exception as e:
            logger.debug(f"Failed to extract sensors from rule body: {e}")
            return set()

    def _group_rules_into_batches(self, rules: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """
        Group rules into batches for efficient LLM processing.

        Strategy:
        1. Group rules by the sensors they reference (rules with same sensors together)
        2. Ensure no batch exceeds MAX_RULES_PER_BATCH
        3. Rules with no sensors go into their own batch

        Args:
            rules: List of rules to group

        Returns:
            List of rule batches
        """
        if len(rules) <= MAX_RULES_PER_BATCH:
            # Small enough to process as single batch
            return [rules]

        from collections import defaultdict

        # Group by sensors
        sensor_groups = defaultdict(list)

        for rule in rules:
            rule_body = rule.get("rule_body", "")
            sensors = self._extract_sensors_from_rule_body(rule_body)

            # Create a key from sorted sensor IDs
            if sensors:
                key = tuple(sorted(sensors))
            else:
                key = ("__no_sensors__",)

            sensor_groups[key].append(rule)

        # Split large groups into smaller batches
        batches = []
        for group_rules in sensor_groups.values():
            # Split group into chunks of MAX_RULES_PER_BATCH
            for i in range(0, len(group_rules), MAX_RULES_PER_BATCH):
                batch = group_rules[i : i + MAX_RULES_PER_BATCH]
                batches.append(batch)

        logger.info(
            f"Grouped {len(rules)} rules into {len(batches)} batches (max {MAX_RULES_PER_BATCH} rules per batch)"
        )

        return batches

    def _build_graph(self) -> Any:
        """Build the LangGraph workflow for consolidation."""
        workflow = StateGraph(ConsolidationState)

        # Add nodes
        workflow.add_node("analyze_rules", self._analyze_rules)
        workflow.add_node("consolidate_with_llm", self._consolidate_with_llm)
        workflow.add_node("verify_consolidated", self._verify_consolidated)
        workflow.add_node("finalize", self._finalize)

        # Define flow
        workflow.add_edge("analyze_rules", "consolidate_with_llm")
        workflow.add_edge("consolidate_with_llm", "verify_consolidated")
        workflow.add_edge("verify_consolidated", "finalize")
        workflow.add_edge("finalize", END)

        # Set entry point
        workflow.set_entry_point("analyze_rules")

        return workflow.compile()

    def _analyze_rules(self, state: ConsolidationState) -> ConsolidationState:
        """
        Analyze rules to identify candidates for consolidation.

        Groups rules into batches by sensor similarity for efficient LLM processing.
        This prevents context window overflow and improves consolidation quality.
        """
        logger.info("📊 Analyzing rules for consolidation opportunities...")

        rules = state.get("rules", [])

        if not rules:
            logger.warning("No rules to analyze")
            state["metadata"] = {"analyzed_rules": 0, "rule_batches": 0}
            state["rule_batches"] = []
            return state

        logger.info(f"Analyzing {len(rules)} rules")

        # Group rules into batches to prevent LLM context overflow
        batches = self._group_rules_into_batches(rules)

        state["rule_batches"] = batches
        state["metadata"] = {
            "analyzed_rules": len(rules),
            "rule_batches": len(batches),
            "avg_batch_size": len(rules) // len(batches) if batches else 0,
        }

        return state

    def _consolidate_with_llm(self, state: ConsolidationState) -> ConsolidationState:
        """
        Use LLM to identify redundant rules and suggest consolidations.

        Processes rules in batches to avoid context window overflow.
        Each batch is processed independently, then results are combined.

        Returns list of consolidation actions with confidence scores.
        """
        logger.info("🤖 Requesting LLM to consolidate rules...")

        rule_batches = state.get("rule_batches", [])
        sensors = state.get("sensors", [])
        confidence_threshold = state.get("confidence_threshold", 0.7)

        if not rule_batches:
            logger.warning("No rule batches to consolidate")
            state["consolidation_actions"] = []
            return state

        # Format sensors for LLM (used for all batches)
        sensors_info = "\n".join(
            [f"- sensor_id: {s['sensor_id']}, name: {s['name']}, unit: {s.get('unit', 'N/A')}" for s in sensors]
        )
        available_statistics = ", ".join(Statistic.get_available_statistics())

        # Process each batch separately
        all_consolidations = []

        for batch_idx, rules_batch in enumerate(rule_batches, 1):
            logger.info(f"Processing batch {batch_idx}/{len(rule_batches)} ({len(rules_batch)} rules)")

            # Format rules for this batch
            rules_info = json.dumps(rules_batch, indent=2)

            # Build prompt for this batch
            prompt = consolidation_prompt(
                sensors_info=sensors_info,
                available_statistics=available_statistics,
                rules_info=rules_info,
            )

            try:
                if self.structured_llm is not None:
                    # Use structured output directly
                    try:
                        validated_response: RuleConsolidations = self.structured_llm.invoke(prompt)
                    except Exception as e:
                        raise LLMInvocationError(
                            message=f"Failed to invoke LLM for batch {batch_idx}",
                            llm_provider=type(self.llm).__name__,
                            original_error=e,
                        ) from e
                else:
                    # Fallback: Request JSON format and parse manually
                    json_prompt = consolidation_json_fallback(prompt)

                    try:
                        response = self.llm.invoke(json_prompt)
                        content = response.content.strip()
                    except Exception as e:
                        raise LLMInvocationError(
                            message=f"Failed to invoke LLM for batch {batch_idx}",
                            llm_provider=type(self.llm).__name__,
                            original_error=e,
                        ) from e

                    # Extract JSON from markdown if present
                    if "```" in content:
                        import re

                        match = re.search(r"```(?:json)?\s*\n(.*?)\n```", content, re.DOTALL)
                        if match:
                            content = match.group(1).strip()

                    # Parse JSON
                    try:
                        result_dict = json.loads(content)
                    except json.JSONDecodeError as e:
                        raise LLMResponseParsingError(
                            message=f"Failed to parse LLM JSON response for batch {batch_idx}",
                            response_content=content,
                            parse_error=e,
                        ) from e

                    # Validate with Pydantic
                    try:
                        validated_response = RuleConsolidations(**result_dict)
                    except ValidationError as e:
                        validation_errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
                        raise LLMValidationError(
                            message=f"LLM response failed validation for batch {batch_idx}",
                            validation_errors=validation_errors,
                        ) from e

                # Convert to dict and add to all_consolidations
                batch_consolidations = [action.model_dump() for action in validated_response.consolidations]
                all_consolidations.extend(batch_consolidations)

                logger.info(f"✓ Batch {batch_idx}: LLM suggested {len(batch_consolidations)} consolidations")

            except (LLMInvocationError, LLMResponseParsingError, LLMValidationError) as e:
                logger.error(f"Batch {batch_idx}: {e.message}")
                if e.details:
                    logger.debug(f"Error details: {e.details}")
            except Exception as e:
                logger.error(f"Batch {batch_idx}: Unexpected error during consolidation: {e}")
                logger.exception("Full error:")

        # Filter all consolidations by confidence threshold
        filtered_consolidations = [c for c in all_consolidations if c["confidence"] >= confidence_threshold]

        logger.info(
            f"✓ Total: {len(all_consolidations)} consolidations from {len(rule_batches)} batches, "
            f"{len(filtered_consolidations)} above threshold ({confidence_threshold})"
        )

        state["consolidation_actions"] = filtered_consolidations

        return state

    def _verify_consolidated(self, state: ConsolidationState) -> ConsolidationState:
        """
        Verify consolidated rules using same verification as extraction.

        Uses AST parsing to validate syntax, sensors, time expressions, and statistics.
        """
        import ast

        from src.agent.domain.time import Statistic, TimeDeltaInterval

        logger.info("✅ Verifying consolidated rules...")

        consolidations = state.get("consolidation_actions", [])
        sensors = state.get("sensors", [])

        if not consolidations:
            state["verified_consolidated_rules"] = []
            state["failed_rules"] = []
            return state

        # Build set of valid sensor IDs
        valid_sensor_ids = {s["sensor_id"] for s in sensors} if sensors else set()
        valid_statistics = set(Statistic.get_available_statistics())

        verified_rules = []
        failed_rules = []

        for consolidation in consolidations:
            output_rule = consolidation.get("output_rule")

            # Skip remove actions (no output rule)
            if not output_rule:
                continue

            rule_body = output_rule.get("rule_body", "")
            rule_name = output_rule.get("rule_name", "unknown")

            try:
                # Step 1: Syntax validation
                tree = ast.parse(rule_body)

                # Step 2-4: Validate status.get() calls
                from src.agent.application.extraction_workflow import StatusCallExtractor

                extractor = StatusCallExtractor()
                extractor.visit(tree)
                status_calls = extractor.calls

                all_valid = True

                for sensor_id, time_expr, statistic in status_calls:
                    # Validate sensor ID
                    if sensor_id not in valid_sensor_ids:
                        logger.warning(f"  ⚠️  Invalid sensor '{sensor_id}' in consolidated rule '{rule_name}'")
                        all_valid = False
                        break

                    # Validate time expression
                    try:
                        time_interval = TimeDeltaInterval.from_str(time_expr)

                        # Validate statistic
                        if time_interval.is_interval():
                            if not statistic or statistic not in valid_statistics:
                                logger.warning(f"  ⚠️  Invalid/missing statistic for interval in rule '{rule_name}'")
                                all_valid = False
                                break
                        else:
                            if statistic:
                                logger.warning(f"  ⚠️  Point query should not have statistic in rule '{rule_name}'")
                                all_valid = False
                                break

                    except Exception as e:
                        logger.warning(f"  ⚠️  Invalid time expression '{time_expr}' in rule '{rule_name}': {e}")
                        all_valid = False
                        break

                if all_valid:
                    # Add consolidation metadata to rule
                    output_rule["consolidation_confidence"] = consolidation.get("confidence")
                    output_rule["consolidation_reasoning"] = consolidation.get("reasoning")
                    output_rule["consolidated_from_ids"] = consolidation.get("input_rule_ids")
                    output_rule["is_consolidated"] = True
                    output_rule["lifecycle_status"] = "consolidated"

                    # Set all verification statuses to OK
                    output_rule["sensor_parsing_status"] = "ok"
                    output_rule["time_parsing_status"] = "ok"
                    output_rule["verification_status"] = "ok"

                    verified_rules.append(output_rule)
                    logger.info(f"✅ Consolidated rule '{rule_name}' passed verification")
                else:
                    failed_rules.append(
                        {
                            "rule": output_rule,
                            "consolidation": consolidation,
                            "reason": "Validation failed",
                        }
                    )

            except SyntaxError as e:
                logger.warning(f"  ⚠️  Syntax error in consolidated rule '{rule_name}': {e}")
                failed_rules.append(
                    {"rule": output_rule, "consolidation": consolidation, "reason": f"Syntax error: {e}"}
                )
            except Exception as e:
                logger.error(f"  ⚠️  Error verifying rule '{rule_name}': {e}")
                failed_rules.append({"rule": output_rule, "consolidation": consolidation, "reason": str(e)})

        logger.info(f"✓ Verification complete: {len(verified_rules)} passed, {len(failed_rules)} failed")

        state["verified_consolidated_rules"] = verified_rules
        state["failed_rules"] = failed_rules

        return state

    def _finalize(self, state: ConsolidationState) -> ConsolidationState:
        """
        Finalize consolidation results.

        Prepares summary and metadata for storage.
        """
        logger.info("📝 Finalizing consolidation results...")

        rules = state.get("rules", [])
        consolidations = state.get("consolidation_actions", [])
        verified_rules = state.get("verified_consolidated_rules", [])
        failed_rules = state.get("failed_rules", [])

        # Count actions by type
        remove_count = sum(1 for c in consolidations if c.get("action_type") == "remove")
        merge_count = sum(1 for c in consolidations if c.get("action_type") == "merge")
        simplify_count = sum(1 for c in consolidations if c.get("action_type") == "simplify")

        # Get IDs of rules that will be superseded
        superseded_ids = set()
        for consolidation in consolidations:
            if consolidation.get("output_rule"):  # merge or simplify
                superseded_ids.update(consolidation.get("input_rule_ids", []))

        # Build results
        results = {
            "input_count": len(rules),
            "output_count": len(verified_rules),
            "superseded_count": len(superseded_ids),
            "remove_count": remove_count,
            "merge_count": merge_count,
            "simplify_count": simplify_count,
            "failed_count": len(failed_rules),
            "consolidated_rules": verified_rules,
            "superseded_rule_ids": list(superseded_ids),
            "failed_consolidations": failed_rules,
        }

        state["results"] = results

        logger.info(
            f"✓ Consolidation complete: {len(rules)} → {len(verified_rules)} rules "
            f"({len(superseded_ids)} superseded, {len(failed_rules)} failed)"
        )

        return state

    def run(
        self,
        rules: list[dict[str, Any]],
        sensors: list[dict[str, Any]],
        confidence_threshold: float = 0.7,
        config: dict | None = None,
    ) -> dict:
        """
        Run the consolidation workflow synchronously.

        Args:
            rules: List of rules to consolidate
            sensors: List of available sensors
            confidence_threshold: Minimum confidence to apply consolidation
            config: Optional LangGraph config

        Returns:
            State dict with results and metadata
        """
        initial_state = {
            "rules": rules,
            "sensors": sensors,
            "confidence_threshold": confidence_threshold,
            "rule_batches": [],  # Will be populated by _analyze_rules
            "consolidation_actions": [],
            "verified_consolidated_rules": [],
            "failed_rules": [],
            "results": {},
            "metadata": {},
            "messages": [],
        }

        result = self.graph.invoke(initial_state, config=config)
        return result

    async def arun(
        self,
        rules: list[dict[str, Any]],
        sensors: list[dict[str, Any]],
        confidence_threshold: float = 0.7,
        config: dict | None = None,
    ) -> dict:
        """
        Run the consolidation workflow asynchronously.

        This is the preferred method for async contexts like background jobs,
        as it doesn't block the event loop during LLM calls.

        Args:
            rules: List of rules to consolidate
            sensors: List of available sensors
            confidence_threshold: Minimum confidence to apply consolidation
            config: Optional LangGraph config

        Returns:
            State dict with results and metadata
        """
        initial_state = {
            "rules": rules,
            "sensors": sensors,
            "confidence_threshold": confidence_threshold,
            "rule_batches": [],  # Will be populated by _analyze_rules
            "consolidation_actions": [],
            "verified_consolidated_rules": [],
            "failed_rules": [],
            "results": {},
            "metadata": {},
            "messages": [],
        }

        result = await self.graph.ainvoke(initial_state, config=config)
        return result
