"""Use case for rule consolidation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.agent.domain.protocols import LLMProvider

if TYPE_CHECKING:
    from src.agent.application.consolidation_workflow import RuleConsolidationWorkflow


class RuleConsolidationUseCase:
    """
    Provides primitives for rule consolidation.

    The API layer orchestrates this to consolidate rules from
    collections or processing jobs.
    """

    def __init__(self, llm_provider: LLMProvider):
        """
        Initialize use case with LLM provider.

        Args:
            llm_provider: LLM provider for consolidation
        """
        self.llm_provider = llm_provider
        self._workflow = None

    def get_workflow(self) -> "RuleConsolidationWorkflow":
        """
        Get the LangGraph workflow for consolidating rules.

        Returns:
            Initialized RuleConsolidationWorkflow
        """
        return self._ensure_workflow()

    def _ensure_workflow(self) -> "RuleConsolidationWorkflow":
        """
        Lazy-load workflow.

        Returns:
            Initialized RuleConsolidationWorkflow instance
        """
        if self._workflow is None:
            from src.agent.application.consolidation_workflow import RuleConsolidationWorkflow

            self._workflow = RuleConsolidationWorkflow(self.llm_provider)
        return self._workflow
