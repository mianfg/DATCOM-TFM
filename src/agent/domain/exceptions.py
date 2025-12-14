"""Custom exceptions for the Agent layer."""


class AgentException(Exception):
    """Base exception for all agent errors."""

    def __init__(self, message: str, details: dict | None = None):
        """
        Initialize agent exception.
        
        Args:
            message: Human-readable error message
            details: Optional dict with additional error context
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)


class WorkflowException(AgentException):
    """Base exception for workflow-related errors."""

    pass


class WorkflowExecutionError(WorkflowException):
    """Raised when a workflow execution fails."""

    pass


class LLMError(WorkflowException):
    """Base exception for LLM-related errors."""

    pass


class LLMInvocationError(LLMError):
    """Raised when LLM invocation fails."""

    def __init__(self, message: str, llm_provider: str | None = None, original_error: Exception | None = None):
        details = {}
        if llm_provider:
            details["llm_provider"] = llm_provider
        if original_error:
            details["original_error"] = str(original_error)
            details["original_error_type"] = type(original_error).__name__
        super().__init__(message, details)


class LLMResponseParsingError(LLMError):
    """Raised when LLM response cannot be parsed."""

    def __init__(self, message: str, response_content: str | None = None, parse_error: Exception | None = None):
        details = {}
        if response_content:
            details["response_preview"] = response_content[:500]
        if parse_error:
            details["parse_error"] = str(parse_error)
            details["parse_error_type"] = type(parse_error).__name__
        super().__init__(message, details)


class LLMValidationError(LLMError):
    """Raised when LLM response fails validation."""

    def __init__(self, message: str, validation_errors: list[str] | None = None):
        details = {}
        if validation_errors:
            details["validation_errors"] = validation_errors
        super().__init__(message, details)


class ConsolidationWorkflowError(WorkflowException):
    """Raised when consolidation workflow fails."""

    pass


class RuleVerificationError(WorkflowException):
    """Raised when rule verification fails."""

    def __init__(self, rule_name: str, reason: str, details: dict | None = None):
        full_details = {"rule_name": rule_name, "reason": reason}
        if details:
            full_details.update(details)
        super().__init__(
            message=f"Rule '{rule_name}' verification failed: {reason}",
            details=full_details
        )

