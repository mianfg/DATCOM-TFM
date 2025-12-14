"""Custom exceptions for the API layer."""


class APIException(Exception):
    """Base exception for all API errors."""

    def __init__(self, message: str, details: dict | None = None):
        """
        Initialize API exception.
        
        Args:
            message: Human-readable error message
            details: Optional dict with additional error context
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ResourceNotFoundException(APIException):
    """Raised when a requested resource is not found."""

    pass


class ConsolidationError(APIException):
    """Base exception for consolidation-related errors."""

    pass


class ConsolidationJobNotFoundError(ConsolidationError, ResourceNotFoundException):
    """Raised when a consolidation job is not found."""

    def __init__(self, job_id: int):
        super().__init__(
            message=f"Consolidation job {job_id} not found",
            details={"consolidation_job_id": job_id}
        )


class ConsolidationAlreadyExecutedError(ConsolidationError):
    """Raised when attempting to execute a consolidation job that has already been executed."""

    def __init__(self, job_id: int, existing_rules_count: int):
        super().__init__(
            message=f"Consolidation job {job_id} has already been executed ({existing_rules_count} rules exist)",
            details={
                "consolidation_job_id": job_id,
                "existing_rules_count": existing_rules_count
            }
        )


class DatabaseError(APIException):
    """Raised for database-related errors."""

    pass


class TransactionError(DatabaseError):
    """Raised when a database transaction fails."""

    pass

