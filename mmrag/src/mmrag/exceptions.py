"""Custom exceptions for the mmrag package."""

class ProcessingError(Exception):
    """Base class for document processing errors."""
    def __init__(self, message="An error occurred during document processing.", user_message: str = None):
        super().__init__(message)
        # User-friendly message for UI display
        self.user_message = user_message or message

class ProcessingTimeoutError(ProcessingError):
    """Raised when document processing exceeds the time limit."""
    def __init__(self, message="Document processing timed out.", timeout_seconds: int = None):
        user_message = f"Processing took too long (limit: {timeout_seconds}s). Try a smaller document or increase the timeout." if timeout_seconds else "Processing took too long."
        super().__init__(message, user_message=user_message)


class MemoryLimitExceededError(ProcessingError):
    """Raised when document processing exceeds the memory limit."""
    def __init__(self, message="Memory limit exceeded during document processing.", usage_mb: float = None, limit_mb: float = None):
        if usage_mb is not None and limit_mb is not None:
            user_message = f"Document processing requires too much memory ({usage_mb:.1f}MB > {limit_mb:.1f}MB limit)."
        else:
            user_message = "Document processing requires too much memory."
        super().__init__(message, user_message=user_message)