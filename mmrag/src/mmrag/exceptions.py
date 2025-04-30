"""Custom exceptions for the mmrag package."""

class ProcessingError(Exception):
    """Base class for document processing errors."""
    def __init__(self, message="An error occurred during document processing."):
        super().__init__(message)

class ProcessingTimeoutError(ProcessingError):
    """Raised when document processing exceeds the time limit."""
    def __init__(self, message="Document processing timed out."):
        super().__init__(message)


class MemoryLimitExceededError(ProcessingError):
    """Raised when document processing exceeds the memory limit."""
    def __init__(self, message="Memory limit exceeded during document processing."):
        super().__init__(message)