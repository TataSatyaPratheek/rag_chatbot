"""Document processing components for mmrag."""

from mmrag.document_processing.base import (
    BoundingBox,
    ChartElement,
    DocumentElement,
    DocumentProcessor,
    ImageElement,
    ProcessedDocument,
    TableElement,
    TextElement,
)
from mmrag.document_processing.factory import get_processor
from mmrag.exceptions import (
    ProcessingError, ProcessingTimeoutError, MemoryLimitExceededError
)

__all__ = [
    # --- Base Classes & Elements ---
    "BoundingBox",
    "ChartElement",
    "DocumentElement",
    "DocumentProcessor",
    "ImageElement",
    "ProcessedDocument",
    "TableElement",
    "TextElement",
    # --- Factory ---
    "get_processor",
    # --- Exceptions ---
    "ProcessingError", "ProcessingTimeoutError", "MemoryLimitExceededError",
]