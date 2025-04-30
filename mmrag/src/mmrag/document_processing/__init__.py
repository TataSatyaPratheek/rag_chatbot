from mmrag.document_processing.base import (
    BoundingBox,
    ChartElement,
    DocumentProcessor, # Added base processor class
    DocumentElement,
    DocumentProcessor,
    ImageElement,
    ProcessedDocument,
    TableElement,
    TextElement,
)
# Import exceptions relevant to processing
from mmrag.exceptions import (
    ProcessingError, ProcessingTimeoutError, MemoryLimitExceededError
)

from mmrag.document_processing.advanced_table import CascadeTabNetDetector # Advanced table detection
from mmrag.document_processing.cache import CachedDocumentProcessor # Caching wrapper
from mmrag.document_processing.enhanced_visual import EnhancedVisualProcessor # Enhanced visual processing
from mmrag.document_processing.factory import get_processor # Processor factory
from mmrag.document_processing.pdf import PDFProcessor
from mmrag.document_processing.ppt import PowerPointProcessor
from mmrag.document_processing.table import TableDetector
from mmrag.document_processing.visual import VisualElementProcessor


__all__ = [
    # --- Base Classes & Elements ---
    "BoundingBox",
    "ChartElement",
    "DocumentElement",
    "DocumentProcessor",
    "ImageElement",
    "ProcessedDocument",
    "TableDetector",
    "TableElement",
    "TextElement", 
    
    # --- Main Processors ---
    "PDFProcessor",
    "PowerPointProcessor",
    
    # --- Specialized Components (Potentially for advanced use/extension) ---
    "TableDetector", # Basic heuristic detector
    "CascadeTabNetDetector",
    "EnhancedVisualProcessor",
    "VisualElementProcessor",
    
    # --- Utilities & Factory ---
    "CachedDocumentProcessor",
    "get_processor",
    
    # --- Exceptions ---
    "ProcessingError", "ProcessingTimeoutError", "MemoryLimitExceededError",
]
