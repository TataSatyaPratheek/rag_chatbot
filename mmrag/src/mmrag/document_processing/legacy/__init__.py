"""Legacy document processing components."""

from .advanced_table import CascadeTabNetDetector # Advanced table detection
from .cache import CachedDocumentProcessor # Caching wrapper
from .enhanced_visual import EnhancedVisualProcessor # Enhanced visual processing
from .pdf import PDFProcessor
from .ppt import PowerPointProcessor
from .table import TableDetector
from .visual import VisualElementProcessor


__all__ = [
    # --- Main Processors ---
    "PDFProcessor",
    "PowerPointProcessor",
    
    # --- Specialized Components (Potentially for advanced use/extension) ---
    "TableDetector", # Basic heuristic detector
    "CascadeTabNetDetector", # Advanced ML-based detector
    "VisualElementProcessor", # Basic image extraction
    "EnhancedVisualProcessor", # Advanced image/chart analysis
    
    # --- Utilities & Factory ---
    "CachedDocumentProcessor",
]
