"""Adapter to make legacy document processors use Docling under the hood."""

from pathlib import Path
from typing import Union, Optional

from mmrag.document_processing.base import DocumentProcessor, ProcessedDocument

try:
    from mmrag.document_processing.docling.processor import DoclingDocumentProcessor
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False


class DoclingAdapter(DocumentProcessor):
    """Adapter that makes legacy processors use Docling."""

    def __init__(self, legacy_processor):
        """Initialize with a legacy processor to adapt."""
        if not DOCLING_AVAILABLE:
            raise ImportError("Docling library is not installed. Cannot use DoclingAdapter.")

        self.legacy_processor = legacy_processor

        # Create Docling processor with equivalent settings
        self.docling_processor = DoclingDocumentProcessor(
            extract_tables=getattr(legacy_processor, 'extract_tables', True),
            extract_images=getattr(legacy_processor, 'extract_images', True),
            advanced_table_detection=getattr(legacy_processor, 'advanced_table_detection', False),
            enable_enhanced_visual=getattr(legacy_processor, 'enable_enhanced_visual', False),
            enable_llm_analysis=getattr(legacy_processor, 'enable_llm_analysis', False),
        )

    def supports(self, document_path: Union[str, Path]) -> bool:
        """Check if the processor supports the given document."""
        # Use Docling's support check
        return self.docling_processor.supports(document_path)

    def process(self, document_path: Union[str, Path]) -> ProcessedDocument:
        """Process a document using Docling but maintain compatibility."""
        # Use Docling processor instead of legacy processor
        return self.docling_processor.process(document_path)