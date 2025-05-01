"""Adapter to make legacy document processors use LlamaParse under the hood."""

from pathlib import Path
from typing import Union, Optional

from mmrag.document_processing.base import DocumentProcessor, ProcessedDocument

try:
    from mmrag.document_processing.llamaparse.processor import LlamaParseDocumentProcessor
    LLAMAPARSE_AVAILABLE = True
except ImportError:
    LLAMAPARSE_AVAILABLE = False


class LlamaParseAdapter(DocumentProcessor):
    """Adapter that makes legacy processors use LlamaParse."""

    def __init__(self, legacy_processor, api_key: Optional[str] = None):
        """Initialize with a legacy processor to adapt."""
        if not LLAMAPARSE_AVAILABLE:
            raise ImportError("LlamaParse library is not installed. Cannot use LlamaParseAdapter.")

        self.legacy_processor = legacy_processor

        # Create LlamaParse processor with equivalent settings
        self.llamaparse_processor = LlamaParseDocumentProcessor(
            api_key=api_key,
            extract_tables=getattr(legacy_processor, 'extract_tables', True),
            extract_images=getattr(legacy_processor, 'extract_images', True),
            advanced_table_detection=getattr(legacy_processor, 'advanced_table_detection', False),
            enable_enhanced_visual=getattr(legacy_processor, 'enable_enhanced_visual', False),
            enable_llm_analysis=getattr(legacy_processor, 'enable_llm_analysis', False),
            timeout_seconds=getattr(legacy_processor, 'timeout_seconds', 60),
            memory_limit_fraction=getattr(legacy_processor, 'memory_limit_fraction', 0.5),
            # LlamaParse specific settings with reasonable defaults
            result_type="markdown",
            use_multimodal=True,
            multimodal_model="anthropic-sonnet-3.5",
        )

    def supports(self, document_path: Union[str, Path]) -> bool:
        """Check if the processor supports the given document."""
        # Use LlamaParse's support check
        return self.llamaparse_processor.supports(document_path)

    def process(self, document_path: Union[str, Path]) -> ProcessedDocument:
        """Process a document using LlamaParse but maintain compatibility."""
        # Use LlamaParse processor instead of legacy processor
        return self.llamaparse_processor.process(document_path)