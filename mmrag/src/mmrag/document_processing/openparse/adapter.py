"""Adapter to make legacy document processors use OpenParse under the hood."""

import logging
from pathlib import Path
from typing import Union, Optional

from mmrag.document_processing.base import DocumentProcessor, ProcessedDocument

try:
    from mmrag.document_processing.openparse.processor import OpenParseDocumentProcessor
    OPENPARSE_AVAILABLE = True
except ImportError:
    OPENPARSE_AVAILABLE = False

logger = logging.getLogger(__name__)


class OpenParseAdapter(DocumentProcessor):
    """Adapter that makes legacy processors use OpenParse."""

    def __init__(
        self,
        legacy_processor, # The legacy processor instance being adapted
        use_semantic_processing: bool = False,
        openai_api_key: Optional[str] = None,
    ):
        """Initialize with a legacy processor to adapt."""
        if not OPENPARSE_AVAILABLE:
            raise ImportError("OpenParse library is not installed. Cannot use OpenParseAdapter.")

        self.legacy_processor = legacy_processor

        # Create OpenParse processor with equivalent settings derived from the legacy one
        self.openparse_processor = OpenParseDocumentProcessor(
            extract_tables=getattr(legacy_processor, 'extract_tables', True),
            extract_images=getattr(legacy_processor, 'extract_images', True), # Note: Limited support in OpenParse
            advanced_table_detection=getattr(legacy_processor, 'advanced_table_detection', False),
            enable_enhanced_visual=getattr(legacy_processor, 'enable_enhanced_visual', False), # Note: Limited support
            enable_llm_analysis=getattr(legacy_processor, 'enable_llm_analysis', False), # LLM analysis is post-processing
            use_semantic_processing=use_semantic_processing,
            openai_api_key=openai_api_key,
        )

    def supports(self, document_path: Union[str, Path]) -> bool:
        """Check if the processor supports the given document using OpenParse's check."""
        return self.openparse_processor.supports(document_path)

    def process(self, document_path: Union[str, Path]) -> ProcessedDocument:
        """Process a document using OpenParse."""
        logger.info(f"Processing {document_path} using OpenParse via adapter.")
        return self.openparse_processor.process(document_path)