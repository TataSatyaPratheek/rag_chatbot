"""Document processor factory."""

import logging
from pathlib import Path
from typing import Union, Optional

from mmrag.document_processing.base import DocumentProcessor

# Import legacy processors
from mmrag.document_processing.legacy.pdf import PDFProcessor
from mmrag.document_processing.legacy.ppt import PowerPointProcessor

# Import Docling processor
try:
    from mmrag.document_processing.docling.processor import DoclingDocumentProcessor
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

logger = logging.getLogger(__name__)

def get_processor(
    document_path: Union[str, Path],
    use_docling: bool = True,  # New parameter to control Docling usage
    **kwargs # Pass other processor options
) -> DocumentProcessor:
    """Get the appropriate processor for a document.

    Args:
        document_path: Path to the document.
        use_docling: Whether to use Docling processor if available.
        **kwargs: Additional options for the processor (e.g., extract_tables, advanced_table_detection).

    Returns:
        Document processor for the document type.

    Raises:
        ValueError: If no processor is available for the document type.
    """
    document_path = Path(document_path)
    suffix = document_path.suffix.lower()

    # Filter kwargs based on processor type
    pdf_kwargs = {k: v for k, v in kwargs.items() if k in ['extract_tables', 'extract_images', 'advanced_table_detection', 'enable_enhanced_visual', 'enable_llm_analysis', 'page_by_page', 'timeout_seconds', 'memory_limit_fraction']}
    ppt_kwargs = {k: v for k, v in kwargs.items() if k in ['extract_tables', 'extract_images', 'enable_llm_analysis']} # PPT doesn't have timeout/memory limits currently

    if suffix == ".pdf":
        # Try Docling first if enabled, available, and supports PDF
        if use_docling and DOCLING_AVAILABLE:
            try:
                docling_processor = DoclingDocumentProcessor(**kwargs)
                if docling_processor.supports(document_path):
                    # Filter kwargs for Docling
                    docling_kwargs = {k: v for k, v in kwargs.items() if k in ['extract_tables', 'extract_images', 'advanced_table_detection', 'enable_enhanced_visual', 'enable_llm_analysis', 'timeout_seconds', 'memory_limit_fraction']}
                    logger.info("Using Docling processor for PDF.")
                    return docling_processor
            except ImportError:
                logger.warning("Docling is enabled but not installed.")
        # Fallback to legacy PDF processor
        logger.info("Using legacy PDF processor.")
        return PDFProcessor(**pdf_kwargs)
    elif suffix in [".pptx", ".ppt"]:
        # Legacy processor for PPT
        logger.info("Using legacy PowerPoint processor.")
        return PowerPointProcessor(**ppt_kwargs)
    else:
        raise ValueError(f"Unsupported document type: {suffix}")