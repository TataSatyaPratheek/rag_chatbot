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

# Import LlamaParse processor
try:
    from mmrag.document_processing.llamaparse.processor import LlamaParseDocumentProcessor
    from mmrag.document_processing.llamaparse.utils import check_llamaparse_api_key
    LLAMAPARSE_AVAILABLE = True
except ImportError:
    LLAMAPARSE_AVAILABLE = False

logger = logging.getLogger(__name__)

def get_processor(
    document_path: Union[str, Path],
    use_docling: bool = False,  # Control Docling usage
    use_llamaparse: bool = False,  # Control LlamaParse usage
    llamaparse_api_key: Optional[str] = None,  # LlamaParse API key
    **kwargs # Pass other processor options
) -> DocumentProcessor:
    """Get the appropriate processor for a document.

    Args:
        document_path: Path to the document.
        use_docling: Whether to use Docling processor if available.
        use_llamaparse: Whether to use LlamaParse processor if available.
        llamaparse_api_key: API key for LlamaParse (optional).
        **kwargs: Additional options for the processor.

    Returns:
        Document processor for the document type.

    Raises:
        ValueError: If no processor is available for the document type.
    """
    document_path = Path(document_path)
    suffix = document_path.suffix.lower()

    # Filter kwargs based on processor type
    pdf_kwargs = {k: v for k, v in kwargs.items() if k in ['extract_tables', 'extract_images', 'advanced_table_detection', 'enable_enhanced_visual', 'enable_llm_analysis', 'page_by_page', 'timeout_seconds', 'memory_limit_fraction']}
    ppt_kwargs = {k: v for k, v in kwargs.items() if k in ['extract_tables', 'extract_images', 'enable_llm_analysis']} 
    docling_kwargs = {k: v for k, v in kwargs.items() if k in ['extract_tables', 'extract_images', 'advanced_table_detection', 'enable_enhanced_visual', 'enable_llm_analysis', 'timeout_seconds', 'memory_limit_fraction']}
    llamaparse_kwargs = {k: v for k, v in kwargs.items() if k in ['extract_tables', 'extract_images', 'advanced_table_detection', 'enable_enhanced_visual', 'enable_llm_analysis', 'timeout_seconds', 'memory_limit_fraction', 'result_type', 'use_multimodal', 'multimodal_model', 'target_pages', 'bbox_params']}

    # Try LlamaParse first if enabled and available
    if use_llamaparse and LLAMAPARSE_AVAILABLE:
        try:
            # Check if API key is available
            if llamaparse_api_key or check_llamaparse_api_key():
                llamaparse_processor = LlamaParseDocumentProcessor(api_key=llamaparse_api_key, **llamaparse_kwargs)
                if llamaparse_processor.supports(document_path):
                    logger.info("Using LlamaParse processor.")
                    return llamaparse_processor
                else:
                    logger.warning(f"LlamaParse does not support {suffix} files.")
            else:
                logger.warning("LlamaParse is enabled but no API key found.")
        except ImportError:
            logger.warning("LlamaParse is enabled but not installed.")
    
    # Try Docling second if enabled, available, and LlamaParse wasn't used
    if use_docling and DOCLING_AVAILABLE:
        try:
            docling_processor = DoclingDocumentProcessor(**docling_kwargs)
            if docling_processor.supports(document_path):
                logger.info("Using Docling processor.")
                return docling_processor
        except ImportError:
            logger.warning("Docling is enabled but not installed.")

    # Fallback to legacy processors
    if suffix == ".pdf":
        logger.info("Using legacy PDF processor.")
        return PDFProcessor(**pdf_kwargs)
    elif suffix in [".pptx", ".ppt"]:
        logger.info("Using legacy PowerPoint processor.")
        return PowerPointProcessor(**ppt_kwargs)
    else:
        raise ValueError(f"Unsupported document type: {suffix}")