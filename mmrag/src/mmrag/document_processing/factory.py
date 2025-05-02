"""Document processor factory."""

import logging
import asyncio
from pathlib import Path
from typing import Union, Optional, List, Callable, Awaitable

from mmrag.document_processing.base import DocumentProcessor, ProcessedDocument

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

# Import OpenParse processor
try:
    from mmrag.document_processing.openparse.processor import OpenParseDocumentProcessor
    OPENPARSE_AVAILABLE = True
except ImportError:
    LLAMAPARSE_AVAILABLE = False

logger = logging.getLogger(__name__)

def get_processor(
    document_path: Union[str, Path],
    use_docling: bool = False,  # Control Docling usage
    use_llamaparse: bool = False,  # Control LlamaParse usage
    llamaparse_api_key: Optional[str] = None,  # LlamaParse API key
    use_openparse: bool = False, # Control OpenParse usage
    openai_api_key: Optional[str] = None, # OpenAI key for OpenParse semantic processing
    **kwargs # Pass other processor options
) -> DocumentProcessor:
    """Get the appropriate processor for a document.

    Args:
        document_path: Path to the document.
        use_docling: Whether to use Docling processor if available.
        use_llamaparse: Whether to use LlamaParse processor if available.
        use_openparse: Whether to use OpenParse processor if available.
        openai_api_key: OpenAI API key for OpenParse semantic processing.
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
    openparse_kwargs = {k: v for k, v in kwargs.items() if k in ['extract_tables', 'extract_images', 'advanced_table_detection', 'enable_enhanced_visual', 'enable_llm_analysis', 'use_semantic_processing', 'timeout_seconds']}

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
    # Try OpenParse if enabled and available (and others weren't chosen)
    if use_openparse and OPENPARSE_AVAILABLE:
        try:
            openparse_processor = OpenParseDocumentProcessor(
                openai_api_key=openai_api_key,
                **openparse_kwargs
            )
            if openparse_processor.supports(document_path):
                logger.info("Using OpenParse processor.")
                return openparse_processor
            else:
                logger.warning(f"OpenParse does not support {suffix} files (or check implementation).")
        except ImportError:
            logger.warning("OpenParse is enabled but not installed.")

    if suffix == ".pdf":
        logger.info("Using legacy PDF processor.")
        return PDFProcessor(**pdf_kwargs)
    elif suffix in [".pptx", ".ppt"]:
        logger.info("Using legacy PowerPoint processor.")
        return PowerPointProcessor(**ppt_kwargs)
    else:
        raise ValueError(f"Unsupported document type: {suffix}")

async def process_multiple(
    document_paths: List[Union[str, Path]],
    progress_callback: Optional[Callable[[str, Optional[float]], Awaitable[None]]] = None, # Optional overall callback
    **kwargs # Passed to get_processor
) -> List[ProcessedDocument]:
    """Process multiple documents concurrently.

    Args:
        document_paths: List of paths to the documents.
        progress_callback: Callback for overall progress (e.g., "Processed 1 of 5").
        **kwargs: Options passed to get_processor for each document.

    Returns:
        List of ProcessedDocument objects. Errors during processing are logged.
    """
    tasks = []
    for i, doc_path_raw in enumerate(document_paths):
        doc_path = Path(doc_path_raw)
        # Note: Individual processor progress is handled within its process method.
        # Here we just create the tasks.
        processor = get_processor(doc_path, **kwargs)
        tasks.append(asyncio.create_task(processor.process(doc_path))) # Add individual callbacks if needed per task

    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_docs = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Error processing document {document_paths[i]}: {result}")
        elif isinstance(result, ProcessedDocument):
            processed_docs.append(result)
            if progress_callback:
                await progress_callback(f"Completed {i+1}/{len(document_paths)}", (i+1)/len(document_paths))
        else:
            logger.warning(f"Unexpected result type for document {document_paths[i]}: {type(result)}")

    return processed_docs