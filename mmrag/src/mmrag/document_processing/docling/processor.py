"""Document processing using Docling."""

import uuid
import hashlib
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Awaitable
import os

# Corrected imports based on docling structure
try:
    # Corrected imports based on docling structure
    from docling.document_converter import DocumentConverter
    from docling_core.types import DoclingDocument
    DOCLING_INSTALLED = True
except ImportError:
    DOCLING_INSTALLED = False
    # Define dummy classes if docling is not installed to avoid runtime errors on import
    class DocumentConverter: pass # Match the corrected import
    class DoclingDocument: pass # Match the corrected import
    class DoclingTextElement: pass
    class DoclingTableElement: pass
    class DoclingImageElement: pass
    class DoclingChartElement: pass

import psutil # Added for memory check
from mmrag.exceptions import MemoryLimitExceededError # noqa: E402

from mmrag.document_processing.base import ( # noqa: E402
    DocumentProcessor, ProcessedDocument, BoundingBox,
    TextElement, TableElement, ImageElement, ChartElement
)
from mmrag.exceptions import ProcessingError, ProcessingTimeoutError # noqa: E402
from mmrag.document_processing.docling.converter import convert_docling_element # noqa: E402

logger = logging.getLogger(__name__)


class DoclingDocumentProcessor(DocumentProcessor):
    """Document processor using Docling."""

    def __init__(
        self,
        extract_tables: bool = True,
        extract_images: bool = True,
        advanced_table_detection: bool = True,
        enable_enhanced_visual: bool = True,
        enable_llm_analysis: bool = False,
        timeout_seconds: int = 30,
        memory_limit_fraction: float = 0.5, # Added memory limit fraction
    ):
        """Initialize the Docling processor.

        Args:
            extract_tables: Whether to extract tables.
            extract_images: Whether to extract images.
            advanced_table_detection: Whether to use advanced table detection.
            enable_enhanced_visual: Whether to use enhanced visual processing (charts).
            enable_llm_analysis: Whether to enable LLM analysis.
            timeout_seconds: Timeout for processing in seconds.
            memory_limit_fraction: Fraction of available memory to allow usage.
        """
        if not DOCLING_INSTALLED:
            raise ImportError("Docling library is not installed. Cannot use DoclingDocumentProcessor.")

        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.advanced_table_detection = advanced_table_detection
        self.enable_enhanced_visual = enable_enhanced_visual
        self.enable_llm_analysis = enable_llm_analysis
        self.timeout_seconds = timeout_seconds
        self.memory_limit_fraction = memory_limit_fraction # Store memory limit fraction

        # Initialize Docling processor with appropriate settings
        self.docling_processor = DocumentConverter( # Use the corrected class name
            extract_tables=extract_tables,
            extract_images=extract_images,
            extract_charts=enable_enhanced_visual,
            advanced_table_detection=advanced_table_detection,
            # timeout_seconds=timeout_seconds, # DocumentConverter might not take this directly
        )

    def supports(self, document_path: Union[str, Path]) -> bool:
        """Check if the processor supports the given document type."""
        document_path = Path(document_path)
        # Docling supports a variety of formats - check with Docling's capabilities
        # Assuming DoclingProcessor has a 'supports' method or similar check
        # For now, let's list common types Docling might support
        return document_path.suffix.lower() in [".pdf", ".pptx", ".ppt", ".docx", ".doc"]
    
    async def process(
        self,
        document_path: Union[str, Path],
        progress_callback: Optional[Callable[[str, Optional[float]], Awaitable[None]]] = None
    ) -> ProcessedDocument:
        """Process a document and extract elements."""
        document_path = Path(document_path)
        async def _update_progress(message: str, progress: Optional[float] = None):
            if progress_callback: await progress_callback(message, progress)

        if not self.supports(document_path):
            raise ValueError(f"Unsupported document type: {document_path.suffix}")

        try:
            # --- Added: Memory Check before calling Docling ---
            process = psutil.Process(os.getpid())
            initial_available_memory = psutil.virtual_memory().available
            memory_limit_bytes = initial_available_memory * self.memory_limit_fraction
            current_rss = process.memory_info().rss
            if current_rss > memory_limit_bytes:
                 raise MemoryLimitExceededError(
                     f"Memory usage exceeded limit before calling Docling.",
                     usage_mb=current_rss / (1024**2),
                     limit_mb=memory_limit_bytes / (1024**2)
                 )
            # --- End Added ---

            await _update_progress(f"Processing with Docling: {document_path.name}...")
            # Run synchronous Docling processing in a thread (already done)
            docling_doc: DoclingDocument = await asyncio.to_thread(
                self.docling_processor.process, document_path
            )
            await _update_progress("Converting Docling elements...")

            elements = [
                mmrag_element for i, docling_element in enumerate(docling_doc.elements)
                if (mmrag_element := convert_docling_element(docling_element, i)) is not None
            ]
            document_id = self._generate_document_id(document_path)
            processed_document = ProcessedDocument(
                document_id=document_id,
                filename=document_path.name, # Use corrected type hint
                doc_type=document_path.suffix.lower().lstrip("."),
                elements=elements,
                metadata=docling_doc.metadata if hasattr(docling_doc, 'metadata') else {},
            )

            if self.enable_llm_analysis:
                await _update_progress("Performing LLM analysis...")
                await _update_progress("Performing LLM analysis...", None)
                # LLM analysis logic remains the same as in PDFProcessor
                try:
                    from mmrag.llm.content_understanding import ContentUnderstanding
                    analyzer = ContentUnderstanding()
                    analysis = await analyzer.analyze_document(processed_document) # Make call async
                    processed_document.analysis = analysis
                except ImportError:
                    logger.warning("LLM analysis requested but mmrag.llm is not available")
                except Exception as e:
                    logger.error(f"Error during LLM analysis: {e}")

            return processed_document

        except Exception as e:
            # Check if the exception string indicates a timeout
            if "timeout" in str(e).lower() or isinstance(e, asyncio.TimeoutError):
                raise ProcessingTimeoutError(f"Processing timed out: {e}", timeout_seconds=self.timeout_seconds) from e
            elif isinstance(e, MemoryError) or "memory" in str(e).lower(): # Catch potential MemoryError from underlying libs
                raise MemoryLimitExceededError(f"Memory limit likely exceeded: {e}") from e
            else:
                raise ProcessingError(f"Error processing document with Docling: {e}") from e

    def _generate_document_id(self, document_path: Path) -> str:
        """Generate a unique ID for the document based on its content."""
        hasher = hashlib.sha256()
        with open(document_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                hasher.update(byte_block)
        return f"{document_path.suffix.lower().lstrip('.')}-{hasher.hexdigest()[:16]}"