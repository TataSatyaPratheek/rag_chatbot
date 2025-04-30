"""Document processing using Docling."""

import uuid
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import os
from mmrag.exceptions import MemoryLimitExceededError

try:
    from docling import Document, DocumentProcessor as DoclingProcessor
    from docling.elements import TextElement as DoclingTextElement
    from docling.elements import TableElement as DoclingTableElement
    from docling.elements import ImageElement as DoclingImageElement
    from docling.elements import ChartElement as DoclingChartElement
    DOCLING_INSTALLED = True
except ImportError:
    DOCLING_INSTALLED = False
    # Define dummy classes if docling is not installed to avoid runtime errors on import
    class DoclingProcessor: pass
    class Document: pass
    class DoclingTextElement: pass
    class DoclingTableElement: pass
    class DoclingImageElement: pass
    class DoclingChartElement: pass

import psutil # Added for memory check

from mmrag.document_processing.base import (
    DocumentProcessor, ProcessedDocument, BoundingBox,
    TextElement, TableElement, ImageElement, ChartElement
)
from mmrag.exceptions import ProcessingError, ProcessingTimeoutError
from mmrag.document_processing.docling.converter import convert_docling_element

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
        self.docling_processor = DoclingProcessor(
            extract_tables=extract_tables,
            extract_images=extract_images,
            extract_charts=enable_enhanced_visual,
            advanced_table_detection=advanced_table_detection,
            timeout_seconds=timeout_seconds,
        )

    def supports(self, document_path: Union[str, Path]) -> bool:
        """Check if the processor supports the given document type."""
        document_path = Path(document_path)
        # Docling supports a variety of formats - check with Docling's capabilities
        # Assuming DoclingProcessor has a 'supports' method or similar check
        # For now, let's list common types Docling might support
        return document_path.suffix.lower() in [".pdf", ".pptx", ".ppt", ".docx", ".doc"]

    def process(self, document_path: Union[str, Path]) -> ProcessedDocument:
        """Process a document and extract elements."""
        document_path = Path(document_path)

        if not self.supports(document_path):
            raise ValueError(f"Unsupported document type: {document_path.suffix}")

        try:
            # --- Added: Memory Check before calling Docling ---
            process = psutil.Process(os.getpid())
            initial_available_memory = psutil.virtual_memory().available
            memory_limit_bytes = initial_available_memory * self.memory_limit_fraction
            current_rss = process.memory_info().rss
            if current_rss > memory_limit_bytes:
                 raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB) before calling Docling.")
            # --- End Added ---

            docling_doc: Document = self.docling_processor.process(document_path)
            document_id = self._generate_document_id(document_path)

            elements = [
                mmrag_element for i, docling_element in enumerate(docling_doc.elements)
                if (mmrag_element := convert_docling_element(docling_element, i)) is not None
            ]

            processed_document = ProcessedDocument(
                document_id=document_id,
                filename=document_path.name,
                doc_type=document_path.suffix.lower().lstrip("."),
                elements=elements,
                metadata=docling_doc.metadata if hasattr(docling_doc, 'metadata') else {},
            )

            if self.enable_llm_analysis:
                # LLM analysis logic remains the same as in PDFProcessor
                try:
                    from mmrag.llm.content_understanding import ContentUnderstanding
                    analyzer = ContentUnderstanding()
                    analysis = analyzer.analyze_document(processed_document)
                    processed_document.analysis = analysis
                except ImportError:
                    logger.warning("LLM analysis requested but mmrag.llm is not available")
                except Exception as e:
                    logger.error(f"Error during LLM analysis: {e}")

            return processed_document

        except Exception as e:
            # Check if the exception string indicates a timeout
            if "timeout" in str(e).lower():
                raise ProcessingTimeoutError(f"Processing timed out: {e}") from e
            elif isinstance(e, MemoryError): # Catch potential MemoryError from underlying libs
                raise ProcessingTimeoutError(f"Processing timed out: {e}") from e
            else:
                raise ProcessingError(f"Error processing document with Docling: {e}") from e

    def _generate_document_id(self, document_path: Path) -> str:
        """Generate a unique ID for the document based on its content."""
        hasher = hashlib.sha256()
        with open(document_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                hasher.update(byte_block)
        return f"{document_path.suffix.lower().lstrip('.')}-{hasher.hexdigest()[:16]}"