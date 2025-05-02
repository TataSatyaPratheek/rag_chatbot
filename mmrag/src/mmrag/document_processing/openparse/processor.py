"""Document processing using OpenParse."""

import uuid
import hashlib
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Awaitable

try:
    import openparse
    from openparse import DocumentParser as OpenParseProcessor
    from openparse import processing
    OPENPARSE_INSTALLED = True
except ImportError:
    OPENPARSE_INSTALLED = False
    # Define dummy classes if not installed
    class OpenParseProcessor: pass
    class processing: pass

from mmrag.document_processing.base import ( # noqa: E402
    DocumentProcessor, ProcessedDocument, BoundingBox,
    TextElement, TableElement, ImageElement, ChartElement
)
from mmrag.exceptions import ProcessingError, ProcessingTimeoutError # noqa: E402
from mmrag.document_processing.openparse.converter import convert_openparse_node # noqa: E402

logger = logging.getLogger(__name__)


class OpenParseDocumentProcessor(DocumentProcessor):
    """Document processor using OpenParse."""

    def __init__(
        self,
        extract_tables: bool = True,
        extract_images: bool = True, # Note: OpenParse doesn't natively extract images yet
        advanced_table_detection: bool = True,
        enable_enhanced_visual: bool = True, # Note: OpenParse doesn't natively extract charts yet
        enable_llm_analysis: bool = False,
        use_semantic_processing: bool = False,
        openai_api_key: Optional[str] = None,
        timeout_seconds: int = 30, # Note: OpenParse doesn't have a direct timeout param
    ):
        """Initialize the OpenParse processor.

        Args:
            extract_tables: Whether to extract tables.
            extract_images: Whether to extract images (currently limited in OpenParse).
            advanced_table_detection: Whether to use advanced table detection (unitable).
            enable_enhanced_visual: Whether to use enhanced visual processing (currently limited).
            enable_llm_analysis: Whether to enable LLM analysis post-processing.
            use_semantic_processing: Whether to use semantic processing pipeline.
            openai_api_key: OpenAI API key for semantic processing.
            timeout_seconds: Timeout for processing in seconds (Note: Not directly supported by OpenParse).
        """
        if not OPENPARSE_INSTALLED:
            raise ImportError("OpenParse library is not installed. Install with 'pip install openparse'")

        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.advanced_table_detection = advanced_table_detection
        self.enable_enhanced_visual = enable_enhanced_visual
        self.enable_llm_analysis = enable_llm_analysis
        self.use_semantic_processing = use_semantic_processing
        self.timeout_seconds = timeout_seconds # Store but note it's not used by openparse directly

        # Setup table arguments based on detection mode
        table_args = None
        if extract_tables:
            if advanced_table_detection:
                table_args = {"parsing_algorithm": "unitable", "min_table_confidence": 0.8}
            else:
                table_args = {"parsing_algorithm": "pymupdf"} # Default heuristic

        # Setup processing pipeline
        processing_pipeline = None
        if use_semantic_processing and openai_api_key:
            processing_pipeline = processing.SemanticIngestionPipeline(
                openai_api_key=openai_api_key,
                model="text-embedding-3-large", # Or configure as needed
                min_tokens=64,
                max_tokens=1024,
            )

        # Initialize OpenParse processor
        self.openparse_processor = OpenParseProcessor(
            table_args=table_args,
            processing_pipeline=processing_pipeline,
        )

    def supports(self, document_path: Union[str, Path]) -> bool:
        """Check if the processor supports the given document type."""
        document_path = Path(document_path)
        # OpenParse supports PDF, DOCX, HTML, EPUB, XML, etc.
        return document_path.suffix.lower() in [".pdf", ".pptx", ".ppt", ".docx", ".doc", ".html", ".htm", ".epub", ".xml"]

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
            await _update_progress(f"Processing with OpenParse: {document_path.name}...")
            # Process document with OpenParse
            # Wrap the synchronous parse call in asyncio.to_thread
            parsed_doc = await asyncio.to_thread(self.openparse_processor.parse, str(document_path))

            # Generate document ID
            document_id = self._generate_document_id(document_path)

            # Convert OpenParse nodes to mmrag elements
            await _update_progress("Converting OpenParse elements...")
            await _update_progress("Converting OpenParse elements...", None)
            elements = [elem for i, node in enumerate(parsed_doc.nodes) if (elem := convert_openparse_node(node, i)) is not None]

            # Create processed document
            processed_document = ProcessedDocument(
                document_id=document_id,
                filename=document_path.name,
                doc_type=document_path.suffix.lower().lstrip("."),
                elements=elements,
                metadata=getattr(parsed_doc, "metadata", {}),
            )

            # Add LLM analysis if requested (post-processing)
            if self.enable_llm_analysis:
                await _update_progress("Performing LLM analysis...")
                await _update_progress("Performing LLM analysis...", None)
                # This part remains the same, using mmrag's LLM capabilities
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
            # OpenParse doesn't have specific timeout errors, check message
            if "timeout" in str(e).lower() or isinstance(e, asyncio.TimeoutError):
                raise ProcessingTimeoutError(f"OpenParse processing may have timed out: {e}", timeout_seconds=self.timeout_seconds) from e
            else:
                raise ProcessingError(f"Error processing document with OpenParse: {e}") from e

    def _generate_document_id(self, document_path: Path) -> str:
        """Generate a unique ID for the document based on its content."""
        hasher = hashlib.sha256()
        with open(document_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                hasher.update(byte_block)
        return f"{document_path.suffix.lower().lstrip('.')}-{hasher.hexdigest()[:16]}"