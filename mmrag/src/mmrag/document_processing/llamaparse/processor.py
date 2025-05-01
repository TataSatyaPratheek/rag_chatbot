"""Document processing using LlamaParse."""

import uuid
import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    from llama_parse import LlamaParse
    LLAMAPARSE_INSTALLED = True
except ImportError:
    LLAMAPARSE_INSTALLED = False
    # Define a dummy class if LlamaParse is not installed
    class LlamaParse:
        pass

import psutil

from mmrag.document_processing.base import (
    DocumentProcessor, ProcessedDocument, BoundingBox,
    TextElement, TableElement, ImageElement, ChartElement
)
from mmrag.exceptions import ProcessingError, ProcessingTimeoutError, MemoryLimitExceededError
from mmrag.document_processing.llamaparse.converter import extract_elements_from_llamaparse

logger = logging.getLogger(__name__)


class LlamaParseDocumentProcessor(DocumentProcessor):
    """Document processor using LlamaParse."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        extract_tables: bool = True,
        extract_images: bool = True,
        advanced_table_detection: bool = True,
        enable_enhanced_visual: bool = True,
        enable_llm_analysis: bool = False,
        timeout_seconds: int = 60,
        memory_limit_fraction: float = 0.5,
        result_type: str = "markdown",
        use_multimodal: bool = True,
        multimodal_model: str = "anthropic-sonnet-3.5",
        target_pages: Optional[str] = None,
        bbox_params: Optional[Dict[str, float]] = None,
    ):
        """Initialize the LlamaParse processor.

        Args:
            api_key: LlamaParse API key (defaults to LLAMA_CLOUD_API_KEY env variable)
            extract_tables: Whether to extract tables
            extract_images: Whether to extract images
            advanced_table_detection: Whether to use advanced table detection
            enable_enhanced_visual: Whether to use enhanced visual processing (charts)
            enable_llm_analysis: Whether to enable LLM analysis
            timeout_seconds: Timeout for processing in seconds
            memory_limit_fraction: Fraction of available memory to allow usage
            result_type: Output format ("markdown" or "text")
            use_multimodal: Whether to use multimodal capabilities
            multimodal_model: Model to use for multimodal parsing ("anthropic-sonnet-3.5" or "openai-gpt4o")
            target_pages: Page numbers to extract (e.g., "0,10,12,22-33")
            bbox_params: Parameters for header/footer removal (e.g., {"bbox_top": 0.1, "bbox_bottom": 0.05})
        """
        if not LLAMAPARSE_INSTALLED:
            raise ImportError("LlamaParse library is not installed. Cannot use LlamaParseDocumentProcessor.")

        # Check for API key in env var if not provided
        self.api_key = api_key or os.environ.get("LLAMA_CLOUD_API_KEY")
        if not self.api_key:
            raise ValueError("LlamaParse API key not provided and LLAMA_CLOUD_API_KEY environment variable not set")

        # Store configuration parameters
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.advanced_table_detection = advanced_table_detection
        self.enable_enhanced_visual = enable_enhanced_visual
        self.enable_llm_analysis = enable_llm_analysis
        self.timeout_seconds = timeout_seconds
        self.memory_limit_fraction = memory_limit_fraction
        self.result_type = result_type
        self.use_multimodal = use_multimodal
        self.multimodal_model = multimodal_model
        self.target_pages = target_pages
        self.bbox_params = bbox_params or {}

        # Initialize LlamaParse
        parser_params = {
            "result_type": self.result_type,
            "timeout_seconds": self.timeout_seconds,
        }
        
        # Add optional parameters
        if self.target_pages:
            parser_params["target_pages"] = self.target_pages
            
        # Add bbox parameters if provided
        for key, value in self.bbox_params.items():
            if key in ["bbox_top", "bbox_bottom", "bbox_left", "bbox_right"]:
                parser_params[key] = value
        
        # Configure multimodal capabilities if enabled
        if self.use_multimodal and self.enable_enhanced_visual:
            parser_params["use_vendor_multimodal_model"] = True
            parser_params["vendor_multimodal_model_name"] = self.multimodal_model
        
        self.parser = LlamaParse(**parser_params)

    def supports(self, document_path: Union[str, Path]) -> bool:
        """Check if the processor supports the given document type."""
        document_path = Path(document_path)
        # LlamaParse supports various formats
        return document_path.suffix.lower() in [
            ".pdf", ".pptx", ".ppt", ".docx", ".doc", ".xlsx", ".html"
        ]

    def process(self, document_path: Union[str, Path]) -> ProcessedDocument:
        """Process a document and extract elements."""
        document_path = Path(document_path)

        if not self.supports(document_path):
            raise ValueError(f"Unsupported document type: {document_path.suffix}")

        try:
            # Memory check before processing
            process = psutil.Process(os.getpid())
            initial_available_memory = psutil.virtual_memory().available
            memory_limit_bytes = initial_available_memory * self.memory_limit_fraction
            current_rss = process.memory_info().rss
            
            if current_rss > memory_limit_bytes:
                raise MemoryLimitExceededError(
                    f"Memory usage ({current_rss / (1024**2):.2f} MB) exceeded limit "
                    f"({memory_limit_bytes / (1024**2):.2f} MB) before calling LlamaParse."
                )

            # Process the document with LlamaParse
            with open(document_path, "rb") as f:
                llamaparse_documents = self.parser.load_data(
                    f, 
                    extra_info={"file_name": document_path.name}
                )

            # Generate document ID
            document_id = self._generate_document_id(document_path)
            
            # Convert LlamaParse output to mmrag elements
            elements = extract_elements_from_llamaparse(
                llamaparse_documents,
                extract_tables=self.extract_tables,
                extract_images=self.extract_images,
                enable_enhanced_visual=self.enable_enhanced_visual
            )

            # Create processed document
            metadata = self._extract_metadata(llamaparse_documents)
            processed_document = ProcessedDocument(
                document_id=document_id,
                filename=document_path.name,
                doc_type=document_path.suffix.lower().lstrip("."),
                elements=elements,
                metadata=metadata,
            )

            # Add LLM analysis if requested
            if self.enable_llm_analysis:
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
            if "timeout" in str(e).lower():
                raise ProcessingTimeoutError(f"Processing timed out: {e}") from e
            elif isinstance(e, MemoryError):
                raise MemoryLimitExceededError(f"Memory limit exceeded: {e}") from e
            else:
                raise ProcessingError(f"Error processing document with LlamaParse: {e}") from e

    def _generate_document_id(self, document_path: Path) -> str:
        """Generate a unique ID for the document based on its content."""
        hasher = hashlib.sha256()
        with open(document_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                hasher.update(byte_block)
        return f"{document_path.suffix.lower().lstrip('.')}-{hasher.hexdigest()[:16]}"

    def _extract_metadata(self, llamaparse_documents: List) -> Dict:
        """Extract metadata from LlamaParse output."""
        metadata = {
            "page_count": len(llamaparse_documents),
            "parser": "LlamaParse",
            "parser_version": getattr(self.parser, "__version__", "unknown"),
            "multimodal_enabled": self.use_multimodal,
            "multimodal_model": self.multimodal_model if self.use_multimodal else None,
        }
        
        # Extract metadata from first document if available
        if llamaparse_documents and hasattr(llamaparse_documents[0], "metadata"):
            doc_metadata = llamaparse_documents[0].metadata
            if isinstance(doc_metadata, dict):
                # Add document metadata, prefixed to avoid collision
                for key, value in doc_metadata.items():
                    metadata[f"doc_{key}"] = value
        
        return metadata