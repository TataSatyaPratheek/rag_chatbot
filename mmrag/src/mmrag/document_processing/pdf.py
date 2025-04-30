"""PDF document processing."""

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import fitz  # PyMuPDF
from pydantic import ValidationError

from mmrag.document_processing.base import (
    BoundingBox,
    DocumentElement,
    DocumentProcessor,
    ProcessedDocument,
    TextElement,
)
from mmrag.document_processing.table import TableDetector
from mmrag.document_processing.visual import VisualElementProcessor
from mmrag.document_processing.advanced_table import CascadeTabNetDetector
from mmrag.document_processing.enhanced_visual import EnhancedVisualProcessor


logger = logging.getLogger(__name__)


class PDFProcessor(DocumentProcessor):
    """PDF document processor."""
    
    def __init__(
        self,
        extract_tables: bool = True,
        extract_images: bool = True,
        table_detector: Optional[TableDetector] = None,
        advanced_table_detection: bool = False,
        visual_processor: Optional[Union[VisualElementProcessor, EnhancedVisualProcessor]] = None,
        enable_enhanced_visual: bool = False,
        enable_llm_analysis: bool = False,
    ):
        """Initialize the PDF processor."""
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.advanced_table_detection = advanced_table_detection
        self.enable_enhanced_visual = enable_enhanced_visual
        self.enable_llm_analysis = enable_llm_analysis
        
        # Use advanced table detection if requested
        if advanced_table_detection:
            try:
                from mmrag.document_processing.advanced_table import CascadeTabNetDetector
                self.table_detector = CascadeTabNetDetector()
            except (ImportError, Exception) as e:
                logger.warning(f"Failed to load advanced table detector: {e}")
                self.table_detector = table_detector or TableDetector()
        else:
            self.table_detector = table_detector or TableDetector()
        
        # Use enhanced visual processor if requested
        if enable_enhanced_visual:
            try:
                from mmrag.document_processing.enhanced_visual import EnhancedVisualProcessor
                self.visual_processor = EnhancedVisualProcessor()
            except (ImportError, Exception) as e:
                logger.warning(f"Failed to load enhanced visual processor: {e}")
                self.visual_processor = visual_processor or VisualElementProcessor()
        else:
            self.visual_processor = visual_processor or VisualElementProcessor()

    
    def supports(self, document_path: Union[str, Path]) -> bool:
        """Check if the processor supports the given document."""
        document_path = Path(document_path)
        return document_path.suffix.lower() == ".pdf"
    
    def process(self, document_path: Union[str, Path]) -> ProcessedDocument:
        """Process a PDF document and extract elements."""
        document_path = Path(document_path)
        
        if not self.supports(document_path):
            raise ValueError(f"Unsupported document type: {document_path.suffix}")
        
        # Generate a document ID based on file content
        document_id = self._generate_document_id(document_path)
        
        # Open the PDF
        doc = fitz.open(document_path)
        
        # Extract metadata
        metadata = self._extract_metadata(doc)
        
        # Extract elements
        elements = []
        
        # Process each page
        for page_idx, page in enumerate(doc):
            # Extract text blocks
            text_elements = self._extract_text_blocks(page, page_idx)
            elements.extend(text_elements)
            
            # Extract tables if enabled
            if self.extract_tables:
                table_elements = self.table_detector.detect_tables(page, page_idx)
                elements.extend(table_elements)
            
            # Extract visual elements if enabled
            if self.extract_images:
                if self.enable_enhanced_visual and isinstance(self.visual_processor, EnhancedVisualProcessor):
                    visual_elements = self.visual_processor.extract_visual_elements(page, page_idx)
                else:
                    visual_elements = self.visual_processor.extract_images(page, page_idx)
                elements.extend(visual_elements)
        
        # Create the processed document
        processed_document = ProcessedDocument(
            document_id=document_id,
            filename=document_path.name,
            doc_type="pdf",
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

    
    def _generate_document_id(self, document_path: Path) -> str:
        """Generate a unique ID for the document based on its content."""
        hasher = hashlib.sha256()
        with open(document_path, "rb") as f:
            # Read and update in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                hasher.update(byte_block)
        return f"pdf-{hasher.hexdigest()[:16]}"
    
    def _extract_metadata(self, doc: fitz.Document) -> Dict:
        """Extract metadata from the PDF document."""
        metadata = {
            "page_count": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "keywords": doc.metadata.get("keywords", ""),
            "creator": doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", ""),
            "creation_date": doc.metadata.get("creationDate", ""),
            "modification_date": doc.metadata.get("modDate", ""),
        }
        return metadata
    
    def _extract_text_blocks(self, page: fitz.Page, page_idx: int) -> List[TextElement]:
        """Extract text blocks from a page."""
        text_elements = []
        
        # Get text blocks
        blocks = page.get_text("blocks")
        
        for i, block in enumerate(blocks):
            x0, y0, x1, y1, text, block_type, block_no = block
            
            # Skip empty blocks or blocks that are not text
            if not text.strip() or block_type != 0:
                continue
            
            try:
                element = TextElement(
                    element_id=f"text-{page_idx}-{i}",
                    content=text.strip(),
                    bbox=BoundingBox(
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        page=page_idx,
                    ),
                    metadata={
                        "block_no": block_no,
                        "block_type": block_type,
                    },
                )
                text_elements.append(element)
            except ValidationError as e:
                logger.warning(f"Failed to create text element: {e}")
        
        return text_elements
