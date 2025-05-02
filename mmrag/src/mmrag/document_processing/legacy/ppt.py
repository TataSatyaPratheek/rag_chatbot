# src/mmrag/document_processing/ppt.py
"""PowerPoint document processing."""

import hashlib
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Awaitable

from pptx import Presentation

from mmrag.document_processing.base import (
    BoundingBox, DocumentElement, DocumentProcessor,
    ProcessedDocument, TextElement, TableElement
)

logger = logging.getLogger(__name__)

class PowerPointProcessor(DocumentProcessor):
    """PowerPoint document processor."""
    
    def __init__(
        self,
        extract_tables: bool = True,
        extract_images: bool = True,
    ):
        """Initialize the PowerPoint processor."""
        self.extract_tables = extract_tables
        self.extract_images = extract_images
    
    def supports(self, document_path: Union[str, Path]) -> bool:
        """Check if the processor supports the given document."""
        document_path = Path(document_path)
        return document_path.suffix.lower() in [".pptx", ".ppt"]
    
    async def process(
        self,
        document_path: Union[str, Path],
        progress_callback: Optional[Callable[[str, Optional[float]], Awaitable[None]]] = None
    ) -> ProcessedDocument:
        """Process a PowerPoint document and extract elements."""
        document_path = Path(document_path)
        
        async def _update_progress(message: str, progress: Optional[float] = None):
            if progress_callback: await progress_callback(message, progress)

        if not self.supports(document_path):
            raise ValueError(f"Unsupported document type: {document_path.suffix}")
        
        # Generate a document ID based on file content
        document_id = self._generate_document_id(document_path)
        
        await _update_progress(f"Opening presentation: {document_path.name}...")
        # --- Wrap synchronous pptx calls ---
        ppt = await asyncio.to_thread(Presentation, document_path)
        
        # Extract metadata
        metadata = self._extract_metadata(ppt)
        
        # Extract elements
        elements = []
        total_slides = len(ppt.slides)
        await _update_progress(f"Found {total_slides} slides.")
        
        # --- Process slides within the async thread wrapper ---
        # Process each slide
        for slide_idx, slide in enumerate(ppt.slides):
            # Extract text elements
            text_elements = self._extract_text_elements(slide, slide_idx)
            elements.extend(text_elements)
            
            # Extract tables if enabled
            if self.extract_tables:
                table_elements = self._extract_table_elements(slide, slide_idx)
                elements.extend(table_elements)
            
            # Extract images if enabled
            if self.extract_images:
                # Image extraction would be implemented here
                # image_elements = await asyncio.to_thread(self._extract_image_elements, slide, slide_idx)
                # elements.extend(image_elements)
                pass
            await _update_progress(f"Processed slide {slide_idx + 1}/{total_slides}", (slide_idx + 1) / total_slides)
        
        # Create the processed document
        processed_document = ProcessedDocument(
            document_id=document_id,
            filename=document_path.name,
            doc_type="pptx",
            elements=elements,
            metadata=metadata,
        )
        # --- End of wrapped synchronous calls ---
        await _update_progress("Presentation processing complete.")
        
        return processed_document
    
    def _generate_document_id(self, document_path: Path) -> str:
        """Generate a unique ID for the document based on its content."""
        hasher = hashlib.sha256()
        with open(document_path, "rb") as f:
            # Read and update in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                hasher.update(byte_block)
        return f"ppt-{hasher.hexdigest()[:16]}"
    
    def _extract_metadata(self, ppt: Presentation) -> Dict:
        """Extract metadata from the presentation."""
        core_props = ppt.core_properties
        metadata = {
            "slide_count": len(ppt.slides),
            "title": core_props.title or "",
            "author": core_props.author or "",
            "subject": core_props.subject or "",
            "keywords": core_props.keywords or "",
            "created": str(core_props.created) if core_props.created else "",
            "modified": str(core_props.modified) if core_props.modified else "",
        }
        return metadata
    
    def _extract_text_elements(self, slide, slide_idx: int) -> List[TextElement]:
        """Extract text elements from a slide."""
        text_elements = []
        
        for shape_idx, shape in enumerate(slide.shapes):
            if hasattr(shape, "text") and shape.text.strip():
                # Get shape position (in EMUs - need to convert to points)
                left = shape.left / 914400 * 72  # EMU to points
                top = shape.top / 914400 * 72
                width = shape.width / 914400 * 72
                height = shape.height / 914400 * 72
                
                element = TextElement(
                    element_id=f"text-{slide_idx}-{shape_idx}",
                    content=shape.text.strip(),
                    bbox=BoundingBox(
                        x0=left,
                        y0=top,
                        x1=left + width,
                        y1=top + height,
                        page=slide_idx,
                    ),
                    metadata={
                        "shape_type": shape.shape_type,
                    },
                )
                text_elements.append(element)
        
        return text_elements
    
    def _extract_table_elements(self, slide, slide_idx: int) -> List[TableElement]:
        """Extract table elements from a slide."""
        table_elements = []
        
        for shape_idx, shape in enumerate(slide.shapes):
            if shape.has_table:
                table = shape.table
                rows = []
                
                # Extract table data
                for row_idx in range(len(table.rows)):
                    row_data = []
                    for col_idx in range(len(table.columns)):
                        cell = table.cell(row_idx, col_idx)
                        row_data.append(cell.text.strip())
                    rows.append(row_data)
                
                # Get table position
                left = shape.left / 914400 * 72
                top = shape.top / 914400 * 72
                width = shape.width / 914400 * 72
                height = shape.height / 914400 * 72
                
                element = TableElement(
                    element_id=f"table-{slide_idx}-{shape_idx}",
                    content=rows,
                    bbox=BoundingBox(
                        x0=left,
                        y0=top,
                        x1=left + width,
                        y1=top + height,
                        page=slide_idx,
                    ),
                    metadata={
                        "num_rows": len(table.rows),
                        "num_cols": len(table.columns),
                    },
                )
                table_elements.append(element)
        
        return table_elements
