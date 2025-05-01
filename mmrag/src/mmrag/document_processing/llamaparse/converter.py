"""Converter utilities for LlamaParse elements."""

import logging
import re
import base64
import io
from typing import Dict, List, Optional, Union, Any
import uuid

from mmrag.document_processing.base import (
    BoundingBox, DocumentElement, TextElement,
    TableElement, ImageElement, ChartElement
)

logger = logging.getLogger(__name__)

def extract_elements_from_llamaparse(
    llamaparse_documents: List,
    extract_tables: bool = True,
    extract_images: bool = True,
    enable_enhanced_visual: bool = False,
) -> List[DocumentElement]:
    """Extract mmrag elements from LlamaParse output.

    Args:
        llamaparse_documents: List of documents returned by LlamaParse
        extract_tables: Whether to extract table elements
        extract_images: Whether to extract image elements
        enable_enhanced_visual: Whether to detect charts in images

    Returns:
        List of mmrag DocumentElements
    """
    elements = []
    element_index = 0
    
    for page_idx, doc in enumerate(llamaparse_documents):
        # Process text blocks
        text_elements = extract_text_elements(doc, page_idx, element_index)
        elements.extend(text_elements)
        element_index += len(text_elements)
        
        # Process tables if enabled
        if extract_tables:
            table_elements = extract_table_elements(doc, page_idx, element_index)
            elements.extend(table_elements)
            element_index += len(table_elements)
        
        # Process images if enabled
        if extract_images:
            image_elements = extract_image_elements(doc, page_idx, element_index)
            elements.extend(image_elements)
            element_index += len(image_elements)
            
            # Process charts if enhanced visual is enabled
            if enable_enhanced_visual:
                chart_elements = extract_chart_elements(doc, page_idx, element_index, image_elements)
                elements.extend(chart_elements)
                element_index += len(chart_elements)
    
    return elements

def extract_text_elements(doc, page_idx: int, start_index: int) -> List[TextElement]:
    """Extract text elements from a LlamaParse document.

    Args:
        doc: LlamaParse document
        page_idx: Page index
        start_index: Starting index for element IDs

    Returns:
        List of TextElements
    """
    elements = []
    
    # Handle markdown content from LlamaParse
    if hasattr(doc, "text") and doc.text:
        # Split content into paragraphs or sections
        # This is a simplified approach - more complex parsing can be implemented
        paragraphs = re.split(r'\n\s*\n', doc.text)
        
        for i, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
                
            # Check if paragraph might be part of a table (avoid duplicating content)
            if '|' in paragraph and '-+-' in paragraph:
                continue
                
            element = TextElement(
                element_id=f"text-{page_idx}-{start_index + i}",
                content=paragraph.strip(),
                # LlamaParse may not provide exact bounding boxes, so we create a logical one
                bbox=BoundingBox(
                    x0=0,
                    y0=i * 100,  # Approximate vertical position
                    x1=500,      # Arbitrary width
                    y1=(i + 1) * 100,
                    page=page_idx,
                ),
                metadata={
                    "source": "llamaparse",
                    "index_in_page": i,
                },
            )
            elements.append(element)
    
    return elements

def extract_table_elements(doc, page_idx: int, start_index: int) -> List[TableElement]:
    """Extract table elements from a LlamaParse document.

    Args:
        doc: LlamaParse document
        page_idx: Page index
        start_index: Starting index for element IDs

    Returns:
        List of TableElements
    """
    elements = []
    table_idx = 0
    
    # Check for Markdown tables in the text
    if hasattr(doc, "text") and doc.text:
        # Find all markdown tables (simplified matching)
        table_pattern = r'(\|[^\n]+\|\n\|[\s-:]+\|\n(?:\|[^\n]+\|\n)+)'
        tables = re.finditer(table_pattern, doc.text)
        
        for table_match in tables:
            table_text = table_match.group(1)
            table_data = parse_markdown_table(table_text)
            
            if table_data and len(table_data) > 1:  # At least header and one data row
                element = TableElement(
                    element_id=f"table-{page_idx}-{start_index + table_idx}",
                    content=table_data,
                    # Approximate bounding box based on match position
                    bbox=BoundingBox(
                        x0=0,
                        y0=0,
                        x1=500,
                        y1=100,
                        page=page_idx,
                    ),
                    metadata={
                        "source": "llamaparse",
                        "num_rows": len(table_data),
                        "num_cols": len(table_data[0]) if table_data else 0,
                        "format": "markdown",
                    },
                )
                elements.append(element)
                table_idx += 1
                
    # Also check for structured tables in LlamaParse output
    # This assumes LlamaParse might have a tables attribute or similar
    if hasattr(doc, "tables") and doc.tables:
        for i, table in enumerate(doc.tables):
            # Convert table to a list of lists format
            table_data = []
            if hasattr(table, "headers") and table.headers:
                table_data.append(table.headers)
            
            if hasattr(table, "rows") and table.rows:
                table_data.extend(table.rows)
            
            if table_data:
                element = TableElement(
                    element_id=f"table-{page_idx}-{start_index + table_idx}",
                    content=table_data,
                    # Use bounding box if available, otherwise approximate
                    bbox=get_bbox_from_llamaparse(table, page_idx) or BoundingBox(
                        x0=0,
                        y0=0,
                        x1=500,
                        y1=100,
                        page=page_idx,
                    ),
                    metadata={
                        "source": "llamaparse",
                        "num_rows": len(table_data),
                        "num_cols": len(table_data[0]) if table_data else 0,
                        "format": "structured",
                    },
                )
                elements.append(element)
                table_idx += 1
    
    return elements

def extract_image_elements(doc, page_idx: int, start_index: int) -> List[ImageElement]:
    """Extract image elements from a LlamaParse document.

    Args:
        doc: LlamaParse document
        page_idx: Page index
        start_index: Starting index for element IDs

    Returns:
        List of ImageElements
    """
    elements = []
    image_idx = 0
    
    # Check for images in LlamaParse output (exact attribute may vary)
    # This handles potential different ways LlamaParse might expose images
    image_sources = []
    
    # Option 1: Check for images attribute
    if hasattr(doc, "images") and doc.images:
        image_sources.extend(doc.images)
    
    # Option 2: Check for image_data attribute
    if hasattr(doc, "image_data") and doc.image_data:
        if isinstance(doc.image_data, list):
            image_sources.extend(doc.image_data)
        else:
            image_sources.append(doc.image_data)
    
    # Option 3: Extract images from markdown
    if hasattr(doc, "text") and doc.text:
        # Find markdown image patterns
        image_matches = re.finditer(r'!\[(.*?)\]\((.*?)\)', doc.text)
        for match in image_matches:
            alt_text = match.group(1)
            image_path = match.group(2)
            image_sources.append({
                "alt_text": alt_text,
                "path": image_path,
                "format": "markdown_reference",
            })
    
    # Process all found images
    for i, img in enumerate(image_sources):
        try:
            # Determine image content and metadata
            image_content = ""
            image_metadata = {
                "source": "llamaparse",
                "index_in_page": i,
            }
            
            # Handle different image formats that LlamaParse might return
            if isinstance(img, dict):
                # Option 1: Path or URL to image
                if "path" in img:
                    image_content = img["path"]
                    if "alt_text" in img:
                        image_metadata["description"] = img["alt_text"]
                
                # Option 2: Base64 encoded data
                elif "data" in img:
                    image_content = img["data"]
                    if not image_content.startswith("data:image/"):
                        image_content = f"data:image/png;base64,{image_content}"
                
                # Add any additional metadata provided
                for key, value in img.items():
                    if key not in ["path", "data"]:
                        image_metadata[key] = value
            
            # If img is a string (base64 or path)
            elif isinstance(img, str):
                image_content = img
                if not image_content.startswith("data:image/") and not image_content.startswith("http"):
                    # Assume it's base64 without prefix
                    image_content = f"data:image/png;base64,{image_content}"
            
            # If img is bytes (raw image data)
            elif isinstance(img, bytes):
                base64_img = base64.b64encode(img).decode("utf-8")
                image_content = f"data:image/png;base64,{base64_img}"
            
            # Create image element
            element = ImageElement(
                element_id=f"image-{page_idx}-{start_index + image_idx}",
                content=image_content,
                # Use bounding box if available, otherwise approximate
                bbox=get_bbox_from_llamaparse(img, page_idx) or BoundingBox(
                    x0=0,
                    y0=0,
                    x1=300,
                    y1=300,
                    page=page_idx,
                ),
                metadata=image_metadata,
            )
            elements.append(element)
            image_idx += 1
            
        except Exception as e:
            logger.warning(f"Failed to extract image: {e}")
    
    return elements

def extract_chart_elements(
    doc, 
    page_idx: int, 
    start_index: int, 
    image_elements: List[ImageElement]
) -> List[ChartElement]:
    """Extract chart elements from a LlamaParse document using multimodal understanding.
    
    When using multimodal LlamaParse, it may identify charts within images.

    Args:
        doc: LlamaParse document
        page_idx: Page index
        start_index: Starting index for element IDs
        image_elements: Previously extracted image elements to cross-reference

    Returns:
        List of ChartElements
    """
    elements = []
    chart_idx = 0
    
    # Check for charts specifically identified by LlamaParse
    if hasattr(doc, "charts") and doc.charts:
        for i, chart in enumerate(doc.charts):
            # Extract chart data
            chart_type = getattr(chart, "type", "unknown")
            chart_data = {}
            
            # Extract any structured data LlamaParse might provide
            if hasattr(chart, "data") and chart.data:
                chart_data = chart.data
            
            # Create chart element
            element = ChartElement(
                element_id=f"chart-{page_idx}-{start_index + chart_idx}",
                content=getattr(chart, "description", f"{chart_type} chart"),
                data=chart_data,
                # Use bounding box if available, otherwise approximate
                bbox=get_bbox_from_llamaparse(chart, page_idx) or BoundingBox(
                    x0=0,
                    y0=0,
                    x1=300,
                    y1=300,
                    page=page_idx,
                ),
                metadata={
                    "source": "llamaparse",
                    "chart_type": chart_type,
                    "index_in_page": i,
                },
            )
            elements.append(element)
            chart_idx += 1
    
    # Check for chart descriptions in the text (when images might be charts)
    # This is a heuristic approach - multimodal LLMs would provide better classification
    if hasattr(doc, "text") and doc.text:
        chart_keywords = [
            r"bar chart", r"line chart", r"pie chart", r"scatter plot", 
            r"histogram", r"graph showing", r"figure \d+:.*data"
        ]
        
        chart_pattern = r'(' + '|'.join(chart_keywords) + r')'
        chart_matches = re.finditer(chart_pattern, doc.text.lower())
        
        for match in chart_matches:
            # Find nearby image that might be this chart
            chart_text_pos = match.start()
            chart_type = match.group(1)
            
            # Extract 100 characters around the match to provide context
            context_start = max(0, chart_text_pos - 50)
            context_end = min(len(doc.text), chart_text_pos + 50)
            chart_context = doc.text[context_start:context_end]
            
            # Only create chart element if we don't have too many already
            if chart_idx < 5:  # Limit to avoid false positives
                element = ChartElement(
                    element_id=f"chart-{page_idx}-{start_index + chart_idx}",
                    content=f"Possible {chart_type} referenced in text",
                    data=None,
                    bbox=BoundingBox(
                        x0=0,
                        y0=0,
                        x1=300,
                        y1=300,
                        page=page_idx,
                    ),
                    metadata={
                        "source": "llamaparse_text_detection",
                        "chart_type": chart_type,
                        "context": chart_context,
                        "confidence": 0.7,
                    },
                )
                elements.append(element)
                chart_idx += 1
    
    return elements

def get_bbox_from_llamaparse(element, page_idx: int) -> Optional[BoundingBox]:
    """Extract bounding box information from LlamaParse element if available.

    Args:
        element: Element from LlamaParse output
        page_idx: Page index

    Returns:
        BoundingBox if available, None otherwise
    """
    # Check if element has bbox or bounding_box attribute
    if hasattr(element, "bbox"):
        bbox = element.bbox
        return BoundingBox(
            x0=getattr(bbox, "x0", 0),
            y0=getattr(bbox, "y0", 0),
            x1=getattr(bbox, "x1", 500),
            y1=getattr(bbox, "y1", 100),
            page=getattr(bbox, "page", page_idx),
        )
    
    if hasattr(element, "bounding_box"):
        bbox = element.bounding_box
        return BoundingBox(
            x0=getattr(bbox, "x0", 0) if isinstance(bbox, object) else (bbox[0] if isinstance(bbox, list) else 0),
            y0=getattr(bbox, "y0", 0) if isinstance(bbox, object) else (bbox[1] if isinstance(bbox, list) else 0),
            x1=getattr(bbox, "x1", 500) if isinstance(bbox, object) else (bbox[2] if isinstance(bbox, list) else 500),
            y1=getattr(bbox, "y1", 100) if isinstance(bbox, object) else (bbox[3] if isinstance(bbox, list) else 100),
            page=getattr(bbox, "page", page_idx) if isinstance(bbox, object) else page_idx,
        )
    
    # Check for position attributes
    if all(hasattr(element, attr) for attr in ["x", "y", "width", "height"]):
        return BoundingBox(
            x0=element.x,
            y0=element.y,
            x1=element.x + element.width,
            y1=element.y + element.height,
            page=page_idx,
        )
    
    return None

def parse_markdown_table(markdown_table: str) -> List[List[str]]:
    """Parse a markdown table into a list of lists.

    Args:
        markdown_table: Markdown table string

    Returns:
        List of lists representing the table
    """
    lines = markdown_table.strip().split('\n')
    table_data = []
    
    for i, line in enumerate(lines):
        # Skip separator lines
        if i == 1 and re.match(r'\|\s*[-:]+\s*\|', line):
            continue
            
        # Extract cells from line
        cells = re.findall(r'\|(.*?)(?=\||$)', line)
        if cells:
            # The regex will include the starting | but not the ending one
            # resulting in an empty string at the beginning
            cells = [cell.strip() for cell in cells if cell.strip()]
            table_data.append(cells)
    
    return table_data