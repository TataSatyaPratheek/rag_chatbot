"""Converter utilities for Docling elements."""

import base64
import logging
from typing import Dict, List, Optional, Union, Any

from docling.elements import (
    Element as DoclingElement,
    TextElement as DoclingTextElement,
    TableElement as DoclingTableElement,
    ImageElement as DoclingImageElement,
    ChartElement as DoclingChartElement,
)

from mmrag.document_processing.base import (
    BoundingBox, DocumentElement, TextElement,
    TableElement, ImageElement, ChartElement
)

logger = logging.getLogger(__name__)


def convert_docling_element(
    element: DoclingElement,
    index: int
) -> Optional[DocumentElement]:
    """Convert a Docling element to an mmrag element.

    Args:
        element: Docling element to convert.
        index: Index of the element (used for ID generation).

    Returns:
        Converted mmrag element or None if conversion failed.
    """
    try:
        # Convert bounding box if available
        bbox = None
        if hasattr(element, "bbox") and element.bbox:
            bbox = BoundingBox(
                x0=element.bbox.x0,
                y0=element.bbox.y0,
                x1=element.bbox.x1,
                y1=element.bbox.y1,
                page=element.bbox.page if hasattr(element.bbox, "page") else 0,
            )

        # Convert by element type
        if isinstance(element, DoclingTextElement):
            return TextElement(
                element_id=f"text-{bbox.page if bbox else 0}-{index}",
                content=element.content,
                bbox=bbox,
                metadata=element.metadata if hasattr(element, "metadata") else {},
            )

        elif isinstance(element, DoclingTableElement):
            return TableElement(
                element_id=f"table-{bbox.page if bbox else 0}-{index}",
                content=element.content,
                bbox=bbox,
                metadata={
                    "num_rows": len(element.content) if isinstance(element.content, list) else 0,
                    "num_cols": len(element.content[0]) if isinstance(element.content, list) and element.content else 0,
                    **(element.metadata if hasattr(element, "metadata") else {}),
                },
            )

        elif isinstance(element, DoclingImageElement):
            # Handle image content (base64 or path)
            content = element.content
            if hasattr(element, "image_data") and element.image_data:
                if not isinstance(element.image_data, str):
                    # Convert bytes to base64
                    img_type = element.metadata.get("image_type", "png") if hasattr(element, "metadata") else "png"
                    base64_img = base64.b64encode(element.image_data).decode("utf-8")
                    content = f"data:image/{img_type};base64,{base64_img}"

            return ImageElement(
                element_id=f"image-{bbox.page if bbox else 0}-{index}",
                content=content,
                bbox=bbox,
                metadata=element.metadata if hasattr(element, "metadata") else {},
            )

        elif isinstance(element, DoclingChartElement):
            return ChartElement(
                element_id=f"chart-{bbox.page if bbox else 0}-{index}",
                content=element.content,
                data=element.data if hasattr(element, "data") else None,
                bbox=bbox,
                metadata=element.metadata if hasattr(element, "metadata") else {},
            )

        else:
            # Generic handling for other element types
            element_type_name = element.__class__.__name__.lower().replace("element", "")
            return DocumentElement(
                element_id=f"{element_type_name}-{bbox.page if bbox else 0}-{index}",
                element_type=element_type_name,
                content=element.content if hasattr(element, "content") else str(element),
                bbox=bbox,
                metadata=element.metadata if hasattr(element, "metadata") else {},
            )

    except Exception as e:
        logger.warning(f"Failed to convert Docling element: {e}")
        return None