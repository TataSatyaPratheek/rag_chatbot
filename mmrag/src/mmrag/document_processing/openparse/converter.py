"""Converter utilities for OpenParse nodes."""

import base64
import logging
from typing import Dict, List, Optional, Union, Any

from mmrag.document_processing.base import (
    BoundingBox, DocumentElement, TextElement,
    TableElement, ImageElement, ChartElement
)

logger = logging.getLogger(__name__)


def convert_openparse_node(
    node: Any,
    index: int
) -> Optional[DocumentElement]:
    """Convert an OpenParse node to an mmrag element.

    Args:
        node: OpenParse node to convert.
        index: Index of the node (used for ID generation).

    Returns:
        Converted mmrag element or None if conversion failed.
    """
    try:
        # Extract bounding box if available
        bbox = None
        if hasattr(node, "bbox") and node.bbox:
            # Handle the case where bbox is a list instead of an object
            if isinstance(node.bbox, list):
                # For list format, typically [x0, y0, x1, y1, page]
                if len(node.bbox) >= 4:
                    page_num = node.bbox[4] if len(node.bbox) > 4 else getattr(node, "page", 0)
                    try:
                        bbox = BoundingBox(
                            x0=float(node.bbox[0]),
                            y0=float(node.bbox[1]),
                            x1=float(node.bbox[2]),
                            y1=float(node.bbox[3]),
                            page=int(page_num),
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not parse bbox list {node.bbox} for node at index {index}: {e}")
            else:
                # Original code for object-based bbox
                page_num = getattr(node.bbox, "page", getattr(node, "page", 0))
                bbox = BoundingBox(
                    x0=node.bbox.x0,
                    y0=node.bbox.y0,
                    x1=node.bbox.x1,
                    y1=node.bbox.y1,
                    page=page_num,
                )

        # Get node type and content
        node_type = getattr(node, "variant", "text")
        content = getattr(node, "text", str(node)) # Use 'text' attribute for content
        metadata = getattr(node, "metadata", {})

        # Convert based on node type
        if "TABLE" in node_type: # OpenParse uses uppercase variants
            return TableElement(
                element_id=f"table-{bbox.page if bbox else 0}-{index}",
                content=content, # OpenParse tables are often returned as markdown strings
                bbox=bbox,
                metadata={
                    "openparse_variant": node_type,
                    **metadata,
                },
            )

        # Note: OpenParse doesn't natively extract images/charts as separate nodes currently.
        # They might be represented textually or require custom pipelines.
        # We'll default other types to TextElement for now.

        else:
            # Default to text element
            return TextElement(
                element_id=f"text-{bbox.page if bbox else 0}-{index}",
                content=content,
                bbox=bbox,
                metadata={ "openparse_variant": node_type, **metadata },
            )

    except Exception as e:
        logger.warning(f"Failed to convert OpenParse node at index {index}: {e}")
        return None