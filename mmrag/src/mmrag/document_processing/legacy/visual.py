"""Visual element processing for documents."""

import base64
import io
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import fitz
from pydantic import ValidationError

from mmrag.document_processing.base import BoundingBox, ChartElement, ImageElement

logger = logging.getLogger(__name__)


class VisualElementProcessor:
    """Processor for visual elements in documents."""
    
    def __init__(self, min_image_size: int = 100, store_images: bool = True):
        """Initialize the visual element processor.
        
        Args:
            min_image_size: Minimum size (width or height) for images to be extracted.
            store_images: Whether to store images as base64 or just metadata.
        """
        self.min_image_size = min_image_size
        self.store_images = store_images
    
    def extract_images(self, page: fitz.Page, page_idx: int) -> List[ImageElement]:
        """Extract images from a page.
        
        Args:
            page: Page to process.
            page_idx: Index of the page.
            
        Returns:
            List of image elements.
        """
        image_elements = []
        
        # Get image blocks
        img_list = page.get_images(full=True)
        
        for img_idx, img in enumerate(img_list):
            try:
                # Extract image metadata
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                
                if base_image:
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Find the image rectangle on the page
                    for img_rect in page.get_image_rects(xref):
                        # Skip small images
                        if (img_rect.width < self.min_image_size or 
                            img_rect.height < self.min_image_size):
                            continue
                        
                        # Create image element
                        content = ""
                        if self.store_images:
                            base64_img = base64.b64encode(image_bytes).decode("utf-8")
                            content = f"data:image/{image_ext};base64,{base64_img}"
                        else:
                            content = f"image-{page_idx}-{img_idx}"
                        
                        image_element = ImageElement(
                            element_id=f"image-{page_idx}-{img_idx}",
                            content=content,
                            bbox=BoundingBox(
                                x0=img_rect.x0,
                                y0=img_rect.y0,
                                x1=img_rect.x1,
                                y1=img_rect.y1,
                                page=page_idx,
                            ),
                            metadata={
                                "width": img_rect.width,
                                "height": img_rect.height,
                                "image_type": image_ext,
                                "xref": xref,
                            },
                        )
                        image_elements.append(image_element)
            except Exception as e:
                logger.warning(f"Failed to extract image: {e}")
        
        return image_elements
    
    def detect_charts(self, page: fitz.Page, page_idx: int) -> List[ChartElement]:
        """Detect charts on a page.
        
        This is a placeholder implementation. In a production environment,
        this would use computer vision or ML models to detect charts.
        
        Args:
            page: Page to process.
            page_idx: Index of the page.
            
        Returns:
            List of chart elements.
        """
        # This is a placeholder - in a real implementation, you would use
        # computer vision techniques to detect charts
        return []
