# src/mmrag/document_processing/advanced_table.py
"""Advanced table detection using deep learning models."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import fitz
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms.v2 import functional as F # Use v2 for consistency if available
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor # Import needed predictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor # Import needed predictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights # Import weights enum



from mmrag.document_processing.base import BoundingBox, TableElement

logger = logging.getLogger(__name__)

class CascadeTabNetDetector:
    """Table detector based on CascadeTabNet architecture.
    
    This is a simplified implementation that focuses on the core functionality
    while being compatible with M1 Mac. A full implementation would include
    the complete model architecture and trained weights.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the CascadeTabNet detector.
        
        Args:
            model_path: Path to pre-trained model weights. If None, uses a base MaskRCNN.
        """
        # For simplicity, we're using MaskRCNN as a base model
        # In a production setting, you would load the actual CascadeTabNet weights
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.classes = [
            '__background__', 'table', 'table_bordered', 'table_borderless', 'table_rotated'
        ]

        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """Load the model architecture and weights.
        
        Args:
            model_path: Path to pre-trained model weights.
            
        Returns:
            Loaded model.
        """
        # For demo purposes, we use MaskRCNN as a base
        # In production, you would implement the full CascadeTabNet architecture
        # Use weights parameter instead of pretrained
        model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        
        # Modify the classifier for our classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # Replace the box predictor head
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(self.classes))
        # Replace the mask predictor head
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, len(self.classes))
        
        # Load weights if provided
        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                logger.info(f"Loaded model weights from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model weights: {e}")
        
        return model
    
    def detect_tables(self, page: fitz.Page, page_idx: int) -> List[TableElement]:
        """Detect tables on a page using deep learning.
        
        Args:
            page: PyMuPDF page object.
            page_idx: Page index.
            
        Returns:
            List of detected table elements.
        """
        # Convert page to image
        pix = page.get_pixmap(dpi=300)
        
        # Save to temporary file (PyMuPDF pixmap to PIL is not straightforward)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pix.save(tmp.name)
            img_path = tmp.name
        
        try:
            # Load image for processing
            image = Image.open(img_path)
            image_tensor = F.to_tensor(image).to(self.device)
            
            # Perform inference
            with torch.no_grad():
                prediction = self.model([image_tensor])[0]
            
            # Process predictions
            table_elements = []
            
            # Filter predictions by score and class
            keep = prediction['scores'] > 0.7
            boxes = prediction['boxes'][keep].cpu().numpy()
            labels = prediction['labels'][keep].cpu().numpy()
            
            # Convert to table elements
            for i, (box, label) in enumerate(zip(boxes, labels)):
                if label > 0:  # Not background
                    class_name = self.classes[label]
                    
                    # Scale coordinates to page coordinates
                    x0, y0, x1, y1 = box
                    page_width, page_height = page.rect.width, page.rect.height
                    scale_x = page_width / image.width
                    scale_y = page_height / image.height
                    
                    page_x0 = x0 * scale_x
                    page_y0 = y0 * scale_y
                    page_x1 = x1 * scale_x
                    page_y1 = y1 * scale_y
                    
                    # Extract table content
                    table_content = self._extract_table_content(page, (page_x0, page_y0, page_x1, page_y1))
                    
                    table_element = TableElement(
                        element_id=f"table-{page_idx}-{i}",
                        content=table_content,
                        bbox=BoundingBox(
                            x0=page_x0,
                            y0=page_y0,
                            x1=page_x1,
                            y1=page_y1,
                            page=page_idx,
                        ),
                        metadata={
                            "table_type": class_name,
                            "confidence": float(prediction['scores'][i]),
                        },
                    )
                    table_elements.append(table_element)
            
            return table_elements
            
        finally:
            # Clean up temporary file
            if os.path.exists(img_path):
                os.unlink(img_path)
    
    def _extract_table_content(self, page: fitz.Page, bbox: Tuple[float, float, float, float]) -> List[List[str]]:
        """Extract table content based on detected bounding box.
        
        This is a simplified implementation. A more advanced version would
        use table structure recognition to identify rows and columns.
        
        Args:
            page: PyMuPDF page object.
            bbox: Table bounding box (x0, y0, x1, y1).
            
        Returns:
            Table content as a list of rows.
        """
        # Extract text within the bounding box
        x0, y0, x1, y1 = bbox
        table_rect = fitz.Rect(x0, y0, x1, y1)
        text_blocks = page.get_text("blocks", clip=table_rect)
        
        # Simple row detection based on y-coordinates
        y_positions = []
        block_texts = []
        
        for block in text_blocks:
            block_x0, block_y0, block_x1, block_y1, text, block_type, block_no = block
            if text.strip():
                y_positions.append((block_y0 + block_y1) / 2)  # Middle of the block
                block_texts.append((text.strip(), block_x0))  # Store text and x-position
        
        # Group by similar y-positions (rows)
        if not y_positions:
            return []
            
        # Cluster y-positions
        rows = {}
        y_tolerance = 5  # pixels
        
        for i, (text, x_pos) in enumerate(block_texts):
            y_mid = y_positions[i]
            y_bin = round(y_mid / y_tolerance) * y_tolerance
            
            if y_bin not in rows:
                rows[y_bin] = []
            
            rows[y_bin].append((text, x_pos))
        
        # Sort rows by y-coordinate and cells by x-coordinate
        sorted_rows = []
        for y_bin in sorted(rows.keys()):
            cells = rows[y_bin]
            cells.sort(key=lambda x: x[1])  # Sort by x-position
            sorted_rows.append([cell[0] for cell in cells])
        
        return sorted_rows
