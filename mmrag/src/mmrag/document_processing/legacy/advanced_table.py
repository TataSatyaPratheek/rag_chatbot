# Fixes for the advanced_table.py file

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
from PIL import Image, UnidentifiedImageError, Image as PILImage # Added import for exception handling and Image alias
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms.v2 import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

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
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available(): # Check for MPS only if CUDA is not available
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.classes = [
            '__background__', 'table', 'table_bordered', 'table_borderless', 'table_rotated'
        ]

        try:
            # First try to load model with device specified directly
            self.model = self._load_model(model_path, device=self.device)
            self.model.eval()
        except Exception as e: # Catch potential errors during model loading on the primary device
            logger.warning(f"Error loading model on primary device ({self.device}): {e}. Falling back to CPU.")
            # Fall back to CPU if needed
            self.device = torch.device('cpu')
            self.model = self._load_model(model_path, device=self.device)
            self.model.eval()
    
    def _load_model(self, model_path: Optional[str], device=None) -> nn.Module:
        """Load the model architecture and weights.
        
        Args:
            model_path: Path to pre-trained model weights.
            device: Device to load the model on.
            
        Returns:
            Loaded model.
        """
        # For demo purposes, we use MaskRCNN as a base
        # In production, you would implement the full CascadeTabNet architecture
        
        # Create model on the specified device
        if device:
            with torch.device(device):
                model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        else:
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
                if device:
                    state_dict = torch.load(model_path, map_location=device)
                else:
                    state_dict = torch.load(model_path, map_location=self.device)
                
                # Filter state dict to only include keys that match the model
                filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
                if not filtered_state_dict:
                    logger.warning(f"No matching keys found in state dict from {model_path}")
                
                # Added strict=False to handle mismatched keys
                model.load_state_dict(filtered_state_dict, strict=False)
                logger.info(f"Loaded model weights from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model weights: {e}")
        
        # Move model to device
        if device:
            model = model.to(device)
        
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
            try:
                # Suppress DecompressionBombWarning
                PILImage.MAX_IMAGE_PIXELS = None
                image = PILImage.open(img_path).convert("RGB") # Ensure 3 channels
                image_tensor = F.to_dtype(F.to_image(image), dtype=torch.float32, scale=True).to(self.device)
                
                # Perform inference
                with torch.no_grad():
                    prediction = self.model([image_tensor])[0]
                
                # Process predictions
                table_elements = []
                
                # Filter predictions by score and class
                keep = prediction['scores'] > 0.7
                boxes = prediction['boxes'][keep].cpu().numpy()
                labels = prediction['labels'][keep].cpu().numpy()
                scores = prediction['scores'][keep].cpu().numpy()
                
                # Convert to table elements
                for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
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
                                "confidence": float(score),
                            },
                        )
                        table_elements.append(table_element)
                
                return table_elements
            except (UnidentifiedImageError, FileNotFoundError) as e:
                # Added error handling for image loading issues
                logger.warning(f"Error loading image: {e}")
                return []
        finally:
            # Clean up temporary file
            if os.path.exists(img_path):
                os.unlink(img_path)