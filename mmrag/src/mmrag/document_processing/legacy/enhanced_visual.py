# src/mmrag/document_processing/enhanced_visual.py
"""Enhanced visual element detection and analysis."""

import base64
import io
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import fitz
import numpy as np
from PIL import Image
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks

from mmrag.document_processing.base import (
    BoundingBox, ChartElement, DocumentElement, ImageElement
)

logger = logging.getLogger(__name__)

class EnhancedVisualProcessor:
    """Enhanced processor for visual elements in documents."""
    
    def __init__(
        self,
        min_image_size: int = 100,
        store_images: bool = True,
        detect_charts: bool = True,
    ):
        """Initialize the enhanced visual element processor.
        
        Args:
            min_image_size: Minimum size (width or height) for images to be extracted.
            store_images: Whether to store images as base64 or just metadata.
            detect_charts: Whether to detect charts in images.
        """
        self.min_image_size = min_image_size
        self.store_images = store_images
        self.detect_charts = detect_charts
    
    def extract_visual_elements(self, page: fitz.Page, page_number: int) -> List[DocumentElement]:
        """Extract visual elements from a page.
        
        Args:
            page: Page to process.
            page_number: Index of the page.
            
        Returns:
            List of visual elements (images, charts).
        """
        elements = []
        
        # Extract images
        image_elements = self._extract_images(page, page_number)
        elements.extend(image_elements)
        
        # Attempt chart detection on images if enabled
        if self.detect_charts and image_elements:
            # Create a temporary directory for saving images
            with tempfile.TemporaryDirectory() as temp_dir:
                # Process each image for chart detection
                for img_element in image_elements:
                    if not self.store_images:
                        # If we're not storing images, we can't detect charts
                        continue
                    
                    # Extract base64 image content
                    content = img_element.content
                    if not content.startswith("data:image/"):
                        continue
                    
                    try:
                        # Parse base64 image
                        image_data = content.split(",", 1)[1]
                        image_bytes = base64.b64decode(image_data)
                        
                        # Save to temporary file
                        img_path = os.path.join(temp_dir, f"image_{img_element.element_id}.png")
                        with open(img_path, "wb") as f:
                            f.write(image_bytes)
                        
                        # Detect chart
                        chart_type, chart_data = self._detect_chart(img_path)
                        
                        if chart_type:
                            # Create chart element
                            chart_element = ChartElement(
                                element_id=f"chart-{page_number}-{uuid.uuid4().hex[:8]}",
                                content=f"{chart_type} chart",
                                data=chart_data,
                                bbox=img_element.bbox,
                                metadata={
                                    "chart_type": chart_type,
                                    "confidence": 0.85,
                                    "from_image": img_element.element_id,
                                },
                            )
                            elements.append(chart_element)
                    except Exception as e:
                        logger.warning(f"Failed to process image for chart detection: {e}")
        
        return elements
    
    def _extract_images(self, page: fitz.Page, page_number: int) -> List[ImageElement]:
        """Extract images from a page.
        
        Args:
            page: Page to process.
            page_number: Index of the page.
            
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
                            content = f"image-{page_number}-{img_idx}"
                        
                        # Extract image features
                        image_features = {}
                        if self.store_images:
                            try:
                                # Load image for feature extraction
                                img_data = io.BytesIO(image_bytes)
                                pil_img = Image.open(img_data)
                                image_features = self._extract_image_features(pil_img)
                            except Exception as e:
                                logger.warning(f"Failed to extract image features: {e}")
                        
                        image_element = ImageElement(
                            element_id=f"image-{page_number}-{img_idx}",
                            content=content,
                            bbox=BoundingBox(
                                x0=img_rect.x0,
                                y0=img_rect.y0,
                                x1=img_rect.x1,
                                y1=img_rect.y1,
                                page=page_number,
                            ),
                            metadata={
                                "width": img_rect.width,
                                "height": img_rect.height,
                                "image_type": image_ext,
                                "xref": xref,
                                **image_features,
                            },
                        )
                        image_elements.append(image_element)
            except Exception as e:
                logger.warning(f"Failed to extract image: {e}")
        
        return image_elements
    
    def _extract_image_features(self, image: Image.Image) -> Dict:
        """Extract features from an image.
        
        Args:
            image: PIL image to process.
            
        Returns:
            Dictionary of image features.
        """
        features = {}
        
        try:
            # Calculate average color
            img_array = np.array(image)
            if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                avg_color = np.mean(img_array, axis=(0, 1))
                features["avg_color_rgb"] = avg_color[:3].tolist()
            
            # Calculate brightness
            if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
                # Convert to grayscale
                gray = 0.2989 * img_array[:, :, 0] + 0.5870 * img_array[:, :, 1] + 0.1140 * img_array[:, :, 2]
                brightness = np.mean(gray) / 255
                features["brightness"] = float(brightness)
            
            # Check if image is likely a photograph or graphic
            is_photo = self._is_photograph(img_array)
            features["is_photograph"] = is_photo
            
            # Estimate image complexity
            complexity = self._estimate_complexity(img_array)
            features["complexity"] = complexity
            
        except Exception as e:
            logger.warning(f"Error extracting image features: {e}")
        
        return features
    
    def _is_photograph(self, img_array: np.ndarray) -> bool:
        """Determine if an image is likely a photograph vs a graphic/chart.
        
        Args:
            img_array: Numpy array of the image.
            
        Returns:
            True if the image is likely a photograph, False otherwise.
        """
        # Convert to grayscale if color image
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            gray = 0.2989 * img_array[:, :, 0] + 0.5870 * img_array[:, :, 1] + 0.1140 * img_array[:, :, 2]
        else:
            gray = img_array
            
        # Calculate standard deviation of pixel values (photos usually have higher variance)
        pixel_std = np.std(gray)
        
        # Check color distribution (photos usually have smoother distributions)
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            r_hist = np.histogram(img_array[:, :, 0], bins=8)[0]
            g_hist = np.histogram(img_array[:, :, 1], bins=8)[0]
            b_hist = np.histogram(img_array[:, :, 2], bins=8)[0]
            
            # Normalize histograms
            r_hist = r_hist / np.sum(r_hist)
            g_hist = g_hist / np.sum(g_hist)
            b_hist = b_hist / np.sum(b_hist)
            
            # Calculate entropy (higher for photos)
            r_entropy = -np.sum(r_hist * np.log2(r_hist + 1e-10))
            g_entropy = -np.sum(g_hist * np.log2(g_hist + 1e-10))
            b_entropy = -np.sum(b_hist * np.log2(b_hist + 1e-10))
            
            color_entropy = (r_entropy + g_entropy + b_entropy) / 3
            
            # Photos usually have higher entropy and standard deviation
            return color_entropy > 2.5 and pixel_std > 30
        else:
            # For grayscale, just use standard deviation
            return pixel_std > 40
    
    def _estimate_complexity(self, img_array: np.ndarray) -> float:
        """Estimate the complexity of an image.
        
        Args:
            img_array: Numpy array of the image.
            
        Returns:
            Complexity score (0-1).
        """
        # Convert to grayscale if color image
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            gray = 0.2989 * img_array[:, :, 0] + 0.5870 * img_array[:, :, 1] + 0.1140 * img_array[:, :, 2]
        else:
            gray = img_array
            
        # Resize for faster processing if needed
        h, w = gray.shape[:2]
        max_dim = 300
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            gray = cv2.resize(gray.astype(np.uint8), (new_w, new_h))
        
        # Calculate edge density
        edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # Calculate texture variation
        texture_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        texture = cv2.filter2D(gray.astype(np.uint8), -1, texture_kernel)
        texture_variation = np.std(texture) / 255
        
        # Combine metrics
        complexity = 0.5 * edge_density + 0.5 * texture_variation
        
        # Normalize to 0-1
        complexity = min(max(complexity, 0), 1)
        
        return float(complexity)
    
    def _detect_chart(self, image_path: str) -> Tuple[str, Dict]:

        """Detect if an image contains a chart and identify its type.
        
        This is a simplified version that can detect basic chart types.
        For production use, consider more sophisticated ML-based approaches.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Tuple of (chart_type, chart_data) or (None, None) if no chart detected.
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return None, None
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to make edges clearer
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            # Detect edges
            edges = canny(thresh, sigma=2.0)
            
            # Detect lines using Hough transform
            tested_angles = np.linspace(-np.pi/2, np.pi/2, 180, endpoint=False)
            h, theta, d = hough_line(edges, theta=tested_angles)
            
            # Find line peaks
            peaks = hough_line_peaks(h, theta, d, min_distance=20, min_angle=10, threshold=0.5*np.max(h))
            
            # Count horizontal and vertical lines
            horizontal_lines = 0
            vertical_lines = 0
            
            for _, angle, dist in zip(*peaks):
                # Classify lines as horizontal or vertical
                if abs(angle) < 0.1 or abs(angle - np.pi) < 0.1:
                    horizontal_lines += 1
                elif abs(angle - np.pi/2) < 0.1 or abs(angle + np.pi/2) < 0.1:
                    vertical_lines += 1
            
            # Check for rectangular shapes (potential bar chart)
            # Handle OpenCV version differences in findContours return values
            contours_result = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours_result) == 2:
                contours, _ = contours_result
            else: # Older OpenCV versions might return 3 values
                _, contours, _ = contours_result
            
            # Count rectangular contours
            rect_count = 0
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # If polygon has 4 vertices, it's potentially a rectangle
                if len(approx) == 4:
                    rect_count += 1
            
            # Check for circular shapes (potential pie chart)
            circles = cv2.HoughCircles(
                gray, 
                cv2.HOUGH_GRADIENT, 
                dp=1, 
                minDist=20, 
                param1=50, 
                param2=30, 
                minRadius=30, 
                maxRadius=300
            )
            
            circle_count = 0 if circles is None else circles.shape[1]
            
            # Identify chart type based on features
            chart_type = None
            chart_data = None
            
            if horizontal_lines >= 4 and vertical_lines >= 1:
                # Likely a line chart
                chart_type = "line"
                chart_data = {
                    "horizontal_lines": horizontal_lines,
                    "vertical_lines": vertical_lines,
                }
            elif rect_count >= 3 and vertical_lines >= 1 and horizontal_lines >= 1:
                # Likely a bar chart
                chart_type = "bar"
                chart_data = {
                    "rectangles": rect_count,
                    "horizontal_lines": horizontal_lines,
                    "vertical_lines": vertical_lines,
                }
            elif circle_count >= 1:
                # Likely a pie chart
                chart_type = "pie"
                chart_data = {
                    "circles": circle_count,
                }
            
            return chart_type, chart_data
            
        except Exception as e:
            logger.warning(f"Error in chart detection: {e}")
            return None, None
