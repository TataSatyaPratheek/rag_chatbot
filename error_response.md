1. bounding box exists in base.py """Base classes for document processing."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box for an element on a page."""
    
    x0: float = Field(..., description="Left coordinate")
    y0: float = Field(..., description="Top coordinate")
    x1: float = Field(..., description="Right coordinate")
    y1: float = Field(..., description="Bottom coordinate")
    page: int = Field(..., description="Page number (0-indexed)")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "page": self.page,
        }


class DocumentElement(BaseModel):
    """Base class for document elements."""
    
    element_id: str = Field(..., description="Unique identifier for the element")
    element_type: str = Field(..., description="Type of element (text, table, image, etc.)")
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box for the element")
    content: Any = Field(..., description="Content of the element")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TextElement(DocumentElement):
    """Text element in a document."""
    
    element_type: str = "text"
    content: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "element_id": "text-1",
                "element_type": "text",
                "bbox": {
                    "x0": 0.0,
                    "y0": 0.0,
                    "x1": 100.0,
                    "y1": 20.0,
                    "page": 0,
                },
                "content": "This is a text element",
                "metadata": {
                    "font_size": 12,
                    "is_bold": False,
                },
            }
        }


class TableElement(DocumentElement):
    """Table element in a document."""
    
    element_type: str = "table"
    content: List[List[str]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "element_id": "table-1",
                "element_type": "table",
                "bbox": {
                    "x0": 0.0,
                    "y0": 0.0,
                    "x1": 100.0,
                    "y1": 50.0,
                    "page": 0,
                },
                "content": [
                    ["Header 1", "Header 2"],
                    ["Value 1", "Value 2"],
                ],
                "metadata": {
                    "num_rows": 2,
                    "num_cols": 2,
                },
            }
        }


class ImageElement(DocumentElement):
    """Image element in a document."""
    
    element_type: str = "image"
    content: str  # Base64 encoded image or path
    
    class Config:
        json_schema_extra = {
            "example": {
                "element_id": "image-1",
                "element_type": "image",
                "bbox": {
                    "x0": 0.0,
                    "y0": 0.0,
                    "x1": 100.0,
                    "y1": 100.0,
                    "page": 0,
                },
                "content": "base64encodedstring",
                "metadata": {
                    "width": 100,
                    "height": 100,
                    "image_type": "png",
                },
            }
        }


class ChartElement(DocumentElement):
    """Chart element in a document."""
    
    element_type: str = "chart"
    content: str  # Description of the chart
    data: Optional[Dict[str, Any]] = None  # Extracted data from the chart
    
    class Config:
        json_schema_extra = {
            "example": {
                "element_id": "chart-1",
                "element_type": "chart",
                "bbox": {
                    "x0": 0.0,
                    "y0": 0.0,
                    "x1": 100.0,
                    "y1": 100.0,
                    "page": 0,
                },
                "content": "Bar chart showing sales by region",
                "data": {
                    "type": "bar",
                    "x_axis": ["North", "South", "East", "West"],
                    "y_axis": [10, 20, 15, 25],
                },
                "metadata": {
                    "chart_type": "bar",
                },
            }
        }


class ProcessedDocument(BaseModel):
    """Processed document with extracted elements."""
    
    document_id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Original filename")
    doc_type: str = Field(..., description="Document type (pdf, ppt, etc.)")
    elements: List[DocumentElement] = Field(default_factory=list, description="Extracted elements")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    analysis: Optional[Dict[str, Any]] = Field(None, description="LLM analysis of the document")
        
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ProcessedDocument":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(**data)


class DocumentProcessor(ABC):
    """Base class for document processors."""
    
    @abstractmethod
    def process(self, document_path: Union[str, Path]) -> ProcessedDocument:
        """Process a document and extract elements."""
        pass
    
    @abstractmethod
    def supports(self, document_path: Union[str, Path]) -> bool:
        """Check if the processor supports the given document."""
        pass
2. the mock_page.get_text.return_value does as you expect in # tests/unit/test_document_processing/test_pdf.py
import unittest
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import fitz

from mmrag.document_processing import PDFProcessor, TextElement, TableElement
from mmrag.document_processing.base import ProcessedDocument, BoundingBox

class TestPDFProcessor:
    """Test suite for the PDF processor."""
    
    def test_supports(self, pdf_processor, sample_pdf_path):
        """Test file type support detection."""
        assert pdf_processor.supports(sample_pdf_path) == True
        assert pdf_processor.supports(Path("test.docx")) == False
        assert pdf_processor.supports(Path("test.pptx")) == False
    
    def test_generate_document_id(self, pdf_processor, sample_pdf_path):
        """Test document ID generation."""
        doc_id = pdf_processor._generate_document_id(sample_pdf_path)
        assert doc_id.startswith("pdf-")
        assert len(doc_id) == 20  # "pdf-" + 16 hex chars
        
        # Same file should generate same ID
        doc_id2 = pdf_processor._generate_document_id(sample_pdf_path)
        assert doc_id == doc_id2
    
    def test_extract_metadata(self, pdf_processor):
        """Test metadata extraction."""
        # Create a mock document
        mock_doc = MagicMock()
        mock_doc.metadata = {
            "title": "Test Document",
            "author": "Test Author",
            "subject": "Test Subject",
            "keywords": "test, pdf, document",
            "creator": "Test Creator",
            "producer": "Test Producer",
            "creationDate": "D:20230101000000",
            "modDate": "D:20230101000000"
        }
        
        metadata = pdf_processor._extract_metadata(mock_doc)
        
        assert metadata["title"] == "Test Document"
        assert metadata["author"] == "Test Author"
        assert metadata["subject"] == "Test Subject"
        assert "page_count" in metadata
    
    def test_extract_text_blocks(self, pdf_processor):
        """Test text block extraction."""
        # Create a mock page
        mock_page = MagicMock()
        mock_page.get_text.return_value = [
            (10, 20, 100, 40, "Test text block 1", 0, 1),
            (10, 50, 100, 70, "Test text block 2", 0, 2),
            (10, 80, 100, 100, "Test text block 3", 0, 3),
        ]
        
        text_elements = pdf_processor._extract_text_blocks(mock_page, 0)
        
        assert len(text_elements) == 3
        assert all(isinstance(el, TextElement) for el in text_elements)
        assert text_elements[0].content == "Test text block 1"
        assert text_elements[1].content == "Test text block 2"
        assert text_elements[2].bbox.page == 0
    
    def test_process_basic(self, pdf_processor, sample_pdf_path):
        """Test basic document processing functionality."""
        doc = pdf_processor.process(sample_pdf_path)
        
        assert isinstance(doc, ProcessedDocument)
        assert doc.document_id.startswith("pdf-")
        assert doc.filename == sample_pdf_path.name
        assert doc.doc_type == "pdf"
        assert len(doc.elements) > 0
        assert isinstance(doc.metadata, dict)
        assert "page_count" in doc.metadata
    
    @patch("mmrag.document_processing.table.TableDetector.detect_tables")
    def test_table_detection_integration(self, mock_detect_tables, pdf_processor, sample_pdf_path):
        """Test that table detection is called during processing."""
        # Mock the table detector to return a sample table
        mock_table = TableElement(
            element_id="table-0-1",
            content=[["Header 1", "Header 2"], ["Data 1", "Data 2"]],
            bbox=BoundingBox(x0=10, y0=10, x1=100, y1=50, page=0),
            metadata={"num_rows": 2, "num_cols": 2}
        )
        mock_detect_tables.return_value = [mock_table]
        
        doc = pdf_processor.process(sample_pdf_path)
        
        # Verify table detection was called
        assert mock_detect_tables.called
        
        # Check that the table was included in the document
        tables = [el for el in doc.elements if el.element_type == "table"]
        assert len(tables) == 1
        assert tables[0].content == [["Header 1", "Header 2"], ["Data 1", "Data 2"]]
    
    @patch("mmrag.llm.content_understanding.ContentUnderstanding.analyze_document")
    def test_llm_analysis_integration(self, mock_analyze, pdf_processor, sample_pdf_path):
        """Test LLM analysis integration."""
        # Configure processor to use LLM analysis
        pdf_processor.enable_llm_analysis = True
        
        # Mock the analysis result
        mock_analyze.return_value = {
            "summary": "This is a test document summary.",
            "topics": ["topic1", "topic2"],
            "entities": {"people": ["Person A"], "organizations": ["Org B"]}
        }
        
        doc = pdf_processor.process(sample_pdf_path)
        
        # Verify LLM analysis was called
        assert mock_analyze.called
        
        # Check that analysis was included in the document
        assert doc.analysis is not None
        assert doc.analysis["summary"] == "This is a test document summary."
        assert len(doc.analysis["topics"]) == 2

3. i added your fix right under mock_page = MagicMock() in import pytest
from unittest.mock import patch, MagicMock
import fitz

from mmrag.document_processing.table import TableDetector
from mmrag.document_processing.base import TableElement, BoundingBox

class TestTableDetector:
    """Test suite for the table detector."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = TableDetector()
        assert detector.min_rows == 2
        assert detector.min_cols == 2
        
        # Custom initialization
        detector = TableDetector(min_rows=3, min_cols=4)
        assert detector.min_rows == 3
        assert detector.min_cols == 4
    
    def test_identify_potential_tables(self):
        """Test table identification logic."""
        # Create mock blocks for a table-like structure
        blocks = [
            {'type': 0, 'lines': [{'spans': [{'text': 'Header 1', 'bbox': (10, 10, 50, 30), 'size': 12}]}]},
            {'type': 0, 'lines': [{'spans': [{'text': 'Header 2', 'bbox': (60, 10, 100, 30), 'size': 12}]}]},
            {'type': 0, 'lines': [{'spans': [{'text': 'Data 1', 'bbox': (10, 40, 50, 60), 'size': 10}]}]},
            {'type': 0, 'lines': [{'spans': [{'text': 'Data 2', 'bbox': (60, 40, 100, 60), 'size': 10}]}]},
        ]
        
        # Initialize detector
        detector = TableDetector()
        
        # Identify tables
        tables = detector._identify_potential_tables(blocks)
        
        # Should find at least one table
        assert len(tables) >= 1
        
        if len(tables) > 0:
            # Check the first table
            rows, bbox = tables[0]
            assert len(rows) >= 2
            assert len(rows[0]) >= 2 if len(rows) > 0 else True
            assert bbox[0] < bbox[2]  # x0 < x1
            assert bbox[1] < bbox[3]  # y0 < y1
    
    def test_detect_tables(self):
        """Test full table detection process."""
        # Create a mock page
        mock_page = MagicMock()
        mock_page.parent = MagicMock()
        
        # Mock get_text to return table-like block structure
        mock_page.get_text.return_value = [
            {'type': 0, 'lines': [{'spans': [{'text': 'Header 1', 'bbox': (10, 10, 50, 30), 'size': 12}]}]},
            {'type': 0, 'lines': [{'spans': [{'text': 'Header 2', 'bbox': (60, 10, 100, 30), 'size': 12}]}]},
            {'type': 0, 'lines': [{'spans': [{'text': 'Data 1', 'bbox': (10, 40, 50, 60), 'size': 10}]}]},
            {'type': 0, 'lines': [{'spans': [{'text': 'Data 2', 'bbox': (60, 40, 100, 60), 'size': 10}]}]},
        ]
        
        # Patch _identify_potential_tables to return a known table
        with patch.object(TableDetector, '_identify_potential_tables') as mock_identify:
            mock_identify.return_value = [
                (
                    [['Header 1', 'Header 2'], ['Data 1', 'Data 2']],
                    (10, 10, 100, 60)
                )
            ]
            
            # Initialize detector
            detector = TableDetector()
            
            # Detect tables
            tables = detector.detect_tables(mock_page, 0)
            
            # Verify results
            assert len(tables) == 1
            assert isinstance(tables[0], TableElement)
            assert tables[0].element_id.startswith("table-0-")
            assert tables[0].content == [['Header 1', 'Header 2'], ['Data 1', 'Data 2']]
            assert tables[0].bbox.x0 == 10
            assert tables[0].bbox.y0 == 10
            assert tables[0].bbox.x1 == 100
            assert tables[0].bbox.y1 == 60
            assert tables[0].bbox.page == 0
            assert "num_rows" in tables[0].metadata
            assert "num_cols" in tables[0].metadata
            assert tables[0].metadata["num_rows"] == 2
            assert tables[0].metadata["num_cols"] == 2

4. there already was a third arg but was called pade_idx, i renamed it to page_number in # src/mmrag/document_processing/enhanced_visual.py
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
    
    def _detect_chart(self, image_path: str) -> Tuple[Optional[str], Optional[Dict]]:
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
            _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
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

5. there was optional associated with str and dict, removed that and added your fix

