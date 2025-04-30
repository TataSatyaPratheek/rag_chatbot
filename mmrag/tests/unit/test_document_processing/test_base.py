import pytest
import json
from pathlib import Path

from mmrag.document_processing.base import (
    BoundingBox, DocumentElement, TextElement, TableElement,
    ImageElement, ChartElement, ProcessedDocument, DocumentProcessor
)

class TestBoundingBox:
    """Test suite for the BoundingBox class."""
    
    def test_initialization(self):
        """Test bounding box initialization."""
        bbox = BoundingBox(x0=10, y0=20, x1=110, y1=120, page=0)
        assert bbox.x0 == 10
        assert bbox.y0 == 20
        assert bbox.x1 == 110
        assert bbox.y1 == 120
        assert bbox.page == 0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        bbox = BoundingBox(x0=10, y0=20, x1=110, y1=120, page=0)
        bbox_dict = bbox.to_dict()
        assert bbox_dict["x0"] == 10
        assert bbox_dict["y0"] == 20
        assert bbox_dict["x1"] == 110
        assert bbox_dict["y1"] == 120
        assert bbox_dict["page"] == 0

class TestDocumentElements:
    """Test suite for document element classes."""
    
    def test_text_element(self):
        """Test text element creation."""
        element = TextElement(
            element_id="text-1",
            content="This is a test.",
            bbox=BoundingBox(x0=10, y0=20, x1=110, y1=40, page=0),
            metadata={"font_size": 12}
        )
        
        assert element.element_id == "text-1"
        assert element.element_type == "text"
        assert element.content == "This is a test."
        assert element.bbox.x0 == 10
        assert element.metadata["font_size"] == 12
    
    def test_table_element(self):
        """Test table element creation."""
        element = TableElement(
            element_id="table-1",
            content=[["Header 1", "Header 2"], ["Data 1", "Data 2"]],
            bbox=BoundingBox(x0=10, y0=20, x1=110, y1=60, page=0),
            metadata={"num_rows": 2, "num_cols": 2}
        )
        
        assert element.element_id == "table-1"
        assert element.element_type == "table"
        assert len(element.content) == 2
        assert element.content[0][0] == "Header 1"
        assert element.metadata["num_rows"] == 2
    
    def test_image_element(self):
        """Test image element creation."""
        element = ImageElement(
            element_id="image-1",
            content="base64encodeddata",
            bbox=BoundingBox(x0=10, y0=20, x1=110, y1=120, page=0),
            metadata={"width": 100, "height": 100}
        )
        
        assert element.element_id == "image-1"
        assert element.element_type == "image"
        assert element.content == "base64encodeddata"
        assert element.metadata["width"] == 100
    
    def test_chart_element(self):
        """Test chart element creation."""
        element = ChartElement(
            element_id="chart-1",
            content="Bar chart showing sales",
            data={"type": "bar", "values": [10, 20, 30]},
            bbox=BoundingBox(x0=10, y0=20, x1=110, y1=120, page=0),
            metadata={"chart_type": "bar"}
        )
        
        assert element.element_id == "chart-1"
        assert element.element_type == "chart"
        assert element.content == "Bar chart showing sales"
        assert element.data["type"] == "bar"
        assert element.metadata["chart_type"] == "bar"

class TestProcessedDocument:
    """Test suite for the ProcessedDocument class."""
    
    def test_initialization(self):
        """Test document initialization."""
        doc = ProcessedDocument(
            document_id="doc-1",
            filename="test.pdf",
            doc_type="pdf",
            elements=[
                TextElement(
                    element_id="text-1",
                    content="This is a test.",
                    bbox=BoundingBox(x0=10, y0=20, x1=110, y1=40, page=0)
                )
            ],
            metadata={"page_count": 1}
        )
        
        assert doc.document_id == "doc-1"
        assert doc.filename == "test.pdf"
        assert doc.doc_type == "pdf"
        assert len(doc.elements) == 1
        assert doc.metadata["page_count"] == 1
    
    def test_to_json(self):
        """Test conversion to JSON."""
        doc = ProcessedDocument(
            document_id="doc-1",
            filename="test.pdf",
            doc_type="pdf",
            elements=[
                TextElement(
                    element_id="text-1",
                    content="This is a test.",
                    bbox=BoundingBox(x0=10, y0=20, x1=110, y1=40, page=0)
                )
            ],
            metadata={"page_count": 1}
        )
        
        json_str = doc.to_json()
        assert isinstance(json_str, str)
        
        # Parse back to verify
        parsed = json.loads(json_str)
        assert parsed["document_id"] == "doc-1"
        assert parsed["filename"] == "test.pdf"
        assert len(parsed["elements"]) == 1
    
    def test_from_json(self):
        """Test creation from JSON."""
        json_str = json.dumps({
            "document_id": "doc-1",
            "filename": "test.pdf",
            "doc_type": "pdf",
            "elements": [
                {
                    "element_id": "text-1",
                    "element_type": "text",
                    "content": "This is a test.",
                    "bbox": {
                        "x0": 10,
                        "y0": 20,
                        "x1": 110,
                        "y1": 40,
                        "page": 0
                    }
                }
            ],
            "metadata": {"page_count": 1}
        })
        
        doc = ProcessedDocument.from_json(json_str)
        assert doc.document_id == "doc-1"
        assert doc.filename == "test.pdf"
        assert len(doc.elements) == 1
        assert doc.elements[0].content == "This is a test."
