# tests/unit/test_document_processing/test_ppt.py
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from mmrag.document_processing.legacy import PowerPointProcessor
from mmrag.document_processing.base import ProcessedDocument, TextElement, TableElement

class TestPowerPointProcessor:
    """Test suite for the PowerPoint processor."""
    
    def test_supports(self, ppt_processor):
        """Test file type support detection."""
        assert ppt_processor.supports(Path("test.pptx")) == True
        assert ppt_processor.supports(Path("test.ppt")) == True
        assert ppt_processor.supports(Path("test.pdf")) == False
        assert ppt_processor.supports(Path("test.docx")) == False
    
    def test_generate_document_id(self, ppt_processor, sample_ppt_path):
        """Test document ID generation."""
        doc_id = ppt_processor._generate_document_id(sample_ppt_path)
        assert doc_id.startswith("ppt-")
        assert len(doc_id) == 20  # "ppt-" + 16 hex chars
        
        # Same file should generate same ID
        doc_id2 = ppt_processor._generate_document_id(sample_ppt_path)
        assert doc_id == doc_id2
    
    def test_extract_metadata(self, ppt_processor):
        """Test metadata extraction."""
        # Create a mock presentation
        mock_ppt = MagicMock()
        mock_core_props = MagicMock()
        mock_core_props.title = "Test Presentation"
        mock_core_props.author = "Test Author"
        mock_core_props.subject = "Test Subject"
        mock_core_props.keywords = "test, ppt, presentation"
        mock_core_props.created = "2023-01-01"
        mock_core_props.modified = "2023-01-02"
        
        mock_ppt.core_properties = mock_core_props
        mock_ppt.slides = [MagicMock(), MagicMock()]  # Two slides
        
        metadata = ppt_processor._extract_metadata(mock_ppt)
        
        assert metadata["title"] == "Test Presentation"
        assert metadata["author"] == "Test Author"
        assert metadata["subject"] == "Test Subject"
        assert metadata["slide_count"] == 2
    
    def test_extract_text_elements(self, ppt_processor):
        """Test text element extraction from slides."""
        # Create a mock slide
        mock_slide = MagicMock()
        
        # Create mock shapes with text
        shape1 = MagicMock()
        shape1.text = "Test text 1"
        shape1.left = 914400  # 1 inch in EMU (914400 EMU = 1 inch)
        shape1.top = 914400
        shape1.width = 2743200  # 3 inches
        shape1.height = 914400  # 1 inch
        shape1.shape_type = 1  # Auto shape
        
        shape2 = MagicMock()
        shape2.text = "Test text 2"
        shape2.left = 914400
        shape2.top = 2743200  # 3 inches
        shape2.width = 2743200
        shape2.height = 914400
        shape2.shape_type = 2  # Text box
        
        # Shape without text should be ignored
        shape3 = MagicMock()
        shape3.text = ""
        
        mock_slide.shapes = [shape1, shape2, shape3]
        
        text_elements = ppt_processor._extract_text_elements(mock_slide, 0)
        
        assert len(text_elements) == 2
        assert all(isinstance(el, TextElement) for el in text_elements)
        assert text_elements[0].content == "Test text 1"
        assert text_elements[1].content == "Test text 2"
        assert text_elements[0].metadata["shape_type"] == 1
        assert text_elements[1].metadata["shape_type"] == 2
    
    def test_extract_table_elements(self, ppt_processor):
        """Test table element extraction from slides."""
        # Create a mock slide
        mock_slide = MagicMock()
        
        # Create a mock shape with a table
        shape = MagicMock()
        shape.has_table = True
        
        # Create a mock table
        table = MagicMock()
        
        # Create mock cells
        cell1 = MagicMock()
        cell1.text = "Header 1"
        cell2 = MagicMock()
        cell2.text = "Header 2"
        cell3 = MagicMock()
        cell3.text = "Data 1"
        cell4 = MagicMock()
        cell4.text = "Data 2"
        
        # Setup table structure
        table.rows = [MagicMock(), MagicMock()]  # 2 rows
        table.columns = [MagicMock(), MagicMock()]  # 2 columns
        
        # Setup cell retrieval
        def get_cell(row, col):
            if row == 0:
                return [cell1, cell2][col]
            else:
                return [cell3, cell4][col]
        
        table.cell = get_cell
        
        shape.table = table
        shape.left = 914400
        shape.top = 914400
        shape.width = 2743200
        shape.height = 1828800  # 2 inches
        
        mock_slide.shapes = [shape]
        
        table_elements = ppt_processor._extract_table_elements(mock_slide, 0)
        
        assert len(table_elements) == 1
        assert isinstance(table_elements[0], TableElement)
        assert table_elements[0].content == [["Header 1", "Header 2"], ["Data 1", "Data 2"]]
        assert table_elements[0].metadata["num_rows"] == 2
        assert table_elements[0].metadata["num_cols"] == 2
    
    def test_process_basic(self, ppt_processor, sample_ppt_path):
        """Test basic document processing functionality."""
        doc = ppt_processor.process(sample_ppt_path)
        
        assert isinstance(doc, ProcessedDocument)
        assert doc.document_id.startswith("ppt-")
        assert doc.filename == sample_ppt_path.name
        assert doc.doc_type == "pptx"
        assert isinstance(doc.metadata, dict)
        assert "slide_count" in doc.metadata
        
        # Should have at least one text element (title)
        text_elements = [el for el in doc.elements if el.element_type == "text"]
        assert len(text_elements) > 0
