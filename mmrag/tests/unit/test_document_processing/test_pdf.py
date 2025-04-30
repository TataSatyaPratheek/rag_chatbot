# tests/unit/test_document_processing/test_pdf.py
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
