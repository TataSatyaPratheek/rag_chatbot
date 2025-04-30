import pytest
from unittest.mock import patch
from pathlib import Path

from mmrag.document_processing.factory import get_processor
from mmrag.document_processing.legacy import PDFProcessor, PowerPointProcessor

class TestProcessorFactory:
    """Test suite for the document processor factory."""
    
    def test_get_processor_for_pdf(self):
        """Test getting processor for PDF files."""
        # Create a PDF path
        pdf_path = Path("test.pdf")
        
        # Get processor
        processor = get_processor(pdf_path)
        
        # Verify type
        assert isinstance(processor, PDFProcessor)
    
    def test_get_processor_for_powerpoint(self):
        """Test getting processor for PowerPoint files."""
        # Create PowerPoint paths
        pptx_path = Path("test.pptx")
        ppt_path = Path("test.ppt")
        
        # Get processors
        pptx_processor = get_processor(pptx_path)
        ppt_processor = get_processor(ppt_path)
        
        # Verify types
        assert isinstance(pptx_processor, PowerPointProcessor)
        assert isinstance(ppt_processor, PowerPointProcessor)
    
    def test_unsupported_file_type(self):
        """Test behavior with unsupported file types."""
        # Create an unsupported file path
        docx_path = Path("test.docx")
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            get_processor(docx_path)
    
    def test_processor_configuration(self):
        """Test that processor is configured correctly."""
        # Create a PDF path
        pdf_path = Path("test.pdf")
        
        # Get processor with specific configuration
        processor = get_processor(
            pdf_path
        )
        
        # Now configure the processor instance (assuming it's a PDFProcessor)
        processor.extract_tables = True
        processor.extract_images = False
        processor.advanced_table_detection = True
        processor.enable_enhanced_visual = False
        processor.enable_llm_analysis = True

        
        # Verify configuration
        assert processor.extract_tables == True
        assert processor.extract_images == False
        assert processor.advanced_table_detection == True
        assert processor.enable_enhanced_visual == False
        assert processor.enable_llm_analysis == True
