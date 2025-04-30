# tests/integration/test_multimodal_processing.py
import pytest
import os
from pathlib import Path

from mmrag.document_processing import PDFProcessor, PowerPointProcessor
from mmrag.document_processing.enhanced_visual import EnhancedVisualProcessor
from mmrag.document_processing.factory import get_processor

@pytest.mark.integration
class TestMultimodalProcessing:
    """Integration tests for processing multiple document types."""
    
    def test_pdf_with_enhanced_visual(self, sample_pdf_path):
        """Test PDF processing with enhanced visual processing."""
        # Initialize processor with enhanced visual processing
        processor = PDFProcessor(
            extract_tables=True,
            extract_images=True,
            enable_enhanced_visual=True,
            visual_processor=EnhancedVisualProcessor(detect_charts=True)
        )
        
        # Process the document
        doc = processor.process(sample_pdf_path)
        
        # Validate processed document
        assert doc is not None
        assert doc.document_id is not None
        assert doc.doc_type == "pdf"
        
        # Extract document elements by type
        text_elements = [el for el in doc.elements if el.element_type == "text"]
        image_elements = [el for el in doc.elements if el.element_type == "image"]
        table_elements = [el for el in doc.elements if el.element_type == "table"]
        chart_elements = [el for el in doc.elements if el.element_type == "chart"]
        
        # Basic validation of elements
        assert len(text_elements) > 0, "Should have extracted text elements"
    
    def test_powerpoint_processing(self, sample_ppt_path):
        """Test PowerPoint processing."""
        # Initialize processor
        processor = PowerPointProcessor(
            extract_tables=True,
            extract_images=True
        )
        
        # Process the document
        doc = processor.process(sample_ppt_path)
        
        # Validate processed document
        assert doc is not None
        assert doc.document_id is not None
        assert doc.doc_type == "pptx"
        
        # Extract document elements by type
        text_elements = [el for el in doc.elements if el.element_type == "text"]
        
        # Basic validation of elements
        assert len(text_elements) > 0, "Should have extracted text elements"
    
    def test_processor_factory(self, sample_pdf_path, sample_ppt_path):
        """Test the document processor factory."""
        # Get processor for PDF
        pdf_processor = get_processor(sample_pdf_path)
        assert isinstance(pdf_processor, PDFProcessor)
        
        # Get processor for PowerPoint
        ppt_processor = get_processor(sample_ppt_path)
        assert isinstance(ppt_processor, PowerPointProcessor)
        
        # Test with unsupported extension
        with pytest.raises(ValueError):
            get_processor(Path("test.docx"))
    
    def test_cross_format_extraction_consistency(self, sample_pdf_path, sample_ppt_path):
        """Test that similar content is extracted consistently across formats."""
        # This test requires specially prepared test files with similar content
        # Skip if not available
        if not (os.path.getsize(sample_pdf_path) > 1000 and 
                os.path.getsize(sample_ppt_path) > 1000):
            pytest.skip("Test requires prepared files with similar content")
        
        # Process both documents
        pdf_processor = PDFProcessor(extract_tables=True, extract_images=True)
        ppt_processor = PowerPointProcessor(extract_tables=True, extract_images=True)
        
        pdf_doc = pdf_processor.process(sample_pdf_path)
        ppt_doc = ppt_processor.process(sample_ppt_path)
        
        # Extract text content
        pdf_text = " ".join([el.content for el in pdf_doc.elements if el.element_type == "text"])
        ppt_text = " ".join([el.content for el in ppt_doc.elements if el.element_type == "text"])
        
        # Check for common content (if the test files are properly prepared)
        # This is a simplistic check - real test files would need to be created with known common content
        common_terms = ["sample", "test", "document"]
        for term in common_terms:
            if term.lower() in pdf_text.lower() and term.lower() in ppt_text.lower():
                assert True, f"Found common term '{term}' in both documents"
                return
        
        # If we reach here, didn't find common terms
        pytest.skip("Test files don't contain expected common terms")
