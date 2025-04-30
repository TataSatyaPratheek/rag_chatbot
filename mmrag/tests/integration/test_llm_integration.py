# tests/integration/test_llm_integration.py
import pytest
import os
import httpx
from unittest.mock import patch

from mmrag.document_processing import PDFProcessor
from mmrag.llm import ContentUnderstanding, OllamaClient

@pytest.mark.integration
class TestLLMIntegration:
    """Integration tests for LLM integration."""
    
    @pytest.fixture(scope="class")
    def ollama_available(self):
        """Check if Ollama server is available."""
        try:
            response = httpx.get("http://localhost:11434/api/version", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def test_document_analysis_with_llm(self, sample_pdf_path, ollama_available):
        """Test document analysis with LLM."""
        if not ollama_available:
            pytest.skip("Ollama server not available")
        
        # Initialize processor with LLM analysis
        processor = PDFProcessor(
            extract_tables=True,
            extract_images=True,
            enable_llm_analysis=True
        )
        
        # Process the document
        doc = processor.process(sample_pdf_path)
        
        # Check that analysis field exists
        assert doc.analysis is not None
        
        # Basic validation of analysis fields
        assert "summary" in doc.analysis
        assert isinstance(doc.analysis["summary"], str)
        assert len(doc.analysis["summary"]) > 0
        
        assert "topics" in doc.analysis
        assert isinstance(doc.analysis["topics"], list)
        
        assert "entities" in doc.analysis
        assert isinstance(doc.analysis["entities"], dict)
    
    def test_element_analysis(self, sample_pdf_path, ollama_available):
        """Test analyzing individual document elements."""
        if not ollama_available:
            pytest.skip("Ollama server not available")
        
        # Process a document to get elements
        processor = PDFProcessor(extract_tables=True, extract_images=True)
        doc = processor.process(sample_pdf_path)
        
        # Find a text element
        text_elements = [el for el in doc.elements if el.element_type == "text"]
        if not text_elements:
            pytest.skip("No text elements found in the test document")
        
        # Initialize content understanding
        analyzer = ContentUnderstanding()
        
        # Analyze a text element
        result = analyzer.analyze_element(text_elements[0])
        
        # Validate the result
        assert result is not None
        assert "type" in result
        assert result["type"] == "text"
        
        # Should have some kind of analysis fields
        analysis_fields = [key for key in result.keys() if key != "type"]
        assert len(analysis_fields) > 0
    
    @patch.object(OllamaClient, "generate_sync")
    def test_analysis_with_mocked_llm(self, mock_generate_sync, sample_pdf_path):
        """Test document analysis with mocked LLM responses."""
        # Mock the LLM responses
        mock_responses = {
            "summary": "This is a mock summary of the document.",
            "topics": ["Topic 1", "Topic 2", "Topic 3"],
            "entities": {
                "people": ["John Doe"],
                "organizations": ["ACME Corp"]
            }
        }
        
        # Setup mock to return appropriate responses based on input
        def mock_response(prompt, **kwargs):
            if "summary" in prompt.lower():
                return mock_responses["summary"]
            elif "topic" in prompt.lower():
                return str(mock_responses["topics"])
            elif "entit" in prompt.lower():
                return str(mock_responses["entities"])
            return "Mock response"
        
        mock_generate_sync.side_effect = mock_response
        
        # Initialize processor with LLM analysis
        processor = PDFProcessor(
            extract_tables=True,
            extract_images=True,
            enable_llm_analysis=True
        )
        
        # Process the document
        doc = processor.process(sample_pdf_path)
        
        # Check that analysis field exists with mock data
        assert doc.analysis is not None
        assert "summary" in doc.analysis
        assert doc.analysis["summary"] == mock_responses["summary"]
