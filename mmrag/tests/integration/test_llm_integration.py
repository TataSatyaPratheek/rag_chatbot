# tests/integration/test_llm_integration.py
import pytest
import os
import httpx
from unittest.mock import patch, MagicMock
import json

import dspy # Import dspy

from mmrag.document_processing import PDFProcessor
from mmrag.llm import ContentUnderstanding, OllamaClient
# Import the DSPy modules we need to mock
from mmrag.guardrails.processors import (
    SummaryModule,
    TopicsModule,
    EntitiesModule,
)
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
    
    # Patch the DSPy modules within ContentUnderstanding
    @patch.object(SummaryModule, "forward", autospec=True)
    @patch.object(TopicsModule, "forward", autospec=True)
    @patch.object(EntitiesModule, "forward", autospec=True)
    def test_analysis_with_mocked_dspy_modules(
        self, mock_entities_forward, mock_topics_forward, mock_summary_forward, sample_pdf_path
    ):
        """Test document analysis with mocked LLM responses."""
        # Define mock DSPy Prediction outputs
        mock_responses = {
            "summary": dspy.Prediction(summary="This is a mock summary of the document."),
            "topics": dspy.Prediction(topics_json='["Topic 1", "Topic 2", "Topic 3"]'),
            "entities": {
                "entities_json": '{"people": ["John Doe"], "organizations": ["ACME Corp"]}'
            }
        }

        # Configure the mocks to return the predefined Prediction objects
        mock_summary_forward.return_value = mock_responses["summary"]
        mock_topics_forward.return_value = mock_responses["topics"]
        mock_entities_forward.return_value = dspy.Prediction(
            entities_json=mock_responses["entities"]["entities_json"]
        )

        # Mock the dspy LM configuration to avoid actual LLM calls during init
        # We can mock dspy.settings.configure or the Ollama class itself if needed,
        # but mocking the forward methods of the modules used is more direct here.
        # Ensure DSPy settings are configured with *some* LM, even if mocked later.
        # If ContentUnderstanding fails without a configured LM, mock that part too.
        # For simplicity, assume ContentUnderstanding handles LM init failure or mock it.
        # Let's mock the Ollama LM init within ContentUnderstanding for safety:
        with patch("mmrag.llm.content_understanding.dspy.Ollama") as mock_dspy_ollama:
            mock_dspy_ollama.return_value = MagicMock() # Return a dummy LM
        
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
        assert doc.analysis["summary"] == mock_responses["summary"].summary
        assert "topics" in doc.analysis
        assert doc.analysis["topics"] == json.loads(mock_responses["topics"].topics_json)
        assert "entities" in doc.analysis
        assert doc.analysis["entities"] == json.loads(mock_responses["entities"]["entities_json"])

        # Verify the mocks were called
        mock_summary_forward.assert_called_once()
        mock_topics_forward.assert_called_once()
        mock_entities_forward.assert_called_once()
