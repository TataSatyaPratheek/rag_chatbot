# tests/unit/test_llm/test_content_understanding.py
import pytest
import json
from unittest.mock import patch, MagicMock

from mmrag.llm.content_understanding import ContentUnderstanding
from mmrag.document_processing.base import (
    ProcessedDocument, TextElement, TableElement, 
    ChartElement, BoundingBox
)

class TestContentUnderstanding:
    """Test suite for content understanding using LLMs."""
    
    def test_initialization(self, mock_ollama_client):
        """Test analyzer initialization."""
        # With default client
        analyzer = ContentUnderstanding()
        assert hasattr(analyzer, "llm_client")
        
        # With custom client
        analyzer = ContentUnderstanding(llm_client=mock_ollama_client)
        assert analyzer.llm_client == mock_ollama_client
    
    def test_prepare_document_text(self, mock_ollama_client):
        """Test document text preparation."""
        analyzer = ContentUnderstanding(llm_client=mock_ollama_client)
        
        # Create a test document
        doc = ProcessedDocument(
            document_id="test-doc-1",
            filename="test.pdf",
            doc_type="pdf",
            elements=[
                TextElement(
                    element_id="text-1",
                    content="This is a test text element.",
                    bbox=BoundingBox(x0=0, y0=0, x1=100, y1=20, page=0)
                ),
                TableElement(
                    element_id="table-1",
                    content=[["Header 1", "Header 2"], ["Data 1", "Data 2"]],
                    bbox=BoundingBox(x0=0, y0=30, x1=100, y1=80, page=0)
                )
            ]
        )
        
        # Prepare document text
        text = analyzer._prepare_document_text(doc)
        
        # Should include text from text element
        assert "This is a test text element." in text
        
        # Should include text from table element
        assert "Header 1 | Header 2" in text
        assert "Data 1 | Data 2" in text
    
    def test_generate_summary(self, mock_ollama_client):
        """Test summary generation."""
        analyzer = ContentUnderstanding(llm_client=mock_ollama_client)
        
        # Mock the LLM response
        mock_ollama_client.generate_sync.return_value = "This is a summary of the document."
        
        # Generate summary
        summary = analyzer._generate_summary("This is the document text.")
        
        # Check that the model was called correctly
        mock_ollama_client.generate_sync.assert_called_once()
        
        # Check that the summary matches the mock response
        assert summary == "This is a summary of the document."
    
    def test_extract_key_topics(self, mock_ollama_client):
        """Test key topics extraction."""
        analyzer = ContentUnderstanding(llm_client=mock_ollama_client)
        
        # Mock the LLM response
        mock_ollama_client.generate_sync.return_value = '["Topic 1", "Topic 2", "Topic 3"]'
        
        # Extract topics
        topics = analyzer._extract_key_topics("This is the document text.")
        
        # Check that the model was called correctly
        mock_ollama_client.generate_sync.assert_called_once()
        
        # Check that the topics were parsed correctly
        assert topics == ["Topic 1", "Topic 2", "Topic 3"]
    
    def test_extract_entities(self, mock_ollama_client):
        """Test entity extraction."""
        analyzer = ContentUnderstanding(llm_client=mock_ollama_client)
        
        # Mock the LLM response
        mock_response = {
            "people": ["John Doe", "Jane Smith"],
            "organizations": ["Acme Corp"],
            "locations": ["New York"]
        }
        mock_ollama_client.generate_sync.return_value = json.dumps(mock_response)
        
        # Extract entities
        entities = analyzer._extract_entities("This is the document text.")
        
        # Check that the model was called correctly
        mock_ollama_client.generate_sync.assert_called_once()
        
        # Check that the entities were parsed correctly
        assert entities == mock_response
        assert "people" in entities
        assert entities["people"] == ["John Doe", "Jane Smith"]
    
    def test_analyze_document(self, mock_ollama_client):
        """Test full document analysis."""
        analyzer = ContentUnderstanding(llm_client=mock_ollama_client)
        
        # Mock the individual analysis methods
        analyzer._generate_summary = MagicMock(return_value="Document summary")
        analyzer._extract_key_topics = MagicMock(return_value=["Topic 1", "Topic 2"])
        analyzer._extract_entities = MagicMock(return_value={"people": ["John Doe"]})
        
        # Create a test document
        doc = ProcessedDocument(
            document_id="test-doc-1",
            filename="test.pdf",
            doc_type="pdf",
            elements=[
                TextElement(
                    element_id="text-1",
                    content="This is a test text element.",
                    bbox=BoundingBox(x0=0, y0=0, x1=100, y1=20, page=0)
                )
            ]
        )
        
        # Analyze the document
        results = analyzer.analyze_document(doc)
        
        # Check that all methods were called
        analyzer._generate_summary.assert_called_once()
        analyzer._extract_key_topics.assert_called_once()
        analyzer._extract_entities.assert_called_once()
        
        # Check that the results contain all expected fields
        assert "summary" in results
        assert "topics" in results
        assert "entities" in results
        assert results["summary"] == "Document summary"
        assert results["topics"] == ["Topic 1", "Topic 2"]
        assert results["entities"] == {"people": ["John Doe"]}
    
    def test_analyze_text_element(self, mock_ollama_client):
        """Test text element analysis."""
        analyzer = ContentUnderstanding(llm_client=mock_ollama_client)
        
        # Mock the LLM response
        mock_response = {
            "summary": "Short summary",
            "sentiment": "positive",
            "purpose": "informative"
        }
        mock_ollama_client.generate_sync.return_value = json.dumps(mock_response)
        
        # Create a text element
        element = TextElement(
            element_id="text-1",
            content="This is a test text element.",
            bbox=BoundingBox(x0=0, y0=0, x1=100, y1=20, page=0)
        )
        
        # Analyze the element
        result = analyzer._analyze_text_element(element, "Some context")
        
        # Check that the model was called correctly
        mock_ollama_client.generate_sync.assert_called_once()
        
        # Check that the result contains all expected fields
        assert "type" in result
        assert "summary" in result
        assert "sentiment" in result
        assert "purpose" in result
        assert result["type"] == "text"
        assert result["summary"] == "Short summary"
        assert result["sentiment"] == "positive"
        assert result["purpose"] == "informative"
