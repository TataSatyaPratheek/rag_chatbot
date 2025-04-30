import pytest
import json
from unittest.mock import patch, MagicMock

from mmrag.guardrails.processors import (
    DocumentProcessingModule,
    TableExtractionModule,
    VisualElementModule
)

class TestDocumentProcessingModule:
    """Test suite for the document processing guardrail module."""
    
    def test_initialization(self):
        """Test module initialization."""
        module = DocumentProcessingModule()
        assert hasattr(module, "predictor")
    
    def test_forward(self):
        """Test forward processing with guardrails."""
        # Create module with mocked predictor
        module = DocumentProcessingModule()
        module.predictor = MagicMock()
        
        # Mock result
        mock_result = MagicMock()
        mock_result.extracted_text = "Extracted text"
        mock_result.potential_tables = "[]"
        mock_result.visual_elements = "[]"
        mock_result.document_structure = "Document structure"
        module.predictor.return_value = mock_result
        
        # Process document
        result = module.forward(
            document_content="Test content",
            document_type="pdf",
            page_number=0
        )
        
        # Verify predictor was called
        module.predictor.assert_called_once()
        call_args = module.predictor.call_args[1]
        assert call_args["document_content"] == "Test content"
        assert call_args["document_type"] == "pdf"
        assert call_args["page_number"] == 0
        
        # Check result
        assert result.extracted_text == "Extracted text"
        assert result.potential_tables == "[]"
        assert result.visual_elements == "[]"
        assert result.document_structure == "Document structure"
    
    def test_validate_output_valid(self):
        """Test validation of valid output."""
        module = DocumentProcessingModule()
        
        # Create valid result
        mock_result = MagicMock()
        mock_result.potential_tables = "[{\"rows\": 2, \"cols\": 3}]"
        mock_result.visual_elements = "[{\"type\": \"chart\"}]"
        
        # Validate
        validated = module.validate_output(mock_result)
        
        # Should not change valid result
        assert validated.potential_tables == mock_result.potential_tables
        assert validated.visual_elements == mock_result.visual_elements
    
    def test_validate_output_invalid(self):
        """Test validation of invalid output."""
        module = DocumentProcessingModule()
        
        # Create invalid result
        mock_result = MagicMock()
        mock_result.potential_tables = "not valid json"
        mock_result.visual_elements = "{\"invalid\": true"  # Missing closing brace
        
        # Validate
        validated = module.validate_output(mock_result)
        
        # Should fix invalid JSON
        assert validated.potential_tables == "[]"
        assert validated.visual_elements == "[]"

class TestTableExtractionModule:
    """Test suite for the table extraction guardrail module."""
    
    def test_initialization(self):
        """Test module initialization."""
        module = TableExtractionModule()
        assert hasattr(module, "predictor")
    
    def test_forward(self):
        """Test forward processing with guardrails."""
        # Create module with mocked predictor
        module = TableExtractionModule()
        module.predictor = MagicMock()
        
        # Mock result
        mock_result = MagicMock()
        mock_result.table_data = "[[\"Header 1\", \"Header 2\"], [\"Data 1\", \"Data 2\"]]"
        mock_result.row_count = "2"
        mock_result.column_count = "2"
        mock_result.headers = "[\"Header 1\", \"Header 2\"]"
        module.predictor.return_value = mock_result
        
        # Process table
        result = module.forward(
            table_region="table text",
            context="surrounding text"
        )
        
        # Verify predictor was called
        module.predictor.assert_called_once()
        call_args = module.predictor.call_args[1]
        assert call_args["table_region"] == "table text"
        assert call_args["context"] == "surrounding text"
        
        # Check result
        assert result.table_data == "[[\"Header 1\", \"Header 2\"], [\"Data 1\", \"Data 2\"]]"
        assert result.row_count == "2"
        assert result.column_count == "2"
        assert result.headers == "[\"Header 1\", \"Header 2\"]"

class TestVisualElementModule:
    """Test suite for the visual element guardrail module."""
    
    def test_initialization(self):
        """Test module initialization."""
        module = VisualElementModule()
        assert hasattr(module, "predictor")
    
    def test_forward(self):
        """Test forward processing with guardrails."""
        # Create module with mocked predictor
        module = VisualElementModule()
        module.predictor = MagicMock()
        
        # Mock result
        mock_result = MagicMock()
        mock_result.description = "A bar chart showing sales data"
        mock_result.extracted_data = "{\"type\": \"bar\", \"data\": [10, 20, 30]}"
        mock_result.element_purpose = "To visualize quarterly sales"
        module.predictor.return_value = mock_result
        
        # Process visual element
        result = module.forward(
            element_type="chart",
            surrounding_text="Sales performance chart"
        )
        
        # Verify predictor was called
        module.predictor.assert_called_once()
        call_args = module.predictor.call_args[1]
        assert call_args["element_type"] == "chart"
        assert call_args["surrounding_text"] == "Sales performance chart"
        
        # Check result
        assert result.description == "A bar chart showing sales data"
        assert result.extracted_data == "{\"type\": \"bar\", \"data\": [10, 20, 30]}"
        assert result.element_purpose == "To visualize quarterly sales"
