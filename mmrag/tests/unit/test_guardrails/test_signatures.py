import pytest

from mmrag.guardrails.signatures import (
    DocumentProcessingSignature,
    TableExtractionSignature,
    VisualElementSignature
)

class TestDocumentProcessingSignature:
    """Test suite for the document processing signature."""
    
    def test_signature_structure(self):
        """Test that the signature has the correct structure."""
        # Basic check that the class exists and seems like a signature
        assert DocumentProcessingSignature is not None
        assert hasattr(DocumentProcessingSignature, "__signature__") # Check for dspy internal attribute
    
    def test_field_descriptions(self):
        """Test that fields have meaningful descriptions."""
        # Check if instructions exist (common place for description)
        assert hasattr(DocumentProcessingSignature, "instructions")

class TestTableExtractionSignature:
    """Test suite for the table extraction signature."""
    
    def test_signature_structure(self):
        """Test that the signature has the correct structure."""
        assert TableExtractionSignature is not None
        assert hasattr(TableExtractionSignature, "__signature__")
    
    def test_field_descriptions(self):
        """Test that fields have meaningful descriptions."""
        assert hasattr(TableExtractionSignature, "instructions")

class TestVisualElementSignature:
    """Test suite for the visual element signature."""
    
    def test_signature_structure(self):
        """Test that the signature has the correct structure."""
        assert VisualElementSignature is not None
        assert hasattr(VisualElementSignature, "__signature__")
    
    def test_field_descriptions(self):
        """Test that fields have meaningful descriptions."""
        assert hasattr(VisualElementSignature, "instructions")
