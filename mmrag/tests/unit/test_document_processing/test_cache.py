import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from mmrag.document_processing.base import ProcessedDocument
from mmrag.document_processing.cache import CachedDocumentProcessor

class TestCachedDocumentProcessor:
    """Test suite for the cached document processor."""
    
    def test_initialization(self):
        """Test initialization of cached processor."""
        # Create a mock processor
        mock_processor = MagicMock()
        
        # Create cached processor
        cached = CachedDocumentProcessor(mock_processor, cache_size=100)
        
        # Verify processor is stored
        assert cached.processor == mock_processor
        assert hasattr(cached, "process")
    
    def test_caching_behavior(self):
        """Test that caching properly reuses results."""
        # Create a mock processor and document
        mock_processor = MagicMock()
        mock_doc = MagicMock(document_id="test-doc-1")
        
        # Configure processor to return a different object each time
        mock_processor.process.side_effect = lambda path: MagicMock(document_id=f"doc-{path}")
        
        # Create cached processor
        cached = CachedDocumentProcessor(mock_processor)
        
        # Process the same path twice
        result1 = cached.process("test/path.pdf")
        result2 = cached.process("test/path.pdf")
        
        # Process a different path
        result3 = cached.process("test/other.pdf")
        
        # First and second calls should return the same result
        assert result1 is result2
        
        # Third call should return a different result
        assert result1 is not result3
        
        # Processor should have been called twice (not 3 times)
        assert mock_processor.process.call_count == 2
    
    def test_cache_invalidation(self):
        """Test that cache is invalidated properly."""
        # Create a mock processor
        mock_processor = MagicMock()
        
        # Configure processor to return a different object each time
        count = 0
        def process_with_count(path):
            nonlocal count
            count += 1
            return MagicMock(document_id=f"doc-{count}")
        
        mock_processor.process.side_effect = process_with_count
        
        # Create cached processor with small cache
        cached = CachedDocumentProcessor(mock_processor, cache_size=2)
        
        # Process multiple paths to fill cache
        result1 = cached.process("test/path1.pdf")
        result2 = cached.process("test/path2.pdf")
        
        # This should evict the first result from cache
        result3 = cached.process("test/path3.pdf")
        
        # Now accessing the first path should create a new result
        result1_new = cached.process("test/path1.pdf")
        
        # Should be different objects due to cache eviction
        assert result1 is not result1_new
        
        # Processor should have been called 4 times
        assert mock_processor.process.call_count == 4
    
    def test_supports_delegation(self):
        """Test that supports method is delegated to the wrapped processor."""
        # Create a mock processor
        mock_processor = MagicMock()
        mock_processor.supports.return_value = True
        
        # Create cached processor
        cached = CachedDocumentProcessor(mock_processor)
        
        # Call supports
        result = cached.supports("test/path.pdf")
        
        # Verify delegation
        assert result is True
        mock_processor.supports.assert_called_once_with("test/path.pdf")
