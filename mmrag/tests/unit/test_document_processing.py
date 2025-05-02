# tests/unit/test_document_processing.py
import pytest
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from mmrag.document_processing.legacy.pdf import PDFProcessor
from mmrag.exceptions import MemoryLimitExceededError, ProcessingTimeoutError
from mmrag.document_processing.base import ProcessedDocument

@pytest.mark.asyncio
async def test_async_pdf_processing_success(sample_pdf_path):
    """Test successful asynchronous processing of a PDF."""
    processor = PDFProcessor(extract_tables=False, extract_images=False)
    
    result = await processor.process(sample_pdf_path)
    
    assert isinstance(result, ProcessedDocument)
    assert result.filename == sample_pdf_path.name
    assert result.doc_type == "pdf"
    assert len(result.elements) > 0
    assert all(elem.element_type == "text" for elem in result.elements) # Basic check

@pytest.mark.asyncio
async def test_async_pdf_processing_memory_limit(sample_pdf_path):
    """Test that MemoryLimitExceededError is raised correctly."""
    # Set a very low memory limit (e.g., 1MB) - adjust based on typical base usage
    processor = PDFProcessor(memory_limit_fraction=0.00001) 
    
    with pytest.raises(MemoryLimitExceededError):
        await processor.process(sample_pdf_path)

@pytest.mark.asyncio
async def test_async_pdf_processing_timeout(sample_pdf_path, monkeypatch):
    """Test that ProcessingTimeoutError is raised correctly."""
    # Mock asyncio.sleep to simulate long processing time
    async def mock_sleep(delay):
        await asyncio.sleep(delay * 100) # Make sleep much longer

    monkeypatch.setattr(asyncio, "sleep", mock_sleep)
    
    processor = PDFProcessor(timeout_seconds=0.1) # Very short timeout
    
    with pytest.raises(ProcessingTimeoutError):
        # This assumes the processor has some internal awaits or uses asyncio.sleep
        # If it's purely CPU bound within a to_thread, this specific mock won't trigger timeout
        # A more robust test might involve mocking time.time()
        await processor.process(sample_pdf_path) 

@pytest.mark.asyncio
async def test_async_pdf_progress_callback(sample_pdf_path):
    """Test that the progress callback is called during processing."""
    processor = PDFProcessor(extract_tables=False, extract_images=False)
    
    # Use AsyncMock for the callback
    mock_callback = AsyncMock()
    
    result = await processor.process(sample_pdf_path, progress_callback=mock_callback)
    
    assert isinstance(result, ProcessedDocument)
    mock_callback.assert_called() # Check if it was called at least once
    
    # Check if it was called with expected arguments (example for the last call)
    last_call_args = mock_callback.call_args_list[-1].args
    assert isinstance(last_call_args[0], str) # Message
    assert isinstance(last_call_args[1], float) or last_call_args[1] is None # Progress fraction or None
    assert last_call_args[1] is None or 0.0 <= last_call_args[1] <= 1.0