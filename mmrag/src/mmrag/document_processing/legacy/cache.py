import functools
from pathlib import Path

class CachedDocumentProcessor:
    """Wrapper for document processors that adds caching."""
    
    def __init__(self, processor, cache_size=128):
        """Initialize the cached processor."""
        self.processor = processor
        self.process = functools.lru_cache(maxsize=cache_size)(self._process)
    
    def _process(self, document_path_str):
        """Process a document with caching."""
        document_path = Path(document_path_str)
        return self.processor.process(document_path)
    
    def supports(self, document_path):
        """Check if the processor supports the given document."""
        return self.processor.supports(document_path)
