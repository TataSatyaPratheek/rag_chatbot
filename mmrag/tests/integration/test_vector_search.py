import pytest
from pathlib import Path

from mmrag.vectordb import ChromaStore
from mmrag.document_processing import PDFProcessor
from mmrag.document_processing.base import ProcessedDocument, TextElement, BoundingBox

@pytest.mark.integration
class TestVectorSearch:
    def test_query_empty_collection(self, temp_dir):
        store = ChromaStore(persist_directory=temp_dir / "chroma_empty")
        results = store.query("nonexistent query", n_results=5)
        assert results is not None
        assert len(results["ids"][0]) == 0

    def test_query_with_filters(self, sample_pdf_path, temp_dir):
        store = ChromaStore(persist_directory=temp_dir / "chroma_filter")
        # Process a document
        processor = PDFProcessor()
        processed_doc = processor.process(sample_pdf_path)
        store.add_document(processed_doc)
        
        # Query with document type filter
        results = store.query("test", n_results=5, where={"doc_type": "pdf"})
        assert results is not None
        assert len(results["ids"][0]) > 0
        
        # Test with element type filter
        text_results = store.query("test", n_results=5, where={"element_type": "text"})
        assert text_results is not None
        
        # Test non-matching filter
        no_results = store.query("test", n_results=5, where={"doc_type": "nonexistent"})
        assert len(no_results["ids"][0]) == 0
