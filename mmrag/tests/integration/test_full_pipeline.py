# tests/integration/test_full_pipeline.py
import pytest
from pathlib import Path

from mmrag.document_processing import PDFProcessor, PowerPointProcessor
from mmrag.vectordb import ChromaStore
from mmrag.document_processing.base import ProcessedDocument

@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for the full document processing and retrieval pipeline."""
    
    def test_process_and_retrieve(self, sample_pdf_path, temp_dir):
        """Test the complete process-store-retrieve pipeline."""
        # Initialize the processor and store
        processor = PDFProcessor(extract_tables=True, extract_images=True)
        store = ChromaStore(persist_directory=temp_dir / "chroma_test")
        
        # Process the document
        processed_doc = processor.process(sample_pdf_path)
        
        # Validate processed document
        assert isinstance(processed_doc, ProcessedDocument)
        assert processed_doc.document_id is not None
        assert processed_doc.filename == sample_pdf_path.name
        assert len(processed_doc.elements) > 0
        
        # Store the document
        store.add_document(processed_doc)
        
        # Query the store with text that should be in the document
        results = store.query("Sample PDF for testing", n_results=3)
        
        # Validate query results
        assert results is not None
        assert len(results["ids"][0]) > 0
        assert len(results["documents"][0]) > 0
        assert len(results["metadatas"][0]) > 0
        
        # Check that the document ID matches
        assert results["metadatas"][0][0]["document_id"] == processed_doc.document_id
    
    def test_multi_document_retrieval(self, sample_pdf_path, sample_ppt_path, temp_dir):
        """Test processing and retrieving from multiple document types."""
        # Initialize processors and store
        pdf_processor = PDFProcessor(extract_tables=True, extract_images=True)
        ppt_processor = PowerPointProcessor(extract_tables=True, extract_images=True)
        store = ChromaStore(persist_directory=temp_dir / "chroma_multi")
        
        # Process the documents
        pdf_doc = pdf_processor.process(sample_pdf_path)
        ppt_doc = ppt_processor.process(sample_ppt_path)
        
        # Store the documents
        store.add_document(pdf_doc)
        store.add_document(ppt_doc)
        
        # Query the store
        results = store.query("Sample", n_results=5)
        
        # Validate query results
        assert results is not None
        assert len(results["ids"][0]) > 0
        
        # Should find elements from both documents
        doc_ids = set()
        for metadata in results["metadatas"][0]:
            doc_ids.add(metadata["document_id"])
        
        # If the test documents contain the query term, should find both
        # Check content by joining text elements, avoid private methods
        pdf_text_content = " ".join([el.content for el in pdf_doc.elements if el.element_type == "text"])
        ppt_text_content = " ".join([el.content for el in ppt_doc.elements if el.element_type == "text"])
        if "Sample" in pdf_text_content and "Sample" in ppt_text_content:
            assert len(doc_ids) == 2
            assert pdf_doc.document_id in doc_ids
            assert ppt_doc.document_id in doc_ids
