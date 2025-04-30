# tests/stress/test_large_documents.py
import pytest
import time
from pathlib import Path
import psutil
import os

from mmrag.document_processing import PDFProcessor
from mmrag.vectordb import ChromaStore

@pytest.mark.stress
class TestLargeDocuments:
    """Stress tests for processing large documents."""
    
    def test_large_pdf_processing(self, large_pdf_path):
        """Test processing a large PDF document."""
        # Skip if the large test file doesn't exist
        if not large_pdf_path.exists():
            pytest.skip(f"Large test file not found: {large_pdf_path}")
        
        # Initialize processor
        processor = PDFProcessor(extract_tables=True, extract_images=True)
        
        # Measure processing time
        start_time = time.time()
        
        # Process the document
        doc = processor.process(large_pdf_path)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Log performance metrics
        print(f"\nLarge PDF processing time: {processing_time:.2f} seconds")
        print(f"Document size: {os.path.getsize(large_pdf_path) / (1024*1024):.2f} MB")
        print(f"Page count: {doc.metadata['page_count']}")
        print(f"Element count: {len(doc.elements)}")
        print(f"Processing speed: {doc.metadata['page_count'] / processing_time:.2f} pages/second")
        
        # Verify that the document was processed correctly
        assert doc is not None
        assert doc.document_id is not None
        assert len(doc.elements) > 0
        
        # Performance requirements (adjust based on your system)
        # These are example thresholds - adjust for your actual requirements
        assert processing_time / doc.metadata['page_count'] < 2.0, "Processing too slow"
    
    def test_memory_usage_large_document(self, large_pdf_path, temp_dir):
        """Test memory usage when processing and storing a large document."""
        # Skip if the large test file doesn't exist
        if not large_pdf_path.exists():
            pytest.skip(f"Large test file not found: {large_pdf_path}")
        
        # Initialize processor and store
        processor = PDFProcessor(extract_tables=True, extract_images=True)
        store = ChromaStore(persist_directory=temp_dir / "chroma_large")
        
        # Get baseline memory usage
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Process the document
        doc = processor.process(large_pdf_path)
        
        # Measure memory after processing
        after_processing = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Store the document
        store.add_document(doc)
        
        # Measure memory after storing
        after_storing = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Log memory usage
        print(f"\nBaseline memory usage: {baseline_memory:.2f} MB")
        print(f"Memory after processing: {after_processing:.2f} MB")
        print(f"Memory after storing: {after_storing:.2f} MB")
        print(f"Processing memory increase: {after_processing - baseline_memory:.2f} MB")
        print(f"Storing memory increase: {after_storing - after_processing:.2f} MB")
        
        # Verify reasonable memory usage
        # These are example thresholds - adjust for your actual requirements
        assert after_processing - baseline_memory < 1000, "Processing used too much memory"
        assert after_storing - after_processing < 500, "Storing used too much memory"
    
    def test_vectordb_query_performance(self, large_pdf_path, temp_dir):
        """Test vector database query performance with a large document."""
        # Skip if the large test file doesn't exist
        if not large_pdf_path.exists():
            pytest.skip(f"Large test file not found: {large_pdf_path}")
        
        # Initialize processor and store
        processor = PDFProcessor(extract_tables=True, extract_images=True)
        store = ChromaStore(persist_directory=temp_dir / "chroma_query_perf")
        
        # Process and store the document
        doc = processor.process(large_pdf_path)
        store.add_document(doc)
        
        # Perform multiple queries and measure time
        queries = [
            "sample",
            "test document",
            "processing performance",
            "this is a specific phrase that should not match anything",
            "page content example"
        ]
        
        total_query_time = 0
        for query in queries:
            start_time = time.time()
            results = store.query(query, n_results=5)
            query_time = time.time() - start_time
            total_query_time += query_time
            
            print(f"Query '{query}': {query_time:.4f} seconds, {len(results['ids'][0])} results")
        
        avg_query_time = total_query_time / len(queries)
        print(f"Average query time: {avg_query_time:.4f} seconds")
        
        # Verify query performance
        assert avg_query_time < 1.0, "Queries too slow"
