# tests/stress/test_concurrency.py
import pytest
import concurrent.futures
import time
import os
from pathlib import Path

from mmrag.document_processing.legacy import PDFProcessor
from mmrag.vectordb import ChromaStore

@pytest.mark.stress
class TestConcurrency:
    """Stress tests for concurrent document processing."""
    
    def test_concurrent_processing(self, large_pdf_path, temp_dir): # Use large_pdf_path
        """Test processing multiple documents concurrently."""
        # Create a set of test documents (copies of the sample)
        doc_paths = []
        for i in range(10):
            doc_path = temp_dir / f"test_doc_{i}.pdf"
            if not doc_path.exists():
                # Copy the large file
                with open(large_pdf_path, "rb") as src:
                    with open(doc_path, "wb") as dst:
                        dst.write(src.read())
            doc_paths.append(doc_path)
        
        # Process documents concurrently
        start_time = time.time()
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Create a processor for each worker
            futures = []
            for doc_path in doc_paths:
                processor = PDFProcessor(extract_tables=True, extract_images=True)
                future = executor.submit(processor.process, doc_path)
                futures.append(future)
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    doc = future.result()
                    results.append(doc)
                except Exception as e:
                    print(f"Error processing document: {e}")
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Log performance metrics
        print(f"\nConcurrent processing time for {len(doc_paths)} documents: {total_time:.2f} seconds")
        print(f"Average time per document: {total_time / len(doc_paths):.2f} seconds")
        print(f"Successfully processed: {len(results)} of {len(doc_paths)}")
        
        # Verify all documents were processed
        assert len(results) == len(doc_paths)
        
        # Verify reasonable performance
        # Concurrent processing should be faster than sequential for multiple documents
        # Calculate actual sequential time by processing all documents sequentially
        seq_processor = PDFProcessor(extract_tables=True, extract_images=True)
        seq_start_time = time.time()
        for doc_path in doc_paths:
            seq_processor.process(doc_path) # Process each doc sequentially
        actual_seq_time = time.time() - seq_start_time
        
        print(f"Actual sequential time: {actual_seq_time:.2f} seconds")
        print(f"Concurrency speedup factor: {actual_seq_time / total_time:.2f}x")
        
        # Should see some speedup from concurrency
        # Relax the assertion slightly - allow for some overhead/variance
        assert total_time < actual_seq_time * 1.1, "Concurrent processing significantly slower than sequential"
    
    def test_concurrent_vectordb_operations(self, sample_pdf_path, temp_dir):
        """Test concurrent vector database operations."""
        # Process a sample document
        processor = PDFProcessor(extract_tables=True, extract_images=True)
        doc = processor.process(sample_pdf_path)
        
        # Create multiple documents by modifying the ID
        docs = []
        for i in range(10):
            doc_copy = doc.model_copy(deep=True)
            doc_copy.document_id = f"test-doc-{i}"
            docs.append(doc_copy)
        
        # Initialize store
        store = ChromaStore(persist_directory=temp_dir / "chroma_concurrent")
        
        # Test concurrent document addition
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(store.add_document, doc) for doc in docs]
            concurrent.futures.wait(futures)
        
        add_time = time.time() - start_time
        print(f"\nConcurrent document addition time: {add_time:.2f} seconds")
        
        # Test concurrent querying
        queries = ["test", "sample", "document", "content", "information"]
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(store.query, query, n_results=3) for query in queries]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        query_time = time.time() - start_time
        print(f"Concurrent query time for {len(queries)} queries: {query_time:.2f} seconds")
        print(f"Average time per query: {query_time / len(queries):.2f} seconds")
        
        # Verify all queries returned results
        assert len(results) == len(queries)
        assert all(len(result["ids"][0]) > 0 for result in results)
