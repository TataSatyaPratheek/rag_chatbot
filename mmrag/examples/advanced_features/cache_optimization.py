"""Example demonstrating caching optimizations for improved performance."""

import argparse
import time
from pathlib import Path

from mmrag.document_processing import PDFProcessor
from mmrag.document_processing.cache import CachedDocumentProcessor
from mmrag.vectordb import ChromaStore
from mmrag.vectordb.cache import EmbeddingCache
from rich.console import Console
from rich.table import Table

console = Console()


def benchmark_caching(file_path: str, iterations: int = 3):
    """Benchmark processing with and without caching."""
    file_path = Path(file_path)
    
    # Create processors
    standard_processor = PDFProcessor(extract_tables=True, extract_images=True)
    cached_processor = CachedDocumentProcessor(PDFProcessor(extract_tables=True, extract_images=True))
    
    results = Table("Operation", "Iteration", "Without Cache (s)", "With Cache (s)", "Speedup")
    
    # Benchmark document processing
    for i in range(iterations):
        # Standard processing
        start_time = time.time()
        standard_doc = standard_processor.process(file_path)
        standard_time = time.time() - start_time
        
        # Cached processing
        start_time = time.time()
        cached_doc = cached_processor.process(str(file_path))  # Cache key must be hashable (string)
        cached_time = time.time() - start_time
        
        # Calculate speedup
        speedup = standard_time / max(cached_time, 0.001)  # Avoid division by zero
        
        results.add_row(
            "Document Processing", 
            str(i+1), 
            f"{standard_time:.4f}", 
            f"{cached_time:.4f}",
            f"{speedup:.2f}x"
        )
    
    # Set up vector stores
    temp_dir = Path("temp_cache_benchmark")
    temp_dir.mkdir(exist_ok=True)
    
    standard_store = ChromaStore(persist_directory=temp_dir / "standard")
    
    # Create a store with embedding cache
    embedding_cache = EmbeddingCache(cache_dir=temp_dir / "embeddings")
    cached_store = ChromaStore(persist_directory=temp_dir / "cached")
    cached_store.embedding_cache = embedding_cache
    
    # First run with both stores to populate database
    standard_store.add_document(standard_doc)
    cached_store.add_document(cached_doc)
    
    # Test queries
    test_queries = [
        "What is the main topic?",
        "Explain the key points",
        "What are the conclusions?",
    ]
    
    # Benchmark queries
    for query in test_queries:
        for i in range(iterations):
            # Standard query
            start_time = time.time()
            standard_results = standard_store.query(query, n_results=3)
            standard_time = time.time() - start_time
            
            # Cached query
            start_time = time.time()
            cached_results = cached_store.query(query, n_results=3)
            cached_time = time.time() - start_time
            
            # Calculate speedup
            speedup = standard_time / max(cached_time, 0.001)
            
            results.add_row(
                f"Query: {query[:20]}...", 
                str(i+1), 
                f"{standard_time:.4f}", 
                f"{cached_time:.4f}",
                f"{speedup:.2f}x"
            )
    
    console.print(results)
    
    # Summary
    console.print("\n[bold]Performance Impact of Caching:[/]")
    console.print("1. Document processing: Subsequent accesses to the same document are nearly instantaneous")
    console.print("2. Embedding generation: Cached embeddings avoid recomputation, especially valuable for large documents")
    console.print("3. Query processing: Caching reduces both embedding time and potentially retrieval time")
    
    return standard_doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark caching optimizations.")
    parser.add_argument("file_path", type=str, help="Path to the document file")
    parser.add_argument("--iterations", "-i", type=int, default=3, help="Number of iterations for benchmarking")
    args = parser.parse_args()
    
    benchmark_caching(args.file_path, args.iterations)
