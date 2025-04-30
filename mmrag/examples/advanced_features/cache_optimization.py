"""Example demonstrating caching optimizations for improved performance."""

import time
import os # Added for psutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import typer
from mmrag.document_processing import PDFProcessor
from mmrag.exceptions import ProcessingTimeoutError, MemoryLimitExceededError # Import exceptions
from mmrag.document_processing.cache import CachedDocumentProcessor
from mmrag.vectordb import ChromaStore
from mmrag.vectordb.cache import EmbeddingCache
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import psutil # Added for memory monitoring
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(help="Benchmark caching optimizations.")
console = Console()


# Add a cache decorator for benchmarking functions
@lru_cache(maxsize=128)
def cached_benchmark_function(data_key: str) -> Dict:
    """Cached function for benchmarking."""
    # This is a placeholder function to demonstrate caching
    time.sleep(0.1)  # Simulate processing
    return {"key": data_key, "processed": True}


def benchmark_caching(
    file_path: Path, 
    iterations: int = 3, 
    clear_cache: bool = False,
    timeout_seconds: int = 60, # Added timeout
    memory_limit_fraction: float = 0.5, # Added memory limit fraction
):
    """Benchmark processing with and without caching."""
    # Create processors
    standard_processor = PDFProcessor(extract_tables=True, extract_images=True)
    cached_processor = CachedDocumentProcessor(PDFProcessor(extract_tables=True, extract_images=True))
    
    # Resource monitoring setup
    start_benchmark_time = time.time()
    process = psutil.Process(os.getpid())
    initial_available_memory = psutil.virtual_memory().available
    memory_limit_bytes = initial_available_memory * memory_limit_fraction
    console.print(f"Resource limits: Timeout={timeout_seconds}s, Memory Limit={memory_limit_bytes / (1024**2):.2f} MB")


    console.print(Panel(f"Benchmarking caching with: [bold blue]{file_path.name}[/]"))
    
    # Create result table
    results = Table("Operation", "Iteration", "Without Cache (s)", "With Cache (s)", "Speedup")
    
    # If requested, clear the cache first to ensure fair benchmark
    if clear_cache:
        # This is a hack to clear the LRU cache - in practice you'd use a more robust approach
        cached_processor = CachedDocumentProcessor(PDFProcessor(extract_tables=True, extract_images=True))
        console.print("[yellow]Cache cleared for benchmarking[/]\n")
    
    # Benchmark document processing
    console.print("[bold]Benchmarking document processing...[/]")
    
    for i in range(iterations):
        with Progress(
            SpinnerColumn(),
            TextColumn(f"Iteration {i+1}/{iterations}..."),
            console=console,
        ) as progress:
            # Standard processing
            # --- Resource Checks ---
            elapsed_time = time.time() - start_benchmark_time
            if elapsed_time > timeout_seconds:
                raise ProcessingTimeoutError(f"Benchmarking exceeded {timeout_seconds} seconds limit.")
                
            current_rss = process.memory_info().rss
            if current_rss > memory_limit_bytes:
                raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
            # --- End Resource Checks ---
            task = progress.add_task("Running standard processor...", total=None)
            start_time = time.time()
            standard_doc = standard_processor.process(file_path)
            standard_time = time.time() - start_time
            progress.update(task, completed=True)
            
            # Cached processing
            # --- Resource Checks ---
            elapsed_time = time.time() - start_benchmark_time
            if elapsed_time > timeout_seconds:
                raise ProcessingTimeoutError(f"Benchmarking exceeded {timeout_seconds} seconds limit.")
                
            current_rss = process.memory_info().rss
            if current_rss > memory_limit_bytes:
                raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
            # --- End Resource Checks ---
            task = progress.add_task("Running cached processor...", total=None)
            start_time = time.time()
            cached_doc = cached_processor.process(str(file_path))  # Cache key must be hashable (string)
            cached_time = time.time() - start_time
            progress.update(task, completed=True)
        
        # Calculate speedup
        speedup = standard_time / max(cached_time, 0.001)  # Avoid division by zero
        
        results.add_row(
            "Document Processing", 
            str(i+1), 
            f"{standard_time:.4f}", 
            f"{cached_time:.4f}",
            f"{speedup:.2f}x"
        )
    
    # Set up vector stores with context manager for cleanup
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        console.print(f"\n[bold]Creating vector stores in temporary directory...[/]")
        
        # Setup progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("Setting up vector stores..."),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing...", total=None)
            
            # Create standard store
            standard_store = ChromaStore(persist_directory=temp_dir / "standard")
            
            # Create a store with embedding cache
            embedding_cache = EmbeddingCache(cache_dir=temp_dir / "embeddings")
            cached_store = ChromaStore(persist_directory=temp_dir / "cached")
            cached_store.embedding_cache = embedding_cache
            
            # First run with both stores to populate database
            progress.update(task, description="Adding document to standard store...")
            standard_store.add_document(standard_doc)
            
            progress.update(task, description="Adding document to cached store...")
            cached_store.add_document(cached_doc)
            
            progress.update(task, completed=True)
        
        # Test queries
        console.print("\n[bold]Benchmarking vector store queries...[/]")
        
        test_queries = [
            "What is the main topic?",
            "Explain the key points",
            "What are the conclusions?",
        ]
        
        # Benchmark queries
        for query_idx, query in enumerate(test_queries):
            console.print(f"\n[cyan]Query {query_idx+1}:[/] \"{query}\"")
            
            for i in range(iterations):
                with Progress(
                    SpinnerColumn(),
                    TextColumn(f"Iteration {i+1}/{iterations}..."),
                    console=console,
                ) as progress:
                    # Standard query
                    # --- Resource Checks ---
                    elapsed_time = time.time() - start_benchmark_time
                    if elapsed_time > timeout_seconds:
                        raise ProcessingTimeoutError(f"Benchmarking exceeded {timeout_seconds} seconds limit.")
                        
                    current_rss = process.memory_info().rss
                    if current_rss > memory_limit_bytes:
                        raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
                    # --- End Resource Checks ---
                    task = progress.add_task("Running standard query...", total=None)
                    start_time = time.time()
                    standard_results = standard_store.query(query, n_results=3)
                    standard_time = time.time() - start_time
                    progress.update(task, completed=True)
                    
                    # Cached query
                    # --- Resource Checks ---
                    elapsed_time = time.time() - start_benchmark_time
                    if elapsed_time > timeout_seconds:
                        raise ProcessingTimeoutError(f"Benchmarking exceeded {timeout_seconds} seconds limit.")
                        
                    current_rss = process.memory_info().rss
                    if current_rss > memory_limit_bytes:
                        raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
                    # --- End Resource Checks ---
                    task = progress.add_task("Running cached query...", total=None)
                    start_time = time.time()
                    cached_results = cached_store.query(query, n_results=3)
                    cached_time = time.time() - start_time
                    progress.update(task, completed=True)
                
                # Calculate speedup
                speedup = standard_time / max(cached_time, 0.001)
                
                results.add_row(
                    f"Query: {query[:20]}...", 
                    str(i+1), 
                    f"{standard_time:.4f}", 
                    f"{cached_time:.4f}",
                    f"{speedup:.2f}x"
                )
    
    console.print("\n[bold]Benchmark Results:[/]")
    console.print(results)
    
    # Summary
    console.print("\n[bold]Performance Impact of Caching:[/]")
    console.print("1. Document processing: Subsequent accesses to the same document are nearly instantaneous")
    console.print("2. Embedding generation: Cached embeddings avoid recomputation, especially valuable for large documents")
    console.print("3. Query processing: Caching reduces both embedding time and potentially retrieval time")
    
    # Add example of in-memory function caching
    console.print("\n[bold]Function-level Caching Example:[/]")
    
    # Benchmark uncached vs cached function
    uncached_times = []
    cached_times = []
    
    # Define an uncached version
    def uncached_function(data_key):
        time.sleep(0.1)  # Simulate processing
        return {"key": data_key, "processed": True}
    
    # Run benchmark
    with Progress(
        SpinnerColumn(),
        TextColumn("Benchmarking function caching..."),
        console=console,
    ) as progress:
        task = progress.add_task("Running benchmark...", total=10)
        
        for i in range(5):
            # Uncached
            start_time = time.time()
            uncached_function("test_data")
            uncached_times.append(time.time() - start_time)
            progress.advance(task)
            
            # Cached
            start_time = time.time()
            cached_benchmark_function("test_data")
            cached_times.append(time.time() - start_time)
            progress.advance(task)
    
    # Display results
    cache_table = Table("Run", "Uncached (s)", "Cached (s)", "Speedup")
    
    for i, (uncached, cached) in enumerate(zip(uncached_times, cached_times)):
        speedup = uncached / max(cached, 0.001)
        cache_table.add_row(
            str(i+1),
            f"{uncached:.4f}",
            f"{cached:.4f}",
            f"{speedup:.2f}x"
        )
    
    console.print(cache_table)
    
    # Memory usage tip
    console.print("\n[bold]Memory Usage Tip:[/]")
    console.print("While caching improves performance, it increases memory usage. For production systems:")
    console.print("1. Consider cache eviction strategies (LRU, TTL, etc.)")
    console.print("2. Monitor memory usage and set appropriate cache size limits")
    console.print("3. Use persistent caching for large datasets")
    
    return standard_doc


@app.command()
def run(
    file_path: Path = typer.Argument(..., help="Path to the document file"),
    iterations: int = typer.Option(3, "--iterations", "-i", help="Number of iterations for benchmarking"),
    clear_cache: bool = typer.Option(False, "--clear-cache", help="Clear cache before benchmarking"),
    timeout: int = typer.Option(60, "--timeout", help="Processing timeout in seconds"),
    mem_limit: float = typer.Option(0.5, "--mem-limit", help="Memory limit as fraction of available memory (0.1-1.0)"),
):
    """Run the caching benchmark."""
    if not file_path.exists():
        console.print(f"[bold red]Error:[/] File {file_path} not found")
        raise typer.Exit(code=1)
    
    if file_path.suffix.lower() != ".pdf":
        console.print(f"[bold red]Error:[/] File {file_path} is not a PDF")
        raise typer.Exit(code=1)
    
    # Validate memory limit
    if not (0.1 <= mem_limit <= 1.0):
        console.print("[bold red]Error:[/] Memory limit must be between 0.1 and 1.0")
        raise typer.Exit(code=1)

    benchmark_caching(
        file_path, 
        iterations, 
        clear_cache,
        timeout_seconds=timeout,
        memory_limit_fraction=mem_limit)


if __name__ == "__main__":
    app()