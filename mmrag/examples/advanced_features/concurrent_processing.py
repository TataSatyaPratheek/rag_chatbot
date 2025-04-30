"""Example of concurrent document processing for improved throughput."""

import argparse
import concurrent.futures
import glob
import os
import time
from pathlib import Path

from mmrag.document_processing.factory import get_processor
from mmrag.vectordb import ChromaStore
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()

def process_document(file_path, enable_tables=True, enable_images=True):
    """Process a single document."""
    try:
        # Get the appropriate processor for this file type
        processor = get_processor(
            file_path,
            extract_tables=enable_tables,
            extract_images=enable_images
        )
        
        # Process the document
        doc = processor.process(file_path)
        
        return {
            "success": True,
            "document": doc,
            "file_path": file_path,
            "element_count": len(doc.elements),
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "document": None,
            "file_path": file_path,
            "element_count": 0,
            "error": str(e)
        }

def process_documents_sequentially(file_paths, enable_tables=True, enable_images=True):
    """Process documents one after another."""
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"Processing {len(file_paths)} documents sequentially...", total=len(file_paths))
        
        for file_path in file_paths:
            progress.update(task, description=f"Processing {Path(file_path).name}")
            result = process_document(file_path, enable_tables, enable_images)
            results.append(result)
            progress.advance(task)
    
    return results

def process_documents_concurrently(file_paths, max_workers=4, enable_tables=True, enable_images=True):
    """Process documents concurrently using a thread pool."""
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"Processing {len(file_paths)} documents concurrently...", total=len(file_paths))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(process_document, path, enable_tables, enable_images): path
                for path in file_paths
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                progress.update(task, description=f"Completed {Path(path).name}")
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        "success": False,
                        "document": None,
                        "file_path": path,
                        "element_count": 0,
                        "error": str(e)
                    })
                progress.advance(task)
    
    return results

def index_documents(documents, collection_name="concurrent_documents"):
    """Index processed documents in a vector store."""
    # Create a temporary directory for ChromaDB
    store_dir = Path("temp_concurrent_processing")
    store_dir.mkdir(exist_ok=True)
    
    # Initialize ChromaDB
    store = ChromaStore(
        persist_directory=store_dir,
        collection_name=collection_name
    )
    
    # Add documents to the store
    for doc in documents:
        store.add_document(doc)
    
    return store

def main(directory, pattern="*.pdf", concurrent=True, max_workers=4, enable_tables=True, enable_images=True):
    """Process all matching documents in a directory."""
    # Find matching files
    file_paths = glob.glob(os.path.join(directory, pattern))
    
    if not file_paths:
        console.print(f"[bold red]No files matching '{pattern}' found in '{directory}'[/]")
        return
    
    console.print(f"Found [bold]{len(file_paths)}[/] files matching '{pattern}' in '{directory}'")
    
    # Measure processing time
    start_time = time.time()
    
    # Process documents
    if concurrent:
        results = process_documents_concurrently(
            file_paths, 
            max_workers=max_workers,
            enable_tables=enable_tables,
            enable_images=enable_images
        )
    else:
        results = process_documents_sequentially(
            file_paths,
            enable_tables=enable_tables,
            enable_images=enable_images
        )
    
    processing_time = time.time() - start_time
    
    # Report results
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    console.print(f"\n[bold green]Processing completed in {processing_time:.2f} seconds[/]")
    console.print(f"Successfully processed: [bold green]{len(successful)}/{len(results)}[/]")
    
    if failed:
        console.print(f"Failed to process: [bold red]{len(failed)}/{len(results)}[/]")
        console.print("[bold]Failures:[/]")
        for failure in failed:
            console.print(f"- {Path(failure['file_path']).name}: {failure['error']}")
    
    # Calculate statistics
    if successful:
        total_elements = sum(r["element_count"] for r in successful)
        avg_elements = total_elements / len(successful)
        
        console.print(f"\n[bold]Processing Stats:[/]")
        console.print(f"Total elements extracted: {total_elements}")
        console.print(f"Average elements per document: {avg_elements:.1f}")
        console.print(f"Documents per second: {len(successful) / processing_time:.2f}")
        
        # Index successfully processed documents
        if len(successful) > 0:
            console.print("\n[bold]Indexing documents in vector store...[/]")
            try:
                documents = [r["document"] for r in successful]
                index_documents(documents)
                console.print(f"[bold green]Successfully indexed {len(documents)} documents[/]")
            except Exception as e:
                console.print(f"[bold red]Error indexing documents: {e}[/]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concurrent document processing example.")
    parser.add_argument("directory", type=str, help="Directory containing documents to process")
    parser.add_argument("--pattern", "-p", type=str, default="*.pdf", help="File pattern to match (default: *.pdf)")
    parser.add_argument("--sequential", "-s", action="store_true", help="Process sequentially instead of concurrently")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Maximum number of worker threads (default: 4)")
    parser.add_argument("--no-tables", action="store_true", help="Disable table extraction")
    parser.add_argument("--no-images", action="store_true", help="Disable image extraction")
    
    args = parser.parse_args()
    
    main(
        args.directory,
        pattern=args.pattern,
        concurrent=not args.sequential,
        max_workers=args.workers,
        enable_tables=not args.no_tables,
        enable_images=not args.no_images
    )
