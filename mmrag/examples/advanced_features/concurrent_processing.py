"""Example of concurrent document processing for improved throughput."""

import asyncio
import concurrent.futures
import glob
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import typer
from mmrag.document_processing.factory import get_processor
from mmrag.vectordb import ChromaStore
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(help="Concurrent document processing example.")
console = Console()


def process_document(file_path: Path, enable_tables: bool = True, enable_images: bool = True) -> Dict[str, Any]:
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
            "file_path": str(file_path),
            "element_count": len(doc.elements),
            "error": None
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {
            "success": False,
            "document": None,
            "file_path": str(file_path),
            "element_count": 0,
            "error": str(e),
            "error_details": error_details
        }


async def process_document_async(file_path: Path, enable_tables: bool = True, enable_images: bool = True) -> Dict[str, Any]:
    """Process a document asynchronously using a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, 
        lambda: process_document(file_path, enable_tables, enable_images)
    )


def process_documents_sequentially(
    file_paths: List[Path], 
    enable_tables: bool = True, 
    enable_images: bool = True
) -> List[Dict[str, Any]]:
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
            progress.update(task, description=f"Processing {file_path.name}")
            result = process_document(file_path, enable_tables, enable_images)
            results.append(result)
            progress.advance(task)
    
    return results


def process_documents_concurrently(
    file_paths: List[Path], 
    max_workers: int = 4, 
    enable_tables: bool = True, 
    enable_images: bool = True
) -> List[Dict[str, Any]]:
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
                progress.update(task, description=f"Completed {path.name}")
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    results.append({
                        "success": False,
                        "document": None,
                        "file_path": str(path),
                        "element_count": 0,
                        "error": str(e),
                        "error_details": error_details
                    })
                progress.advance(task)
    
    return results


async def process_documents_asyncio(
    file_paths: List[Path], 
    max_concurrency: int = 4, 
    enable_tables: bool = True, 
    enable_images: bool = True
) -> List[Dict[str, Any]]:
    """Process documents using asyncio for concurrency.
    
    This can be more efficient than thread pools for I/O-bound operations.
    """
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_with_semaphore(file_path: Path) -> Dict[str, Any]:
        async with semaphore:
            return await process_document_async(file_path, enable_tables, enable_images)
    
    # Create tasks for all files
    tasks = [process_with_semaphore(path) for path in file_paths]
    
    # Process with progress display
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        progress_task = progress.add_task(f"Processing {len(file_paths)} documents with asyncio...", total=len(file_paths))
        
        for i, task in enumerate(asyncio.as_completed(tasks)):
            try:
                result = await task
                results.append(result)
                file_path = Path(result["file_path"])
                progress.update(progress_task, description=f"Completed {file_path.name}", advance=1)
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                console.print(f"[bold red]Error in task:[/] {e}")
                # Since we don't know which file this was for, we'll create a generic error entry
                results.append({
                    "success": False,
                    "document": None,
                    "file_path": "unknown",
                    "element_count": 0,
                    "error": str(e),
                    "error_details": error_details
                })
                progress.advance(progress_task)
    
    return results


def index_documents(documents: List, collection_name: str = "concurrent_documents") -> ChromaStore:
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
        if doc is not None:
            store.add_document(doc)
    
    return store


@app.command()
def main(
    directory: Path = typer.Argument(..., help="Directory containing documents to process"),
    pattern: str = typer.Option("*.pdf", "--pattern", "-p", help="File pattern to match"),
    mode: str = typer.Option(
        "concurrent", 
        "--mode", "-m", 
        help="Processing mode: sequential, concurrent, or asyncio"
    ),
    workers: int = typer.Option(4, "--workers", "-w", help="Maximum number of worker threads/tasks"),
    no_tables: bool = typer.Option(False, "--no-tables", help="Disable table extraction"),
    no_images: bool = typer.Option(False, "--no-images", help="Disable image extraction"),
    index: bool = typer.Option(True, "--index/--no-index", help="Index documents in vector store"),
):
    """Process documents with different concurrency models."""
    # Validate directory
    if not directory.exists() or not directory.is_dir():
        console.print(f"[bold red]Error:[/] Directory {directory} not found")
        raise typer.Exit(code=1)
    
    # Find matching files
    file_paths = list(directory.glob(pattern))
    
    if not file_paths:
        console.print(f"[bold red]No files matching '{pattern}' found in '{directory}'[/]")
        raise typer.Exit(code=1)
    
    console.print(Panel(f"Found [bold]{len(file_paths)}[/] files matching '{pattern}' in '{directory}'"))
    
    # Validate concurrency mode
    valid_modes = ["sequential", "concurrent", "asyncio"]
    if mode not in valid_modes:
        console.print(f"[bold red]Error:[/] Invalid processing mode: {mode}")
        console.print(f"Valid modes: {', '.join(valid_modes)}")
        raise typer.Exit(code=1)
    
    # Validate worker count
    if workers < 1:
        console.print(f"[bold red]Error:[/] Worker count must be at least 1")
        raise typer.Exit(code=1)
    
    console.print(f"[bold]Processing Mode:[/] {mode}")
    console.print(f"[bold]Worker Count:[/] {workers}")
    console.print(f"[bold]Processing Options:[/] Tables: {'Disabled' if no_tables else 'Enabled'}, Images: {'Disabled' if no_images else 'Enabled'}")
    
    # Measure processing time
    start_time = time.time()
    
    # Process documents based on selected mode
    try:
        if mode == "sequential":
            results = process_documents_sequentially(
                file_paths, 
                enable_tables=not no_tables,
                enable_images=not no_images
            )
        elif mode == "concurrent":
            results = process_documents_concurrently(
                file_paths, 
                max_workers=workers,
                enable_tables=not no_tables,
                enable_images=not no_images
            )
        elif mode == "asyncio":
            # For asyncio mode, we need to run the event loop
            results = asyncio.run(process_documents_asyncio(
                file_paths, 
                max_concurrency=workers,
                enable_tables=not no_tables,
                enable_images=not no_images
            ))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Processing interrupted by user[/]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"\n[bold red]Error during processing:[/] {e}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)
    
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
        stats_table = Table("Metric", "Value")
        stats_table.add_row("Total elements extracted", str(total_elements))
        stats_table.add_row("Average elements per document", f"{avg_elements:.1f}")
        stats_table.add_row("Documents per second", f"{len(successful) / processing_time:.2f}")
        stats_table.add_row("Processing time per document", f"{processing_time / len(successful):.2f} seconds")
        console.print(stats_table)
        
        # Index successfully processed documents
        if len(successful) > 0 and index:
            console.print("\n[bold]Indexing documents in vector store...[/]")
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Indexing documents...", total=None)
                    documents = [r["document"] for r in successful]
                    index_documents(documents)
                console.print(f"[bold green]Successfully indexed {len(documents)} documents[/]")
            except Exception as e:
                console.print(f"[bold red]Error indexing documents: {e}[/]")
                import traceback
                console.print(traceback.format_exc())
        elif not index:
            console.print("\n[yellow]Document indexing skipped (--no-index specified)[/]")
    
    # Performance comparison
    if mode != "sequential" and len(successful) > 0:
        speedup = len(successful) / processing_time
        console.print(f"\n[bold]Concurrency Benefit:[/] {speedup:.2f} documents/second")
        console.print(f"With {workers} workers using {mode} mode")


if __name__ == "__main__":
    app()