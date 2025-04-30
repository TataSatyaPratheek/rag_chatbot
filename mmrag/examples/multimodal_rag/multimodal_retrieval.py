"""Enhanced example of multimodal document processing and retrieval."""

import json
import os # Added for psutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import typer
from mmrag.document_processing.factory import get_processor
from mmrag.exceptions import ProcessingTimeoutError, MemoryLimitExceededError # Import exceptions
from mmrag.vectordb import ChromaStore
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
import psutil # Added for memory monitoring
from rich.markdown import Markdown
from rich.tree import Tree

app = typer.Typer(help="Process and search multimodal documents.")
console = Console()


def process_multimodal(
    file_path: Path, 
    query: Optional[str] = None,
    extract_tables: bool = True,
    extract_images: bool = True,
    advanced_tables: bool = False,
    enhanced_visual: bool = False,
    store_dir: Optional[Path] = None,
    collection_name: str = "multimodal_retrieval",
    n_results: int = 5,
    visual_only: bool = False,
    output_json: Optional[Path] = None,
    timeout_seconds: int = 60, # Added timeout (longer for processing + indexing)
    memory_limit_fraction: float = 0.5, # Added memory limit fraction
) -> Dict[str, Any]:
    """Process a document with multimodal content and perform retrieval.
    
    Args:
        file_path: Path to the document file
        query: Optional query to search for
        extract_tables: Whether to extract tables
        extract_images: Whether to extract images
        advanced_tables: Whether to use advanced table detection
        enhanced_visual: Whether to use enhanced visual processing
        store_dir: Directory to store the vector database
        collection_name: Name of the vector collection
        n_results: Number of results to return for queries
        visual_only: Only search for visual elements (images, charts)
        output_json: Path to save results as JSON
        timeout_seconds: Maximum processing time in seconds.
        memory_limit_fraction: Maximum fraction of available memory to use.
        
    Returns:
        Dictionary with processing and query results
    """
    # Validate file exists
    if not file_path.exists():
        console.print(f"[bold red]Error:[/] File {file_path} does not exist")
        raise typer.Exit(code=1)
    
    # Resource monitoring setup
    start_process_time = time.time()
    process = psutil.Process(os.getpid())
    initial_available_memory = psutil.virtual_memory().available
    memory_limit_bytes = initial_available_memory * memory_limit_fraction
    console.print(f"Resource limits: Timeout={timeout_seconds}s, Memory Limit={memory_limit_bytes / (1024**2):.2f} MB")

    # Get appropriate processor for this file type
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing document processor...", total=None)
            
            processor = get_processor(
                file_path,
                extract_tables=extract_tables,
                extract_images=extract_images,
                advanced_table_detection=advanced_tables,
                enable_enhanced_visual=enhanced_visual,
            )
            
            progress.update(task, completed=True, description=f"Using {processor.__class__.__name__} for {file_path.suffix} files")
    except ValueError as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Unexpected error initializing processor:[/] {e}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)
    
    # Process the document
    try:
        start_time = time.time()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Processing {file_path.name}...", total=None)
            
            # --- Resource Checks ---
            elapsed_time = time.time() - start_process_time
            if elapsed_time > timeout_seconds:
                raise ProcessingTimeoutError(f"Processing exceeded {timeout_seconds} seconds limit.")
                
            current_rss = process.memory_info().rss
            if current_rss > memory_limit_bytes:
                raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
            # --- End Resource Checks ---
            document = processor.process(file_path)
            processing_time = time.time() - start_time
            progress.update(task, completed=True, description=f"Processed {file_path.name} in {processing_time:.2f}s")
    except Exception as e:
        console.print(f"[bold red]Error processing document:[/] {e}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)
    
    # Count element types
    element_types = {}
    for element in document.elements:
        element_types[element.element_type] = element_types.get(element.element_type, 0) + 1
    
    # Display element counts
    console.print("\n[bold]Extracted elements:[/]")
    element_table = Table("Element Type", "Count", "Percentage")
    
    total_elements = len(document.elements)
    for element_type, count in element_types.items():
        percentage = (count / total_elements * 100) if total_elements > 0 else 0
        element_table.add_row(element_type, str(count), f"{percentage:.1f}%")
    
    console.print(element_table)
    
    # Determine vector store directory
    if store_dir:
        store_dir = Path(store_dir)
        store_dir.mkdir(exist_ok=True, parents=True)
        temp_dir = None
    else:
        # Use temporary directory if no store_dir provided
        temp_dir = tempfile.TemporaryDirectory()
        store_dir = Path(temp_dir.name)
    
    # Set up vector store
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Setting up vector store...", total=None)
            
            store = ChromaStore(
                persist_directory=store_dir,
                collection_name=collection_name
            )
            
            progress.update(task, description="Adding document to vector store...")
            
            # --- Resource Checks ---
            elapsed_time = time.time() - start_process_time
            if elapsed_time > timeout_seconds:
                raise ProcessingTimeoutError(f"Indexing exceeded {timeout_seconds} seconds limit.")
                
            current_rss = process.memory_info().rss
            if current_rss > memory_limit_bytes:
                raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) before indexing exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
            # --- End Resource Checks ---

            # Add document to vector store
            indexing_start = time.time()
            store.add_document(document)
            indexing_time = time.time() - indexing_start
            
            progress.update(task, completed=True, description=f"Indexed document in {indexing_time:.2f}s")
    except Exception as e:
        console.print(f"[bold red]Error setting up vector store:[/] {e}")
        if temp_dir:
            temp_dir.cleanup()
        raise typer.Exit(code=1)
    
    # Create results dictionary
    results = {
        "document_id": document.document_id,
        "filename": file_path.name,
        "processing_stats": {
            "processing_time": processing_time,
            "indexing_time": indexing_time,
            "total_elements": total_elements,
            "element_types": element_types
        },
        "query_results": None
    }
    
    # Perform retrieval if query provided
    if query:
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Searching for: {query}", total=None)
                
                # --- Resource Checks ---
                elapsed_time = time.time() - start_process_time
                if elapsed_time > timeout_seconds:
                    raise ProcessingTimeoutError(f"Query preparation exceeded {timeout_seconds} seconds limit.")
                    
                current_rss = process.memory_info().rss
                if current_rss > memory_limit_bytes:
                    raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) before query exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
                # --- End Resource Checks ---

                # Prepare where filter for visual-only search
                where = None
                if visual_only:
                    where = {"element_type": {"$in": ["image", "chart"]}}
                
                # Perform search
                query_start = time.time()
                search_results = store.query(
                    query_text=query,
                    n_results=n_results,
                    where=where
                )
                query_time = time.time() - query_start

                # --- Resource Checks ---
                elapsed_time = time.time() - start_process_time
                if elapsed_time > timeout_seconds:
                    raise ProcessingTimeoutError(f"Query exceeded {timeout_seconds} seconds limit.")
                    
                current_rss = process.memory_info().rss
                if current_rss > memory_limit_bytes:
                    raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) after query exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
                # --- End Resource Checks ---
                
                progress.update(task, completed=True, description=f"Found {len(search_results['ids'][0])} results in {query_time:.3f}s")
            
            # Format results for display and output
            formatted_results = []
            
            for i, (doc_content, metadata, doc_id) in enumerate(zip(
                search_results["documents"][0],
                search_results["metadatas"][0],
                search_results["ids"][0]
            )):
                # Get distance/score if available
                score = None
                if "distances" in search_results and search_results["distances"][0]:
                    distance = search_results["distances"][0][i]
                    score = round((1 - (distance / 2)) * 100)  # Convert to percentage
                
                # Format result
                formatted_result = {
                    "rank": i+1,
                    "element_id": metadata.get("element_id", "unknown"),
                    "element_type": metadata.get("element_type", "unknown"),
                    "page": metadata.get("page", 0),
                    "content": doc_content[:500] + ("..." if len(doc_content) > 500 else ""),
                    "relevance": score
                }
                
                formatted_results.append(formatted_result)
            
            # Store in results dictionary
            results["query_results"] = {
                "query": query,
                "time": query_time,
                "count": len(search_results["ids"][0]),
                "results": formatted_results
            }
            
            # Display results
            console.print(f"\n[bold green]Retrieved {len(search_results['ids'][0])} results:[/]")
            
            results_table = Table("Rank", "Element Type", "Page", "Content", "Relevance")
            
            for result in formatted_results:
                # Truncate content for display
                display_content = result["content"]
                if len(display_content) > 100:
                    display_content = display_content[:100] + "..."
                
                results_table.add_row(
                    str(result["rank"]),
                    result["element_type"],
                    str(result["page"]),
                    display_content,
                    f"{result['relevance']}%" if result["relevance"] is not None else "N/A"
                )
            
            console.print(results_table)
        except Exception as e:
            console.print(f"[bold red]Error querying vector store:[/] {e}")
            import traceback
            console.print(traceback.format_exc())
            results["query_error"] = str(e)
    
    # Save results if requested
    if output_json:
        try:
            with open(output_json, "w") as f:
                json.dump(results, f, indent=2)
            console.print(f"\nResults saved to: [bold blue]{output_json}[/]")
        except Exception as e:
            console.print(f"[bold red]Error saving results:[/] {e}")
    
    # Clean up temporary directory if used
    if temp_dir:
        temp_dir.cleanup()
    
    return results


def display_document_structure(document_path: Path, max_pages: int = 5) -> None:
    """Display the structure of a document without processing it fully.
    
    Args:
        document_path: Path to the document file
        max_pages: Maximum number of pages to analyze
    """
    try:
        # Get processor for this file type
        processor = get_processor(document_path)
        
        # For PDF files, we can try to get more info
        if document_path.suffix.lower() == ".pdf":
            try:
                import fitz  # PyMuPDF
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task(f"Analyzing {document_path.name}...", total=None)
                    
                    # Open document
                    doc = fitz.open(document_path)
                    page_count = len(doc)
                    
                    # Create structure tree
                    tree = Tree(f"[bold blue]{document_path.name}[/] ({page_count} pages)")
                    
                    # Process limited number of pages
                    pages_to_process = min(page_count, max_pages)
                    
                    for page_idx in range(pages_to_process):
                        page = doc[page_idx]
                        
                        # Get page info
                        page_branch = tree.add(f"Page {page_idx+1}")
                        
                        # Text blocks
                        blocks = page.get_text("blocks")
                        text_count = len([b for b in blocks if b[6] == 0])  # Type 0 is text
                        
                        # Images
                        images = page.get_images()
                        image_count = len(images)
                        
                        # Tables (approximation based on rectangles)
                        # This is a simple heuristic, not accurate
                        rects = page.search_for("", quads=True)
                        potential_tables = len(rects) > 10
                        
                        # Add elements to page branch
                        if text_count > 0:
                            page_branch.add(f"Text blocks: {text_count}")
                        if image_count > 0:
                            page_branch.add(f"Images: {image_count}")
                        if potential_tables:
                            page_branch.add("Potential tables detected")
                    
                    # Add note about limited pages
                    if page_count > max_pages:
                        tree.add(f"[dim]... and {page_count - max_pages} more pages[/]")
                    
                    progress.update(task, completed=True)
                
                console.print(Panel(tree, title="Document Structure"))
            except ImportError:
                console.print("[yellow]Note:[/] Install PyMuPDF for enhanced document structure analysis")
                console.print(f"Document: [bold blue]{document_path.name}[/]")
        else:
            console.print(f"Document: [bold blue]{document_path.name}[/]")
    except Exception as e:
        console.print(f"[bold red]Error analyzing document structure:[/] {e}")


@app.command()
def process(
    file_path: Path = typer.Argument(..., help="Path to the document file"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Query to search for in the document"),
    no_tables: bool = typer.Option(False, "--no-tables", help="Disable table extraction"),
    no_images: bool = typer.Option(False, "--no-images", help="Disable image extraction"),
    advanced_tables: bool = typer.Option(False, "--advanced-tables", help="Use advanced table detection"),
    enhanced_visual: bool = typer.Option(False, "--enhanced-visual", help="Use enhanced visual processing"),
    results: int = typer.Option(5, "--results", "-r", help="Number of results to return"),
    visual_only: bool = typer.Option(False, "--visual-only", help="Only search visual elements"),
    output_json: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results to JSON file"),
    timeout: int = typer.Option(60, "--timeout", help="Processing timeout in seconds"),
    mem_limit: float = typer.Option(0.5, "--mem-limit", help="Memory limit as fraction of available memory (0.1-1.0)"),
    analyze_only: bool = typer.Option(False, "--analyze-only", help="Only analyze document structure without processing"),
):
    """Process a multimodal document and perform retrieval."""
    # Check if file exists
    if not file_path.exists():
        console.print(f"[bold red]Error:[/] File {file_path} does not exist")
        raise typer.Exit(code=1)

    # Validate memory limit
    if not (0.1 <= mem_limit <= 1.0):
        console.print(f"[bold red]Error:[/] File {file_path} does not exist")
        raise typer.Exit(code=1)
    
    # Just analyze document structure if requested
    if analyze_only:
        display_document_structure(file_path)
        return
    
    try:
        # Process document
        results = process_multimodal(
            file_path=file_path,
            query=query,
            extract_tables=not no_tables,
            extract_images=not no_images,
            advanced_tables=advanced_tables,
            enhanced_visual=enhanced_visual,
            n_results=results,
            visual_only=visual_only,
            output_json=output_json,
            timeout_seconds=timeout,
            memory_limit_fraction=mem_limit,
        )
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Processing interrupted by user[/]")
        raise typer.Exit(code=1)


@app.command()
def interactive(
    file_path: Path = typer.Argument(..., help="Path to the document file"),
    no_tables: bool = typer.Option(False, "--no-tables", help="Disable table extraction"),
    no_images: bool = typer.Option(False, "--no-images", help="Disable image extraction"),
    advanced_tables: bool = typer.Option(False, "--advanced-tables", help="Use advanced table detection"),
    enhanced_visual: bool = typer.Option(False, "--enhanced-visual", help="Use enhanced visual processing"),
    store_dir: Optional[Path] = typer.Option(None, "--store-dir", "-s", help="Directory to store the vector database"),
    collection: str = typer.Option("multimodal_interactive", "--collection", "-c", help="Collection name"),
    timeout: int = typer.Option(60, "--timeout", help="Processing timeout in seconds"),
    mem_limit: float = typer.Option(0.5, "--mem-limit", help="Memory limit as fraction of available memory (0.1-1.0)"),
):
    """Run an interactive query session with a multimodal document."""
    # Check if file exists
    if not file_path.exists():
        console.print(f"[bold red]Error:[/] File {file_path} does not exist")
        raise typer.Exit(code=1)

    # Validate memory limit
    if not (0.1 <= mem_limit <= 1.0):
        console.print("[bold red]Error:[/] Memory limit must be between 0.1 and 1.0")
        raise typer.Exit(code=1)
    
    # Process document first
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Processing {file_path.name}...", total=None)
            
            # Resource monitoring setup for initial processing
            start_time = time.time()
            process = psutil.Process(os.getpid())
            initial_available_memory = psutil.virtual_memory().available
            memory_limit_bytes = initial_available_memory * mem_limit
            console.print(f"Resource limits: Timeout={timeout}s, Memory Limit={memory_limit_bytes / (1024**2):.2f} MB")
            
            # Get appropriate processor
            processor = get_processor(
                file_path,
                extract_tables=not no_tables,
                extract_images=not no_images,
                advanced_table_detection=advanced_tables,
                enable_enhanced_visual=enhanced_visual,
            )
            
            # --- Resource Checks ---
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                raise ProcessingTimeoutError(f"Processing exceeded {timeout} seconds limit.")
                
            current_rss = process.memory_info().rss
            if current_rss > memory_limit_bytes:
                raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
            # --- End Resource Checks ---

            # Process document
            document = processor.process(file_path)
            
            progress.update(task, completed=True)
        
        # Count element types
        element_types = {}
        for element in document.elements:
            element_types[element.element_type] = element_types.get(element.element_type, 0) + 1
        
        # Display element counts
        console.print("\n[bold]Extracted elements:[/]")
        for element_type, count in element_types.items():
            console.print(f"  {element_type}: {count}")
        
        # Determine vector store directory
        if store_dir:
            store_dir = Path(store_dir)
            store_dir.mkdir(exist_ok=True, parents=True)
            temp_dir = None
        else:
            # Use temporary directory if no store_dir provided
            temp_dir = tempfile.TemporaryDirectory()
            store_dir = Path(temp_dir.name)
        
        # Set up vector store
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Setting up vector store...", total=None)
            
            store = ChromaStore(
                persist_directory=store_dir,
                collection_name=collection
            )
            
            # --- Resource Checks ---
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                raise ProcessingTimeoutError(f"Indexing exceeded {timeout} seconds limit.")
                
            current_rss = process.memory_info().rss
            if current_rss > memory_limit_bytes:
                raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) before indexing exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
            # --- End Resource Checks ---

            # Add document to vector store
            store.add_document(document)
            
            progress.update(task, completed=True)
        
        # Start interactive query session
        console.print(Panel.fit(
            "Interactive Query Session\n"
            "Type a query to search the document, or use these special commands:\n"
            "- 'filter <type>' to filter by element type (e.g., 'filter table')\n"
            "- 'filter clear' to clear element type filter\n"
            "- 'results <n>' to set number of results (e.g., 'results 10')\n"
            "- 'quit' to exit",
            title=f"Querying {file_path.name}",
            border_style="blue"
        ))
        
        # Query loop
        current_filter = None
        n_results = 5
        # Reset start time for query phase timeout
        start_query_phase_time = time.time()
        query_timeout = 15 # Shorter timeout for interactive queries
        query_memory_limit_bytes = memory_limit_bytes # Reuse memory limit
        
        while True:
            # Show current filter if any
            filter_text = f" (filtered to: {current_filter})" if current_filter else ""
            
            # Get user input
            user_input = console.input(f"\n[bold cyan]Query{filter_text}:[/] ")
            
            if user_input.lower() == "quit":
                break
                
            elif user_input.lower().startswith("filter "):
                filter_type = user_input[7:].strip().lower()
                
                if filter_type in ["none", "clear"]:
                    current_filter = None
                    console.print("[bold yellow]Filter cleared[/]")
                else:
                    # Validate filter type
                    if filter_type in element_types:
                        current_filter = filter_type
                        console.print(f"[bold yellow]Filter set to: {current_filter}[/]")
                    else:
                        console.print(f"[bold red]Error:[/] No elements of type '{filter_type}' found")
                        console.print(f"Available types: {', '.join(element_types.keys())}")
                
                continue
                
            elif user_input.lower().startswith("results "):
                try:
                    n_results = int(user_input[8:].strip())
                    console.print(f"[bold yellow]Number of results set to {n_results}[/]")
                except ValueError:
                    console.print("[bold red]Error:[/] Invalid number")
                
                continue
            
            # Process query
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task(f"Searching: {user_input}", total=None)
                    
                    # --- Resource Checks ---
                    elapsed_time = time.time() - start_query_phase_time
                    if elapsed_time > query_timeout:
                        raise ProcessingTimeoutError(f"Query exceeded {query_timeout} seconds limit.")
                        
                    current_rss = process.memory_info().rss
                    if current_rss > query_memory_limit_bytes:
                        raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) during query exceeded limit ({query_memory_limit_bytes / (1024**2):.2f} MB).")
                    # --- End Resource Checks ---

                    # Prepare where filter
                    where = {"element_type": current_filter} if current_filter else None
                    
                    # Perform search
                    start_time = time.time()
                    results = store.query(
                        query_text=user_input,
                        n_results=n_results,
                        where=where
                    )
                    query_time = time.time() - start_time
                    
                    progress.update(task, completed=True, description=f"Found {len(results['ids'][0])} results in {query_time:.3f}s")
                
                # Display results
                console.print(f"\n[bold green]Retrieved {len(results['ids'][0])} results:[/]")
                
                # Check if we have results
                if len(results["ids"][0]) == 0:
                    console.print("[yellow]No results found for this query[/]")
                    continue
                
                # Create results table
                results_table = Table("Rank", "Element Type", "Page", "Content")
                
                for i, (doc_content, metadata) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0]
                )):
                    element_type = metadata.get("element_type", "unknown")
                    page = metadata.get("page", "unknown")
                    
                    # Truncate content for display
                    display_content = doc_content[:100] + "..." if len(doc_content) > 100 else doc_content
                    
                    results_table.add_row(
                        str(i+1),
                        element_type,
                        str(page),
                        display_content
                    )
                
                console.print(results_table)
            except Exception as e:
                console.print(f"[bold red]Error:[/] {e}")
        
        # Clean up
        console.print("[bold green]Query session ended[/]")
        if temp_dir:
            temp_dir.cleanup()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Query session interrupted by user[/]")
        # Clean up temp directory if needed
        if 'temp_dir' in locals() and temp_dir:
            temp_dir.cleanup()
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        import traceback
        console.print(traceback.format_exc())
        # Clean up temp directory if needed
        if 'temp_dir' in locals() and temp_dir:
            temp_dir.cleanup()


@app.command()
def batch(
    directory: Path = typer.Argument(..., help="Directory containing documents to process"),
    query: str = typer.Argument(..., help="Query to search for in all documents"),
    pattern: str = typer.Option("*.pdf", "--pattern", "-p", help="File pattern to match"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit number of files to process"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Output directory for results"),
    results_per_doc: int = typer.Option(3, "--results-per-doc", "-r", help="Number of results per document"),
    top_overall: int = typer.Option(10, "--top-overall", "-t", help="Number of top results to display across all documents"),
    timeout: int = typer.Option(60, "--timeout", help="Processing timeout per file in seconds"),
    mem_limit: float = typer.Option(0.5, "--mem-limit", help="Memory limit per file as fraction of available memory (0.1-1.0)"),
):
    """Process multiple documents and search for a query."""
    # Validate directory exists
    if not directory.exists() or not directory.is_dir():
        console.print(f"[bold red]Error:[/] Directory {directory} does not exist")
        raise typer.Exit(code=1)

    # Validate memory limit
    if not (0.1 <= mem_limit <= 1.0):
        if not directory.exists() or not directory.is_dir():
            console.print(f"[bold red]Error:[/] Directory {directory} does not exist")
            raise typer.Exit(code=1)
    
    # Find matching files
    files = list(directory.glob(pattern))
    files = [f for f in files if f.is_file()]
    
    if not files:
        console.print(f"[bold red]Error:[/] No files matching pattern '{pattern}' found in {directory}")
        raise typer.Exit(code=1)
    
    # Apply limit if specified
    if limit and limit > 0 and limit < len(files):
        console.print(f"Limiting to {limit} files (out of {len(files)} found)")
        files = files[:limit]
    else:
        console.print(f"Found {len(files)} files matching pattern '{pattern}'")
    
    # Create output directory if specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"Results will be saved to: [bold blue]{output_dir}[/]")
    
    # Initialize results collection
    all_results = []
    
    # Process each file
    for i, file_path in enumerate(files):
        console.print(f"\n[bold]Processing file {i+1}/{len(files)}:[/] {file_path.name}")
        
        try:
            # Process document and query
            with tempfile.TemporaryDirectory() as temp_dir:
                results = process_multimodal(
                    file_path=file_path,
                    query=query,
                    extract_tables=True,
                    extract_images=True,
                    n_results=results_per_doc,
                    store_dir=Path(temp_dir),
                    timeout_seconds=timeout, # Pass timeout
                    memory_limit_fraction=mem_limit, # Pass mem limit
                )
            
            # Add filename and document info to results
            if "query_results" in results and results["query_results"] and "results" in results["query_results"]:
                for result in results["query_results"]["results"]:
                    result["filename"] = file_path.name
                    result["document_id"] = results["document_id"]
                
                # Add results to collection
                all_results.extend(results["query_results"]["results"])
            
            # Save individual results if output directory specified
            if output_dir:
                output_file = output_dir / f"{file_path.stem}_query_results.json"
                try:
                    with open(output_file, "w") as f:
                        json.dump(results, f, indent=2)
                except Exception as e:
                    console.print(f"[bold red]Error saving results for {file_path.name}:[/] {e}")
        except Exception as e:
            console.print(f"[bold red]Error processing {file_path.name}:[/] {e}")
            import traceback
            console.print(traceback.format_exc())
    
    # Sort all results by relevance
    all_results.sort(key=lambda x: x.get("relevance", 0) or 0, reverse=True)
    
    # Display top overall results
    console.print(f"\n[bold]Top {top_overall} Results Across All Documents:[/]")
    
    if not all_results:
        console.print("[yellow]No results found across all documents[/]")
    else:
        # Create results table
        results_table = Table("Rank", "Filename", "Element Type", "Content", "Relevance")
        
        for i, result in enumerate(all_results[:top_overall]):
            # Truncate content for display
            display_content = result["content"]
            if len(display_content) > 100:
                display_content = display_content[:100] + "..."
            
            results_table.add_row(
                str(i+1),
                result["filename"],
                result["element_type"],
                display_content,
                f"{result['relevance']}%" if result.get("relevance") is not None else "N/A"
            )
        
        console.print(results_table)
    
    # Save combined results if output directory specified
    if output_dir:
        combined_output = output_dir / "combined_results.json"
        try:
            with open(combined_output, "w") as f:
                json.dump({
                    "query": query,
                    "total_documents": len(files),
                    "total_results": len(all_results),
                    "top_results": all_results[:top_overall]
                }, f, indent=2)
            console.print(f"Combined results saved to: [bold blue]{combined_output}[/]")
        except Exception as e:
            console.print(f"[bold red]Error saving combined results:[/] {e}")


if __name__ == "__main__":
    app()