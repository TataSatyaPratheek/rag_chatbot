"""Enhanced example of vector search using ChromaDB."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import typer
from mmrag.document_processing import PDFProcessor
from mmrag.vectordb import ChromaStore
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

app = typer.Typer(help="Vector search demo for document retrieval.")
console = Console()


def index_document(
    file_path: Path, 
    persist_dir: Optional[Path] = None, 
    collection_name: str = "vector_search_demo",
    advanced_tables: bool = False,
    enhanced_visual: bool = False
) -> Tuple[ChromaStore, Any]:
    """Process and index a document with enhanced options."""
    # Initialize document processor with configurable options
    processor = PDFProcessor(
        extract_tables=True,
        extract_images=True,
        advanced_table_detection=advanced_tables,
        enable_enhanced_visual=enhanced_visual
    )
    
    # Process document with progress indicator
    with Progress(
        SpinnerColumn(), 
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Processing document: {file_path.name}", total=None)
        start_time = time.time()
        document = processor.process(file_path)
        processing_time = time.time() - start_time
        progress.update(task, completed=True)
    
    # Set up vector store
    if persist_dir is None:
        persist_dir = Path("temp_vector_search")
    else:
        persist_dir = Path(persist_dir)
    
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    # Index document with progress indicator
    with Progress(
        SpinnerColumn(), 
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Indexing document in vector database...", total=None)
        start_time = time.time()
        store = ChromaStore(
            persist_directory=persist_dir,
            collection_name=collection_name
        )
        store.add_document(document)
        indexing_time = time.time() - start_time
        progress.update(task, completed=True)
    
    # Print processing stats
    console.print(f"[bold green]Successfully indexed document:[/] {file_path.name}")
    console.print(f"Document ID: {document.document_id}")
    console.print(f"Processing time: {processing_time:.2f}s")
    console.print(f"Indexing time: {indexing_time:.2f}s")
    
    # Report element counts
    element_types = {}
    for element in document.elements:
        element_types[element.element_type] = element_types.get(element.element_type, 0) + 1
    
    # Show element breakdown
    table = Table("Element Type", "Count", "Percentage")
    total_elements = len(document.elements)
    
    for element_type, count in element_types.items():
        percentage = (count / total_elements * 100) if total_elements > 0 else 0 # Fixed HTML entity &gt;
        table.add_row(element_type, str(count), f"{percentage:.1f}%")
    
    console.print(table)
    
    return store, document


def search_documents(
    store: ChromaStore, 
    query: str, 
    n_results: int = 5, 
    element_type: Optional[str] = None,
    document_id: Optional[str] = None,
    highlight_keywords: bool = True
) -> Dict:
    """Search documents by semantic similarity with enhanced display options."""
    console.print(f"\nSearching for: [bold cyan]'{query}'[/]")
    
    # Prepare where filter
    where = {}
    if element_type:
        where["element_type"] = element_type
    if document_id:
        where["document_id"] = document_id
    
    # Query the database with timing
    start_time = time.time()
    results = store.query(
        query_text=query,
        n_results=n_results,
        where=where if where else None
    )
    search_time = time.time() - start_time
    
    # Format and display results
    console.print(f"\n[bold green]Search Results:[/] Found {len(results['ids'][0])} matches in {search_time:.3f}s")
    
    table = Table("Rank", "Content", "Element Type", "Page", "Relevance")
    
    # If no distance values are provided, we'll use a default
    distances = results.get("distances", [[0] * len(results["ids"][0])])
    
    # Extract keywords for highlighting
    keywords = set(query.lower().split())
    
    for i, (doc_content, metadata, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        distances[0]
    )):
        # Get metadata
        element_type = metadata.get("element_type", "unknown")
        page = metadata.get("page", "unknown")
        
        # Calculate relevance score (1 - distance, normalized to percentage)
        # Lower distance = higher relevance
        relevance = round((1 - (distance / 2)) * 100) if distance else "N/A"
        
        # Truncate long content for display
        content = doc_content[:200] + "..." if len(doc_content) > 200 else doc_content # Fixed HTML entity &gt;
        
        # Highlight keywords if enabled
        if highlight_keywords:
            highlighted_content = Text()
            words = content.split()
            for word in words:
                if word.lower().strip(".,!?;:()[]{}\"'") in keywords:
                    highlighted_content.append(word, style="bold yellow")
                else:
                    highlighted_content.append(word)
                highlighted_content.append(" ")
            
            # Add to table with styled content
            table.add_row(
                str(i+1),
                highlighted_content,
                element_type,
                str(page),
                f"{relevance}%" if isinstance(relevance, int) else relevance
            )
        else:
            # Add to table without highlighting
            table.add_row(
                str(i+1),
                content,
                element_type,
                str(page),
                f"{relevance}%" if isinstance(relevance, int) else relevance
            )
    
    console.print(table)
    
    # Add some search statistics
    if distances[0]:
        avg_score = sum((1 - (d / 2)) * 100 for d in distances[0]) / len(distances[0])
        console.print(f"Average relevance score: [bold]{avg_score:.1f}%[/]")
    
    return results


def view_result_detail(
    results: Dict, 
    index: int, 
    document: Optional[Any] = None
) -> None:
    """View detailed information for a single search result."""
    if not results or "documents" not in results or not results["documents"][0]:
        console.print("[bold red]No results to display[/]")
        return
    
    if index < 0 or index >= len(results["documents"][0]): # Fixed HTML entity &lt; and &gt;=
        console.print(f"[bold red]Invalid result index: {index}[/]")
        return
    
    # Get result data
    doc_content = results["documents"][0][index]
    metadata = results["metadatas"][0][index]
    doc_id = results["ids"][0][index]
    
    # Calculate distance/score if available
    relevance = "N/A"
    if "distances" in results and results["distances"][0]:
        distance = results["distances"][0][index]
        relevance = f"{round((1 - (distance / 2)) * 100)}%"
    
    # Create detail panel
    element_type = metadata.get("element_type", "unknown")
    page = metadata.get("page", "unknown")
    document_id = metadata.get("document_id", "unknown")
    filename = metadata.get("filename", "unknown")
    
    panel_title = f"Result #{index+1} Details"
    
    # Build tree for structured display
    tree = Tree(f"[bold]{element_type.upper()}[/] from [cyan]{filename}[/] (page {page})")
    
    # Add metadata
    meta_branch = tree.add("[bold]Metadata[/]")
    for key, value in metadata.items():
        if key not in ["element_id", "document_id", "element_type", "page", "filename"]:
            meta_branch.add(f"{key}: {value}")
    
    # Add relevance information
    tree.add(f"[bold]Relevance score:[/] {relevance}")
    tree.add(f"[bold]Element ID:[/] {metadata.get('element_id', 'unknown')}")
    tree.add(f"[bold]Document ID:[/] {document_id}")
    
    # Add content with proper formatting based on element type
    content_branch = tree.add("[bold]Content:[/]")
    
    if element_type == "text":
        # For text, just show the content
        content_branch.add(doc_content)
    elif element_type == "table":
        # For tables, try to display in tabular format
        if document and hasattr(document, "elements"):
            # Find the original table element for better display
            for element in document.elements:
                if element.element_id == metadata.get("element_id"):
                    # Create a rich table for display
                    if isinstance(element.content, list) and len(element.content) > 0:
                        table = Table()
                        
                        # Add headers
                        for i, cell in enumerate(element.content[0]):
                            table.add_column(str(cell) if cell else f"Column {i+1}")
                        
                        # Add data rows (limit to 10 for display)
                        max_rows = min(len(element.content), 11)
                        for row in element.content[1:max_rows]:
                            table.add_row(*[str(cell) if cell is not None else "" for cell in row]) # Handle None cells
                        
                        # Add the table to the content branch
                        content_branch.add(table)
                        
                        if len(element.content) > 11:
                            content_branch.add(f"[dim](Showing 10 of {len(element.content)-1} rows)[/]")
                    break
            else:
                # If we didn't find the original element, fall back to text
                content_branch.add(doc_content)
        else:
            # No document reference, just show as text
            content_branch.add(doc_content)
    elif element_type in ["image", "chart"]:
        # For visual elements, show metadata
        content_branch.add(f"[Image/Chart metadata: {metadata}]")
    
    # Display the panel
    console.print(Panel(tree, title=panel_title))


def explore_vector_database(store: ChromaStore, document: Optional[Any] = None) -> None:
    """Interactive exploration of vector database contents."""
    console.print("\n[bold]Vector Database Explorer[/]")
    console.print("Type 'quit' to exit, 'filter <type>' to filter by element type")
    console.print("Type 'view <number>' to view details of a result")
    console.print("Type 'help' for more commands")
    
    current_filter = None
    current_document_id = None
    current_results = None
    n_results = 5
    highlight = True
    
    while True:
        # Show current filter if any
        filter_display = ""
        if current_filter:
            filter_display += f" (Filter: {current_filter})"
        if current_document_id:
            filter_display += f" (Document: {current_document_id})"
        
        # Get user input
        user_input = console.input(f"\n[bold cyan]Query{filter_display}:[/] ")
        
        if user_input.lower() == "quit":
            break
            
        elif user_input.lower() == "help":
            console.print("\n[bold]Available Commands:[/]")
            help_table = Table("Command", "Description")
            help_table.add_row("quit", "Exit the explorer")
            help_table.add_row("filter <type>", "Filter results by element type (e.g., text, table)")
            help_table.add_row("filter clear", "Clear the element type filter")
            help_table.add_row("document <id>", "Filter results by document ID")
            help_table.add_row("document clear", "Clear the document filter")
            help_table.add_row("view <number>", "View details of a search result by number")
            help_table.add_row("results <number>", "Set number of results to display (default: 5)")
            help_table.add_row("highlight on/off", "Toggle keyword highlighting")
            help_table.add_row("help", "Show this help message")
            console.print(help_table)
            continue
            
        elif user_input.lower().startswith("filter "):
            filter_type = user_input[7:].strip().lower()
            
            if filter_type in ["none", "clear"]:
                current_filter = None
                console.print("[bold yellow]Filter cleared[/]")
            else:
                current_filter = filter_type
                console.print(f"[bold yellow]Filter set to: {current_filter}[/]")
            
            continue
            
        elif user_input.lower().startswith("document "):
            doc_id = user_input[9:].strip()
            
            if doc_id in ["none", "clear"]:
                current_document_id = None
                console.print("[bold yellow]Document filter cleared[/]")
            else:
                current_document_id = doc_id
                console.print(f"[bold yellow]Document filter set to: {current_document_id}[/]")
            
            continue
            
        elif user_input.lower().startswith("results "):
            try:
                n_results = int(user_input[8:].strip())
                console.print(f"[bold yellow]Number of results set to: {n_results}[/]")
            except ValueError:
                console.print("[bold red]Invalid number format[/]")
            continue
            
        elif user_input.lower().startswith("view "):
            try:
                result_index = int(user_input[5:].strip()) - 1  # Convert to 0-based index
                if current_results:
                    view_result_detail(current_results, result_index, document)
                else:
                    console.print("[bold yellow]No search results to view. Run a search first.[/]")
            except ValueError:
                console.print("[bold red]Invalid result number[/]")
            continue
            
        elif user_input.lower() == "highlight on":
            highlight = True
            console.print("[bold yellow]Keyword highlighting enabled[/]")
            continue
            
        elif user_input.lower() == "highlight off":
            highlight = False
            console.print("[bold yellow]Keyword highlighting disabled[/]")
            continue
        
        # Perform search with current filter
        current_results = search_documents(
            store, 
            user_input, 
            n_results=n_results, 
            element_type=current_filter,
            document_id=current_document_id,
            highlight_keywords=highlight
        )


@app.command()
def interactive(
    document: Optional[Path] = typer.Option(None, "--document", "-d", help="Document to index"),
    persist_dir: Optional[Path] = typer.Option(None, "--persist-dir", "-p", help="Directory to persist vector database"),
    collection: str = typer.Option("vector_search_demo", "--collection", "-c", help="Collection name"),
    advanced_tables: bool = typer.Option(False, "--advanced-tables", help="Use advanced table detection"),
    enhanced_visual: bool = typer.Option(False, "--enhanced-visual", help="Use enhanced visual processing"),
):
    """Run in interactive exploration mode."""
    store = None
    processed_doc = None
    
    # Setup vector store
    if document:
        # Check if document exists
        if not document.exists():
            console.print(f"[bold red]Error:[/] Document {document} not found")
            raise typer.Exit(code=1)
        
        # Index document
        store, processed_doc = index_document(
            document, 
            persist_dir, 
            collection,
            advanced_tables=advanced_tables,
            enhanced_visual=enhanced_visual
        )
    else:
        # Try to load existing vector store
        try:
            persist_dir = persist_dir or Path("temp_vector_search")
            persist_dir.mkdir(parents=True, exist_ok=True)
            store = ChromaStore(persist_directory=persist_dir, collection_name=collection)
            console.print(f"Using existing vector database at: {persist_dir}")
            
            # Check if the collection has documents
            collection_info = store.collection.count()
            if collection_info == 0:
                console.print("[bold yellow]Warning:[/] Collection is empty")
                console.print("Please specify a document to index with --document")
                raise typer.Exit(code=1)
            else:
                console.print(f"Collection contains {collection_info} elements")
        except Exception as e:
            console.print(f"[bold red]Error loading vector database:[/] {e}")
            console.print("Please specify a document to index with --document")
            raise typer.Exit(code=1)
    
    # Run the explorer
    explore_vector_database(store, processed_doc)


@app.command()
def query(
    query_text: str = typer.Argument(..., help="Query text"),
    document: Optional[Path] = typer.Option(None, "--document", "-d", help="Document to index"),
    n_results: int = typer.Option(5, "--results", "-r", help="Number of results to return"),
    filter_type: Optional[str] = typer.Option(None, "--filter", "-f", help="Filter by element type"),
    persist_dir: Optional[Path] = typer.Option(None, "--persist-dir", "-p", help="Directory to persist vector database"),
    collection: str = typer.Option("vector_search_demo", "--collection", "-c", help="Collection name"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results to JSON file"),
    advanced_tables: bool = typer.Option(False, "--advanced-tables", help="Use advanced table detection"),
    enhanced_visual: bool = typer.Option(False, "--enhanced-visual", help="Use enhanced visual processing"),
):
    """Run a single query and display results."""
    store = None
    processed_doc = None
    
    # Setup vector store
    if document:
        # Check if document exists
        if not document.exists():
            console.print(f"[bold red]Error:[/] Document {document} not found")
            raise typer.Exit(code=1)
        
        # Index document
        store, processed_doc = index_document(
            document, 
            persist_dir, 
            collection,
            advanced_tables=advanced_tables,
            enhanced_visual=enhanced_visual
        )
    else:
        # Try to load existing vector store
        try:
            persist_dir = persist_dir or Path("temp_vector_search")
            persist_dir.mkdir(parents=True, exist_ok=True)
            store = ChromaStore(persist_directory=persist_dir, collection_name=collection)
            console.print(f"Using existing vector database at: {persist_dir}")
        except Exception as e:
            console.print(f"[bold red]Error loading vector database:[/] {e}")
            console.print("Please specify a document to index with --document")
            raise typer.Exit(code=1)
    
    # Run the query
    results = search_documents(
        store, 
        query_text, 
        n_results=n_results, 
        element_type=filter_type
    )
    
    # Save results if requested
    if output:
        try:
            with open(output, "w") as f:
                # Convert to serializable format
                output_data = {
                    "query": query_text,
                    "results": []
                }
                
                for i, (doc_content, metadata, doc_id) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["ids"][0]
                )):
                    output_data["results"].append({
                        "rank": i + 1,
                        "content": doc_content,
                        "metadata": metadata,
                        "id": doc_id
                    })
                
                json.dump(output_data, f, indent=2)
            
            console.print(f"[bold green]Results saved to:[/] {output}")
        except Exception as e:
            console.print(f"[bold red]Error saving results:[/] {e}")


if __name__ == "__main__":
    app()