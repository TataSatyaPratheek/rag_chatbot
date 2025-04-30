"""Command-line interface for mmrag."""

import json
import logging
import sys
from pathlib import Path
import subprocess
from typing import List, Optional
import typer
from rich.console import Console # Keep this import
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# Make sure these imports are direct, not from another module
from mmrag.document_processing.factory import get_processor
from mmrag.document_processing.base import ProcessedDocument
from mmrag.exceptions import ProcessingTimeoutError, MemoryLimitExceededError # Import new exceptions
from rich.progress import TimeElapsedColumn # Import TimeElapsedColumn
from mmrag.vectordb import ChromaStore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger("mmrag")
console = Console()
app = typer.Typer()


@app.command()
def process(
    file_path: Path = typer.Argument(..., help="Path to the document to process"),
    output_path: Optional[Path] = typer.Option(None, help="Path to save processed document JSON"),
    extract_tables: bool = typer.Option(True, help="Extract tables from the document"),
    extract_images: bool = typer.Option(True, help="Extract images from the document"),
    advanced_tables: bool = typer.Option(False, help="Use advanced table detection (ML-based)"),
    enhanced_visual: bool = typer.Option(False, help="Use enhanced visual element detection"),
    llm_analysis: bool = typer.Option(False, help="Enable LLM content analysis"),
):
    use_docling: bool = typer.Option(True, help="Use Docling processor if available"),  # New option
    """Process a document and extract its elements."""
    if not file_path.exists():
        console.print(f"[bold red]Error:[/] File not found: {file_path}")
        raise typer.Exit(code=1)

    # Get appropriate processor - explicitly call get_processor to ensure our mock works
    try:
        processor = get_processor(
            file_path,
            use_docling=use_docling, # Pass the new parameter
            # Pass other options directly to the factory
            extract_tables=extract_tables,
            extract_images=extract_images,
            advanced_table_detection=advanced_tables,
            enable_enhanced_visual=enhanced_visual,
            enable_llm_analysis=llm_analysis,
        )
    except ValueError as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(), # Add elapsed time column
        console=console,
    ) as progress:
        progress.add_task(description="Processing document...", total=None)

        # Process the document
        try:
            processed_doc = processor.process(file_path)
        except Exception as e:
            # Catch specific processing errors
            if isinstance(e, (ProcessingTimeoutError, MemoryLimitExceededError)):
                console.print(f"[bold red]Processing stopped:[/] {e}")
            progress.stop()
            console.print(f"[bold red]Error processing document:[/] {e}")
            raise typer.Exit(code=1)

    # Print summary
    console.print("\n[bold green]Document processed successfully![/]")
    console.print(f"Document ID: {processed_doc.document_id}")
    console.print(f"File: {processed_doc.filename}")
    console.print(f"Type: {processed_doc.doc_type}")
    console.print(f"Elements extracted: {len(processed_doc.elements)}")

    # Element type counts
    element_types = {}
    for element in processed_doc.elements:
        element_types[element.element_type] = element_types.get(element.element_type, 0) + 1

    console.print("\n[bold]Element types:[/]")
    for element_type, count in element_types.items():
        console.print(f"  {element_type}: {count}")

    # Save output if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(processed_doc.to_json())
        console.print(f"\nOutput saved to: {output_path}")

    return processed_doc


@app.command()
def store(
    file_path: Path = typer.Argument(..., help="Path to the document to process and store"),
    collection_name: str = typer.Option("document_elements", help="ChromaDB collection name"),
    extract_tables: bool = typer.Option(True, help="Extract tables"),
    extract_images: bool = typer.Option(True, help="Extract images"),
    advanced_tables: bool = typer.Option(False, help="Use advanced table detection"),
    enhanced_visual: bool = typer.Option(False, help="Use enhanced visual element detection"),
    llm_analysis: bool = typer.Option(False, help="Enable LLM content analysis"),
    use_docling: bool = typer.Option(True, help="Use Docling processor if available"), # New option
):
    """Process a document and store it in the vector database."""
    # Check if the file exists first
    if not file_path.exists():
        console.print(f"[bold red]Error:[/] File not found: {file_path}")
        raise typer.Exit(code=1)
    
    # Get appropriate processor - ensure mocks work
    try:
        processor = get_processor(
            file_path,
            use_docling=use_docling, # Pass the new parameter
            # Pass other options directly to the factory
            extract_tables=extract_tables,
            extract_images=extract_images,
            advanced_table_detection=advanced_tables,
            enable_enhanced_visual=enhanced_visual,
            enable_llm_analysis=llm_analysis,
        )
    except ValueError as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)

    # Process the document
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(), # Add elapsed time column
        console=console,
    ) as progress:
        progress.add_task(description="Processing document...", total=None)
        
        try:
            processed_doc = processor.process(file_path)
        except Exception as e:
            # Catch specific processing errors
            if isinstance(e, (ProcessingTimeoutError, MemoryLimitExceededError)):
                console.print(f"[bold red]Processing stopped:[/] {e}")
            progress.stop()
            console.print(f"[bold red]Error processing document:[/] {e}")
            raise typer.Exit(code=1)

    # Store in vector database
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(description="Storing in vector database...", total=None)

        try:
            store = ChromaStore(collection_name=collection_name)
            store.add_document(processed_doc)
        except Exception as e:
            progress.stop()
            console.print(f"[bold red]Error storing document:[/] {e}")
            raise typer.Exit(code=1)

    console.print("\n[bold green]Document stored successfully![/]")
    console.print(f"Document ID: {processed_doc.document_id}")
    console.print(f"Collection: {collection_name}")


@app.command()
def query(
    query_text: str = typer.Argument(..., help="Query text"),
    n_results: int = typer.Option(5, help="Number of results to return"),
    collection_name: str = typer.Option("document_elements", help="ChromaDB collection name"),
    document_id: Optional[str] = typer.Option(None, help="Filter by document ID"),
):
    """Query the vector database."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(description="Querying vector database...", total=None)

        # Initialize vector store - ensure mocks work
        try:
            store = ChromaStore(collection_name=collection_name)
        
            # Prepare filter
            where = {"document_id": document_id} if document_id else None

            results = store.query(
                query_text=query_text,
                n_results=n_results,
                where=where,
            )
        except Exception as e:
            progress.stop()
            console.print(f"[bold red]Error querying database:[/] {e}")
            raise typer.Exit(code=1)

    # Print results
    console.print("\n[bold green]Query Results:[/]")
    console.print(f"Query: {query_text}")
    console.print(f"Results found: {len(results['ids'][0])}")

    for i, (doc_id, doc, metadata) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
    )):
        console.print(f"\n[bold cyan]Result {i+1}:[/]")
        console.print(f"Document: {metadata['filename']} (ID: {metadata['document_id']})")
        console.print(f"Element Type: {metadata['element_type']}")
        console.print(f"Page: {metadata['page']}")
        console.print(f"Content: {doc[:200]}{'...' if len(doc) > 200 else ''}")


@app.command()
def delete(
    document_id: str = typer.Argument(..., help="Document ID to delete"),
    collection_name: str = typer.Option("document_elements", help="ChromaDB collection name"),
):
    """Delete a document from the vector database."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(description=f"Deleting document {document_id}...", total=None)

        # Initialize vector store - ensure mocks work
        try:
            store = ChromaStore(collection_name=collection_name)
            store.delete_document(document_id)
        except Exception as e:
            progress.stop()
            console.print(f"[bold red]Error deleting document:[/] {e}")
            raise typer.Exit(code=1)

    console.print(f"\n[bold green]Document {document_id} deleted successfully![/]")