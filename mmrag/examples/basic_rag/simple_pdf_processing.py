"""Basic example of PDF processing and content extraction."""

import os # Added for psutil
import time # Added for time
import typer
from pathlib import Path
from typing import Optional, Dict, Any

from mmrag.document_processing.legacy import PDFProcessor
# --- Added: Import exceptions and psutil ---
from mmrag.exceptions import ProcessingTimeoutError, MemoryLimitExceededError
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import psutil

app = typer.Typer(help="Process a PDF file and display extracted content.")
console = Console()


def process_pdf(
    pdf_path: Path, 
    output_json: bool = False,
    extract_tables: bool = True,
    extract_images: bool = True,
    advanced_tables: bool = False,
    enhanced_visual: bool = False,
    # --- Added: timeout and memory limit parameters ---
    timeout_seconds: int = 30,
    memory_limit_fraction: float = 0.5,
) -> None:
    """Process a PDF file and display extracted content."""
    # Validate input file
    if not pdf_path.exists():
        console.print(f"[bold red]Error:[/] File {pdf_path} not found")
        raise typer.Exit(code=1)
    
    if pdf_path.suffix.lower() != ".pdf":
        console.print(f"[bold red]Error:[/] File {pdf_path} is not a PDF")
        raise typer.Exit(code=1)

    # --- Added: Resource monitoring setup ---
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_available_memory = psutil.virtual_memory().available
    memory_limit_bytes = initial_available_memory * memory_limit_fraction
    console.print(f"Resource limits: Timeout={timeout_seconds}s, Memory Limit={memory_limit_bytes / (1024**2):.2f} MB")
    
    # Create processor with options
    processor = PDFProcessor(
        extract_tables=extract_tables,
        extract_images=extract_images,
        advanced_table_detection=advanced_tables,
        enable_enhanced_visual=enhanced_visual,
    )
    
    console.print(Panel(f"Processing: [bold blue]{pdf_path}[/]"))
    
    try:
        # Process the document with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing document...", total=None)

            # --- Added: Resource checks ---
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise ProcessingTimeoutError(f"Processing exceeded {timeout_seconds} seconds limit.")
            current_rss = process.memory_info().rss
            if current_rss > memory_limit_bytes:
                raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
            # --- End Added ---
            document = processor.process(pdf_path)
            progress.update(task, completed=True, description="Document processed successfully!")
        
        # Print summary information
        console.print(f"\n[bold green]Document ID:[/] {document.document_id}")
        console.print(f"[bold green]Filename:[/] {document.filename}")
        console.print(f"[bold green]Page count:[/] {document.metadata.get('page_count', 'Unknown')}")
        
        # Display element counts
        element_types: Dict[str, int] = {}
        for element in document.elements:
            element_types[element.element_type] = element_types.get(element.element_type, 0) + 1
        
        console.print("\n[bold]Extracted elements:[/]")
        table = Table("Element Type", "Count")
        for element_type, count in element_types.items():
            table.add_row(element_type, str(count))
        console.print(table)
        
        # Display sample content
        console.print("\n[bold]Sample content:[/]")
        for element_type in element_types:
            # Get the first element of this type
            for element in document.elements:
                if element.element_type == element_type:
                    console.print(f"\n[bold cyan]{element_type.upper()}[/] (ID: {element.element_id}):")
                    if element_type == "text":
                        # Show truncated text for readability
                        content = element.content[:200] + "..." if len(element.content) > 200 else element.content
                        console.print(content)
                    elif element_type == "table":
                        # Create a rich table for display
                        if isinstance(element.content, list) and len(element.content) > 0:
                            rich_table = Table()
                            for i, cell in enumerate(element.content[0]):
                                rich_table.add_column(str(cell) if i < len(element.content[0]) else f"Column {i+1}")
                            for row in element.content[1:min(len(element.content), 6)]:  # Limit rows for display
                                rich_table.add_row(*[str(cell) for cell in row])
                            if len(element.content) > 6:
                                console.print(rich_table)
                                console.print(f"[dim](Showing 5 of {len(element.content)-1} data rows)[/]")
                            else:
                                console.print(rich_table)
                    elif element_type in ["image", "chart"]:
                        console.print(f"[Image/Chart metadata: {element.metadata}]")
                    break
        
        # Save as JSON if requested
        if output_json:
            output_path = pdf_path.with_suffix(".json")
            with open(output_path, "w") as f:
                f.write(document.to_json())
            console.print(f"\nSaved document JSON to [bold blue]{output_path}[/]")
        
        return document
    
    except Exception as e:
        console.print(f"[bold red]Error processing document:[/] {str(e)}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)


@app.command()
def main(
    pdf_path: Path = typer.Argument(..., help="Path to the PDF file"),
    json: bool = typer.Option(False, "--json", help="Output as JSON"),
    no_tables: bool = typer.Option(False, "--no-tables", help="Disable table extraction"),
    no_images: bool = typer.Option(False, "--no-images", help="Disable image extraction"),
    advanced_tables: bool = typer.Option(False, "--advanced-tables", help="Enable advanced table detection"),
    enhanced_visual: bool = typer.Option(False, "--enhanced-visual", help="Enable enhanced visual processing"),
    # --- Added: timeout and memory limit options ---
    timeout: int = typer.Option(30, "--timeout", help="Processing timeout in seconds"),
    mem_limit: float = typer.Option(0.5, "--mem-limit", help="Memory limit as fraction of available memory (0.1-1.0)"),
) -> None:
    """Process a PDF file and display extracted content."""
    # --- Added: Validate memory limit ---
    if not (0.1 <= mem_limit <= 1.0):
        console.print("[bold red]Error:[/] Memory limit must be between 0.1 and 1.0")
        raise typer.Exit(code=1)

    process_pdf(
        pdf_path, 
        output_json=json,
        extract_tables=not no_tables,
        extract_images=not no_images,
        advanced_tables=advanced_tables,
        enhanced_visual=enhanced_visual,
        timeout_seconds=timeout, # Pass timeout
        memory_limit_fraction=mem_limit, # Pass mem limit
    )


if __name__ == "__main__":
    app()