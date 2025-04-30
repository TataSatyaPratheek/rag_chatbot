"""Enhanced example of processing complex multimodal documents."""

import json
import os # Added for psutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import typer
from mmrag.document_processing.factory import get_processor
from mmrag.exceptions import ProcessingTimeoutError, MemoryLimitExceededError # Import exceptions
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TaskProgressColumn
from rich.table import Table
from rich.tree import Tree
from rich.markup import escape
from rich.markdown import Markdown
import psutil # Added for memory monitoring

app = typer.Typer(help="Process complex multimodal documents with advanced features.")
console = Console()


def process_complex_document(
    file_path: Path, 
    enable_advanced_tables: bool = True, 
    enable_enhanced_visual: bool = True,
    enable_llm_analysis: bool = False, 
    output_format: Optional[str] = "json",
    output_dir: Optional[Path] = None,
    extract_tables: bool = True,
    extract_images: bool = True,
    show_details: bool = False,
    model: str = "llama3.2:latest",
    timeout_seconds: int = 30, # Added timeout
    memory_limit_fraction: float = 0.5, # Added memory limit fraction
) -> Dict[str, Any]:
    """Process a complex document with multiple modalities.
    
    Args:
        file_path: Path to the document file
        enable_advanced_tables: Whether to use advanced table detection
        enable_enhanced_visual: Whether to use enhanced visual element detection
        enable_llm_analysis: Whether to enable LLM content analysis
        output_format: Output format (json, html, md, or None)
        output_dir: Directory to save output files
        extract_tables: Whether to extract tables
        extract_images: Whether to extract images
        show_details: Whether to show detailed debug information
        model: LLM model to use for analysis
        timeout_seconds: Maximum processing time in seconds.
        memory_limit_fraction: Maximum fraction of available memory to use.
        
    Returns:
        Dictionary with processing results
    """
    # Validate file path
    if not file_path.exists():
        console.print(f"[bold red]Error:[/] File {file_path} not found")
        raise typer.Exit(code=1)

    # Resource monitoring setup
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_available_memory = psutil.virtual_memory().available
    memory_limit_bytes = initial_available_memory * memory_limit_fraction
    console.print(f"Resource limits: Timeout={timeout_seconds}s, Memory Limit={memory_limit_bytes / (1024**2):.2f} MB")
    
    # Determine file type and get appropriate processor
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
                advanced_table_detection=enable_advanced_tables,
                enable_enhanced_visual=enable_enhanced_visual,
                enable_llm_analysis=enable_llm_analysis
            )
            
            processor_name = processor.__class__.__name__
            progress.update(task, completed=True, description=f"Using {processor_name} for {file_path.suffix} files")
    except ValueError as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Unexpected error initializing processor:[/] {e}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)
    
    # Show processor configuration
    if show_details:
        console.print("\n[bold]Processor Configuration:[/]")
        config_table = Table("Setting", "Value")
        config_table.add_row("Processor Type", processor_name)
        config_table.add_row("Advanced Tables", str(enable_advanced_tables))
        config_table.add_row("Enhanced Visual", str(enable_enhanced_visual))
        config_table.add_row("LLM Analysis", str(enable_llm_analysis))
        config_table.add_row("Extract Tables", str(extract_tables))
        config_table.add_row("Extract Images", str(extract_images))
        config_table.add_row("LLM Model", model if enable_llm_analysis else "N/A")
        
        console.print(config_table)
    
    # Process document with metrics tracking
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TaskProgressColumn(),
            console=console
        ) as progress:
            # Create an indeterminate task for processing
            task = progress.add_task(f"Processing {file_path.name}...", total=None)
            
            # --- Resource Checks ---
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise ProcessingTimeoutError(f"Processing exceeded {timeout_seconds} seconds limit.")
                
            current_rss = process.memory_info().rss
            if current_rss > memory_limit_bytes:
                raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
            # --- End Resource Checks ---

            # Process document
            document = processor.process(file_path)
            
            # Update task to completed
            processing_time = time.time() - start_time
            
            progress.update(task, completed=True, description=f"Document processed in {processing_time:.2f}s")
    except Exception as e:
        console.print(f"[bold red]Error processing document:[/] {e}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)
    
    # Display document information
    console.print(Panel(f"[bold green]Successfully processed:[/] {file_path.name}", subtitle=f"Document ID: {document.document_id}"))
    
    # Count element types
    element_counts = {}
    for element in document.elements:
        element_type = element.element_type
        element_counts[element_type] = element_counts.get(element_type, 0) + 1
    
    # Display element breakdown
    table = Table("Element Type", "Count", "Percentage")
    total_elements = len(document.elements)
    
    for element_type, count in element_counts.items():
        percentage = (count / total_elements) * 100 if total_elements > 0 else 0
        table.add_row(element_type, str(count), f"{percentage:.1f}%")
    
    console.print(table)
    
    # Display performance metrics
    console.print("\n[bold]Performance Metrics:[/]")
    metrics_table = Table("Metric", "Value")
    metrics_table.add_row("Processing Time", f"{processing_time:.2f} seconds")
    metrics_table.add_row("Elements Extracted", str(total_elements))
    metrics_table.add_row("Elements per Second", f"{total_elements / processing_time:.1f}")
    metrics_table.add_row("Peak Memory Usage (RSS)", f"{process.memory_info().rss / (1024**2):.2f} MB")
    
    console.print(metrics_table)
    
    # Display document structure
    if document.metadata.get("page_count"):
        console.print(f"\n[bold]Document Structure:[/] {document.metadata.get('page_count')} pages")
        
        # Create a tree visualization
        doc_tree = Tree(f"{file_path.name}")
        
        pages = {}
        for element in document.elements:
            if element.bbox:
                page = element.bbox.page
                if page not in pages:
                    pages[page] = {"text": 0, "table": 0, "image": 0, "chart": 0}
                
                element_type = element.element_type
                if element_type in pages[page]:
                    pages[page][element_type] += 1
        
        # Sort pages
        for page_num in sorted(pages.keys()):
            page_counts = pages[page_num]
            page_branch = doc_tree.add(f"Page {page_num + 1}")
            
            for element_type, count in page_counts.items():
                if count > 0:
                    page_branch.add(f"{element_type}: {count}")
        
        console.print(doc_tree)
    
    # Display document analysis if available
    if document.analysis:
        console.print(f"\n[bold]Document Analysis:[/]")
        console.print(Panel(
            document.analysis.get("summary", "No summary available"), 
            title="Document Summary",
            border_style="green"
        ))
        
        if "topics" in document.analysis and document.analysis["topics"]:
            console.print("\n[bold]Key Topics:[/]")
            for topic in document.analysis["topics"]:
                console.print(f"- {topic}")
        
        if "entities" in document.analysis and document.analysis["entities"]:
            console.print("\n[bold]Named Entities:[/]")
            entities_tree = Tree("Entities")
            for entity_type, entities in document.analysis["entities"].items():
                entity_branch = entities_tree.add(entity_type.capitalize())
                for entity in entities:
                    entity_branch.add(entity)
            console.print(entities_tree)
    
    # Create a results dictionary
    results = {
        "document_id": document.document_id,
        "filename": file_path.name,
        "doc_type": document.doc_type,
        "metadata": document.metadata,
        "element_counts": element_counts,
        "performance": {
            "processing_time": processing_time,
            "total_elements": total_elements,
            "elements_per_second": total_elements / processing_time,
            "peak_memory_rss_mb": process.memory_info().rss / (1024**2)
        }
    }
    
    # Add analysis if available
    if document.analysis:
        results["analysis"] = document.analysis
    
    # Save output
    if output_format:
        # Determine output path
        if output_dir:
            output_dir.mkdir(exist_ok=True, parents=True)
            output_base = output_dir / file_path.stem
        else:
            output_base = file_path.with_suffix("")
        
        if output_format == "json":
            output_path = f"{output_base}.json"
            try:
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)
                console.print(f"\nSaved output to: [bold blue]{output_path}[/]")
            except Exception as e:
                console.print(f"[bold red]Error saving JSON output:[/] {e}")
        
        elif output_format == "html":
            output_path = f"{output_base}.html"
            try:
                with open(output_path, "w") as f:
                    # Generate HTML
                    html = generate_html_report(document, results, file_path.name)
                    f.write(html)
                console.print(f"\nSaved HTML report to: [bold blue]{output_path}[/]")
            except Exception as e:
                console.print(f"[bold red]Error saving HTML output:[/] {e}")
        
        elif output_format == "md":
            output_path = f"{output_base}.md"
            try:
                with open(output_path, "w") as f:
                    # Generate Markdown
                    md = generate_markdown_report(document, results, file_path.name)
                    f.write(md)
                console.print(f"\nSaved Markdown report to: [bold blue]{output_path}[/]")
            except Exception as e:
                console.print(f"[bold red]Error saving Markdown output:[/] {e}")
    
    # Show sample elements if requested
    if show_details:
        console.print("\n[bold]Sample Elements:[/]")
        show_sample_elements(document)
    
    return results


def show_sample_elements(document: Any) -> None:
    """Show sample elements from the document.
    
    Args:
        document: The processed document
    """
    # Get element types
    element_types = set(element.element_type for element in document.elements)
    
    for element_type in element_types:
        # Find first element of this type
        for element in document.elements:
            if element.element_type == element_type:
                console.print(f"\n[bold cyan]{element_type.upper()}[/] (ID: {element.element_id}):")
                
                if element_type == "text":
                    # Show truncated text
                    content = element.content
                    if len(content) > 200:
                        content = content[:197] + "..."
                    console.print(content)
                
                elif element_type == "table":
                    # Display table
                    if isinstance(element.content, list) and len(element.content) > 0:
                        # Create rich table for display
                        rows = len(element.content)
                        cols = len(element.content[0]) if rows > 0 else 0
                        
                        console.print(f"Dimensions: {rows} rows × {cols} columns")
                        
                        # Only show the table if it's not too large
                        if rows <= 10 and cols <= 10:
                            rich_table = Table()
                            
                            # Add headers
                            for i, cell in enumerate(element.content[0]):
                                rich_table.add_column(str(cell) if i < len(element.content[0]) else f"Column {i+1}")
                            
                            # Add data rows (limit to 5)
                            max_rows = min(rows, 6)
                            for row in element.content[1:max_rows]:
                                rich_table.add_row(*[str(cell) for cell in row])
                            
                            console.print(rich_table)
                            
                            if rows > 6:
                                console.print(f"[dim](Showing 5 of {rows-1} data rows)[/]")
                        else:
                            console.print(f"[dim](Table too large to display: {rows} rows × {cols} columns)[/]")
                
                elif element_type in ["image", "chart"]:
                    # Show metadata
                    console.print(f"Metadata: {element.metadata}")
                    
                    # For charts, show any available data
                    if element_type == "chart" and hasattr(element, "data") and element.data:
                        console.print(f"Chart Data: {element.data}")
                
                # Only show one example of each type
                break


def generate_html_report(document: Any, results: Dict[str, Any], filename: str) -> str:
    """Generate an HTML report for the document.
    
    Args:
        document: The processed document
        results: The processing results
        filename: The document filename
        
    Returns:
        HTML report as a string
    """
    # Basic HTML structure
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Document Analysis - {filename}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .topic {{ background-color: #e9f7ef; margin: 5px; padding: 8px; border-radius: 4px; display: inline-block; }}
        .entity-type {{ margin-bottom: 10px; }}
        .entity {{ background-color: #ebf5fb; display: inline-block; margin: 3px; padding: 5px; border-radius: 3px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metrics {{ background-color: #f0f4f8; padding: 10px; border-radius: 5px; margin-bottom: 15px; }}
        .chart {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <h1>Document Analysis: {filename}</h1>
    <div class="metadata">
        <p><strong>Document ID:</strong> {document.document_id}</p>
        <p><strong>Document Type:</strong> {document.doc_type}</p>
    </div>
    
    <h2>Element Counts</h2>
    <table>
        <tr>
            <th>Element Type</th>
            <th>Count</th>
            <th>Percentage</th>
        </tr>
"""
    
    # Add element counts
    total_elements = sum(results["element_counts"].values())
    for element_type, count in results["element_counts"].items():
        percentage = (count / total_elements) * 100 if total_elements > 0 else 0
        html += f"""
        <tr>
            <td>{element_type}</td>
            <td>{count}</td>
            <td>{percentage:.1f}%</td>
        </tr>"""
    
    html += """
    </table>
    
    <h2>Performance Metrics</h2>
    <div class="metrics">
"""
    
    # Add performance metrics
    for key, value in results["performance"].items():
        # Format key for display
        display_key = key.replace("_", " ").title()
        
        # Format value based on type
        if isinstance(value, float):
            display_value = f"{value:.2f}"
            if "time" in key:
                display_value += " seconds"
            elif "memory" in key:
                display_value += " MB"
        else:
            display_value = str(value)
        
        html += f"        <p><strong>{display_key}:</strong> {display_value}</p>\n"
    
    html += "    </div>\n"
    
    # Add analysis if available
    if document.analysis:
        html += """
    <h2>Document Analysis</h2>
    <div class="summary">
"""
        
        if "summary" in document.analysis:
            html += f"        <p>{document.analysis['summary']}</p>\n"
        
        html += "    </div>\n"
        
        # Add topics
        if "topics" in document.analysis and document.analysis["topics"]:
            html += """
    <h3>Key Topics</h3>
    <div class="topics">
"""
            
            for topic in document.analysis["topics"]:
                html += f'        <span class="topic">{topic}</span>\n'
            
            html += "    </div>\n"
        
        # Add entities
        if "entities" in document.analysis and document.analysis["entities"]:
            html += """
    <h3>Named Entities</h3>
"""
            
            for entity_type, entities in document.analysis["entities"].items():
                html += f'    <div class="entity-type"><h4>{entity_type.capitalize()}</h4>\n'
                for entity in entities:
                    html += f'        <span class="entity">{entity}</span>\n'
                html += "    </div>\n"
    
    # Add page structure if available
    if document.metadata.get("page_count"):
        html += f"""
    <h2>Document Structure</h2>
    <p>Pages: {document.metadata.get('page_count')}</p>
    <table>
        <tr>
            <th>Page</th>
            <th>Text Elements</th>
            <th>Tables</th>
            <th>Images</th>
            <th>Charts</th>
        </tr>
"""
        
        # Count elements by page
        pages = {}
        for element in document.elements:
            if element.bbox:
                page = element.bbox.page
                if page not in pages:
                    pages[page] = {"text": 0, "table": 0, "image": 0, "chart": 0}
                
                element_type = element.element_type
                if element_type in pages[page]:
                    pages[page][element_type] += 1
        
        # Add page rows
        for page_num in sorted(pages.keys()):
            page_counts = pages[page_num]
            html += f"""
        <tr>
            <td>{page_num + 1}</td>
            <td>{page_counts.get('text', 0)}</td>
            <td>{page_counts.get('table', 0)}</td>
            <td>{page_counts.get('image', 0)}</td>
            <td>{page_counts.get('chart', 0)}</td>
        </tr>"""
        
        html += """
    </table>
"""
    
    # Close HTML
    html += """
</body>
</html>
"""
    
    return html


def generate_markdown_report(document: Any, results: Dict[str, Any], filename: str) -> str:
    """Generate a Markdown report for the document.
    
    Args:
        document: The processed document
        results: The processing results
        filename: The document filename
        
    Returns:
        Markdown report as a string
    """
    # Basic Markdown structure
    md = f"# Document Analysis: {filename}\n\n"
    md += f"**Document ID:** {document.document_id}\n"
    md += f"**Document Type:** {document.doc_type}\n\n"
    
    # Add element counts
    md += "## Element Counts\n\n"
    md += "| Element Type | Count | Percentage |\n"
    md += "| ------------ | ----- | ---------- |\n"
    
    total_elements = sum(results["element_counts"].values())
    for element_type, count in results["element_counts"].items():
        percentage = (count / total_elements) * 100 if total_elements > 0 else 0
        md += f"| {element_type} | {count} | {percentage:.1f}% |\n"
    
    md += "\n## Performance Metrics\n\n"
    
    # Add performance metrics
    for key, value in results["performance"].items():
        # Format key for display
        display_key = key.replace("_", " ").title()
        
        # Format value based on type
        if isinstance(value, float):
            display_value = f"{value:.2f}"
            if "time" in key:
                display_value += " seconds"
            elif "memory" in key:
                display_value += " MB"
        else:
            display_value = str(value)
        
        md += f"- **{display_key}:** {display_value}\n"
    
    # Add analysis if available
    if document.analysis:
        md += "\n## Document Analysis\n\n"
        
        if "summary" in document.analysis:
            md += f"{document.analysis['summary']}\n\n"
        
        # Add topics
        if "topics" in document.analysis and document.analysis["topics"]:
            md += "### Key Topics\n\n"
            
            for topic in document.analysis["topics"]:
                md += f"- {topic}\n"
            
            md += "\n"
        
        # Add entities
        if "entities" in document.analysis and document.analysis["entities"]:
            md += "### Named Entities\n\n"
            
            for entity_type, entities in document.analysis["entities"].items():
                md += f"#### {entity_type.capitalize()}\n\n"
                for entity in entities:
                    md += f"- {entity}\n"
                md += "\n"
    
    # Add page structure if available
    if document.metadata.get("page_count"):
        md += f"\n## Document Structure\n\n"
        md += f"Pages: {document.metadata.get('page_count')}\n\n"
        md += "| Page | Text Elements | Tables | Images | Charts |\n"
        md += "| ---- | ------------- | ------ | ------ | ------ |\n"
        
        # Count elements by page
        pages = {}
        for element in document.elements:
            if element.bbox:
                page = element.bbox.page
                if page not in pages:
                    pages[page] = {"text": 0, "table": 0, "image": 0, "chart": 0}
                
                element_type = element.element_type
                if element_type in pages[page]:
                    pages[page][element_type] += 1
        
        # Add page rows
        for page_num in sorted(pages.keys()):
            page_counts = pages[page_num]
            md += f"| {page_num + 1} | {page_counts.get('text', 0)} | {page_counts.get('table', 0)} | {page_counts.get('image', 0)} | {page_counts.get('chart', 0)} |\n"
    
    return md


@app.command()
def process(
    file_path: Path = typer.Argument(..., help="Path to the document file"),
    no_advanced_tables: bool = typer.Option(False, "--no-advanced-tables", help="Disable advanced table detection"),
    no_enhanced_visual: bool = typer.Option(False, "--no-enhanced-visual", help="Disable enhanced visual processing"),
    llm: bool = typer.Option(False, "--llm", help="Enable LLM analysis"),
    output: str = typer.Option("json", "--output", "-o", help="Output format (json, html, md, none)"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-d", help="Directory for output files"),
    no_tables: bool = typer.Option(False, "--no-tables", help="Disable table extraction"),
    no_images: bool = typer.Option(False, "--no-images", help="Disable image extraction"),
    details: bool = typer.Option(False, "--details", help="Show detailed processing information"),
    model: str = typer.Option("llama3.2:latest", "--model", "-m", help="LLM model to use (if LLM is enabled)"),
    timeout: int = typer.Option(30, "--timeout", help="Processing timeout in seconds"),
    mem_limit: float = typer.Option(0.5, "--mem-limit", help="Memory limit as fraction of available memory (0.1-1.0)"),
):
    """Process a complex multimodal document."""
    # Validate output format
    if output not in ["json", "html", "md", "none"]:
        console.print(f"[bold red]Error:[/] Invalid output format: {output}")
        console.print("Supported formats: json, html, md, none")
        raise typer.Exit(code=1)
    
    # Validate memory limit
    if not (0.1 <= mem_limit <= 1.0):
        console.print("[bold red]Error:[/] Memory limit must be between 0.1 and 1.0")
        raise typer.Exit(code=1)

    try:
        process_complex_document(
            file_path=file_path,
            enable_advanced_tables=not no_advanced_tables,
            enable_enhanced_visual=not no_enhanced_visual,
            enable_llm_analysis=llm,
            output_format=None if output == "none" else output,
            output_dir=output_dir,
            extract_tables=not no_tables,
            extract_images=not no_images,
            show_details=details,
            model=model,
            timeout_seconds=timeout,
            memory_limit_fraction=mem_limit,
        )
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Processing interrupted by user[/]")
        raise typer.Exit(code=1)


@app.command()
def batch(
    directory: Path = typer.Argument(..., help="Directory containing documents to process"),
    pattern: str = typer.Option("*.*", "--pattern", "-p", help="File pattern to match"),
    no_advanced_tables: bool = typer.Option(False, "--no-advanced-tables", help="Disable advanced table detection"),
    no_enhanced_visual: bool = typer.Option(False, "--no-enhanced-visual", help="Disable enhanced visual processing"),
    llm: bool = typer.Option(False, "--llm", help="Enable LLM analysis"),
    output: str = typer.Option("json", "--output", "-o", help="Output format (json, html, md, none)"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-d", help="Directory for output files"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit number of files to process"),
    timeout: int = typer.Option(30, "--timeout", help="Processing timeout per file in seconds"),
    mem_limit: float = typer.Option(0.5, "--mem-limit", help="Memory limit per file as fraction of available memory (0.1-1.0)"),
    summarize: bool = typer.Option(True, "--summarize/--no-summarize", help="Create a summary report"),
):
    """Batch process multiple complex documents."""
    # Validate directory exists
    if not directory.exists() or not directory.is_dir():
        console.print(f"[bold red]Error:[/] Directory {directory} does not exist")
        raise typer.Exit(code=1)
    
    # Validate output format
    if output not in ["json", "html", "md", "none"]:
        console.print(f"[bold red]Error:[/] Invalid output format: {output}")
        console.print("Supported formats: json, html, md, none")
        raise typer.Exit(code=1)

    # Validate memory limit
    if not (0.1 <= mem_limit <= 1.0):
        console.print("[bold red]Error:[/] Memory limit must be between 0.1 and 1.0")
        raise typer.Exit(code=1)
    
    # Find matching files
    files = list(directory.glob(pattern))
    files = [f for f in files if f.is_file()]
    
    if not files:
        console.print(f"[bold red]Error:[/] No files matching pattern '{pattern}' found in {directory}")
        raise typer.Exit(code=1)
    
    # Get supported file types
    supported_extensions = [".pdf", ".pptx", ".ppt"]
    files = [f for f in files if f.suffix.lower() in supported_extensions]
    
    if not files:
        console.print(f"[bold red]Error:[/] No supported files found in {directory}")
        console.print(f"Supported extensions: {', '.join(supported_extensions)}")
        raise typer.Exit(code=1)
    
    # Apply limit if specified
    if limit and limit > 0 and limit < len(files):
        console.print(f"Limiting to {limit} files (out of {len(files)} found)")
        files = files[:limit]
    else:
        console.print(f"Found {len(files)} files to process")
    
    # Create output directory if specified
    if output != "none" and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"Results will be saved to: [bold blue]{output_dir}[/]")
    
    # Process each file
    results = []
    success_count = 0
    error_count = 0
    
    for i, file_path in enumerate(files):
        console.print(f"\n[bold]Processing file {i+1}/{len(files)}:[/] {file_path.name}")
        
        try:
            # Determine output directory for this file
            file_output_dir = output_dir / file_path.stem if output_dir else None
            if file_output_dir:
                file_output_dir.mkdir(exist_ok=True, parents=True)
            
            # Process document
            result = process_complex_document(
                file_path=file_path,
                enable_advanced_tables=not no_advanced_tables,
                enable_enhanced_visual=not no_enhanced_visual,
                enable_llm_analysis=llm,
                output_format=None if output == "none" else output,
                output_dir=file_output_dir,
                extract_tables=True,
                extract_images=True,
                show_details=False,
                timeout_seconds=timeout,
                memory_limit_fraction=mem_limit,
            )
            
            # Add to results
            results.append({
                "filename": file_path.name,
                "document_id": result["document_id"],
                "element_counts": result["element_counts"],
                "processing_time": result["performance"]["processing_time"],
                "success": True
            })
            
            success_count += 1
        except Exception as e:
            console.print(f"[bold red]Error processing {file_path.name}:[/] {e}")
            import traceback
            console.print(traceback.format_exc())
            
            results.append({
                "filename": file_path.name,
                "success": False,
                "error": str(e)
            })
            
            error_count += 1
    
    # Display summary
    console.print(f"\n[bold]Batch Processing Complete:[/] {success_count} successful, {error_count} failed")
    
    # Create summary report if requested
    if summarize and results and output_dir:
        try:
            # Create summary JSON
            summary_path = output_dir / "processing_summary.json"
            with open(summary_path, "w") as f:
                json.dump({
                    "files_processed": len(files),
                    "files_successful": success_count,
                    "files_failed": error_count,
                    "file_results": results
                }, f, indent=2)
            
            # Create summary HTML
            html_summary_path = output_dir / "processing_summary.html"
            with open(html_summary_path, "w") as f:
                html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Batch Processing Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .success {{ color: green; }}
        .error {{ color: red; }}
    </style>
</head>
<body>
    <h1>Batch Processing Summary</h1>
    <p>Processed {len(results)} files with {success_count} successful and {error_count} failed.</p>
    
    <table>
        <tr>
            <th>Filename</th>
            <th>Status</th>
            <th>Processing Time</th>
            <th>Elements</th>
        </tr>
"""
                
                for result in results:
                    status = "success" if result.get("success", False) else "error"
                    status_text = "Success" if result.get("success", False) else f"Error: {result.get('error', 'Unknown')}"
                    processing_time = f"{result.get('processing_time', 0):.2f}s" if result.get("success", False) else "N/A"
                    
                    # Get total elements
                    total_elements = "N/A"
                    if result.get("success", False) and "element_counts" in result:
                        total_elements = str(sum(result["element_counts"].values()))
                    
                    html_content += f"""
        <tr>
            <td>{result["filename"]}</td>
            <td class="{status}">{status_text}</td>
            <td>{processing_time}</td>
            <td>{total_elements}</td>
        </tr>"""
                
                html_content += """
    </table>
</body>
</html>
"""
                f.write(html_content)
            
            console.print(f"Summary reports saved to: [bold blue]{output_dir}[/]")
        except Exception as e:
            console.print(f"[bold red]Error creating summary reports:[/] {e}")


@app.command()
def analyze(
    file_path: Path = typer.Argument(..., help="Path to the document file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to save analysis results"),
):
    """Analyze a document structure without full processing."""
    # Validate file exists
    if not file_path.exists():
        console.print(f"[bold red]Error:[/] File {file_path} does not exist")
        raise typer.Exit(code=1)
    
    try:
        # Get processor for this file type
        try:
            processor = get_processor(file_path)
            processor_name = processor.__class__.__name__
        except ValueError as e:
            console.print(f"[bold red]Error:[/] {e}")
            raise typer.Exit(code=1)
        
        # For PDF files, we can analyze structure without full processing
        if file_path.suffix.lower() == ".pdf":
            try:
                import fitz  # PyMuPDF
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task(f"Analyzing {file_path.name}...", total=None)
                    
                    # Open document
                    doc = fitz.open(file_path)
                    page_count = len(doc)
                    
                    # Document metadata
                    metadata = {
                        "title": doc.metadata.get("title", ""),
                        "author": doc.metadata.get("author", ""),
                        "subject": doc.metadata.get("subject", ""),
                        "keywords": doc.metadata.get("keywords", ""),
                        "creator": doc.metadata.get("creator", ""),
                        "producer": doc.metadata.get("producer", ""),
                        "creation_date": doc.metadata.get("creationDate", ""),
                        "modification_date": doc.metadata.get("modDate", ""),
                        "page_count": page_count,
                    }
                    
                    # Analyze page content
                    page_analysis = []
                    
                    for page_idx in range(min(page_count, 20)):  # Limit to 20 pages for performance
                        page = doc[page_idx]
                        
                        # Count elements on page
                        text_blocks = len(page.get_text("blocks"))
                        images = len(page.get_images())
                        
                        # Analyze layout
                        blocks = page.get_text("dict")["blocks"]
                        
                        # Count potential tables based on rectangular areas
                        table_candidates = 0
                        for block in blocks:
                            if block.get("type") == 1:  # Image block
                                continue
                            
                            lines = block.get("lines", [])
                            if len(lines) > 3:
                                # Check if lines have similar width (potential table)
                                widths = [line["bbox"][2] - line["bbox"][0] for line in lines]
                                if widths and max(widths) - min(widths) < 10:
                                    table_candidates += 1
                        
                        page_analysis.append({
                            "page_number": page_idx + 1,
                            "text_blocks": text_blocks,
                            "images": images,
                            "potential_tables": table_candidates,
                            "width": page.rect.width,
                            "height": page.rect.height,
                        })
                    
                    progress.update(task, completed=True)
                
                # Display document info
                console.print(Panel(f"[bold blue]{file_path.name}[/]", subtitle=f"PDF Document, {page_count} pages"))
                
                # Display metadata
                console.print("\n[bold]Document Metadata:[/]")
                meta_table = Table("Property", "Value")
                for key, value in metadata.items():
                    if value:  # Only show non-empty values
                        meta_table.add_row(key.replace("_", " ").title(), str(value))
                console.print(meta_table)
                
                # Display page analysis
                console.print("\n[bold]Page Analysis:[/]")
                page_table = Table("Page", "Size", "Text Blocks", "Images", "Potential Tables")
                for page in page_analysis:
                    page_table.add_row(
                        str(page["page_number"]),
                        f"{page['width']:.0f}×{page['height']:.0f}",
                        str(page["text_blocks"]),
                        str(page["images"]),
                        str(page["potential_tables"])
                    )
                console.print(page_table)
                
                # Save analysis if requested
                if output:
                    try:
                        with open(output, "w") as f:
                            json.dump({
                                "filename": file_path.name,
                                "file_type": "pdf",
                                "metadata": metadata,
                                "page_analysis": page_analysis
                            }, f, indent=2)
                        console.print(f"\nAnalysis saved to: [bold blue]{output}[/]")
                    except Exception as e:
                        console.print(f"[bold red]Error saving analysis:[/] {e}")
            except ImportError:
                console.print("[yellow]Note:[/] Install PyMuPDF for enhanced PDF analysis")
                console.print("pip install pymupdf")
        else:
            console.print(f"File: [bold blue]{file_path.name}[/]")
            console.print(f"Type: {file_path.suffix.lower()}")
            console.print(f"Processor: {processor_name}")
            console.print("\nNote: Detailed analysis without processing is only available for PDF files")
    except Exception as e:
        console.print(f"[bold red]Error analyzing document:[/] {e}")
        import traceback
        console.print(traceback.format_exc())


@app.command()
def info(file_format: Optional[str] = typer.Argument(None, help="File format to get information for")):
    """Display information about supported file formats and processing capabilities."""
    # Map of supported file formats
    formats = {
        "pdf": {
            "name": "PDF Document",
            "extensions": [".pdf"],
            "processor": "PDFProcessor",
            "features": [
                "Text extraction", 
                "Table detection (basic and advanced)",
                "Image extraction",
                "Chart detection",
                "Enhanced visual processing",
                "LLM analysis"
            ],
            "notes": "Most comprehensive support with advanced visual and table detection"
        },
        "ppt": {
            "name": "PowerPoint Presentation",
            "extensions": [".ppt", ".pptx"],
            "processor": "PowerPointProcessor",
            "features": [
                "Text extraction", 
                "Table extraction",
                "Image extraction (basic)"
            ],
            "notes": "Good support for structured text and simple tables"
        }
    }
    
    if file_format:
        # Show details for a specific format
        format_key = file_format.lower()
        if format_key in formats:
            format_info = formats[format_key]
            console.print(Panel(f"[bold]{format_info['name']}[/]", subtitle=f"Processor: {format_info['processor']}"))
            console.print(f"Extensions: {', '.join(format_info['extensions'])}")
            
            console.print("\n[bold]Supported Features:[/]")
            for feature in format_info['features']:
                console.print(f"- {feature}")
            
            console.print(f"\n[bold]Notes:[/] {format_info['notes']}")
        else:
            console.print(f"[bold red]Error:[/] Unknown format: {file_format}")
            console.print(f"Supported formats: {', '.join(formats.keys())}")
    else:
        # Show overview of all formats
        console.print(Panel("[bold]Supported File Formats[/]"))
        
        format_table = Table("Format", "Extensions", "Processor", "Feature Count")
        for format_key, format_info in formats.items():
            format_table.add_row(
                format_info['name'],
                ", ".join(format_info['extensions']),
                format_info['processor'],
                str(len(format_info['features']))
            )
        
        console.print(format_table)
        console.print("\nUse 'info <format>' for details about a specific format")


if __name__ == "__main__":
    app()