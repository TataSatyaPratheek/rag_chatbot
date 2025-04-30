"""Enhanced example of table extraction from documents."""

import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import typer
from mmrag.document_processing import PDFProcessor
from mmrag.document_processing.advanced_table import CascadeTabNetDetector
from mmrag.document_processing.table import TableDetector
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table as RichTable
from rich.tree import Tree
from rich.markdown import Markdown

app = typer.Typer(help="Extract and analyze tables from documents.")
console = Console()


def extract_tables(
    file_path: Path,
    advanced: bool = True,
    output_format: Optional[str] = "csv",
    output_dir: Optional[Path] = None,
    table_filter: Optional[int] = None,
    min_rows: int = 2,
    min_cols: int = 2,
    analyze_tables: bool = False,
    show_extraction_details: bool = False,
) -> Dict[str, Any]:
    """Extract tables from a document using basic or advanced detection.
    
    Args:
        file_path: Path to the document file
        advanced: Whether to use advanced table detection
        output_format: Output format (csv, json, html, or None)
        output_dir: Directory to save extracted tables
        table_filter: Extract only the specified table index
        min_rows: Minimum number of rows for a valid table
        min_cols: Minimum number of columns for a valid table
        analyze_tables: Whether to analyze table content using patterns
        show_extraction_details: Show detailed information about the extraction process
        
    Returns:
        Dictionary with extraction results
    """
    # Validate file exists
    if not file_path.exists():
        console.print(f"[bold red]Error:[/] File {file_path} does not exist")
        raise typer.Exit(code=1)
    
    # Set up the table detector
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing table detector...", total=None)
            
            if advanced:
                try:
                    # Try to use advanced table detection
                    table_detector = CascadeTabNetDetector()
                    detector_name = "advanced (CascadeTabNet)"
                except Exception as e:
                    console.print(f"[bold yellow]Warning:[/] Could not initialize advanced table detector: {e}")
                    console.print("Falling back to basic table detection")
                    table_detector = TableDetector(min_rows=min_rows, min_cols=min_cols)
                    detector_name = "basic"
                    advanced = False
            else:
                table_detector = TableDetector(min_rows=min_rows, min_cols=min_cols)
                detector_name = "basic"
                
            progress.update(task, completed=True, description=f"Using {detector_name} table detection")
    except Exception as e:
        console.print(f"[bold red]Error initializing table detector:[/] {e}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)
    
    # Set up document processor
    processor = PDFProcessor(
        extract_tables=True,
        extract_images=False,  # Focus only on tables
        table_detector=table_detector,
        advanced_table_detection=advanced
    )
    
    # Process the document
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Processing document: {file_path.name}", total=None)
            
            # Track extraction time
            start_time = time.time()
            document = processor.process(file_path)
            extraction_time = time.time() - start_time
            
            progress.update(task, completed=True, description=f"Processed document in {extraction_time:.2f}s")
    except Exception as e:
        console.print(f"[bold red]Error processing document:[/] {e}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)
    
    # Extract tables
    tables = [element for element in document.elements if element.element_type == "table"]
    
    # Check if we found any tables
    if not tables:
        console.print("[bold yellow]No tables detected in the document.[/]")
        return {"document_id": document.document_id, "tables": []}
    
    console.print(f"\nFound [bold]{len(tables)}[/] tables in the document")
    
    # Display extraction details if requested
    if show_extraction_details:
        console.print("\n[bold]Table Extraction Details:[/]")
        console.print(f"Document ID: {document.document_id}")
        console.print(f"Document Pages: {document.metadata.get('page_count', 'Unknown')}")
        console.print(f"Table Detection Method: {detector_name}")
        console.print(f"Extraction Time: {extraction_time:.2f}s")
        console.print(f"Average Time Per Table: {extraction_time / len(tables):.2f}s")
    
    # Set up output directory if specified
    if output_format and output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
        console.print(f"Saving tables to: [bold blue]{output_dir}[/]")
    elif output_format:
        output_dir = Path(f"tables_{document.document_id}")
        output_dir.mkdir(exist_ok=True, parents=True)
        console.print(f"Saving tables to: [bold blue]{output_dir}[/]")
    
    # Prepare results
    results = {
        "document_id": document.document_id,
        "filename": file_path.name,
        "table_count": len(tables),
        "extraction_method": detector_name,
        "extraction_time": extraction_time,
        "tables": []
    }
    
    # Process specific table if filter is provided
    if table_filter is not None:
        if table_filter < 0 or table_filter >= len(tables):
            console.print(f"[bold red]Error:[/] Table index {table_filter} is out of range (0-{len(tables)-1})")
            raise typer.Exit(code=1)
        
        tables = [tables[table_filter]]
        console.print(f"[bold yellow]Note:[/] Extracting only table #{table_filter}")
    
    # Process each table
    for i, table_element in enumerate(tables):
        table_number = i + 1
        console.print(f"\n[bold cyan]Table {table_number}:[/]")
        
        # Get table metadata
        console.print(f"Location: Page {table_element.bbox.page + 1}, Position: ({table_element.bbox.x0:.1f}, {table_element.bbox.y0:.1f})")
        
        # Get table content
        table_data = table_element.content
        
        # Display table information
        rows = len(table_data) if isinstance(table_data, list) else 0
        cols = len(table_data[0]) if rows > 0 and isinstance(table_data[0], list) else 0
        
        console.print(f"Dimensions: {rows} rows × {cols} columns")
        
        # Analyze table structure if requested
        table_analysis = None
        if analyze_tables and rows > 0 and cols > 0:
            try:
                # Simple analysis of table structure
                table_analysis = analyze_table_structure(table_data)
                
                # Display analysis
                console.print("\n[bold]Table Analysis:[/]")
                
                if table_analysis["has_header"]:
                    console.print("- Has header row: [green]Yes[/]")
                else:
                    console.print("- Has header row: [yellow]No[/]")
                
                if table_analysis["data_types"]:
                    console.print("- Column Data Types:")
                    for col_idx, data_type in table_analysis["data_types"].items():
                        console.print(f"  - Column {col_idx}: {data_type}")
                
                if table_analysis["patterns"]:
                    console.print("- Detected Patterns:")
                    for pattern_name, pattern_desc in table_analysis["patterns"].items():
                        console.print(f"  - {pattern_name}: {pattern_desc}")
            except Exception as e:
                console.print(f"[bold yellow]Error analyzing table:[/] {e}")
                table_analysis = None
        
        # Display table content
        if rows > 0 and cols > 0:
            # Create Rich table for display
            rich_table = RichTable(title=f"Table {table_number} Content")
            
            # Add headers (assuming first row is header)
            for col_idx in range(cols):
                # Use header text if available, otherwise default to "Column X"
                header = str(table_data[0][col_idx]) if col_idx < len(table_data[0]) else f"Column {col_idx+1}"
                rich_table.add_column(header)
            
            # Add data rows (skip header)
            max_rows_to_display = 10  # Limit number of rows to display
            rows_to_display = min(rows - 1, max_rows_to_display)
            
            for row_idx in range(1, rows_to_display + 1):
                row = table_data[row_idx]
                # Ensure row has the right number of cells
                cells = [str(row[col_idx]) if col_idx < len(row) else "" for col_idx in range(cols)]
                rich_table.add_row(*cells)
            
            # Add note if table is truncated
            if rows - 1 > max_rows_to_display:
                console.print(f"[dim](Showing {max_rows_to_display} of {rows-1} data rows)[/]")
            
            console.print(rich_table)
            
            # Save table if output format is specified
            if output_format and output_dir:
                # Create filename
                file_stem = file_path.stem
                table_filename = f"{file_stem}_table_{table_number}"
                
                if output_format == "csv":
                    # Save as CSV
                    csv_path = output_dir / f"{table_filename}.csv"
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        for row in table_data:
                            writer.writerow(row)
                    console.print(f"Saved table as CSV: [bold blue]{csv_path}[/]")
                
                elif output_format == "json":
                    # Save as JSON
                    json_path = output_dir / f"{table_filename}.json"
                    with open(json_path, "w") as f:
                        # Create a more structured representation
                        structured_table = {
                            "headers": table_data[0] if rows > 0 else [],
                            "data": table_data[1:] if rows > 1 else [],
                            "metadata": {
                                "page": table_element.bbox.page,
                                "position": {
                                    "x0": table_element.bbox.x0,
                                    "y0": table_element.bbox.y0,
                                    "x1": table_element.bbox.x1,
                                    "y1": table_element.bbox.y1,
                                },
                                "rows": rows,
                                "columns": cols,
                                "analysis": table_analysis
                            }
                        }
                        json.dump(structured_table, f, indent=2)
                    console.print(f"Saved table as JSON: [bold blue]{json_path}[/]")
                
                elif output_format == "html":
                    # Save as HTML
                    html_path = output_dir / f"{table_filename}.html"
                    with open(html_path, "w") as f:
                        html = "<html><head><title>Extracted Table</title>"
                        html += "<style>table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ddd;padding:8px;}th{background-color:#f2f2f2;}</style>"
                        html += "</head><body>"
                        html += f"<h1>Table {table_number}</h1>"
                        html += f"<p>Page: {table_element.bbox.page + 1}</p>"
                        
                        # Add analysis if available
                        if table_analysis:
                            html += "<h2>Table Analysis</h2>"
                            html += "<ul>"
                            if table_analysis["has_header"]:
                                html += "<li>Has header row: Yes</li>"
                            else:
                                html += "<li>Has header row: No</li>"
                                
                            if table_analysis["data_types"]:
                                html += "<li>Column Data Types:<ul>"
                                for col_idx, data_type in table_analysis["data_types"].items():
                                    html += f"<li>Column {col_idx}: {data_type}</li>"
                                html += "</ul></li>"
                            
                            if table_analysis["patterns"]:
                                html += "<li>Detected Patterns:<ul>"
                                for pattern_name, pattern_desc in table_analysis["patterns"].items():
                                    html += f"<li>{pattern_name}: {pattern_desc}</li>"
                                html += "</ul></li>"
                            html += "</ul>"
                        
                        # Add table content
                        html += "<table>"
                        
                        # Add headers
                        html += "<tr>"
                        for cell in table_data[0]:
                            html += f"<th>{cell}</th>"
                        html += "</tr>"
                        
                        # Add data rows
                        for row in table_data[1:]:
                            html += "<tr>"
                            for cell in row:
                                html += f"<td>{cell}</td>"
                            html += "</tr>"
                        
                        html += "</table></body></html>"
                        f.write(html)
                    console.print(f"Saved table as HTML: [bold blue]{html_path}[/]")
                
                elif output_format == "md":
                    # Save as Markdown
                    md_path = output_dir / f"{table_filename}.md"
                    with open(md_path, "w") as f:
                        md = f"# Table {table_number}\n\n"
                        md += f"Page: {table_element.bbox.page + 1}\n\n"
                        
                        # Add analysis if available
                        if table_analysis:
                            md += "## Table Analysis\n\n"
                            if table_analysis["has_header"]:
                                md += "- Has header row: Yes\n"
                            else:
                                md += "- Has header row: No\n"
                                
                            if table_analysis["data_types"]:
                                md += "- Column Data Types:\n"
                                for col_idx, data_type in table_analysis["data_types"].items():
                                    md += f"  - Column {col_idx}: {data_type}\n"
                            
                            if table_analysis["patterns"]:
                                md += "- Detected Patterns:\n"
                                for pattern_name, pattern_desc in table_analysis["patterns"].items():
                                    md += f"  - {pattern_name}: {pattern_desc}\n"
                            md += "\n"
                        
                        # Add table content
                        md += "| " + " | ".join([str(cell) for cell in table_data[0]]) + " |\n"
                        md += "| " + " | ".join(["---" for _ in range(cols)]) + " |\n"
                        
                        # Add data rows
                        for row in table_data[1:]:
                            md += "| " + " | ".join([str(cell) for cell in row]) + " |\n"
                        
                        f.write(md)
                    console.print(f"Saved table as Markdown: [bold blue]{md_path}[/]")
        else:
            console.print(Panel("[bold yellow]Empty or malformatted table[/]"))
        
        # Add to results
        table_result = {
            "table_number": table_number,
            "element_id": table_element.element_id,
            "page": table_element.bbox.page + 1,
            "position": {
                "x0": table_element.bbox.x0,
                "y0": table_element.bbox.y0,
                "x1": table_element.bbox.x1,
                "y1": table_element.bbox.y1,
            },
            "rows": rows,
            "columns": cols,
            "data": table_data,
        }
        
        if table_analysis:
            table_result["analysis"] = table_analysis
            
        results["tables"].append(table_result)
    
    if output_format and output_dir:
        console.print(f"\nAll tables saved to: [bold blue]{output_dir}[/]")
        
        # Save overall results
        if len(tables) > 1:
            summary_path = output_dir / f"{file_path.stem}_tables_summary.json"
            try:
                with open(summary_path, "w") as f:
                    json.dump(results, f, indent=2)
                console.print(f"Tables summary saved to: [bold blue]{summary_path}[/]")
            except Exception as e:
                console.print(f"[bold red]Error saving summary:[/] {e}")
    
    return results


def analyze_table_structure(table_data: List[List[Any]]) -> Dict[str, Any]:
    """Analyze the structure and content of a table.
    
    Args:
        table_data: The table data as a list of rows, each containing a list of cells
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        "has_header": False,
        "data_types": {},
        "patterns": {},
    }
    
    if not table_data or len(table_data) < 2:
        return analysis
    
    # Check if first row looks like a header
    first_row = table_data[0]
    data_rows = table_data[1:]
    
    # Header detection heuristics
    header_score = 0
    
    # Check if first row is shorter than other rows (often means it's a header)
    if len(first_row) <= max(len(row) for row in data_rows):
        header_score += 1
    
    # Check if first row has different formatting (e.g., all strings while data is numeric)
    first_row_types = [type(cell) for cell in first_row]
    data_row_types = [type(cell) for row in data_rows for cell in row]
    
    if str in first_row_types and (int in data_row_types or float in data_row_types):
        header_score += 1
    
    # Check if first row has shorter content (headers are often shorter than data)
    first_row_lens = [len(str(cell)) for cell in first_row]
    data_row_lens = [len(str(cell)) for row in data_rows for cell in row]
    
    if sum(first_row_lens) / len(first_row_lens) < sum(data_row_lens) / len(data_row_lens):
        header_score += 1
    
    analysis["has_header"] = header_score >= 2
    
    # Analyze data types by column
    cols = min(len(row) for row in table_data)
    
    for col_idx in range(cols):
        column = [row[col_idx] for row in data_rows]
        
        # Skip empty columns
        if not column:
            continue
        
        # Try to infer data type
        numeric_count = sum(1 for cell in column if isinstance(cell, (int, float)) or (isinstance(cell, str) and cell.strip().replace(".", "", 1).replace("-", "", 1).isdigit()))
        date_count = sum(1 for cell in column if isinstance(cell, str) and ("/" in cell or "-" in cell) and len(cell) <= 12)
        
        # Determine predominant type
        if numeric_count / len(column) > 0.7:
            analysis["data_types"][col_idx] = "numeric"
        elif date_count / len(column) > 0.7:
            analysis["data_types"][col_idx] = "date"
        else:
            analysis["data_types"][col_idx] = "text"
    
    # Look for patterns in the data
    # Incremental/decremental pattern
    for col_idx, col_type in analysis["data_types"].items():
        if col_type == "numeric":
            column = [float(row[col_idx]) if isinstance(row[col_idx], (int, float)) or (isinstance(row[col_idx], str) and row[col_idx].strip().replace(".", "", 1).replace("-", "", 1).isdigit()) else 0 for row in data_rows]
            
            # Skip columns with too few values
            if len(column) < 3:
                continue
            
            # Check for incremental pattern
            increments = [column[i+1] - column[i] for i in range(len(column)-1)]
            if all(inc > 0 for inc in increments):
                analysis["patterns"]["incremental"] = f"Column {col_idx} values are consistently increasing"
            elif all(inc < 0 for inc in increments):
                analysis["patterns"]["decremental"] = f"Column {col_idx} values are consistently decreasing"
    
    # Look for repeating patterns
    for col_idx in range(cols):
        column = [row[col_idx] for row in data_rows]
        
        # Skip columns with too few values or all identical values
        if len(column) < 4 or all(cell == column[0] for cell in column):
            continue
        
        # Check for repeating pattern
        for pattern_len in range(1, min(4, len(column) // 2)):
            pattern = column[:pattern_len]
            matches = 0
            
            for i in range(0, len(column) - pattern_len, pattern_len):
                if column[i:i+pattern_len] == pattern:
                    matches += 1
            
            if matches > 1 and matches * pattern_len > len(column) / 2:
                analysis["patterns"]["repeating"] = f"Column {col_idx} has a repeating pattern of length {pattern_len}"
                break
    
    return analysis


@app.command()
def extract(
    file_path: Path = typer.Argument(..., help="Path to the document file"),
    basic: bool = typer.Option(False, "--basic", help="Use basic table detection instead of advanced"),
    format: str = typer.Option("csv", "--format", "-f", help="Output format for tables (csv, json, html, md, or none)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for extracted tables"),
    table: Optional[int] = typer.Option(None, "--table", "-t", help="Extract only the specified table index"),
    min_rows: int = typer.Option(2, "--min-rows", help="Minimum number of rows for a valid table"),
    min_cols: int = typer.Option(2, "--min-cols", help="Minimum number of columns for a valid table"),
    analyze: bool = typer.Option(False, "--analyze", "-a", help="Analyze table content for patterns"),
    details: bool = typer.Option(False, "--details", "-d", help="Show detailed information about the extraction process"),
):
    """Extract tables from a document."""
    if format not in ["csv", "json", "html", "md", "none"]:
        console.print(f"[bold red]Error:[/] Invalid output format: {format}")
        console.print("Supported formats: csv, json, html, md, none")
        raise typer.Exit(code=1)
    
    try:
        extract_tables(
            file_path=file_path,
            advanced=not basic,
            output_format=None if format == "none" else format,
            output_dir=output,
            table_filter=table,
            min_rows=min_rows,
            min_cols=min_cols,
            analyze_tables=analyze,
            show_extraction_details=details
        )
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Table extraction interrupted by user[/]")
        raise typer.Exit(code=1)


@app.command()
def batch(
    directory: Path = typer.Argument(..., help="Directory containing documents to process"),
    pattern: str = typer.Option("*.pdf", "--pattern", "-p", help="File pattern to match"),
    basic: bool = typer.Option(False, "--basic", help="Use basic table detection instead of advanced"),
    format: str = typer.Option("csv", "--format", "-f", help="Output format for tables (csv, json, html, md, or none)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for extracted tables"),
    analyze: bool = typer.Option(False, "--analyze", "-a", help="Analyze table content for patterns"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit number of files to process"),
):
    """Batch extract tables from multiple documents."""
    # Validate directory exists
    if not directory.exists() or not directory.is_dir():
        console.print(f"[bold red]Error:[/] Directory {directory} does not exist")
        raise typer.Exit(code=1)
    
    # Validate output format
    if format not in ["csv", "json", "html", "md", "none"]:
        console.print(f"[bold red]Error:[/] Invalid output format: {format}")
        console.print("Supported formats: csv, json, html, md, none")
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
    if output:
        output.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    results_summary = []
    
    for i, file_path in enumerate(files):
        console.print(f"\n[bold]Processing file {i+1}/{len(files)}:[/] {file_path.name}")
        
        # Determine output directory for this file
        if output:
            file_output_dir = output / file_path.stem
            file_output_dir.mkdir(exist_ok=True)
        else:
            file_output_dir = None
        
        try:
            # Extract tables
            result = extract_tables(
                file_path=file_path,
                advanced=not basic,
                output_format=None if format == "none" else format,
                output_dir=file_output_dir,
                analyze_tables=analyze
            )
            
            # Add to summary
            results_summary.append({
                "filename": file_path.name,
                "tables_found": len(result["tables"]),
                "success": True
            })
        except Exception as e:
            console.print(f"[bold red]Error processing {file_path.name}:[/] {e}")
            import traceback
            console.print(traceback.format_exc())
            
            results_summary.append({
                "filename": file_path.name,
                "tables_found": 0,
                "success": False,
                "error": str(e)
            })
    
    # Display summary
    console.print("\n[bold]Batch Processing Summary:[/]")
    
    summary_table = RichTable("Filename", "Status", "Tables Found")
    
    total_tables = 0
    successful_files = 0
    
    for result in results_summary:
        if result["success"]:
            status = "[green]Success[/]"
            total_tables += result["tables_found"]
            successful_files += 1
        else:
            status = f"[red]Failed:[/] {result.get('error', 'Unknown error')}"
        
        summary_table.add_row(
            result["filename"],
            status,
            str(result["tables_found"])
        )
    
    console.print(summary_table)
    console.print(f"\nProcessed {len(files)} files: {successful_files} successful, {len(files) - successful_files} failed")
    console.print(f"Total tables extracted: {total_tables}")
    
    # Save summary if output directory specified
    if output:
        summary_path = output / "extraction_summary.json"
        try:
            with open(summary_path, "w") as f:
                json.dump({
                    "files_processed": len(files),
                    "files_successful": successful_files,
                    "total_tables": total_tables,
                    "file_results": results_summary
                }, f, indent=2)
            console.print(f"Summary saved to: [bold blue]{summary_path}[/]")
        except Exception as e:
            console.print(f"[bold red]Error saving summary:[/] {e}")


@app.command()
def convert(
    input_file: Path = typer.Argument(..., help="Path to the input file (CSV, JSON, HTML)"),
    output_format: str = typer.Argument(..., help="Output format (csv, json, html, md)"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """Convert a table from one format to another."""
    # Validate input file exists
    if not input_file.exists():
        console.print(f"[bold red]Error:[/] Input file {input_file} does not exist")
        raise typer.Exit(code=1)
    
    # Validate output format
    if output_format not in ["csv", "json", "html", "md"]:
        console.print(f"[bold red]Error:[/] Invalid output format: {output_format}")
        console.print("Supported formats: csv, json, html, md")
        raise typer.Exit(code=1)
    
    # Determine input format from file extension
    input_format = input_file.suffix.lower()[1:]  # Remove the dot
    
    if input_format not in ["csv", "json", "html", "md"]:
        console.print(f"[bold red]Error:[/] Unsupported input format: {input_format}")
        console.print("Supported formats: csv, json, html, md")
        raise typer.Exit(code=1)
    
    # Determine output file
    if not output_file:
        output_file = input_file.with_suffix(f".{output_format}")
    
    # Check if output file exists
    if output_file.exists():
        overwrite = typer.confirm(f"Output file {output_file} already exists. Overwrite?")
        if not overwrite:
            console.print("Conversion cancelled")
            raise typer.Exit(code=0)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Converting table...", total=None)
            
            # Read input file
            table_data = None
            
            if input_format == "csv":
                with open(input_file, "r", newline="") as f:
                    reader = csv.reader(f)
                    table_data = list(reader)
            
            elif input_format == "json":
                with open(input_file, "r") as f:
                    json_data = json.load(f)
                    
                    # Handle different JSON formats
                    if isinstance(json_data, list):
                        # Direct list of rows
                        table_data = json_data
                    elif "data" in json_data:
                        # Our format: {"headers": [...], "data": [...]}
                        headers = json_data.get("headers", [])
                        data = json_data.get("data", [])
                        table_data = [headers] + data
                    elif "tables" in json_data and len(json_data["tables"]) > 0:
                        # Summary format with multiple tables
                        table_data = json_data["tables"][0]["data"]
            
            elif input_format == "md":
                with open(input_file, "r") as f:
                    md_content = f.read()
                    
                    # Extract table rows (very simple parser)
                    rows = []
                    table_lines = [line for line in md_content.split("\n") if line.strip().startswith("|")]
                    
                    for i, line in enumerate(table_lines):
                        # Skip separator line
                        if i == 1 and all(cell.strip().startswith("-") for cell in line.split("|")[1:-1]):
                            continue
                        
                        # Parse cells
                        cells = [cell.strip() for cell in line.split("|")[1:-1]]
                        rows.append(cells)
                    
                    table_data = rows
            
            # Validate table data
            if not table_data or not isinstance(table_data, list) or len(table_data) == 0:
                progress.update(task, description="Error: Could not extract table data")
                console.print("[bold red]Error:[/] Could not extract table data from input file")
                raise typer.Exit(code=1)
            
            # Write output file
            if output_format == "csv":
                with open(output_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    for row in table_data:
                        writer.writerow(row)
            
            elif output_format == "json":
                with open(output_file, "w") as f:
                    # Use our structured format
                    json_data = {
                        "headers": table_data[0] if len(table_data) > 0 else [],
                        "data": table_data[1:] if len(table_data) > 1 else [],
                        "metadata": {
                            "rows": len(table_data),
                            "columns": len(table_data[0]) if len(table_data) > 0 else 0,
                            "source": str(input_file)
                        }
                    }
                    json.dump(json_data, f, indent=2)
            
            elif output_format == "html":
                with open(output_file, "w") as f:
                    html = "<html><head><title>Converted Table</title>"
                    html += "<style>table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ddd;padding:8px;}th{background-color:#f2f2f2;}</style>"
                    html += "</head><body>"
                    html += f"<h1>Converted from {input_file.name}</h1>"
                    
                    # Add table content
                    html += "<table>"
                    
                    # Add headers
                    html += "<tr>"
                    for cell in table_data[0]:
                        html += f"<th>{cell}</th>"
                    html += "</tr>"
                    
                    # Add data rows
                    for row in table_data[1:]:
                        html += "<tr>"
                        for cell in row:
                            html += f"<td>{cell}</td>"
                        html += "</tr>"
                    
                    html += "</table></body></html>"
                    f.write(html)
            
            elif output_format == "md":
                with open(output_file, "w") as f:
                    md = f"# Converted Table\n\n"
                    
                    # Add table content
                    md += "| " + " | ".join([str(cell) for cell in table_data[0]]) + " |\n"
                    md += "| " + " | ".join(["---" for _ in range(len(table_data[0]))]) + " |\n"
                    
                    # Add data rows
                    for row in table_data[1:]:
                        md += "| " + " | ".join([str(cell) for cell in row]) + " |\n"
                    
                    f.write(md)
            
            progress.update(task, completed=True, description=f"Converted table from {input_format} to {output_format}")
        
        console.print(f"[bold green]Table converted successfully![/]")
        console.print(f"Output saved to: [bold blue]{output_file}[/]")
    except Exception as e:
        console.print(f"[bold red]Error converting table:[/] {e}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()