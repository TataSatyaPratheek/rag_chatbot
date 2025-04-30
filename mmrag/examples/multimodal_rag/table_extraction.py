"""Example of table extraction from documents."""

import argparse
import csv
import json
from pathlib import Path

from mmrag.document_processing import PDFProcessor
from mmrag.document_processing.advanced_table import CascadeTabNetDetector
from mmrag.document_processing.table import TableDetector
from rich.console import Console
from rich.panel import Panel
from rich.table import Table as RichTable

console = Console()

def extract_tables(file_path, advanced=True, output_format="csv", output_dir=None):
    """Extract tables from a document using basic or advanced detection."""
    # Set up the document processor
    if advanced:
        try:
            # Try to use advanced table detection
            table_detector = CascadeTabNetDetector()
            console.print("Using [bold cyan]advanced[/] table detection (CascadeTabNet)")
        except Exception as e:
            console.print(f"[bold yellow]Warning:[/] Could not initialize advanced table detector: {e}")
            console.print("Falling back to basic table detection")
            table_detector = TableDetector()
            advanced = False
    else:
        table_detector = TableDetector()
        console.print("Using [bold cyan]basic[/] table detection")
    
    processor = PDFProcessor(
        extract_tables=True,
        extract_images=False,  # Focus only on tables
        table_detector=table_detector,
        advanced_table_detection=advanced
    )
    
    # Process the document
    console.print(f"Processing document: [bold blue]{file_path}[/]")
    document = processor.process(file_path)
    
    console.print(f"[bold green]Successfully processed document:[/] {Path(file_path).name}")
    console.print(f"Document ID: {document.document_id}")
    
    # Extract tables
    tables = [element for element in document.elements if element.element_type == "table"]
    
    if not tables:
        console.print("[bold yellow]No tables detected in the document.[/]")
        return document
    
    console.print(f"\nFound [bold]{len(tables)}[/] tables in the document")
    
    # Set up output directory if specified
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Path(f"tables_{document.document_id}")
    
    # Create output directory if it doesn't exist
    if output_format:
        output_path.mkdir(exist_ok=True, parents=True)
    
    # Process each table
    for i, table_element in enumerate(tables):
        table_number = i + 1
        console.print(f"\n[bold cyan]Table {table_number}:[/]")
        console.print(f"Location: Page {table_element.bbox.page + 1}, Position: ({table_element.bbox.x0:.1f}, {table_element.bbox.y0:.1f})")
        
        # Get table content
        table_data = table_element.content
        
        # Display table information
        rows = len(table_data) if isinstance(table_data, list) else 0
        cols = len(table_data[0]) if rows > 0 and isinstance(table_data[0], list) else 0
        
        console.print(f"Dimensions: {rows} rows × {cols} columns")
        
        # Display table content
        if rows > 0 and cols > 0:
            rich_table = RichTable(title=f"Table {table_number}")
            
            # Add headers (assuming first row is header)
            for col_idx in range(cols):
                # Use header text if available, otherwise default to "Column X"
                header = str(table_data[0][col_idx]) if col_idx < len(table_data[0]) else f"Column {col_idx+1}"
                rich_table.add_column(header)
            
            # Add data rows (skip header)
            for row_idx in range(1, rows):
                row = table_data[row_idx]
                # Ensure row has the right number of cells
                cells = [str(row[col_idx]) if col_idx < len(row) else "" for col_idx in range(cols)]
                rich_table.add_row(*cells)
            
            console.print(rich_table)
            
            # Save table if output format is specified
            if output_format:
                # Create filename
                file_stem = Path(file_path).stem
                table_filename = f"{file_stem}_table_{table_number}"
                
                if output_format == "csv":
                    # Save as CSV
                    csv_path = output_path / f"{table_filename}.csv"
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.writer(f)
                        for row in table_data:
                            writer.writerow(row)
                    console.print(f"Saved table as CSV: [bold blue]{csv_path}[/]")
                
                elif output_format == "json":
                    # Save as JSON
                    json_path = output_path / f"{table_filename}.json"
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
                                }
                            }
                        }
                        json.dump(structured_table, f, indent=2)
                    console.print(f"Saved table as JSON: [bold blue]{json_path}[/]")
                
                elif output_format == "html":
                    # Save as HTML
                    html_path = output_path / f"{table_filename}.html"
                    with open(html_path, "w") as f:
                        html = "<html><head><title>Extracted Table</title>"
                        html += "<style>table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ddd;padding:8px;}</style>"
                        html += "</head><body>"
                        html += f"<h1>Table {table_number}</h1>"
                        html += f"<p>Page: {table_element.bbox.page + 1}</p>"
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
        else:
            console.print(Panel("[bold yellow]Empty or malformatted table[/]"))
    
    if output_format:
        console.print(f"\nAll tables saved to: [bold blue]{output_path}[/]")
    
    return document

def main():
    """Run the table extraction example."""
    parser = argparse.ArgumentParser(description="Table extraction from documents.")
    parser.add_argument("file_path", type=str, help="Path to the document file")
    parser.add_argument("--basic", action="store_true", help="Use basic table detection instead of advanced")
    parser.add_argument("--format", "-f", type=str, choices=["csv", "json", "html", "none"], default="csv",
                        help="Output format for tables (default: csv)")
    parser.add_argument("--output", "-o", type=str, help="Output directory for extracted tables")
    
    args = parser.parse_args()
    
    extract_tables(
        args.file_path,
        advanced=not args.basic,
        output_format=None if args.format == "none" else args.format,
        output_dir=args.output
    )

if __name__ == "__main__":
    main()
