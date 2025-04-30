"""Basic example of PDF processing and content extraction."""

import argparse
from pathlib import Path

from mmrag.document_processing import PDFProcessor
from rich.console import Console
from rich.table import Table

console = Console()


def process_pdf(pdf_path: str, output_json: bool = False):
    """Process a PDF file and display extracted content."""
    processor = PDFProcessor(
        extract_tables=True,
        extract_images=True,
    )
    
    console.print(f"Processing [bold blue]{pdf_path}[/]...")
    
    document = processor.process(pdf_path)
    
    # Print summary information
    console.print(f"\n[bold green]Document ID:[/] {document.document_id}")
    console.print(f"[bold green]Filename:[/] {document.filename}")
    console.print(f"[bold green]Page count:[/] {document.metadata.get('page_count', 'Unknown')}")
    
    # Display element counts
    element_types = {}
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
                        for row in element.content[1:]:
                            rich_table.add_row(*[str(cell) for cell in row])
                        console.print(rich_table)
                elif element_type in ["image", "chart"]:
                    console.print(f"[Image/Chart metadata: {element.metadata}]")
                break
    
    # Save as JSON if requested
    if output_json:
        output_path = Path(pdf_path).with_suffix(".json")
        with open(output_path, "w") as f:
            f.write(document.to_json())
        console.print(f"\nSaved document JSON to [bold blue]{output_path}[/]")
    
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a PDF file and display extracted content.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    process_pdf(args.pdf_path, args.json)
