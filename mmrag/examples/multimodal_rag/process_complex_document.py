"""Example of processing complex multimodal documents."""

import argparse
import json
import os
from pathlib import Path

from mmrag.document_processing.factory import get_processor
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

console = Console()

def process_complex_document(file_path, enable_advanced_tables=True, enable_enhanced_visual=True,
                             enable_llm_analysis=False, output_format="json"):
    """Process a complex document with multiple modalities."""
    # Determine file type and get appropriate processor
    try:
        processor = get_processor(
            file_path,
            extract_tables=True,
            extract_images=True,
            advanced_table_detection=enable_advanced_tables,
            enable_enhanced_visual=enable_enhanced_visual,
            enable_llm_analysis=enable_llm_analysis
        )
    except ValueError as e:
        console.print(f"[bold red]Error:[/] {e}")
        return None
    
    console.print(f"Processing document: [bold blue]{file_path}[/]")
    console.print(f"Using processor: [bold cyan]{processor.__class__.__name__}[/]")
    console.print(f"Advanced tables: {'Enabled' if enable_advanced_tables else 'Disabled'}")
    console.print(f"Enhanced visual: {'Enabled' if enable_enhanced_visual else 'Disabled'}")
    console.print(f"LLM analysis: {'Enabled' if enable_llm_analysis else 'Disabled'}")
    
    # Process the document with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Processing document...", total=1)
        
        # Process document
        document = processor.process(file_path)
        
        progress.update(task, completed=1)
    
    # Display document information
    console.print(f"\n[bold green]Successfully processed document:[/] {Path(file_path).name}")
    console.print(f"Document ID: {document.document_id}")
    console.print(f"Element count: {len(document.elements)}")
    
    # Count element types
    element_counts = {}
    for element in document.elements:
        element_counts[element.element_type] = element_counts.get(element.element_type, 0) + 1
    
    # Display element breakdown
    table = Table("Element Type", "Count", "Percentage")
    
    for element_type, count in element_counts.items():
        percentage = (count / len(document.elements)) * 100
        table.add_row(element_type, str(count), f"{percentage:.1f}%")
    
    console.print(table)
    
    # Display document structure
    if document.metadata.get("page_count"):
        console.print(f"\n[bold]Document Structure:[/] {document.metadata.get('page_count')} pages")
        
        # Create a tree visualization
        doc_tree = Tree(f"{Path(file_path).name}")
        
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
        console.print(Panel(document.analysis.get("summary", "No summary available"), title="Summary"))
        
        if "topics" in document.analysis and document.analysis["topics"]:
            console.print("\n[bold]Key Topics:[/]")
            for topic in document.analysis["topics"]:
                console.print(f"- {topic}")
    
    # Save output
    if output_format:
        output_path = Path(file_path).with_suffix(f".{output_format}")
        
        if output_format == "json":
            with open(output_path, "w") as f:
                f.write(document.to_json())
        elif output_format == "html":
            # Simple HTML conversion
            html_content = "<html><head><title>Document Analysis</title></head><body>"
            html_content += f"<h1>{Path(file_path).name}</h1>"
            
            # Add document info
            html_content += f"<h2>Document Information</h2>"
            html_content += f"<p>Document ID: {document.document_id}</p>"
            html_content += f"<p>Element count: {len(document.elements)}</p>"
            
            # Add element breakdown
            html_content += f"<h2>Element Types</h2>"
            html_content += "<table border='1'><tr><th>Element Type</th><th>Count</th></tr>"
            for element_type, count in element_counts.items():
                html_content += f"<tr><td>{element_type}</td><td>{count}</td></tr>"
            html_content += "</table>"
            
            # Add sample elements
            html_content += f"<h2>Sample Elements</h2>"
            for element_type in element_counts:
                for element in document.elements:
                    if element.element_type == element_type:
                        html_content += f"<h3>{element_type.capitalize()} Element</h3>"
                        
                        if element_type == "text":
                            html_content += f"<div>{element.content}</div>"
                        elif element_type == "table":
                            html_content += "<table border='1'>"
                            if isinstance(element.content, list):
                                for row in element.content:
                                    html_content += "<tr>"
                                    if isinstance(row, list):
                                        for cell in row:
                                            html_content += f"<td>{cell}</td>"
                                    html_content += "</tr>"
                            html_content += "</table>"
                        elif element_type in ["image", "chart"]:
                            if hasattr(element, "content") and isinstance(element.content, str) and element.content.startswith("data:image/"):
                                html_content += f"<img src='{element.content}' style='max-width:100%'><br>"
                                html_content += f"<p>{element.metadata}</p>"
                        
                        break  # Just show one example of each type
            
            html_content += "</body></html>"
            
            with open(output_path, "w") as f:
                f.write(html_content)
        
        console.print(f"\nSaved output to: [bold blue]{output_path}[/]")
    
    return document

def main():
    """Run the complex document processing example."""
    parser = argparse.ArgumentParser(description="Process complex multimodal documents.")
    parser.add_argument("file_path", type=str, help="Path to the document file")
    parser.add_argument("--no-advanced-tables", action="store_true", help="Disable advanced table detection")
    parser.add_argument("--no-enhanced-visual", action="store_true", help="Disable enhanced visual processing")
    parser.add_argument("--llm", action="store_true", help="Enable LLM analysis")
    parser.add_argument("--output", "-o", type=str, choices=["json", "html", "none"], default="json", 
                        help="Output format (default: json)")
    
    args = parser.parse_args()
    
    process_complex_document(
        args.file_path,
        enable_advanced_tables=not args.no_advanced_tables,
        enable_enhanced_visual=not args.no_enhanced_visual,
        enable_llm_analysis=args.llm,
        output_format=None if args.output == "none" else args.output
    )

if __name__ == "__main__":
    main()
