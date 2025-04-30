"""Example of multimodal document processing and retrieval."""

import argparse
from pathlib import Path
import tempfile

from mmrag.document_processing.factory import get_processor
from mmrag.vectordb import ChromaStore
from rich.console import Console

console = Console()


def process_multimodal(file_path: str, query: str = None):
    """Process a document with multimodal content and perform retrieval."""
    file_path = Path(file_path)
    
    # Get the appropriate processor for this file type
    try:
        processor = get_processor(
            file_path,
            extract_tables=True,
            extract_images=True,
            advanced_table_detection=True,
            enable_enhanced_visual=True,
        )
    except ValueError:
        console.print(f"[bold red]Error:[/] Unsupported file type: {file_path.suffix}")
        return
    
    console.print(f"Processing [bold blue]{file_path}[/] with [bold cyan]{processor.__class__.__name__}[/]...")
    
    # Process the document
    document = processor.process(file_path)
    
    # Display element counts
    element_types = {}
    for element in document.elements:
        element_types[element.element_type] = element_types.get(element.element_type, 0) + 1
    
    console.print("\n[bold]Extracted elements:[/]")
    for element_type, count in element_types.items():
        console.print(f"  {element_type}: {count}")
    
    # Set up vector store in a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        console.print("\nSetting up vector store...")
        store = ChromaStore(persist_directory=temp_dir)
        
        # Add document to vector store
        store.add_document(document)
        console.print(f"Added document with ID: [bold]{document.document_id}[/]")
        
        # Perform retrieval if query provided
        if query:
            console.print(f"\nQuerying: [bold cyan]{query}[/]")
            results = store.query(query, n_results=3)
            
            console.print(f"\n[bold green]Retrieved {len(results['ids'][0])} results:[/]")
            for i, (doc_content, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                console.print(f"\n[bold]Result {i+1}[/] ([cyan]{metadata['element_type']}[/] from page {metadata['page']}):")
                # Truncate long text for display
                display_content = doc_content[:300] + "..." if len(doc_content) > 300 else doc_content
                console.print(display_content)
                console.print(f"[dim]Element ID: {metadata['element_id']}[/]")
    
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a multimodal document and perform retrieval.")
    parser.add_argument("file_path", type=str, help="Path to the document file")
    parser.add_argument("--query", "-q", type=str, help="Query to search for in the document")
    args = parser.parse_args()
    
    process_multimodal(args.file_path, args.query)
