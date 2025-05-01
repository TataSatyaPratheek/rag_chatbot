#!/usr/bin/env python3
"""
Example demonstrating how to use LlamaParse for document processing.

This script shows how to:
1. Process a document with LlamaParse
2. Extract text, tables, and visual elements
3. Build a vector index for RAG
4. Query the index with natural language questions

Usage:
    python llamaparse_example.py process PATH_TO_DOCUMENT [--multimodal] [--output OUTPUT_PATH]
    python llamaparse_example.py query PATH_TO_DOCUMENT "Your query here"

Requirements:
    - LlamaParse API key set as LLAMA_CLOUD_API_KEY in environment
    - mmrag with llama-parse and llama-index installed
"""

import argparse
import os
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from mmrag.document_processing.factory import get_processor
from mmrag.document_processing.llamaparse.utils import (
    check_llamaparse_api_key, 
    get_llamaparse_cost_estimate,
    get_optimal_llamaparse_settings
)
from mmrag.vectordb.chroma import ChromaStore

console = Console()

def process_document(args):
    """Process a document with LlamaParse and output results."""
    file_path = Path(args.document_path)
    output_path = Path(args.output) if args.output else None
    use_multimodal = args.multimodal
    
    # Check for API key
    if not check_llamaparse_api_key():
        console.print("[bold red]Error:[/] LLAMA_CLOUD_API_KEY environment variable not set.")
        console.print("Please set it with your LlamaParse API key.")
        return 1
    
    # Get cost estimate
    console.print(Panel.fit("Cost Estimate", title="LlamaParse"))
    cost_estimate = get_llamaparse_cost_estimate(
        file_path=str(file_path),
        use_multimodal=use_multimodal,
        multimodal_model="anthropic-sonnet-3.5"
    )
    for key, value in cost_estimate.items():
        console.print(f"{key}: {value}")
    
    # Confirm processing
    console.print()
    if not args.yes:
        confirm = input("Continue with processing? (y/n): ")
        if confirm.lower() != 'y':
            console.print("Processing cancelled.")
            return 0
    
    # Get optimal settings
    settings = get_optimal_llamaparse_settings(str(file_path))
    
    # Override with command line args
    if use_multimodal:
        settings["use_multimodal"] = True
    
    console.print(Panel.fit("Processing Settings", title="LlamaParse"))
    for key, value in settings.items():
        console.print(f"{key}: {value}")
    
    # Process document
    console.print("\n[bold]Processing document...[/]")
    start_time = time.time()
    
    processor = get_processor(
        file_path,
        use_docling=False,
        use_llamaparse=True,
        extract_tables=True,
        extract_images=True,
        advanced_table_detection=True,
        enable_enhanced_visual=True,
        use_multimodal=settings["use_multimodal"],
        multimodal_model=settings["multimodal_model"],
        timeout_seconds=settings["timeout_seconds"],
    )
    
    # Process the document
    processed_doc = processor.process(file_path)
    
    # Print processing results
    duration = time.time() - start_time
    console.print(f"\n[bold green]Document processed successfully in {duration:.2f} seconds![/]")
    console.print(f"Document ID: {processed_doc.document_id}")
    console.print(f"File: {processed_doc.filename}")
    console.print(f"Elements extracted: {len(processed_doc.elements)}")
    
    # Count elements by type
    element_counts = {}
    for element in processed_doc.elements:
        element_type = element.element_type
        element_counts[element_type] = element_counts.get(element_type, 0) + 1
    
    console.print("\n[bold]Element types:[/]")
    for element_type, count in element_counts.items():
        console.print(f"  {element_type}: {count}")
    
    # Save output if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(processed_doc.to_json())
        console.print(f"\nOutput saved to: {output_path}")
    
    # Create vector index for document
    store_dir = Path("./.chroma_llamaparse")
    console.print(f"\n[bold]Building vector index...[/]")
    store = ChromaStore(persist_directory=store_dir, collection_name=processed_doc.document_id)
    store.add_document(processed_doc)
    console.print(f"Vector index built and stored in: {store_dir}")
    
    # Display sample queries
    console.print("\n[bold]Sample queries to try:[/]")
    console.print(f"  python {sys.argv[0]} query {file_path} \"What is the main topic of this document?\"")
    console.print(f"  python {sys.argv[0]} query {file_path} \"Extract all tables with numerical data\"")
    console.print(f"  python {sys.argv[0]} query {file_path} \"Summarize the key points\"")
    
    return 0

def query_document(args):
    """Query a processed document with natural language."""
    file_path = Path(args.document_path)
    query_text = args.query_text
    
    # Generate document ID to find the right collection
    import hashlib
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            hasher.update(byte_block)
    doc_type = file_path.suffix.lower().lstrip(".")
    document_id = f"{doc_type}-{hasher.hexdigest()[:16]}"
    
    # Initialize vector store
    store_dir = Path("./.chroma_llamaparse")
    store = ChromaStore(persist_directory=store_dir, collection_name=document_id)
    
    # Query the vector store
    console.print(f"\n[bold]Querying: [cyan]{query_text}[/][/]")
    results = store.query(query_text=query_text, n_results=5)
    
    # Print results
    console.print("\n[bold green]Results:[/]")
    for i, (doc_id, doc, metadata) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
    )):
        console.print(f"\n[bold cyan]Result {i+1}:[/]")
        console.print(f"Element Type: {metadata['element_type']}")
        console.print(f"Page: {metadata['page']}")
        console.print(f"Content: {doc}")
    
    return 0

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Process and query documents with LlamaParse.")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process a document with LlamaParse")
    process_parser.add_argument("document_path", help="Path to document to process")
    process_parser.add_argument("--multimodal", action="store_true", help="Enable multimodal processing")
    process_parser.add_argument("--output", help="Path to save output JSON")
    process_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query a processed document")
    query_parser.add_argument("document_path", help="Path to the document to query")
    query_parser.add_argument("query_text", help="Query text")
    
    args = parser.parse_args()
    
    # Execute command
    if args.command == "process":
        return process_document(args)
    elif args.command == "query":
        return query_document(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())