"""Basic example of vector search using ChromaDB."""

import argparse
from pathlib import Path

from mmrag.document_processing import PDFProcessor
from mmrag.vectordb import ChromaStore
from rich.console import Console
from rich.table import Table

console = Console()

def index_document(file_path, persist_dir=None, collection_name="vector_search_demo"):
    """Process and index a document."""
    # Initialize document processor
    processor = PDFProcessor(
        extract_tables=True,
        extract_images=True
    )
    
    # Process document
    console.print(f"Processing document: [bold blue]{file_path}[/]")
    document = processor.process(file_path)
    
    # Set up vector store
    if persist_dir is None:
        persist_dir = Path("temp_vector_search")
    else:
        persist_dir = Path(persist_dir)
    
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    store = ChromaStore(
        persist_directory=persist_dir,
        collection_name=collection_name
    )
    
    # Index document
    console.print("Indexing document in vector database...")
    store.add_document(document)
    
    # Report statistics
    element_types = {}
    for element in document.elements:
        element_types[element.element_type] = element_types.get(element.element_type, 0) + 1
    
    console.print(f"[bold green]Successfully indexed document:[/] {Path(file_path).name}")
    console.print(f"Document ID: {document.document_id}")
    
    # Show element breakdown
    table = Table("Element Type", "Count")
    for element_type, count in element_types.items():
        table.add_row(element_type, str(count))
    console.print(table)
    
    return store, document

def search_documents(store, query, n_results=5, element_type=None):
    """Search documents by semantic similarity."""
    console.print(f"\nSearching for: [bold cyan]'{query}'[/]")
    
    # Prepare where filter
    where = {}
    if element_type:
        where["element_type"] = element_type
    
    # Query the database
    results = store.query(
        query_text=query,
        n_results=n_results,
        where=where if where else None
    )
    
    # Format and display results
    console.print(f"\n[bold green]Search Results:[/] Found {len(results['ids'][0])} matches")
    
    table = Table("Rank", "Content", "Element Type", "Page", "Relevance")
    
    for i, (doc_content, metadata, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0] if "distances" in results else [0] * len(results["documents"][0])
    )):
        # Get metadata
        element_type = metadata.get("element_type", "unknown")
        page = metadata.get("page", "unknown")
        
        # Calculate relevance score (1 - distance, normalized to percentage)
        # Lower distance = higher relevance
        relevance = round((1 - (distance / 2)) * 100) if distance else "N/A"
        
        # Truncate long content for display
        content = doc_content[:100] + "..." if len(doc_content) > 100 else doc_content
        
        # Add to table
        table.add_row(
            str(i+1),
            content,
            element_type,
            str(page),
            f"{relevance}%" if isinstance(relevance, int) else relevance
        )
    
    console.print(table)
    
    return results

def explore_vector_database(store):
    """Interactive exploration of vector database contents."""
    console.print("\n[bold]Vector Database Explorer[/]")
    console.print("Type 'quit' to exit, 'filter <type>' to filter by element type")
    
    current_filter = None
    
    while True:
        # Show current filter if any
        filter_display = f" (Filtered to: {current_filter})" if current_filter else ""
        
        # Get user input
        user_input = console.input(f"\n[bold cyan]Query{filter_display}:[/] ")
        
        if user_input.lower() == "quit":
            break
            
        elif user_input.lower().startswith("filter "):
            filter_type = user_input[7:].strip().lower()
            
            if filter_type == "none" or filter_type == "clear":
                current_filter = None
                console.print("[bold yellow]Filter cleared[/]")
            else:
                current_filter = filter_type
                console.print(f"[bold yellow]Filter set to: {current_filter}[/]")
            
            continue
        
        # Perform search with current filter
        search_documents(store, user_input, element_type=current_filter)

def main():
    """Run the vector search demo."""
    parser = argparse.ArgumentParser(description="Vector search demo for document retrieval.")
    parser.add_argument("--document", "-d", type=str, help="Document to index")
    parser.add_argument("--query", "-q", type=str, help="Query to search for")
    parser.add_argument("--filter", "-f", type=str, help="Filter results by element type")
    parser.add_argument("--results", "-r", type=int, default=5, help="Number of results to return")
    parser.add_argument("--persist-dir", "-p", type=str, help="Directory to persist vector database")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Setup vector store
    persist_dir = args.persist_dir or "temp_vector_search"
    persist_dir = Path(persist_dir)
    
    if args.document:
        # Index document
        store, document = index_document(args.document, persist_dir)
    else:
        # Try to load existing vector store
        try:
            persist_dir.mkdir(parents=True, exist_ok=True)
            store = ChromaStore(persist_directory=persist_dir)
            console.print(f"Using existing vector database at: {persist_dir}")
        except Exception as e:
            console.print(f"[bold red]Error loading vector database:[/] {e}")
            console.print("Please specify a document to index with --document")
            return
    
    # Handle query or interactive mode
    if args.query:
        search_documents(store, args.query, args.results, args.filter)
    elif args.interactive:
        explore_vector_database(store)
    else:
        console.print("\nUse --query to search or --interactive for interactive mode")

if __name__ == "__main__":
    main()
