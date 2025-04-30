"""Example of integrating Ollama for local LLM analysis."""

import argparse
from pathlib import Path

from mmrag.document_processing import PDFProcessor
from mmrag.llm import OllamaClient, ContentUnderstanding
from rich.console import Console
from rich.panel import Panel

console = Console()


def analyze_with_ollama(file_path: str, model_name: str = "llama3.2:latest"):
    """Process a document and analyze it with a local Ollama model."""
    try:
        # Test Ollama connection
        client = OllamaClient(model=model_name)
        test_response = client.generate_sync("Hello, are you working?", max_tokens=20)
        console.print(f"[bold green]Ollama connection successful[/] - Model: {model_name}")
    except Exception as e:
        console.print(f"[bold red]Error connecting to Ollama:[/] {e}")
        console.print("Make sure Ollama is installed and running, and the model is available.")
        console.print("Installation: curl -fsSL https://ollama.com/install.sh | sh")
        console.print(f"Pull model: ollama pull {model_name}")
        return
    
    # Process the document
    file_path = Path(file_path)
    console.print(f"\nProcessing [bold blue]{file_path}[/]...")
    
    processor = PDFProcessor(
        extract_tables=True,
        extract_images=True,
        enable_llm_analysis=False,  # We'll do this manually
    )
    
    document = processor.process(file_path)
    
    # Set up content analyzer with Ollama
    analyzer = ContentUnderstanding(llm_client=client)
    
    # Analyze the document
    console.print("\n[bold]Analyzing document content with LLM...[/]")
    analysis = analyzer.analyze_document(document)
    
    # Display analysis results
    console.print(Panel(analysis["summary"], title="Document Summary", expand=False))
    
    console.print("\n[bold]Key Topics:[/]")
    for topic in analysis["topics"]:
        console.print(f"- {topic}")
    
    console.print("\n[bold]Entities:[/]")
    for entity_type, entities in analysis["entities"].items():
        console.print(f"[bold cyan]{entity_type}:[/] {', '.join(entities)}")
    
    # Analyze specific elements
    console.print("\n[bold]Analyzing individual elements...[/]")
    
    # Find a table element to analyze
    table_elements = [e for e in document.elements if e.element_type == "table"]
    if table_elements:
        console.print("\n[bold cyan]Table Analysis:[/]")
        table_analysis = analyzer.analyze_element(table_elements[0])
        if "description" in table_analysis:
            console.print(f"Description: {table_analysis['description']}")
        if "insights" in table_analysis:
            console.print(f"Insights: {table_analysis['insights']}")
    
    # Find a chart element to analyze
    chart_elements = [e for e in document.elements if e.element_type == "chart"]
    if chart_elements:
        console.print("\n[bold cyan]Chart Analysis:[/]")
        chart_analysis = analyzer.analyze_element(chart_elements[0])
        if "interpretation" in chart_analysis:
            console.print(f"Interpretation: {chart_analysis['interpretation']}")
    
    return document, analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a document with Ollama LLM.")
    parser.add_argument("file_path", type=str, help="Path to the document file")
    parser.add_argument("--model", "-m", type=str, default="llama3.2:latest", 
                        help="Ollama model to use (default: llama3.2:latest)")
    args = parser.parse_args()
    
    analyze_with_ollama(args.file_path, args.model)
