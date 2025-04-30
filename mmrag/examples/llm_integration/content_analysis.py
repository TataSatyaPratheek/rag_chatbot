"""Example of document content analysis using local LLMs."""

import argparse
import json
from pathlib import Path

from mmrag.document_processing import PDFProcessor
from mmrag.llm import OllamaClient, ContentUnderstanding
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

console = Console()

def analyze_document_content(file_path, model_name="llama3.2:latest", output_json=False):
    """Analyze document content using a local LLM."""
    # Initialize LLM client
    try:
        client = OllamaClient(model=model_name)
        console.print(f"Using LLM model: [bold cyan]{model_name}[/]")
    except Exception as e:
        console.print(f"[bold red]Error initializing LLM client:[/] {e}")
        console.print("Make sure Ollama is installed and running, and the model is available.")
        return None
    
    # Initialize document processor
    processor = PDFProcessor(
        extract_tables=True,
        extract_images=True
    )
    
    # Process document
    console.print(f"Processing document: [bold blue]{file_path}[/]")
    document = processor.process(file_path)
    
    console.print(f"[bold green]Successfully processed document:[/] {Path(file_path).name}")
    console.print(f"Document ID: {document.document_id}")
    console.print(f"Extracted {len(document.elements)} elements")
    
    # Set up content analyzer
    analyzer = ContentUnderstanding(llm_client=client)
    
    # Analyze document
    console.print("\n[bold]Analyzing document content...[/]")
    document_analysis = analyzer.analyze_document(document)
    
    # Display document analysis
    console.print(Panel(document_analysis["summary"], title="Document Summary", border_style="green"))
    
    # Display topics
    console.print("\n[bold]Key Topics:[/]")
    for topic in document_analysis["topics"]:
        console.print(f"- {topic}")
    
    # Display entities
    if document_analysis["entities"]:
        console.print("\n[bold]Named Entities:[/]")
        entities_tree = Tree("Entities")
        for entity_type, entities in document_analysis["entities"].items():
            entity_branch = entities_tree.add(entity_type)
            for entity in entities:
                entity_branch.add(entity)
        console.print(entities_tree)
    
    # Analyze specific elements by type
    element_types = set(element.element_type for element in document.elements)
    element_analyses = {}
    
    for element_type in element_types:
        # Get the first few elements of this type
        elements = [e for e in document.elements if e.element_type == element_type][:2]
        
        if elements:
            console.print(f"\n[bold]Analyzing {element_type} elements...[/]")
            type_analyses = []
            
            for element in elements:
                analysis = analyzer.analyze_element(element)
                type_analyses.append({
                    "element_id": element.element_id,
                    "analysis": analysis
                })
                
                # Display analysis
                console.print(f"\n[bold cyan]{element_type.upper()}[/] (ID: {element.element_id}):")
                
                if element_type == "text":
                    if "summary" in analysis:
                        console.print(f"Summary: {analysis['summary']}")
                    if "sentiment" in analysis:
                        console.print(f"Sentiment: {analysis['sentiment']}")
                elif element_type == "table":
                    if "description" in analysis:
                        console.print(f"Description: {analysis['description']}")
                    if "insights" in analysis:
                        console.print(f"Insights: {analysis['insights']}")
                elif element_type in ["image", "chart"]:
                    if "interpretation" in analysis:
                        console.print(f"Interpretation: {analysis['interpretation']}")
                    if "relevance" in analysis:
                        console.print(f"Relevance: {analysis['relevance']}")
            
            element_analyses[element_type] = type_analyses
    
    # Combine all analyses
    complete_analysis = {
        "document_id": document.document_id,
        "filename": document.filename,
        "document_analysis": document_analysis,
        "element_analyses": element_analyses
    }
    
    # Save to JSON if requested
    if output_json:
        output_path = Path(file_path).with_suffix(".analysis.json")
        with open(output_path, "w") as f:
            json.dump(complete_analysis, f, indent=2)
        console.print(f"\nSaved analysis to: [bold blue]{output_path}[/]")
    
    return complete_analysis

def main():
    """Run the content analysis example."""
    parser = argparse.ArgumentParser(description="Document content analysis using local LLMs.")
    parser.add_argument("file_path", type=str, help="Path to the document file")
    parser.add_argument("--model", "-m", type=str, default="llama3.2:latest", help="LLM model to use")
    parser.add_argument("--json", "-j", action="store_true", help="Save analysis to JSON file")
    
    args = parser.parse_args()
    
    analyze_document_content(args.file_path, args.model, args.json)

if __name__ == "__main__":
    main()
