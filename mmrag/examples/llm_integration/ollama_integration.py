"""Enhanced example of integrating Ollama for local LLM analysis."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import typer
import httpx
from mmrag.document_processing import PDFProcessor
from mmrag.llm import OllamaClient, ContentUnderstanding
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.markdown import Markdown

app = typer.Typer(help="Analyze documents with local LLMs via Ollama.")
console = Console()


async def test_ollama_connection(model_name: str) -> Tuple[bool, str]:
    """Test if Ollama is available and the model is loaded.
    
    Args:
        model_name: Name of the model to test
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Test Ollama connection
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check if Ollama is running
            try:
                response = await client.get("http://localhost:11434/api/tags")
                response.raise_for_status()
            except (httpx.ConnectError, httpx.TimeoutException):
                return False, "Ollama server is not running"
            
            # Check if the model is available
            models = response.json().get("models", [])
            if not any(model["name"].startswith(model_name) for model in models):
                return False, f"Model '{model_name}' is not available, you may need to pull it first"
            
            # Test a simple query to make sure the model works
            try:
                data = {
                    "model": model_name,
                    "prompt": "Hello, are you working?",
                    "stream": False
                }
                response = await client.post("http://localhost:11434/api/generate", json=data, timeout=10.0)
                response.raise_for_status()
                return True, "Ollama connection successful"
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                return False, f"Error testing model: {str(e)}"
    except Exception as e:
        return False, f"Error connecting to Ollama: {str(e)}"


async def analyze_with_ollama_async(
    file_path: Path, 
    model_name: str = "llama3.2:latest",
    analysis_types: List[str] = ["document", "elements", "entities"],
    temperature: float = 0.3,
    output_format: Optional[str] = None,
    timeout: int = 60
) -> Tuple[bool, Dict[str, Any]]:
    """Process a document and analyze it with a local Ollama model asynchronously.
    
    Args:
        file_path: Path to the document file
        model_name: Name of the Ollama model to use
        analysis_types: Types of analysis to perform
        temperature: Temperature for the LLM
        output_format: Output format (json or md)
        timeout: Timeout for LLM requests in seconds
        
    Returns:
        Tuple of (success, results)
    """
    # Test Ollama connection first
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Testing Ollama connection...", total=None)
        
        ollama_ok, message = await test_ollama_connection(model_name)
        
        if ollama_ok:
            progress.update(task, completed=True, description=f"✓ {message} - Model: {model_name}")
        else:
            progress.update(task, completed=True, description=f"✗ {message}")
            console.print(f"[bold red]Error:[/] {message}")
            console.print("Make sure Ollama is installed and running, and the model is available.")
            console.print("Installation: curl -fsSL https://ollama.com/install.sh | sh")
            console.print(f"Pull model: ollama pull {model_name}")
            return False, {"error": message}
    
    # Process the document
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Processing document: {file_path.name}", total=None)
            
            processor = PDFProcessor(
                extract_tables=True,
                extract_images=True,
                enable_llm_analysis=False,  # We'll do this manually
            )
            
            document = processor.process(file_path)
            
            progress.update(task, completed=True)
    except Exception as e:
        console.print(f"[bold red]Error processing document:[/] {e}")
        import traceback
        console.print(traceback.format_exc())
        return False, {"error": str(e)}
    
    # Create results dictionary
    results = {
        "document_id": document.document_id,
        "filename": document.filename,
        "elements_count": len(document.elements),
        "element_types": {},
        "analysis": {},
    }
    
    # Count element types
    for element in document.elements:
        element_type = element.element_type
        results["element_types"][element_type] = results["element_types"].get(element_type, 0) + 1
    
    # Set up content analyzer with Ollama
    client = OllamaClient(model=model_name)
    client.timeout = timeout
    analyzer = ContentUnderstanding(llm_client=client)
    
    # Configure analysis options based on the requested types
    run_document_analysis = "document" in analysis_types
    run_element_analysis = "elements" in analysis_types
    run_entity_analysis = "entities" in analysis_types
    
    # Apply analysis
    if run_document_analysis:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing document content with LLM...", total=None)
            
            try:
                # Start timer for performance tracking
                start_time = time.time()
                
                # Run analysis
                analysis = analyzer.analyze_document(document)
                
                # Record time
                analysis_time = time.time() - start_time
                
                # Store results
                results["analysis"]["document"] = analysis
                results["analysis"]["performance"] = {
                    "document_analysis_time": analysis_time
                }
                
                progress.update(task, completed=True)
            except Exception as e:
                progress.update(task, completed=True, description=f"✗ Error analyzing document: {str(e)}")
                console.print(f"[bold red]Error analyzing document:[/] {e}")
                # Continue with other analyses if possible
    
    # Analyze specific elements if requested
    if run_element_analysis:
        results["analysis"]["elements"] = {}
        
        # Select elements to analyze
        element_samples = {}
        for element in document.elements:
            element_type = element.element_type
            if element_type not in element_samples:
                element_samples[element_type] = element
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing individual elements...", total=len(element_samples))
            
            for element_type, element in element_samples.items():
                progress.update(task, description=f"Analyzing {element_type} element...")
                
                try:
                    # Analyze element
                    element_analysis = analyzer.analyze_element(element)
                    
                    # Store results
                    results["analysis"]["elements"][element_type] = {
                        "element_id": element.element_id,
                        "analysis": element_analysis
                    }
                except Exception as e:
                    console.print(f"[bold yellow]Warning:[/] Error analyzing {element_type} element: {e}")
                    results["analysis"]["elements"][element_type] = {
                        "element_id": element.element_id,
                        "error": str(e)
                    }
                
                progress.advance(task)
    
    # If entity extraction is enabled but we haven't done document analysis yet, do it now
    if run_entity_analysis and "document" not in results["analysis"]:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Extracting entities...", total=None)
            
            try:
                # Start timer for performance tracking
                start_time = time.time()
                
                # Run analysis just to get entities
                analysis = analyzer.analyze_document(document)
                
                # Record time
                analysis_time = time.time() - start_time
                
                # Store just the entities
                results["analysis"]["entities"] = analysis.get("entities", {})
                
                progress.update(task, completed=True)
            except Exception as e:
                progress.update(task, completed=True, description=f"✗ Error extracting entities: {str(e)}")
                console.print(f"[bold red]Error extracting entities:[/] {e}")
    
    # Output results
    console.print("\n[bold green]Analysis completed successfully![/]")
    
    # Save to file if requested
    if output_format:
        output_path = file_path.with_suffix(f".{output_format}")
        try:
            with open(output_path, "w") as f:
                if output_format == "json":
                    json.dump(results, f, indent=2)
                elif output_format == "md":
                    # Create markdown output
                    md_content = f"# Analysis of {file_path.name}\n\n"
                    md_content += f"- Document ID: {results['document_id']}\n"
                    md_content += f"- Elements: {results['elements_count']}\n\n"
                    
                    md_content += "## Element Types\n\n"
                    for element_type, count in results["element_types"].items():
                        md_content += f"- {element_type}: {count}\n"
                    
                    if "document" in results["analysis"]:
                        md_content += "\n## Document Analysis\n\n"
                        doc_analysis = results["analysis"]["document"]
                        
                        md_content += "### Summary\n\n"
                        md_content += f"{doc_analysis.get('summary', 'No summary available')}\n\n"
                        
                        if "topics" in doc_analysis:
                            md_content += "### Key Topics\n\n"
                            for topic in doc_analysis.get("topics", []):
                                md_content += f"- {topic}\n"
                    
                    if "entities" in results["analysis"]:
                        md_content += "\n## Entities\n\n"
                        entities = results["analysis"].get("entities", {})
                        for entity_type, entity_list in entities.items():
                            md_content += f"### {entity_type}\n\n"
                            for entity in entity_list:
                                md_content += f"- {entity}\n"
                    
                    if "elements" in results["analysis"]:
                        md_content += "\n## Element Analysis\n\n"
                        for element_type, element_info in results["analysis"]["elements"].items():
                            md_content += f"### {element_type.capitalize()}\n\n"
                            analysis = element_info.get("analysis", {})
                            for key, value in analysis.items():
                                if key != "type":
                                    md_content += f"**{key}**: {value}\n\n"
                    
                    f.write(md_content)
            
            console.print(f"Results saved to: [bold blue]{output_path}[/]")
        except Exception as e:
            console.print(f"[bold red]Error saving results:[/] {e}")
    
    return True, results


def analyze_with_ollama(
    file_path: Path, 
    model_name: str = "llama3.2:latest",
    analysis_types: List[str] = ["document", "elements", "entities"],
    temperature: float = 0.3,
    output_format: Optional[str] = None,
    timeout: int = 60
) -> Dict[str, Any]:
    """Process a document and analyze it with a local Ollama model (synchronous version).
    
    Args:
        file_path: Path to the document file
        model_name: Name of the Ollama model to use
        analysis_types: Types of analysis to perform
        temperature: Temperature for the LLM
        output_format: Output format (json or md)
        timeout: Timeout for LLM requests in seconds
        
    Returns:
        Results dictionary
    """
    # For synchronous version, we'll use asyncio.run
    import asyncio
    
    # Check if running in an event loop already
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            console.print("[yellow]Warning: Running in existing event loop, some async features may not work properly[/]")
            # Use the synchronous OllamaClient methods
            # This is a simplified implementation that doesn't mirror the async version exactly
            
            # Test Ollama connection
            client = OllamaClient(model=model_name)
            client.timeout = timeout
            
            try:
                test_response = client.generate_sync("Hello, are you working?", max_tokens=20)
                console.print(f"[bold green]Ollama connection successful[/] - Model: {model_name}")
            except Exception as e:
                console.print(f"[bold red]Error connecting to Ollama:[/] {e}")
                console.print("Make sure Ollama is installed and running, and the model is available.")
                return {"error": str(e)}
            
            # Process the document
            try:
                processor = PDFProcessor(
                    extract_tables=True,
                    extract_images=True,
                    enable_llm_analysis=False,
                )
                document = processor.process(file_path)
            except Exception as e:
                console.print(f"[bold red]Error processing document:[/] {e}")
                return {"error": str(e)}
            
            # Set up content analyzer
            analyzer = ContentUnderstanding(llm_client=client)
            
            # Analyze document
            results = {
                "document_id": document.document_id,
                "filename": document.filename,
                "elements_count": len(document.elements),
                "analysis": {}
            }
            
            if "document" in analysis_types:
                analysis = analyzer.analyze_document(document)
                results["analysis"]["document"] = analysis
            
            return results
        else:
            # We can use asyncio.run
            success, results = asyncio.run(analyze_with_ollama_async(
                file_path,
                model_name,
                analysis_types,
                temperature,
                output_format,
                timeout
            ))
            return results
    except RuntimeError:
        # No event loop, we can use asyncio.run
        success, results = asyncio.run(analyze_with_ollama_async(
            file_path,
            model_name,
            analysis_types,
            temperature,
            output_format,
            timeout
        ))
        return results


def display_analysis_results(results: Dict[str, Any]) -> None:
    """Display analysis results in a structured format.
    
    Args:
        results: The analysis results
    """
    if "error" in results:
        console.print(f"[bold red]Analysis failed:[/] {results['error']}")
        return
    
    # Display document summary
    if "analysis" in results and "document" in results["analysis"]:
        document_analysis = results["analysis"]["document"]
        
        # Summary
        if "summary" in document_analysis:
            console.print(Panel(
                document_analysis["summary"],
                title="Document Summary",
                border_style="green"
            ))
        
        # Topics
        if "topics" in document_analysis:
            console.print("\n[bold]Key Topics:[/]")
            for topic in document_analysis["topics"]:
                console.print(f"- {topic}")
        
        # Entities
        if "entities" in document_analysis:
            console.print("\n[bold]Entities:[/]")
            
            entity_table = Table("Entity Type", "Entities")
            for entity_type, entities in document_analysis["entities"].items():
                entity_table.add_row(
                    entity_type.capitalize(),
                    ", ".join(entities[:10]) + ("..." if len(entities) > 10 else "")
                )
            
            console.print(entity_table)
    
    # Display element analysis
    if "analysis" in results and "elements" in results["analysis"]:
        console.print("\n[bold]Element Analysis:[/]")
        
        for element_type, element_info in results["analysis"]["elements"].items():
            console.print(f"\n[bold cyan]{element_type.upper()}[/] Analysis:")
            
            if "error" in element_info:
                console.print(f"[yellow]Error analyzing {element_type}:[/] {element_info['error']}")
                continue
            
            analysis = element_info.get("analysis", {})
            
            if element_type == "text":
                if "summary" in analysis:
                    console.print(f"Summary: {analysis['summary']}")
                if "sentiment" in analysis:
                    console.print(f"Sentiment: {analysis['sentiment']}")
                if "purpose" in analysis:
                    console.print(f"Purpose: {analysis['purpose']}")
            elif element_type == "table":
                if "description" in analysis:
                    console.print(f"Description: {analysis['description']}")
                if "insights" in analysis:
                    console.print(f"Insights: {analysis['insights']}")
                if "trends" in analysis:
                    console.print(f"Trends: {analysis['trends']}")
            elif element_type in ["image", "chart"]:
                if "interpretation" in analysis:
                    console.print(f"Interpretation: {analysis['interpretation']}")
                if "relevance" in analysis:
                    console.print(f"Relevance: {analysis['relevance']}")
            else:
                for key, value in analysis.items():
                    if key != "type":
                        console.print(f"{key.capitalize()}: {value}")


@app.command()
def analyze(
    file_path: Path = typer.Argument(..., help="Path to the document file"),
    model: str = typer.Option("llama3.2:latest", "--model", "-m", help="Ollama model to use"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output format (json or md)"),
    timeout: int = typer.Option(60, "--timeout", "-t", help="Timeout for LLM requests in seconds"),
    analysis_type: List[str] = typer.Option(
        ["document", "elements", "entities"],
        "--analysis", "-a",
        help="Types of analysis to perform"
    ),
    temperature: float = typer.Option(0.3, "--temperature", help="Temperature for the LLM"),
):
    """Analyze a document with Ollama LLM."""
    # Validate file exists
    if not file_path.exists():
        console.print(f"[bold red]Error:[/] File {file_path} does not exist")
        raise typer.Exit(code=1)
    
    # Validate output format
    if output and output not in ["json", "md"]:
        console.print(f"[bold red]Error:[/] Output format {output} not supported")
        console.print("Supported formats: json, md")
        raise typer.Exit(code=1)
    
    # Validate temperature
    if temperature < 0 or temperature > 1:
        console.print(f"[bold red]Error:[/] Temperature must be between 0 and 1")
        raise typer.Exit(code=1)
    
    # Validate analysis types
    valid_types = ["document", "elements", "entities"]
    for analysis_type in analysis_type:
        if analysis_type not in valid_types:
            console.print(f"[bold red]Error:[/] Analysis type {analysis_type} not supported")
            console.print(f"Supported types: {', '.join(valid_types)}")
            raise typer.Exit(code=1)
    
    console.print(f"Analyzing [bold blue]{file_path.name}[/] with Ollama using model [bold]{model}[/]")
    
    try:
        # Run analysis
        results = analyze_with_ollama(
            file_path,
            model_name=model,
            analysis_types=analysis_type,
            temperature=temperature,
            output_format=output,
            timeout=timeout
        )
        
        # Display results
        display_analysis_results(results)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Analysis interrupted by user[/]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/] {e}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)


@app.command()
def batch(
    directory: Path = typer.Argument(..., help="Directory containing documents to analyze"),
    pattern: str = typer.Option("*.pdf", "--pattern", "-p", help="File pattern to match"),
    model: str = typer.Option("llama3.2:latest", "--model", "-m", help="Ollama model to use"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    format: str = typer.Option("json", "--format", "-f", help="Output format (json or md)"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit number of files to process"),
):
    """Batch analyze multiple documents with Ollama LLM."""
    # Validate directory exists
    if not directory.exists() or not directory.is_dir():
        console.print(f"[bold red]Error:[/] Directory {directory} does not exist")
        raise typer.Exit(code=1)
    
    # Find matching files
    files = list(directory.glob(pattern))
    
    if not files:
        console.print(f"[bold red]Error:[/] No files matching pattern '{pattern}' found in {directory}")
        raise typer.Exit(code=1)
    
    # Apply limit if specified
    if limit and limit > 0:
        files = files[:limit]
    
    console.print(f"Found {len(files)} files to analyze")
    
    # Create output directory if specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    results_summary = []
    
    for i, file_path in enumerate(files):
        console.print(f"\n[bold]Processing file {i+1}/{len(files)}:[/] {file_path.name}")
        
        try:
            # Determine output path
            output_path = None
            if output_dir:
                output_path = output_dir / f"{file_path.stem}.{format}"
            
            # Run analysis
            results = analyze_with_ollama(
                file_path,
                model_name=model,
                output_format=format if output_dir else None,
                analysis_types=["document"]  # Simplified analysis for batch processing
            )
            
            # Save results if output directory is specified but not already saved
            if output_dir and not output_path.exists():
                try:
                    with open(output_path, "w") as f:
                        if format == "json":
                            json.dump(results, f, indent=2)
                        elif format == "md":
                            # Create simple markdown
                            md_content = f"# Analysis of {file_path.name}\n\n"
                            if "analysis" in results and "document" in results["analysis"]:
                                doc_analysis = results["analysis"]["document"]
                                md_content += "## Summary\n\n"
                                md_content += f"{doc_analysis.get('summary', 'No summary available')}\n\n"
                            f.write(md_content)
                except Exception as e:
                    console.print(f"[bold red]Error saving results for {file_path.name}:[/] {e}")
            
            # Add to summary
            document_summary = "Error during analysis"
            if "analysis" in results and "document" in results["analysis"]:
                document_summary = results["analysis"]["document"].get("summary", "No summary")
                if len(document_summary) > 100:
                    document_summary = document_summary[:100] + "..."
            
            results_summary.append({
                "filename": file_path.name,
                "document_id": results.get("document_id", "unknown"),
                "elements": results.get("elements_count", 0),
                "summary": document_summary
            })
        except Exception as e:
            console.print(f"[bold red]Error processing {file_path.name}:[/] {e}")
            results_summary.append({
                "filename": file_path.name,
                "error": str(e)
            })
    
    # Display summary
    console.print("\n[bold]Batch Processing Summary:[/]")
    
    summary_table = Table("Filename", "Status", "Elements", "Summary")
    for result in results_summary:
        if "error" in result:
            summary_table.add_row(
                result["filename"],
                "[red]Failed[/]",
                "",
                result["error"]
            )
        else:
            summary_table.add_row(
                result["filename"],
                "[green]Success[/]",
                str(result["elements"]),
                result["summary"]
            )
    
    console.print(summary_table)
    
    # Save overall summary if output directory is specified
    if output_dir:
        try:
            summary_path = output_dir / "analysis_summary.json"
            with open(summary_path, "w") as f:
                json.dump(results_summary, f, indent=2)
            console.print(f"\nSummary saved to: [bold blue]{summary_path}[/]")
        except Exception as e:
            console.print(f"[bold red]Error saving summary:[/] {e}")


if __name__ == "__main__":
    app()