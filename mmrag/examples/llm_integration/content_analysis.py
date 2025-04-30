"""Enhanced example of document content analysis using local LLMs."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import typer
from mmrag.document_processing import PDFProcessor
from mmrag.llm import OllamaClient, ContentUnderstanding
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

app = typer.Typer(help="Document content analysis using local LLMs.")
console = Console()


def analyze_document_content(
    file_path: Path,
    model_name: str = "llama3.2:latest", 
    output_json: bool = False,
    output_html: bool = False,
    extract_tables: bool = True,
    extract_images: bool = True,
    advanced_tables: bool = False,
    analyze_elements: bool = True,
    timeout: int = 120,
) -> Optional[Dict[str, Any]]:
    """Analyze document content using a local LLM with enhanced features.
    
    Args:
        file_path: Path to the document file
        model_name: Name of the LLM model to use
        output_json: Whether to save analysis to JSON
        output_html: Whether to save analysis to HTML
        extract_tables: Whether to extract tables
        extract_images: Whether to extract images
        advanced_tables: Whether to use advanced table detection
        analyze_elements: Whether to analyze individual elements
        timeout: Timeout for LLM requests in seconds
        
    Returns:
        Dictionary containing the analysis results, or None if an error occurred
    """
    # Initialize LLM client
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing LLM client...", total=None)
            
            client = OllamaClient(model=model_name, timeout=timeout)
            test_response = client.generate_sync("Hello, are you working?", max_tokens=20)
            
            progress.update(task, completed=True)
            console.print(f"Using LLM model: [bold cyan]{model_name}[/]")
    except Exception as e:
        console.print(f"[bold red]Error initializing LLM client:[/] {e}")
        console.print("Make sure Ollama is installed and running, and the model is available.")
        console.print("Installation: curl -fsSL https://ollama.com/install.sh | sh")
        console.print(f"Pull model: ollama pull {model_name}")
        return None
    
    # Initialize document processor
    processor = PDFProcessor(
        extract_tables=extract_tables,
        extract_images=extract_images,
        advanced_table_detection=advanced_tables,
    )
    
    # Process document
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Processing document: {file_path.name}", total=None)
            document = processor.process(file_path)
            progress.update(task, completed=True)
        
        console.print(f"[bold green]Successfully processed document:[/] {file_path.name}")
        console.print(f"Document ID: {document.document_id}")
        console.print(f"Extracted {len(document.elements)} elements")
    except Exception as e:
        console.print(f"[bold red]Error processing document:[/] {e}")
        import traceback
        console.print(traceback.format_exc())
        return None
    
    # Set up content analyzer
    analyzer = ContentUnderstanding(llm_client=client)
    
    # Count elements by type for reporting
    element_types = {}
    for element in document.elements:
        element_type = element.element_type
        element_types[element_type] = element_types.get(element_type, 0) + 1
    
    # Display element counts
    element_table = Table("Element Type", "Count")
    for element_type, count in element_types.items():
        element_table.add_row(element_type, str(count))
    console.print(element_table)
    
    # Analyze document
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing document content...", total=None)
            
            # Track start time for performance metrics
            start_time = time.time()
            document_analysis = analyzer.analyze_document(document)
            analysis_time = time.time() - start_time
            
            progress.update(task, completed=True)
            console.print(f"Document analysis completed in {analysis_time:.2f} seconds")
    except Exception as e:
        console.print(f"[bold red]Error analyzing document:[/] {e}")
        console.print("Continuing with other analyses...")
        document_analysis = {
            "summary": "Error analyzing document content",
            "topics": [],
            "entities": {}
        }
    
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
            entity_branch = entities_tree.add(entity_type.capitalize())
            for entity in entities:
                entity_branch.add(entity)
        console.print(entities_tree)
    
    # Analyze specific elements by type
    element_analyses = {}
    
    if analyze_elements:
        for element_type in element_types:
            # Get a sample of elements of this type (up to 2)
            elements = [e for e in document.elements if e.element_type == element_type][:2]
            
            if elements:
                console.print(f"\n[bold]Analyzing {element_type} elements...[/]")
                type_analyses = []
                
                for element in elements:
                    try:
                        with Progress(
                            SpinnerColumn(),
                            TextColumn(f"Analyzing {element_type} (ID: {element.element_id})..."),
                            console=console
                        ) as progress:
                            task = progress.add_task("Analyzing...", total=None)
                            analysis = analyzer.analyze_element(element)
                            progress.update(task, completed=True)
                        
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
                    except Exception as e:
                        console.print(f"[bold yellow]Error analyzing {element_type} element:[/] {e}")
                        type_analyses.append({
                            "element_id": element.element_id,
                            "error": str(e)
                        })
                
                element_analyses[element_type] = type_analyses
    
    # Combine all analyses
    complete_analysis = {
        "document_id": document.document_id,
        "filename": document.filename,
        "document_analysis": document_analysis,
        "element_analyses": element_analyses,
        "metadata": {
            "model": model_name,
            "analysis_time": analysis_time if 'analysis_time' in locals() else None,
            "element_counts": element_types
        }
    }
    
    # Save to output files if requested
    if output_json or output_html:
        output_base = file_path.with_suffix("")
        
        # Save JSON
        if output_json:
            output_path = f"{output_base}.analysis.json"
            try:
                with open(output_path, "w") as f:
                    json.dump(complete_analysis, f, indent=2)
                console.print(f"\nSaved analysis to: [bold blue]{output_path}[/]")
            except Exception as e:
                console.print(f"[bold red]Error saving JSON output:[/] {e}")
        
        # Save HTML
        if output_html:
            output_path = f"{output_base}.analysis.html"
            try:
                with open(output_path, "w") as f:
                    # Generate HTML
                    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Document Analysis - {file_path.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .topic {{ background-color: #e9f7ef; margin: 5px; padding: 8px; border-radius: 4px; display: inline-block; }}
        .entity-type {{ margin-bottom: 10px; }}
        .entity {{ background-color: #ebf5fb; display: inline-block; margin: 3px; padding: 5px; border-radius: 3px; }}
        .element-analysis {{ margin-bottom: 20px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
        .metadata {{ color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Document Analysis: {file_path.name}</h1>
    <div class="metadata">
        <p>Document ID: {document.document_id}</p>
        <p>Analysis performed with model: {model_name}</p>
    </div>
    
    <h2>Document Summary</h2>
    <div class="summary">
        <p>{document_analysis["summary"]}</p>
    </div>
    
    <h2>Key Topics</h2>
    <div class="topics">
    """
                    
                    # Add topics
                    for topic in document_analysis["topics"]:
                        html += f'<span class="topic">{topic}</span>\n'
                    
                    html += """
    </div>
    
    <h2>Named Entities</h2>
    """
                    
                    # Add entities
                    if document_analysis["entities"]:
                        for entity_type, entities in document_analysis["entities"].items():
                            html += f'<div class="entity-type"><h3>{entity_type.capitalize()}</h3>'
                            for entity in entities:
                                html += f'<span class="entity">{entity}</span>\n'
                            html += '</div>\n'
                    else:
                        html += '<p>No entities detected</p>\n'
                    
                    # Add element analyses
                    if element_analyses:
                        html += '<h2>Element Analyses</h2>\n'
                        
                        for element_type, analyses in element_analyses.items():
                            html += f'<h3>{element_type.capitalize()} Elements</h3>\n'
                            
                            for analysis_item in analyses:
                                html += f'<div class="element-analysis">\n'
                                html += f'<p><strong>Element ID:</strong> {analysis_item["element_id"]}</p>\n'
                                
                                if "error" in analysis_item:
                                    html += f'<p><em>Error: {analysis_item["error"]}</em></p>\n'
                                elif "analysis" in analysis_item:
                                    analysis = analysis_item["analysis"]
                                    for key, value in analysis.items():
                                        if key != "type":
                                            html += f'<p><strong>{key.capitalize()}:</strong> {value}</p>\n'
                                
                                html += '</div>\n'
                    
                    # Close HTML
                    html += """
</body>
</html>
"""
                    
                    f.write(html)
                console.print(f"Saved HTML report to: [bold blue]{output_path}[/]")
            except Exception as e:
                console.print(f"[bold red]Error saving HTML output:[/] {e}")
    
    return complete_analysis


@app.command()
def analyze(
    file_path: Path = typer.Argument(..., help="Path to the document file"),
    model: str = typer.Option("llama3.2:latest", "--model", "-m", help="LLM model to use"),
    json: bool = typer.Option(False, "--json", "-j", help="Save analysis to JSON file"),
    html: bool = typer.Option(False, "--html", help="Save analysis to HTML report"),
    no_tables: bool = typer.Option(False, "--no-tables", help="Disable table extraction"),
    no_images: bool = typer.Option(False, "--no-images", help="Disable image extraction"),
    advanced_tables: bool = typer.Option(False, "--advanced-tables", help="Use advanced table detection"),
    no_element_analysis: bool = typer.Option(False, "--no-element-analysis", help="Skip individual element analysis"),
    timeout: int = typer.Option(120, "--timeout", "-t", help="Timeout for LLM requests in seconds"),
):
    """Analyze a document with a local LLM."""
    # Validate file exists
    if not file_path.exists():
        console.print(f"[bold red]Error:[/] File {file_path} does not exist")
        raise typer.Exit(code=1)
    
    try:
        # Run analysis
        analyze_document_content(
            file_path=file_path,
            model_name=model,
            output_json=json,
            output_html=html,
            extract_tables=not no_tables,
            extract_images=not no_images,
            advanced_tables=advanced_tables,
            analyze_elements=not no_element_analysis,
            timeout=timeout
        )
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Analysis interrupted by user[/]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error during analysis:[/] {e}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)


@app.command()
def batch(
    directory: Path = typer.Argument(..., help="Directory containing documents to analyze"),
    pattern: str = typer.Option("*.pdf", "--pattern", "-p", help="File pattern to match"),
    model: str = typer.Option("llama3.2:latest", "--model", "-m", help="LLM model to use"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    json: bool = typer.Option(True, "--json/--no-json", help="Save analysis to JSON files"),
    html: bool = typer.Option(False, "--html/--no-html", help="Save analysis to HTML reports"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit number of files to process"),
    summarize: bool = typer.Option(True, "--summarize/--no-summarize", help="Create a summary report"),
):
    """Batch analyze multiple documents."""
    # Validate directory exists
    if not directory.exists() or not directory.is_dir():
        console.print(f"[bold red]Error:[/] Directory {directory} does not exist")
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
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"Results will be saved to: [bold blue]{output_dir}[/]")
    elif json or html:
        console.print("[yellow]Warning:[/] No output directory specified, files will be saved alongside source files")
    
    # Process each file
    results = []
    success_count = 0
    error_count = 0
    
    for i, file_path in enumerate(files):
        console.print(f"\n[bold]Processing file {i+1}/{len(files)}:[/] {file_path.name}")
        
        try:
            # Determine output paths
            if output_dir:
                output_base = output_dir / file_path.stem
                json_path = output_base.with_suffix(".analysis.json") if json else None
                html_path = output_base.with_suffix(".analysis.html") if html else None
            else:
                json_path = None
                html_path = None
                
            # Run analysis with simplified options for batch processing
            result = analyze_document_content(
                file_path=file_path,
                model_name=model,
                output_json=json and json_path is None,  # Only save if not using custom path
                output_html=html and html_path is None,  # Only save if not using custom path
                extract_tables=True,
                extract_images=True,
                advanced_tables=False,
                analyze_elements=False,  # Skip element analysis for batch processing
                timeout=60
            )
            
            if result:
                # Save results if output directory is specified
                if output_dir:
                    if json:
                        try:
                            with open(json_path, "w") as f:
                                json.dump(result, f, indent=2)
                        except Exception as e:
                            console.print(f"[bold red]Error saving JSON for {file_path.name}:[/] {e}")
                    
                    if html:
                        try:
                            with open(html_path, "w") as f:
                                # Generate simple HTML report
                                html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Analysis - {file_path.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background-color: #f0f0f0; padding: 10px; }}
    </style>
</head>
<body>
    <h1>Analysis of {file_path.name}</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p>{result["document_analysis"]["summary"]}</p>
    </div>
    <div class="topics">
        <h2>Topics</h2>
        <ul>
"""
                                for topic in result["document_analysis"]["topics"]:
                                    html_content += f"            <li>{topic}</li>\n"
                                
                                html_content += """
        </ul>
    </div>
</body>
</html>
"""
                                f.write(html_content)
                        except Exception as e:
                            console.print(f"[bold red]Error saving HTML for {file_path.name}:[/] {e}")
                
                # Add to results
                results.append({
                    "filename": file_path.name,
                    "document_id": result["document_id"],
                    "summary": result["document_analysis"]["summary"],
                    "topics": result["document_analysis"]["topics"],
                    "success": True
                })
                
                success_count += 1
            else:
                # Add error to results
                results.append({
                    "filename": file_path.name,
                    "success": False,
                    "error": "Analysis failed"
                })
                
                error_count += 1
        except Exception as e:
            console.print(f"[bold red]Error processing {file_path.name}:[/] {e}")
            results.append({
                "filename": file_path.name,
                "success": False,
                "error": str(e)
            })
            
            error_count += 1
    
    # Display summary
    console.print(f"\n[bold]Batch Processing Complete:[/] {success_count} successful, {error_count} failed")
    
    # Create summary report if requested
    if summarize and results and output_dir:
        try:
            # Create summary JSON
            summary_path = output_dir / "analysis_summary.json"
            with open(summary_path, "w") as f:
                json.dump(results, f, indent=2)
            
            # Create summary HTML
            html_summary_path = output_dir / "analysis_summary.html"
            with open(html_summary_path, "w") as f:
                html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Batch Analysis Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .success {{ color: green; }}
        .error {{ color: red; }}
    </style>
</head>
<body>
    <h1>Batch Analysis Summary</h1>
    <p>Processed {len(results)} files with {success_count} successful and {error_count} failed.</p>
    
    <table>
        <tr>
            <th>Filename</th>
            <th>Status</th>
            <th>Summary</th>
        </tr>
"""
                
                for result in results:
                    status = "success" if result.get("success", False) else "error"
                    status_text = "Success" if result.get("success", False) else f"Error: {result.get('error', 'Unknown')}"
                    summary = result.get("summary", "N/A") if result.get("success", False) else ""
                    
                    if len(summary) > 200:
                        summary = summary[:200] + "..."
                    
                    html_content += f"""
        <tr>
            <td>{result["filename"]}</td>
            <td class="{status}">{status_text}</td>
            <td>{summary}</td>
        </tr>"""
                
                html_content += """
    </table>
</body>
</html>
"""
                f.write(html_content)
            
            console.print(f"Summary reports saved to: [bold blue]{output_dir}[/]")
        except Exception as e:
            console.print(f"[bold red]Error creating summary reports:[/] {e}")


if __name__ == "__main__":
    app()