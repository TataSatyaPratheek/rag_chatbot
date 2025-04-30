"""Enhanced example of using Hugging Face transformers for document analysis."""

import json
import os # Added for psutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set

import typer
from mmrag.document_processing.legacy import PDFProcessor
from rich.console import Console
from mmrag.exceptions import ProcessingTimeoutError, MemoryLimitExceededError # Import exceptions
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TaskProgressColumn
from rich.table import Table
from rich.markdown import Markdown
import psutil # Added for memory monitoring
 
app = typer.Typer(help="Document analysis using Hugging Face transformers.")
console = Console()


def check_transformers_installed() -> bool:
    """Check if the transformers package is installed.
    
    Returns:
        True if installed, False otherwise
    """
    try:
        import transformers
        return True
    except ImportError:
        return False


def analyze_with_huggingface(
    file_path: Path,
    summarize: bool = True,
    qa: bool = True,
    zero_shot: bool = True,
    output_format: Optional[str] = None,
    output_dir: Optional[Path] = None,
    interactive_qa: bool = False,
    cache_models: bool = True,
    device: Optional[str] = None,
    show_timing: bool = False,
    timeout_seconds: int = 120, # Added timeout
    memory_limit_fraction: float = 0.5, # Added memory limit fraction
) -> Dict[str, Any]:
    """Analyze a document using Hugging Face transformers.
    
    Args:
        file_path: Path to the document file
        summarize: Whether to perform document summarization
        qa: Whether to perform question answering
        zero_shot: Whether to perform zero-shot classification
        output_format: Output format (json, html, md, or None)
        output_dir: Directory to save output files
        interactive_qa: Whether to enter interactive Q&A mode
        cache_models: Whether to cache models in the Hugging Face cache
        device: Device to use for inference (cpu, cuda, mps, or None for auto)
        show_timing: Whether to show timing information
        timeout_seconds: Maximum processing time in seconds.
        memory_limit_fraction: Maximum fraction of available memory to use.
        
    Returns:
        Dictionary with analysis results
    """
    # Check if transformers is installed
    if not check_transformers_installed():
        console.print("[bold red]Error:[/] transformers package not installed.")
        console.print("Install it with: pip install transformers torch")
        raise typer.Exit(code=1)
    
    # Now it's safe to import transformers
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
    except ImportError as e:
        console.print(f"[bold red]Error:[/] {e}")
        console.print("Make sure you have PyTorch installed as well: pip install torch")
        raise typer.Exit(code=1)
    
    # Resource monitoring setup
    start_analysis_time = time.time()
    process = psutil.Process(os.getpid())
    initial_available_memory = psutil.virtual_memory().available
    memory_limit_bytes = initial_available_memory * memory_limit_fraction
    console.print(f"Resource limits: Timeout={timeout_seconds}s, Memory Limit={memory_limit_bytes / (1024**2):.2f} MB")

    # Determine device if not specified
    if not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    console.print(f"Using device: [bold cyan]{device}[/]")
    
    # Initialize document processor
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Processing document: {file_path.name}", total=None)
            
            # --- Resource Checks ---
            elapsed_time = time.time() - start_analysis_time
            if elapsed_time > timeout_seconds:
                raise ProcessingTimeoutError(f"Processing exceeded {timeout_seconds} seconds limit.")
                
            current_rss = process.memory_info().rss
            if current_rss > memory_limit_bytes:
                raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
            # --- End Resource Checks ---

            processor = PDFProcessor(
                extract_tables=True,
                extract_images=True
            )
            
            # Process document
            document = processor.process(file_path)
            
            progress.update(task, completed=True)
    except Exception as e:
        console.print(f"[bold red]Error processing document:[/] {e}")
        import traceback
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)
    
    console.print(f"[bold green]Successfully processed document:[/] {file_path.name}")
    console.print(f"Document ID: {document.document_id}")
    console.print(f"Extracted {len(document.elements)} elements")
    
    # Create results dictionary
    results = {
        "document_id": document.document_id,
        "filename": file_path.name,
        "element_count": len(document.elements),
        "analyses": {}
    }
    
    # Initialize pipelines based on requirements
    models = {}
    metrics = {}
    
    # Configure model loading options
    pipeline_kwargs = {"device": device}
    
    # Add cache_dir option if caching is enabled
    if cache_models:
        from huggingface_hub import snapshot_download
        from transformers.utils import DEFAULT_CACHE_DIR
        pipeline_kwargs["cache_dir"] = DEFAULT_CACHE_DIR
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        if summarize:
            task = progress.add_task("Loading summarization model...", total=None)
            
            start_time = time.time()
            try:
                models["summarizer"] = pipeline(
                    "summarization", 
                    model="facebook/bart-large-cnn",
                    **pipeline_kwargs
                )
                load_time = time.time() - start_time
                metrics["summarizer_load_time"] = load_time
                
                progress.update(task, completed=True, description=f"Loaded summarization model in {load_time:.2f}s")
            except Exception as e:
                progress.update(task, completed=True, description=f"Error loading summarization model: {str(e)}")
                console.print(f"[bold yellow]Warning:[/] Could not load summarization model: {e}")
        
        if qa:
            task = progress.add_task("Loading question-answering model...", total=None)
            
            start_time = time.time()
            try:
                models["qa"] = pipeline(
                    "question-answering", 
                    model="deepset/roberta-base-squad2",
                    **pipeline_kwargs
                )
                load_time = time.time() - start_time
                metrics["qa_load_time"] = load_time
                
                progress.update(task, completed=True, description=f"Loaded question-answering model in {load_time:.2f}s")
            except Exception as e:
                progress.update(task, completed=True, description=f"Error loading question-answering model: {str(e)}")
                console.print(f"[bold yellow]Warning:[/] Could not load question-answering model: {e}")
        
        if zero_shot:
            task = progress.add_task("Loading zero-shot classification model...", total=None)
            
            start_time = time.time()
            try:
                models["zero_shot"] = pipeline(
                    "zero-shot-classification", 
                    model="facebook/bart-large-mnli",
                    **pipeline_kwargs
                )
                load_time = time.time() - start_time
                metrics["zero_shot_load_time"] = load_time
                
                progress.update(task, completed=True, description=f"Loaded zero-shot model in {load_time:.2f}s")
            except Exception as e:
                progress.update(task, completed=True, description=f"Error loading zero-shot model: {str(e)}")
                console.print(f"[bold yellow]Warning:[/] Could not load zero-shot model: {e}")
    
    # Get text elements
    text_elements = [e for e in document.elements if e.element_type == "text"]
    
    # Show warning if no text elements found
    if not text_elements:
        console.print("[bold yellow]Warning:[/] No text elements found in the document.")
        console.print("This may affect analysis results.")
    
    # Process with summarization
    if summarize and "summarizer" in models:
        console.print("\n[bold]Document Summarization:[/]")
        
        # Combine text elements for overall summary
        combined_text = " ".join([e.content for e in text_elements])
        
        # Truncate if too long (most models have limits)
        max_length = 1024
        if len(combined_text) > max_length:
            console.print(f"[yellow]Note:[/] Document text truncated from {len(combined_text)} to {max_length} tokens for summarization")
            combined_text = combined_text[:max_length]
        
        # Generate summary
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("Generating summary..."),
                console=console
            ) as progress:
                task = progress.add_task("Generating...", total=None)

                # --- Resource Checks ---
                elapsed_time = time.time() - start_analysis_time
                if elapsed_time > timeout_seconds:
                    raise ProcessingTimeoutError(f"Summarization exceeded {timeout_seconds} seconds limit.")
                    
                current_rss = process.memory_info().rss
                if current_rss > memory_limit_bytes:
                    raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) before summarization exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
                # --- End Resource Checks ---
                
                start_time = time.time()
                summary = models["summarizer"](
                    combined_text, 
                    max_length=150, 
                    min_length=50, 
                    do_sample=False
                )
                inference_time = time.time() - start_time
                metrics["summarization_time"] = inference_time
                
                progress.update(task, completed=True, description=f"Summary generated in {inference_time:.2f}s")
            
            # Display summary
            summary_text = summary[0]['summary_text']
            console.print(Panel(summary_text, title="Document Summary", border_style="green"))
            
            # Store in results
            results["analyses"]["summary"] = summary_text
        except Exception as e:
            console.print(f"[bold red]Error generating summary:[/] {e}")
            results["analyses"]["summary_error"] = str(e)
    
    # Process with zero-shot classification
    if zero_shot and "zero_shot" in models:
        console.print("\n[bold]Document Topic Classification:[/]")
        
        # Define candidate labels
        candidate_labels = [
            "business", "technology", "science", "politics", "health", 
            "education", "finance", "entertainment", "sports", "travel"
        ]
        
        # Combine text elements
        combined_text = " ".join([e.content[:200] for e in text_elements[:5]])
        
        # Classify document
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("Classifying document..."),
                console=console
            ) as progress:
                task = progress.add_task("Classifying...", total=None)

                # --- Resource Checks ---
                elapsed_time = time.time() - start_analysis_time
                if elapsed_time > timeout_seconds:
                    raise ProcessingTimeoutError(f"Classification exceeded {timeout_seconds} seconds limit.")
                    
                current_rss = process.memory_info().rss
                if current_rss > memory_limit_bytes:
                    raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) before classification exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
                # --- End Resource Checks ---
                
                start_time = time.time()
                classification = models["zero_shot"](
                    combined_text, 
                    candidate_labels, 
                    multi_label=True
                )
                inference_time = time.time() - start_time
                metrics["classification_time"] = inference_time
                
                progress.update(task, completed=True, description=f"Classification completed in {inference_time:.2f}s")
            
            # Display classification results
            console.print("\n[bold cyan]Document Topics:[/]")
            
            topic_table = Table("Topic", "Confidence")
            topic_results = []
            
            for label, score in zip(classification["labels"], classification["scores"]):
                topic_table.add_row(label, f"{score:.2f}")
                topic_results.append({"topic": label, "confidence": score})
            
            console.print(topic_table)
            
            # Store in results
            results["analyses"]["topics"] = topic_results
        except Exception as e:
            console.print(f"[bold red]Error classifying document:[/] {e}")
            results["analyses"]["topics_error"] = str(e)
    
    # Process with question-answering
    if qa and "qa" in models:
        console.print("\n[bold]Question Answering:[/]")
        
        # Define some general questions about the document
        default_questions = [
            "What is the main topic?",
            "Who is involved?",
            "When did this happen?",
            "What is the main conclusion?"
        ]
        
        # Combine text elements
        context = " ".join([e.content for e in text_elements])
        
        # Truncate context if too long
        max_length = 512
        if len(context) > max_length:
            console.print(f"[yellow]Note:[/] Document text truncated from {len(context)} to {max_length} tokens for question answering")
            context = context[:max_length]
        
        # Answer questions
        qa_results = []
        
        console.print("\n[bold cyan]Automatic Q&A:[/]")
        for question in default_questions:
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn(f"Answering: {question}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Answering...", total=None)

                    # --- Resource Checks ---
                    elapsed_time = time.time() - start_analysis_time
                    if elapsed_time > timeout_seconds:
                        raise ProcessingTimeoutError(f"Q&A exceeded {timeout_seconds} seconds limit.")
                        
                    current_rss = process.memory_info().rss
                    if current_rss > memory_limit_bytes:
                        raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) before Q&A exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
                    # --- End Resource Checks ---
                    
                    start_time = time.time()
                    answer = models["qa"](
                        question=question, 
                        context=context
                    )
                    inference_time = time.time() - start_time
                    
                    progress.update(task, completed=True)
                
                console.print(f"Q: {question}")
                console.print(f"A: {answer['answer']} (confidence: {answer['score']:.2f})")
                console.print("")
                
                # Store in results
                qa_results.append({
                    "question": question,
                    "answer": answer['answer'],
                    "confidence": answer['score'],
                    "inference_time": inference_time
                })
            except Exception as e:
                console.print(f"[bold yellow]Error answering question '{question}':[/] {e}")
                qa_results.append({
                    "question": question,
                    "error": str(e)
                })
        
        # Store in results
        results["analyses"]["question_answering"] = qa_results
    
    # Display performance metrics if requested
    if show_timing and metrics:
        console.print("\n[bold]Performance Metrics:[/]")
        
        metrics_table = Table("Operation", "Time (seconds)")
        for operation, time_value in metrics.items():
            # Format the operation name
            display_name = operation.replace("_", " ").title()
            metrics_table.add_row(display_name, f"{time_value:.2f}")
        
        console.print(metrics_table)
        
        # Store in results
        results["performance"] = metrics
    
    # Save output if requested
    if output_format:
        # Determine output path
        if output_dir:
            output_dir.mkdir(exist_ok=True, parents=True)
            output_path = output_dir / f"{file_path.stem}_analysis.{output_format}"
        else:
            output_path = file_path.with_suffix(f".{output_format}")
        
        try:
            if output_format == "json":
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)
            elif output_format == "html":
                with open(output_path, "w") as f:
                    # Generate HTML
                    html = generate_html_report(results, file_path.name)
                    f.write(html)
            elif output_format == "md":
                with open(output_path, "w") as f:
                    # Generate Markdown
                    md = generate_markdown_report(results, file_path.name)
                    f.write(md)
            
            console.print(f"\nAnalysis saved to: [bold blue]{output_path}[/]")
        except Exception as e:
            console.print(f"[bold red]Error saving output:[/] {e}")
    
    # Interactive Q&A mode
    if interactive_qa and qa and "qa" in models: # Pass limits to interactive mode
        # --- Fix: Pass timeout and memory limit to interactive mode ---
        run_interactive_qa(
            models["qa"], context, file_path.name, 
            timeout_seconds=timeout_seconds, # Pass timeout
            memory_limit_fraction=memory_limit_fraction # Pass mem limit
        )
    return results


def run_interactive_qa(
    qa_model: Any, 
    context: str, 
    filename: str,
    timeout_seconds: int = 30, # Added timeout
    memory_limit_fraction: float = 0.5, # Added memory limit fraction
    # --- End Fix ---
) -> None:
    """
    Args:
        qa_model: The question answering pipeline
        context: Document text to use as context
        filename: Name of the document file
    """
    console.print(Panel.fit(
        "Interactive Question Answering\n"
        "Ask questions about the document (type 'quit' to exit)",
        title=f"Q&A: {filename}",
        border_style="blue"
    ))
    
    # Resource monitoring setup for interactive mode
    process = psutil.Process(os.getpid())
    initial_available_memory = psutil.virtual_memory().available
    memory_limit_bytes = initial_available_memory * memory_limit_fraction
    start_interactive_time = time.time() # Track start time for overall timeout

    while True:
        try:
            question = console.input("\n[bold cyan]Question:[/] ")
            if question.lower() in ["quit", "exit", "q"]:
                break
            
            with Progress(
                SpinnerColumn(),
                TextColumn("Answering..."),
                console=console
            ) as progress:
                task = progress.add_task("Generating answer...", total=None)

                # --- Resource Checks ---
                elapsed_time = time.time() - start_interactive_time
                if elapsed_time > timeout_seconds:
                    raise ProcessingTimeoutError(f"Interactive Q&A exceeded {timeout_seconds} seconds limit.")
                    
                current_rss = process.memory_info().rss
                if current_rss > memory_limit_bytes:
                    raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) during Q&A exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
                # --- End Resource Checks ---
                
                start_time = time.time()
                answer = qa_model(question=question, context=context)
                inference_time = time.time() - start_time
                
                progress.update(task, completed=True, description=f"Answered in {inference_time:.2f}s")
            
            console.print(f"[bold green]Answer:[/] {answer['answer']}")
            console.print(f"Confidence: {answer['score']:.2f}")
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Interactive mode interrupted[/]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")


def generate_html_report(results: Dict[str, Any], filename: str) -> str:
    """Generate an HTML report from analysis results.
    
    Args:
        results: Analysis results
        filename: Name of the document file
        
    Returns:
        HTML report as a string
    """
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Document Analysis - {filename}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .topic {{ background-color: #e9f7ef; margin: 5px; padding: 8px; border-radius: 4px; display: inline-block; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metrics {{ background-color: #f0f4f8; padding: 10px; border-radius: 5px; margin-bottom: 15px; }}
        .qa-item {{ margin-bottom: 15px; background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
        .question {{ font-weight: bold; color: #2980b9; }}
        .answer {{ margin-top: 5px; }}
        .confidence {{ color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Document Analysis: {filename}</h1>
    <div class="metadata">
        <p><strong>Document ID:</strong> {results['document_id']}</p>
        <p><strong>Elements:</strong> {results['element_count']}</p>
    </div>
"""
    
    # Add summary if available
    if "analyses" in results and "summary" in results["analyses"]:
        html += f"""
    <h2>Document Summary</h2>
    <div class="summary">
        <p>{results['analyses']['summary']}</p>
    </div>
"""
    
    # Add topics if available
    if "analyses" in results and "topics" in results["analyses"]:
        html += """
    <h2>Document Topics</h2>
    <table>
        <tr>
            <th>Topic</th>
            <th>Confidence</th>
        </tr>
"""
        
        for topic in results["analyses"]["topics"]:
            html += f"""
        <tr>
            <td>{topic['topic']}</td>
            <td>{topic['confidence']:.2f}</td>
        </tr>"""
        
        html += """
    </table>
"""
    
    # Add question answering if available
    if "analyses" in results and "question_answering" in results["analyses"]:
        html += """
    <h2>Question Answering</h2>
    <div class="qa-results">
"""
        
        for qa_item in results["analyses"]["question_answering"]:
            if "error" in qa_item:
                html += f"""
        <div class="qa-item">
            <div class="question">Q: {qa_item['question']}</div>
            <div class="answer">Error: {qa_item['error']}</div>
        </div>"""
            else:
                html += f"""
        <div class="qa-item">
            <div class="question">Q: {qa_item['question']}</div>
            <div class="answer">A: {qa_item['answer']}</div>
            <div class="confidence">Confidence: {qa_item['confidence']:.2f}</div>
        </div>"""
        
        html += """
    </div>
"""
    
    # Add performance metrics if available
    if "performance" in results:
        html += """
    <h2>Performance Metrics</h2>
    <div class="metrics">
        <table>
            <tr>
                <th>Operation</th>
                <th>Time (seconds)</th>
            </tr>
"""
        
        for operation, time_value in results["performance"].items():
            # Format the operation name
            display_name = operation.replace("_", " ").title()
            html += f"""
            <tr>
                <td>{display_name}</td>
                <td>{time_value:.2f}</td>
            </tr>"""
        
        html += """
        </table>
    </div>
"""
    
    # Close HTML
    html += """
</body>
</html>
"""
    
    return html


def generate_markdown_report(results: Dict[str, Any], filename: str) -> str:
    """Generate a Markdown report from analysis results.
    
    Args:
        results: Analysis results
        filename: Name of the document file
        
    Returns:
        Markdown report as a string
    """
    md = f"# Document Analysis: {filename}\n\n"
    md += f"**Document ID:** {results['document_id']}\n"
    md += f"**Elements:** {results['element_count']}\n\n"
    
    # Add summary if available
    if "analyses" in results and "summary" in results["analyses"]:
        md += f"## Document Summary\n\n{results['analyses']['summary']}\n\n"
    
    # Add topics if available
    if "analyses" in results and "topics" in results["analyses"]:
        md += "## Document Topics\n\n"
        md += "| Topic | Confidence |\n"
        md += "| ----- | ---------- |\n"
        
        for topic in results["analyses"]["topics"]:
            md += f"| {topic['topic']} | {topic['confidence']:.2f} |\n"
        
        md += "\n"
    
    # Add question answering if available
    if "analyses" in results and "question_answering" in results["analyses"]:
        md += "## Question Answering\n\n"
        
        for qa_item in results["analyses"]["question_answering"]:
            if "error" in qa_item:
                md += f"**Q:** {qa_item['question']}\n\n"
                md += f"**Error:** {qa_item['error']}\n\n"
            else:
                md += f"**Q:** {qa_item['question']}\n\n"
                md += f"**A:** {qa_item['answer']}\n\n"
                md += f"*Confidence: {qa_item['confidence']:.2f}*\n\n"
    
    # Add performance metrics if available
    if "performance" in results:
        md += "## Performance Metrics\n\n"
        md += "| Operation | Time (seconds) |\n"
        md += "| --------- | -------------- |\n"
        
        for operation, time_value in results["performance"].items():
            # Format the operation name
            display_name = operation.replace("_", " ").title()
            md += f"| {display_name} | {time_value:.2f} |\n"
    
    return md


@app.command()
def analyze(
    file_path: Path = typer.Argument(..., help="Path to the document file"),
    no_summarize: bool = typer.Option(False, "--no-summarize", help="Skip summarization"),
    no_qa: bool = typer.Option(False, "--no-qa", help="Skip question answering"),
    no_zero_shot: bool = typer.Option(False, "--no-zero-shot", help="Skip zero-shot classification"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output format (json, html, md)"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-d", help="Directory for output files"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Enter interactive Q&A mode after analysis"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable model caching"),
    device: Optional[str] = typer.Option(None, "--device", help="Device to use (cpu, cuda, mps)"),
    timing: bool = typer.Option(False, "--timing", "-t", help="Show timing information"),
    timeout: int = typer.Option(120, "--timeout", help="Processing timeout in seconds"),
    mem_limit: float = typer.Option(0.5, "--mem-limit", help="Memory limit as fraction of available memory (0.1-1.0)"),
):
    """Analyze a document using Hugging Face transformers."""
    # Validate file exists
    if not file_path.exists():
        console.print(f"[bold red]Error:[/] File {file_path} does not exist")
        raise typer.Exit(code=1)

    # Validate memory limit
    if not (0.1 <= mem_limit <= 1.0):
        console.print(f"[bold red]Error:[/] File {file_path} does not exist")
        raise typer.Exit(code=1)
    
    # Validate output format
    if output and output not in ["json", "html", "md"]:
        console.print(f"[bold red]Error:[/] Invalid output format: {output}")
        console.print("Supported formats: json, html, md")
        raise typer.Exit(code=1)
    
    # Validate device
    if device and device not in ["cpu", "cuda", "mps"]:
        console.print(f"[bold red]Error:[/] Invalid device: {device}")
        console.print("Supported devices: cpu, cuda, mps")
        raise typer.Exit(code=1)
    
    try:
        analyze_with_huggingface(
            file_path=file_path,
            summarize=not no_summarize,
            qa=not no_qa,
            zero_shot=not no_zero_shot,
            output_format=output,
            output_dir=output_dir,
            interactive_qa=interactive,
            cache_models=not no_cache,
            device=device,
            show_timing=timing,
            timeout_seconds=timeout,
            memory_limit_fraction=mem_limit,
        )
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Analysis interrupted by user[/]")
        raise typer.Exit(code=1)


@app.command()
def batch(
    directory: Path = typer.Argument(..., help="Directory containing documents to analyze"),
    pattern: str = typer.Option("*.pdf", "--pattern", "-p", help="File pattern to match"),
    no_summarize: bool = typer.Option(False, "--no-summarize", help="Skip summarization"),
    no_qa: bool = typer.Option(False, "--no-qa", help="Skip question answering"),
    no_zero_shot: bool = typer.Option(False, "--no-zero-shot", help="Skip zero-shot classification"),
    output: str = typer.Option("json", "--output", "-o", help="Output format (json, html, md, none)"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-d", help="Directory for output files"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit number of files to process"),
    timeout: int = typer.Option(60, "--timeout", help="Processing timeout per file in seconds"), # Shorter timeout for batch
    mem_limit: float = typer.Option(0.5, "--mem-limit", help="Memory limit per file as fraction of available memory (0.1-1.0)"),
    device: Optional[str] = typer.Option(None, "--device", help="Device to use (cpu, cuda, mps)"),
):
    """Batch analyze multiple documents using Hugging Face transformers."""
    # Validate memory limit
    if not (0.1 <= mem_limit <= 1.0):
        console.print("[bold red]Error:[/] Memory limit must be between 0.1 and 1.0")
        raise typer.Exit(code=1)

    # Validate directory exists
    if not directory.exists() or not directory.is_dir():
        console.print(f"[bold red]Error:[/] Directory {directory} does not exist")
        raise typer.Exit(code=1)
    
    # Validate output format
    if output not in ["json", "html", "md", "none"]:
        console.print(f"[bold red]Error:[/] Invalid output format: {output}")
        console.print("Supported formats: json, html, md, none")
        raise typer.Exit(code=1)
    
    # Validate device
    if device and device not in ["cpu", "cuda", "mps"]:
        console.print(f"[bold red]Error:[/] Invalid device: {device}")
        console.print("Supported devices: cpu, cuda, mps")
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
    if output != "none" and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"Results will be saved to: [bold blue]{output_dir}[/]")
    
    # Process each file
    results = []
    success_count = 0
    error_count = 0
    
    for i, file_path in enumerate(files):
        console.print(f"\n[bold]Processing file {i+1}/{len(files)}:[/] {file_path.name}")
        
        try:
            # Determine output directory for this file
            file_output_dir = output_dir / file_path.stem if output_dir else None
            if file_output_dir:
                file_output_dir.mkdir(exist_ok=True, parents=True)
            
            # Only use output format if not none
            output_format = None if output == "none" else output
            
            # Analyze document
            result = analyze_with_huggingface(
                file_path=file_path,
                summarize=not no_summarize,
                qa=not no_qa,
                zero_shot=not no_zero_shot,
                output_format=output_format,
                output_dir=file_output_dir,
                interactive_qa=False,
                device=device,
                timeout_seconds=timeout, # Pass timeout
                memory_limit_fraction=mem_limit, # Pass mem limit
            )
            
            # Add to results
            results.append({
                "filename": file_path.name,
                "document_id": result["document_id"],
                "element_count": result["element_count"],
                "success": True
            })
            
            success_count += 1
        except Exception as e:
            console.print(f"[bold red]Error processing {file_path.name}:[/] {e}")
            import traceback
            console.print(traceback.format_exc())
            
            results.append({
                "filename": file_path.name,
                "success": False,
                "error": str(e)
            })
            
            error_count += 1
    
    # Display summary
    console.print(f"\n[bold]Batch Processing Complete:[/] {success_count} successful, {error_count} failed")
    
    # Create summary report if requested
    if output_dir:
        try:
            # Create summary JSON
            summary_path = output_dir / "analysis_summary.json"
            with open(summary_path, "w") as f:
                json.dump({
                    "files_processed": len(files),
                    "files_successful": success_count,
                    "files_failed": error_count,
                    "file_results": results
                }, f, indent=2)
            
            console.print(f"Summary saved to: [bold blue]{summary_path}[/]")
        except Exception as e:
            console.print(f"[bold red]Error creating summary:[/] {e}")


@app.command()
def supported_models(
    download: bool = typer.Option(False, "--download", "-d", help="Download models to cache"),
):
    """Show information about supported models."""
    # Check if transformers is installed
    if not check_transformers_installed():
        console.print("[bold red]Error:[/] transformers package not installed.")
        console.print("Install it with: pip install transformers")
        raise typer.Exit(code=1)
    
    # Now it's safe to import
    from transformers import AutoConfig
    
    # Define supported models
    supported_models = [
        {
            "task": "summarization",
            "model": "facebook/bart-large-cnn",
            "description": "BART model fine-tuned on CNN Daily Mail dataset for summarization",
            "size": "1.6 GB"
        },
        {
            "task": "question-answering",
            "model": "deepset/roberta-base-squad2",
            "description": "RoBERTa model fine-tuned on SQuAD v2 for question answering",
            "size": "500 MB"
        },
        {
            "task": "zero-shot-classification",
            "model": "facebook/bart-large-mnli",
            "description": "BART model fine-tuned on MultiNLI for zero-shot text classification",
            "size": "1.6 GB"
        }
    ]
    
    # Display models table
    console.print(Panel("[bold]Supported Models for Document Analysis[/]"))
    
    models_table = Table("Task", "Model", "Description", "Size")
    for model in supported_models:
        models_table.add_row(
            model["task"],
            model["model"],
            model["description"],
            model["size"]
        )
    
    console.print(models_table)
    
    # Download models if requested
    if download:
        if not check_transformers_installed():
            console.print("[bold red]Error:[/] transformers package not installed.")
            console.print("Install it with: pip install transformers")
            raise typer.Exit(code=1)
        
        from huggingface_hub import snapshot_download
        
        console.print("\n[bold]Downloading models to cache...[/]")
        console.print("[yellow]Note:[/] This may take some time depending on your internet connection")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            for model in supported_models:
                task = progress.add_task(f"Downloading {model['model']}...", total=None)
                
                try:
                    # Download model files
                    snapshot_download(model["model"])
                    progress.update(task, completed=True, description=f"✓ Downloaded {model['model']}")
                except Exception as e:
                    progress.update(task, completed=True, description=f"✗ Error downloading {model['model']}: {str(e)}")
                    console.print(f"[bold red]Error downloading {model['model']}:[/] {e}")
        
        console.print("\n[bold green]Downloads complete![/]")
        console.print("Models are now cached and will load faster in subsequent runs")


if __name__ == "__main__":
    app()