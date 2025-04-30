"""Run mmrag examples on provided documents with improved user experience."""

import os
import subprocess
import sys
import tempfile
import time # Import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(help="Run mmrag examples on document files.")
console = Console()

# Map file types to compatible examples
FILE_TYPE_MAP = {
    ".pdf": {
        "basic_rag/simple_pdf_processing.py",
        "basic_rag/vector_search.py",
        "basic_rag/rag_chatbot.py",
        "llm_integration/ollama_integration.py",
        "llm_integration/content_analysis.py",
        "multimodal_rag/multimodal_retrieval.py",
        "multimodal_rag/process_complex_document.py",
        "multimodal_rag/table_extraction.py",
        "advanced_features/cache_optimization.py",
    },
    ".pptx": {
        "multimodal_rag/process_complex_document.py",
        "multimodal_rag/multimodal_retrieval.py",
        "llm_integration/ollama_integration.py",
    },
    ".ppt": {
        "multimodal_rag/process_complex_document.py",
        "multimodal_rag/multimodal_retrieval.py",
        "llm_integration/ollama_integration.py",
    },
}


def get_compatible_examples(file_path: Path) -> Set[str]:
    """Get examples compatible with the file type."""
    file_suffix = file_path.suffix.lower()
    return FILE_TYPE_MAP.get(file_suffix, set())


def run_example(
    example_path: Path,
    file_path: Path,
    timeout: int, # Added timeout parameter
    verbose: bool = False
) -> bool:
    """Run a single example on a file."""
    example_name = os.path.basename(example_path)
    example_dir = os.path.dirname(example_path)
    example_rel_path = example_path.relative_to(example_path.parent.parent)
    
    start_run_time = time.time() # Record start time
    console.print(f"Running {example_rel_path} on {file_path.name}")
    
    # Create command with appropriate arguments
    cmd = [
        sys.executable,  # Use the same Python interpreter
        str(example_path),
    ]

    # --- Restructured Logic ---
    # Handle specific command structures first
    if "basic_rag/rag_chatbot.py" in str(example_path):
        # Needs the 'run' command and the file path passed via --documents option
        cmd.extend(["run", "--documents", str(file_path)])
    elif "basic_rag/vector_search.py" in str(example_path):
        cmd.extend(["interactive", "--document", str(file_path)])
    elif "basic_rag/simple_pdf_processing.py" in str(example_path):
        cmd.extend([str(file_path)])
    elif "llm_integration/ollama_integration.py" in str(example_path):
        cmd.extend(["analyze", str(file_path), "--model", "llama3.2:latest"])
    elif "llm_integration/huggingface_integration.py" in str(example_path):
        # Assuming 'analyze' command structure similar to ollama
        cmd.extend(["analyze", str(file_path)])
    elif "llm_integration/content_analysis.py" in str(example_path):
        cmd.extend(["analyze", str(file_path)])
    
    elif "advanced_features/cache_optimization.py" in str(example_path):
        # This script doesn't need special handling
        cmd.extend([str(file_path)])
    
    elif "advanced_features/concurrent_processing.py" in str(example_path):
        # Special handling - create temp dir with the file since it expects a directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy file to temp directory
            import shutil
            target_file = Path(temp_dir) / file_path.name
            shutil.copy2(file_path, target_file)
            
            # Build command
            temp_cmd = [
                sys.executable,
                str(example_path),
                temp_dir,
                "--pattern", file_path.name
            ]
            
            # Run the example with the temp directory
            try:
                if verbose:
                    # Pass timeout to subprocess
                    result = subprocess.run(temp_cmd, check=True, timeout=timeout)
                else:
                    result = subprocess.run(
                        temp_cmd, 
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=timeout # Pass timeout to subprocess
                    )
                console.print(f"[green]✓[/] Example completed successfully")
                return True
            except subprocess.CalledProcessError as e:
                console.print(f"[bold red]✗ Error running example:[/] {e}")
                if not verbose and e.stderr: # Check stderr for specific errors
                    if "ProcessingTimeoutError" in e.stderr or "MemoryLimitExceededError" in e.stderr:
                         console.print(f"[yellow]Processing stopped due to resource limits.[/]")
                    console.print(f"[red]Error details:[/] {e.stderr}")
                return False
            except subprocess.TimeoutExpired:
                console.print(f"[bold red]✗ Timeout:[/] Example exceeded {timeout}s limit and was terminated.")
                return False
            except Exception as e:
                console.print(f"[bold red]✗ Unexpected error:[/] {e}")
                return False
            # concurrent_processing handled above, return early
            return True # Return success status from the try block
    elif "multimodal_rag/process_complex_document.py" in str(example_path):
        cmd.extend(["process", str(file_path)])
    elif "multimodal_rag/multimodal_retrieval.py" in str(example_path):
        cmd.extend(["process", str(file_path)])
    elif "multimodal_rag/table_extraction.py" in str(example_path):
        cmd.extend(["extract", str(file_path)])
    
    else:
        # Default behavior for other scripts
        cmd.append(str(file_path))
    
    try: # This try block now handles the default case and scripts not handled above
        # Run the example
        if verbose:
            # Show output directly
            # Pass timeout to subprocess
            result = subprocess.run(cmd, check=True, timeout=timeout)
        else:
            # Capture output
            result = subprocess.run(
                cmd, 
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout # Pass timeout to subprocess
            )
        
        elapsed_run_time = time.time() - start_run_time # Calculate elapsed time
        console.print(f"[green]✓[/] Example completed successfully in {elapsed_run_time:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]✗ Error running example:[/] Command '{' '.join(cmd)}' returned non-zero exit status {e.returncode}")
        # Check stderr for specific resource errors
        if not verbose and e.stderr and ("ProcessingTimeoutError" in e.stderr or "MemoryLimitExceededError" in e.stderr): # Check stderr
             console.print(f"[yellow]Processing likely stopped due to resource limits.[/]")
        if not verbose and e.stderr:
            console.print(f"[red]Error details:[/] {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        console.print(f"[bold red]✗ Timeout:[/] Example exceeded {timeout}s limit and was terminated.")
        return False
    except Exception as e:
        console.print(f"[bold red]✗ Unexpected error:[/] {e}")
        return False

@app.command()
def run_all(
    directory: Path = typer.Argument(..., help="Directory containing documents to process"),
    example_dir: Optional[Path] = typer.Option(None, "--examples", "-e", help="Directory containing example scripts"),
    pattern: str = typer.Option("*.*", "--pattern", "-p", help="File pattern to match"),
    timeout: int = typer.Option(90, "--timeout", help="Timeout in seconds for each example run"), # Added timeout option
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Run all compatible examples on matching files."""
    # Resolve paths
    directory = directory.resolve()
    
    if example_dir is None:
        # Try to find examples directory relative to this script
        script_dir = Path(__file__).resolve().parent
        possible_paths = [
            script_dir / "examples",
            script_dir.parent / "examples",
            script_dir / "src" / "mmrag" / "examples",
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_dir():
                example_dir = path
                break
        
        if example_dir is None:
            console.print("[bold red]Error:[/] Could not find examples directory")
            console.print("Please specify with --examples option")
            raise typer.Exit(code=1)
    else:
        example_dir = example_dir.resolve()
    
    # Check if directories exist
    if not directory.exists() or not directory.is_dir():
        console.print(f"[bold red]Error:[/] Directory {directory} not found")
        raise typer.Exit(code=1)
    
    if not example_dir.exists() or not example_dir.is_dir():
        console.print(f"[bold red]Error:[/] Examples directory {example_dir} not found")
        raise typer.Exit(code=1)
    
    # Find matching files
    files = list(directory.glob(pattern))
    files = [f for f in files if f.is_file() and f.suffix.lower() in FILE_TYPE_MAP]
    
    if not files:
        console.print(f"[bold red]No compatible files found in {directory}[/]")
        console.print(f"Supported file types: {', '.join(FILE_TYPE_MAP.keys())}")
        raise typer.Exit(code=1)
    
    console.print(f"Found [bold]{len(files)}[/] compatible files in {directory}")
    
    # Track results
    results = []
    
    # Process each file
    for file_path in files:
        console.print(Panel(f"Processing: [bold blue]{file_path.name}[/]"))
        
        # Get compatible examples
        compatible_examples = get_compatible_examples(file_path)
        console.print(f"Running {len(compatible_examples)} compatible examples...")
        
        file_results = []
        
        # Run each compatible example
        for example_name in compatible_examples:
            example_path = example_dir / example_name
            
            if not example_path.exists():
                console.print(f"[yellow]⚠️ Example not found:[/] {example_name}")
                continue
            
            # Run the example
            success = run_example(example_path, file_path, timeout=timeout, verbose=verbose) # Pass timeout
            
            file_results.append({
                "example": example_name,
                "success": success
            })
            
            console.print("---")
        
        results.append({
            "file": file_path.name,
            "examples": file_results
        })
        
        console.print("")
    
    # Print summary
    console.print(Panel("[bold]Results Summary[/]"))
    
    summary_table = Table("File", "Examples Run", "Successful", "Failed")
    
    total_examples = 0
    successful_examples = 0
    
    for result in results:
        file_name = result["file"]
        examples = result["examples"]
        
        total = len(examples)
        successful = sum(1 for e in examples if e["success"])
        failed = total - successful
        
        total_examples += total
        successful_examples += successful
        
        summary_table.add_row(
            file_name,
            str(total),
            f"[green]{successful}[/]",
            f"[red]{failed}[/]" if failed > 0 else "0"
        ) # Fixed HTML entity &gt;
    
    console.print(summary_table)
    
    # Overall success rate
    success_rate = (successful_examples / total_examples) * 100 if total_examples > 0 else 0
    console.print(f"Overall success rate: [bold]{success_rate:.1f}%[/] ({successful_examples}/{total_examples})")


@app.command()
def run_single(
    example: str = typer.Argument(..., help="Example name (path relative to examples directory)"),
    file: Path = typer.Argument(..., help="Path to document file"),
    example_dir: Optional[Path] = typer.Option(None, "--examples", "-e", help="Directory containing example scripts"),
    timeout: int = typer.Option(90, "--timeout", help="Timeout in seconds for the example run"), # Added timeout option
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Run a specific example on a file."""
    # Resolve paths
    file = file.resolve()
    
    if example_dir is None:
        # Try to find examples directory relative to this script
        script_dir = Path(__file__).resolve().parent
        possible_paths = [
            script_dir / "examples",
            script_dir.parent / "examples",
            script_dir / "src" / "mmrag" / "examples",
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_dir():
                example_dir = path
                break
        
        if example_dir is None:
            console.print("[bold red]Error:[/] Could not find examples directory")
            console.print("Please specify with --examples option")
            raise typer.Exit(code=1)
    else:
        example_dir = example_dir.resolve()
    
    # Check if file exists
    if not file.exists() or not file.is_file():
        console.print(f"[bold red]Error:[/] File {file} not found")
        raise typer.Exit(code=1)
    
    # Check if example exists
    example_path = example_dir / example
    if not example_path.exists():
        console.print(f"[bold red]Error:[/] Example {example} not found")
        raise typer.Exit(code=1)
    
    # Run the example
    success = run_example(example_path, file, timeout=timeout, verbose=verbose) # Pass timeout
    
    if not success:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()