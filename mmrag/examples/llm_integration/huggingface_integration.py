"""Example of using Hugging Face transformers for document analysis."""

import argparse
from pathlib import Path

from mmrag.document_processing import PDFProcessor
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def analyze_with_huggingface(file_path, summarize=True, qa=True, zero_shot=True):
    """Analyze a document using Hugging Face transformers."""
    try:
        from transformers import pipeline
    except ImportError:
        console.print("[bold red]Error:[/] transformers package not installed.")
        console.print("Install it with: pip install transformers")
        return
    
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
    
    # Initialize pipelines based on requirements
    models = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        if summarize:
            task = progress.add_task("Loading summarization model...", total=1)
            models["summarizer"] = pipeline("summarization", model="facebook/bart-large-cnn")
            progress.update(task, completed=1)
        
        if qa:
            task = progress.add_task("Loading question-answering model...", total=1)
            models["qa"] = pipeline("question-answering", model="deepset/roberta-base-squad2")
            progress.update(task, completed=1)
        
        if zero_shot:
            task = progress.add_task("Loading zero-shot classification model...", total=1)
            models["zero_shot"] = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            progress.update(task, completed=1)
    
    # Get text elements
    text_elements = [e for e in document.elements if e.element_type == "text"]
    
    # Process with summarization
    if summarize and "summarizer" in models:
        console.print("\n[bold]Document Summarization:[/]")
        
        # Combine text elements for overall summary
        combined_text = " ".join([e.content for e in text_elements])
        
        # Truncate if too long (most models have limits)
        max_length = 1024
        if len(combined_text) > max_length:
            combined_text = combined_text[:max_length]
        
        # Generate summary
        with Progress(SpinnerColumn(), TextColumn("Generating summary..."), console=console) as progress:
            task = progress.add_task("Generating summary...", total=1)
            summary = models["summarizer"](combined_text, max_length=150, min_length=50, do_sample=False)
            progress.update(task, completed=1)
        
        # Display summary
        console.print(f"\n[bold cyan]Summary:[/] {summary[0]['summary_text']}")
    
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
        with Progress(SpinnerColumn(), TextColumn("Classifying document..."), console=console) as progress:
            task = progress.add_task("Classifying document...", total=1)
            classification = models["zero_shot"](combined_text, candidate_labels, multi_label=True)
            progress.update(task, completed=1)
        
        # Display classification results
        console.print("\n[bold cyan]Document Topics:[/]")
        for label, score in zip(classification["labels"], classification["scores"]):
            console.print(f"- {label}: {score:.2f}")
    
    # Process with question-answering
    if qa and "qa" in models:
        console.print("\n[bold]Question Answering:[/]")
        
        # Define some general questions about the document
        questions = [
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
            context = context[:max_length]
        
        # Answer questions
        console.print("\n[bold cyan]Automatic Q&A:[/]")
        for question in questions:
            with Progress(SpinnerColumn(), TextColumn(f"Answering: {question}"), console=console) as progress:
                task = progress.add_task(f"Answering: {question}", total=1)
                answer = models["qa"](question=question, context=context)
                progress.update(task, completed=1)
            
            console.print(f"Q: {question}")
            console.print(f"A: {answer['answer']} (confidence: {answer['score']:.2f})")
            console.print("")
    
    # Interactive Q&A mode
    if qa and "qa" in models:
        console.print("\n[bold]Interactive Question Answering[/]")
        console.print("Ask questions about the document (type 'quit' to exit)")
        
        context = " ".join([e.content for e in text_elements])
        max_length = 512
        if len(context) > max_length:
            context = context[:max_length]
        
        while True:
            question = console.input("\n[bold cyan]Question:[/] ")
            if question.lower() == "quit":
                break
            
            with Progress(SpinnerColumn(), TextColumn("Answering..."), console=console) as progress:
                task = progress.add_task("Answering...", total=1)
                answer = models["qa"](question=question, context=context)
                progress.update(task, completed=1)
            
            console.print(f"[bold green]Answer:[/] {answer['answer']}")
            console.print(f"Confidence: {answer['score']:.2f}")

def main():
    """Run the Hugging Face integration example."""
    parser = argparse.ArgumentParser(description="Document analysis using Hugging Face transformers.")
    parser.add_argument("file_path", type=str, help="Path to the document file")
    parser.add_argument("--no-summarize", action="store_true", help="Skip summarization")
    parser.add_argument("--no-qa", action="store_true", help="Skip question answering")
    parser.add_argument("--no-zero-shot", action="store_true", help="Skip zero-shot classification")
    
    args = parser.parse_args()
    
    analyze_with_huggingface(
        args.file_path,
        summarize=not args.no_summarize,
        qa=not args.no_qa, 
        zero_shot=not args.no_zero_shot
    )

if __name__ == "__main__":
    main()
