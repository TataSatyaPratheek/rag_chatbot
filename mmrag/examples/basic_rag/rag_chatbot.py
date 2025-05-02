"""Improved RAG Chatbot implementation for PDF documents."""

import contextlib
import json
import time
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import httpx
import typer
from mmrag.document_processing.legacy import PDFProcessor
from mmrag.document_processing.factory import get_processor # Use factory
from mmrag.exceptions import ProcessingTimeoutError, MemoryLimitExceededError # Import processing exceptions
from mmrag.llm import LLMClientError # Import LLM exception from correct location
from mmrag.vectordb import ChromaStore
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
import psutil # Added for memory monitoring
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(help="RAG Chatbot for document Q&A.")
console = Console()


class RAGChatbot:
    """A RAG chatbot for document question answering."""
    
    def __init__(
        self, 
        persist_dir: Optional[Path] = None, 
        collection_name: str = "rag_chatbot",
        llm_base_url: str = "http://localhost:11434",
        llm_model: str = "llama3.2:latest",
        timeout_seconds: int = 30, # Added timeout
        memory_limit_fraction: float = 0.5, # Added memory limit fraction
    ):
        """Initialize the RAG chatbot."""
        # Set up vector store
        if persist_dir is None:
            persist_dir = Path("temp_rag_chatbot")
        
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.store = ChromaStore(
            persist_directory=self.persist_dir,
            collection_name=collection_name
        )
        
        # Set up document processor
        self.processor = PDFProcessor(
            extract_tables=True,
            extract_images=True
        )
        
        # Set up LLM client
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        
        # Track conversation history
        self.history = []
        
        # Track loaded documents
        self.loaded_documents = {}

        # Store resource limits
        self.timeout_seconds = timeout_seconds
        self.memory_limit_fraction = memory_limit_fraction
    
    def load_document(self, file_path: Path) -> Optional[Any]:
        """Load and process a document."""
        try:
            # Resource monitoring setup
            start_time = time.time()
            process = psutil.Process(os.getpid())
            initial_available_memory = psutil.virtual_memory().available
            memory_limit_bytes = initial_available_memory * self.memory_limit_fraction
            console.print(f"Resource limits for loading: Timeout={self.timeout_seconds}s, Memory Limit={memory_limit_bytes / (1024**2):.2f} MB")

            with Progress(
                SpinnerColumn(),
                # --- Resource Checks ---
                # Note: Checks are placed here to run before the potentially long `processor.process`
                # More granular checks could be added inside the processor if needed.
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Processing document: {file_path.name}", total=None)
                document = self.processor.process(file_path)
                
                # Add to vector store
                elapsed_time = time.time() - start_time
                if elapsed_time > self.timeout_seconds:
                    raise ProcessingTimeoutError(f"Document loading exceeded {self.timeout_seconds} seconds limit.")
                    
                current_rss = process.memory_info().rss
                if current_rss > memory_limit_bytes:
                    raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
                # --- End Resource Checks ---

                progress.update(task, description=f"Indexing document: {file_path.name}")
                self.store.add_document(document)
                
                progress.update(task, completed=True)
            
            console.print(f"[bold green]Successfully loaded document:[/] {file_path.name}")
            console.print(f"Document ID: {document.document_id}")
            console.print(f"Extracted {len(document.elements)} elements")
            
            # Track loaded document
            self.loaded_documents[document.document_id] = {
                "path": str(file_path),
                "name": file_path.name,
                "elements": len(document.elements)
            }
            
            return document
        except Exception as e:
            console.print(f"[bold red]Error loading document:[/] {e}")
            import traceback
            console.print(traceback.format_exc())
            return None
    
    def retrieve_context(self, query: str, n_results: int = 5) -> str:
        """Retrieve relevant context for a query."""
        try:
            results = self.store.query(query, n_results=n_results)
            
            if results and len(results["documents"][0]) > 0:
                context_parts = []
                
                for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                    doc_id = metadata.get("document_id", "unknown")
                    element_type = metadata.get("element_type", "unknown")
                    page = metadata.get("page", "unknown")
                    
                    # Get file name if available
                    file_name = "unknown"
                    if doc_id in self.loaded_documents:
                        file_name = self.loaded_documents[doc_id]["name"]
                    
                    context_parts.append(f"[Excerpt {i+1} - {element_type} from {file_name}, page {page}]\n{doc}\n")
                
                return "\n".join(context_parts)
            else:
                return "No relevant context found in the documents."
        except Exception as e:
            console.print(f"[bold red]Error retrieving context:[/] {e}")
            return "Error retrieving context."
    
    async def query_llm_async(self, prompt: str) -> str:
        """Query the LLM using Ollama asynchronously."""
        try:
            # Prepare request data
            data = {
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False
            }
            
            # Send request asynchronously
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.llm_base_url}/generate",
                    json=data
                )
                
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code} - {e.response.text}"
        except LLMClientError as e: # Catch specific LLM client errors
            return f"LLM Error: {e}"
        except Exception as e:
            return f"Error querying LLM: {e}"
    
    def query_llm(self, prompt: str) -> str:
        """Query the LLM using Ollama synchronously."""
        try:
            # Prepare request data
            data = {
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False
            }
            
            # Send request
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self.llm_base_url}/generate",
                    json=data
                )
                
                response.raise_for_status()
                result = response.json()
                return result.get("response", "")
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code} - {e.response.text}"
        except LLMClientError as e: # Catch specific LLM client errors
            return f"LLM Error: {e}"
        except Exception as e:
            return f"Error querying LLM: {e}"
    
    async def answer_query_async(self, query: str) -> str:
        """Answer a user query using RAG asynchronously."""
        # Resource monitoring setup for query
        start_query_time = time.time()
        process = psutil.Process(os.getpid())
        initial_available_memory = psutil.virtual_memory().available
        memory_limit_bytes = initial_available_memory * self.memory_limit_fraction
        # Note: Timeout for LLM query is handled by httpx timeout, 
        # but we can add an overall timeout for the whole function if needed.
        # overall_timeout = self.timeout_seconds * 2 # Example: double the loading timeout

        console.print(f"Resource limits for query: Memory Limit={memory_limit_bytes / (1024**2):.2f} MB")


        # Step 1: Retrieve relevant context
        with Progress(
            SpinnerColumn(),
            TextColumn("Retrieving relevant information..."),
            console=console
        ) as progress:
            task = progress.add_task("Searching...", total=None)
            context = self.retrieve_context(query)
            progress.update(task, completed=True)

        # --- Resource Check after retrieval ---
        current_rss = process.memory_info().rss
        if current_rss > memory_limit_bytes:
            raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) after retrieval exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
        # elapsed_query_time = time.time() - start_query_time
        # if elapsed_query_time > overall_timeout:
        #     raise ProcessingTimeoutError(f"Query processing exceeded {overall_timeout} seconds limit.")
        
        # Step 2: Construct prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

CONTEXT:
{context}

USER QUERY:
{query}

Respond to the user query based only on the information provided in the context. 
If the context doesn't contain relevant information to answer the query, politely state that you don't have that information.
"""
        
        # Step 3: Query LLM
        with Progress(
            SpinnerColumn(),
            TextColumn("Generating answer..."),
            console=console
        ) as progress:
            task = progress.add_task("Thinking...", total=None)
            response = await self.query_llm_async(prompt)
            progress.update(task, completed=True)

        # --- Resource Check after LLM ---
        current_rss = process.memory_info().rss
        if current_rss > memory_limit_bytes:
            raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) after LLM query exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
        # elapsed_query_time = time.time() - start_query_time
        # if elapsed_query_time > overall_timeout:
        #     raise ProcessingTimeoutError(f"Query processing exceeded {overall_timeout} seconds limit.")
        
        # Step 4: Update conversation history
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response})
        
        return response
    
    def answer_query(self, query: str) -> str:
        """Answer a user query using RAG."""
        # Resource monitoring setup for query
        start_query_time = time.time()
        process = psutil.Process(os.getpid())
        initial_available_memory = psutil.virtual_memory().available
        memory_limit_bytes = initial_available_memory * self.memory_limit_fraction
        # Note: Timeout for LLM query is handled by httpx timeout, 
        # but we can add an overall timeout for the whole function if needed.
        # overall_timeout = self.timeout_seconds * 2 # Example: double the loading timeout

        console.print(f"Resource limits for query: Memory Limit={memory_limit_bytes / (1024**2):.2f} MB")


        # Step 1: Retrieve relevant context
        with Progress(
            SpinnerColumn(),
            TextColumn("Retrieving relevant information..."),
            console=console
        ) as progress:
            task = progress.add_task("Searching...", total=None)
            context = self.retrieve_context(query)
            progress.update(task, completed=True)

        # --- Resource Check after retrieval ---
        current_rss = process.memory_info().rss
        if current_rss > memory_limit_bytes:
            raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) after retrieval exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
        # elapsed_query_time = time.time() - start_query_time
        # if elapsed_query_time > overall_timeout:
        #     raise ProcessingTimeoutError(f"Query processing exceeded {overall_timeout} seconds limit.")
        
        # Step 2: Construct prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

CONTEXT:
{context}

USER QUERY:
{query}

Respond to the user query based only on the information provided in the context. 
If the context doesn't contain relevant information to answer the query, politely state that you don't have that information.
"""
        
        # Step 3: Query LLM
        with Progress(
            SpinnerColumn(),
            TextColumn("Generating answer..."),
            console=console
        ) as progress:
            task = progress.add_task("Thinking...", total=None)
            response = self.query_llm(prompt)
            progress.update(task, completed=True)

        # --- Resource Check after LLM ---
        current_rss = process.memory_info().rss
        if current_rss > memory_limit_bytes:
            raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) after LLM query exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
        # elapsed_query_time = time.time() - start_query_time
        # if elapsed_query_time > overall_timeout:
        #     raise ProcessingTimeoutError(f"Query processing exceeded {overall_timeout} seconds limit.")
        
        # Step 4: Update conversation history
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response})
        
        return response
    
    def save_conversation(self, file_path: Optional[Path] = None) -> Optional[Path]:
        """Save the conversation history to a file."""
        if not self.history:
            console.print("[yellow]No conversation to save[/]")
            return None
        
        if file_path is None:
            timestamp = Path("conversation.json")
            file_path = timestamp
        
        try:
            with open(file_path, "w") as f:
                json.dump({
                    "history": self.history,
                    "documents": self.loaded_documents
                }, f, indent=2)
            
            console.print(f"[bold green]Conversation saved to:[/] {file_path}")
            return file_path
        except Exception as e:
            console.print(f"[bold red]Error saving conversation:[/] {e}")
            return None
    
    def load_conversation(self, file_path: Path) -> bool:
        """Load a conversation history from a file."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            self.history = data.get("history", [])
            
            # Just load the metadata about documents, not the actual documents
            self.loaded_documents = data.get("documents", {})
            
            console.print(f"[bold green]Loaded conversation with {len(self.history)//2} exchanges[/]")
            return True
        except Exception as e:
            console.print(f"[bold red]Error loading conversation:[/] {e}")
            return False
    
    def chat(self):
        """Run an interactive chat session."""
        console.print(Panel.fit(
            "RAG Chatbot - Ask questions about your documents\n"
            "Type 'quit' to exit, 'load <path>' to load a document, 'clear' to clear history, 'save' to save conversation",
            title="RAG Chatbot",
            border_style="blue"
        ))
        
        # Display info about loaded documents if any
        if self.loaded_documents:
            console.print("\n[bold]Loaded Documents:[/]")
            for doc_id, doc_info in self.loaded_documents.items():
                console.print(f"- {doc_info['name']} ({doc_info['elements']} elements)")
        
        while True:
            # Get user input
            user_input = Prompt.ask("\n[bold cyan]You")
            
            # Check for commands
            if user_input.lower() == "quit":
                break
                
            elif user_input.lower() == "clear":
                self.history = []
                console.print("[bold yellow]Conversation history cleared.[/]")
                continue
                
            elif user_input.lower() == "save":
                self.save_conversation()
                continue
                
            elif user_input.lower().startswith("load "):
                file_path = user_input[5:].strip()
                path = Path(file_path)
                if path.exists():
                    if path.suffix.lower() == ".json":
                        # Try to load as conversation
                        self.load_conversation(path)
                    else:
                        # Try to load as document
                        self.load_document(path)
                else:
                    console.print(f"[bold red]File not found:[/] {file_path}")
                continue
            
            # Process regular query
            console.print("\n[bold green]Assistant:[/]")
            try:
                import asyncio
                # Use async version if running in an environment that supports it
                if hasattr(asyncio, "run"):
                    response = asyncio.run(self.answer_query_async(user_input))
                else:
                    response = self.answer_query(user_input)
                
                # Display formatted response
                console.print(Markdown(response))
            except KeyboardInterrupt:
                console.print("\n[yellow]Query cancelled[/]")
            except Exception as e:
                console.print(f"[bold red]Error:[/] {e}")
                import traceback
                console.print(traceback.format_exc())


@app.command()
def run(
    documents: Optional[List[Path]] = typer.Option(None, "--documents", "-d", help="Documents to load at startup"),
    model: str = typer.Option("llama3.2:latest", "--model", "-m", help="LLM model to use"),
    persist_dir: Optional[Path] = typer.Option(None, "--persist-dir", "-p", help="Directory to persist vector database"),
    conversation: Optional[Path] = typer.Option(None, "--conversation", "-c", help="Load a saved conversation"),
    timeout: int = typer.Option(30, "--timeout", help="Processing timeout for loading in seconds"),
    mem_limit: float = typer.Option(0.5, "--mem-limit", help="Memory limit as fraction of available memory (0.1-1.0)"),
):
    """Run the RAG chatbot."""
    # Validate memory limit
    if not (0.1 <= mem_limit <= 1.0):
        console.print("[bold red]Error:[/] Memory limit must be between 0.1 and 1.0")
        raise typer.Exit(code=1)

    # Initialize chatbot
    chatbot = RAGChatbot(persist_dir=persist_dir, llm_model=model, timeout_seconds=timeout, memory_limit_fraction=mem_limit)
    
    # Load conversation if specified
    if conversation and conversation.exists():
        chatbot.load_conversation(conversation)
    
    # Load documents if specified
    if documents:
        for doc_path in documents:
            if doc_path.exists():
                chatbot.load_document(doc_path)
            else:
                console.print(f"[bold red]Document not found:[/] {doc_path}")
    
    # Start chat session
    try:
        chatbot.chat()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Chatbot session ended by user.[/]")
        
        # Offer to save the conversation
        if chatbot.history and Prompt.ask("[yellow]Save conversation?[/]", choices=["y", "n"], default="y") == "y":
            chatbot.save_conversation()


@app.command()
def batch(
    document: Path = typer.Argument(..., help="Document to query"),
    questions: Path = typer.Argument(..., help="File with list of questions, one per line"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for answers"),
    model: str = typer.Option("llama3.2:latest", "--model", "-m", help="LLM model to use"),
    timeout: int = typer.Option(30, "--timeout", help="Processing timeout for loading in seconds"),
    mem_limit: float = typer.Option(0.5, "--mem-limit", help="Memory limit as fraction of available memory (0.1-1.0)"),
):
    """Run batch queries against a document."""
    # Validate memory limit
    if not (0.1 <= mem_limit <= 1.0):
        console.print("[bold red]Error:[/] Memory limit must be between 0.1 and 1.0")
        raise typer.Exit(code=1)

    if not document.exists():
        console.print(f"[bold red]Document not found:[/] {document}")
        raise typer.Exit(code=1)
    
    if not questions.exists():
        console.print(f"[bold red]Questions file not found:[/] {questions}")
        raise typer.Exit(code=1)
    
    # Initialize chatbot
    with tempfile.TemporaryDirectory() as temp_dir:
        chatbot = RAGChatbot(persist_dir=temp_dir, llm_model=model, timeout_seconds=timeout, memory_limit_fraction=mem_limit)
        
        # Load document
        console.print(f"Loading document: [bold blue]{document}[/]")
        doc = chatbot.load_document(document)
        
        if not doc:
            console.print("[bold red]Failed to load document.[/]")
            raise typer.Exit(code=1)
        
        # Load questions
        with open(questions, "r") as f:
            question_list = [q.strip() for q in f.readlines() if q.strip()]
        
        console.print(f"Loaded {len(question_list)} questions")
        
        # Process questions
        results = []
        for i, question in enumerate(question_list):
            console.print(f"\n[bold]Question {i+1}/{len(question_list)}:[/] {question}")
            
            answer = chatbot.answer_query(question)
            results.append({"question": question, "answer": answer})
            
            console.print("[bold green]Answer:[/]")
            console.print(Markdown(answer))
        
        # Save results
        if output:
            with open(output, "w") as f:
                json.dump(results, f, indent=2)
            console.print(f"\n[bold green]Results saved to:[/] {output}")


if __name__ == "__main__":
    app()