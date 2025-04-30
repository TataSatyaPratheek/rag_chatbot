"""Example of generating streaming responses with RAG."""

import asyncio
import json
import os # Added for psutil
import signal
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Union, Any

import httpx
import typer
from mmrag.document_processing.legacy import PDFProcessor
from mmrag.exceptions import ProcessingTimeoutError, MemoryLimitExceededError # Import exceptions
from mmrag.vectordb import ChromaStore
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
import psutil # Added for memory monitoring
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(help="Streaming RAG chatbot example.")
console = Console()


class StreamingRAGChatbot:
    """A RAG chatbot that generates streaming responses."""
    
    def __init__(
        self, 
        collection_name: str = "streaming_rag",
        base_url: str = "http://localhost:11434/api",
        model: str = "llama3.2:latest",
        timeout_seconds: int = 30, # Added timeout
        memory_limit_fraction: float = 0.5, # Added memory limit fraction
    ):
        """Initialize the chatbot."""
        # Set up vector store
        store_dir = Path("temp_streaming_rag")
        store_dir.mkdir(exist_ok=True)
        
        self.store = ChromaStore(
            persist_directory=store_dir,
            collection_name=collection_name
        )
        
        # Initialize Ollama client
        self.base_url = base_url
        self.model = model
        
        # Set up document processor
        self.processor = PDFProcessor(
            extract_tables=True,
            extract_images=True
        )
        
        # Track loaded documents
        self.loaded_documents = []

        # Store resource limits
        self.timeout_seconds = timeout_seconds
        self.memory_limit_fraction = memory_limit_fraction

    def load_document(self, file_path: Path) -> Optional[Any]:
        """Load a document into the system."""
        try:
            # Resource monitoring setup
            start_time = time.time()
            process = psutil.Process(os.getpid())
            initial_available_memory = psutil.virtual_memory().available
            memory_limit_bytes = initial_available_memory * self.memory_limit_fraction
            console.print(f"Resource limits for loading: Timeout={self.timeout_seconds}s, Memory Limit={memory_limit_bytes / (1024**2):.2f} MB")

            # Process the document
            with Progress(
                # --- Resource Checks ---
                # Note: Checks are placed here to run before the potentially long `processor.process`
                # More granular checks could be added inside the processor if needed.
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Processing document: {file_path.name}", total=None)
                document = self.processor.process(file_path)
                
                elapsed_time = time.time() - start_time
                if elapsed_time > self.timeout_seconds:
                    raise ProcessingTimeoutError(f"Document loading exceeded {self.timeout_seconds} seconds limit.")
                    
                current_rss = process.memory_info().rss
                if current_rss > memory_limit_bytes:
                    raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
                # --- End Resource Checks ---

                # Store in vector database
                progress.update(task, description=f"Indexing document: {file_path.name}")
                self.store.add_document(document)
                
                progress.update(task, completed=True)
            
            # Track loaded document
            self.loaded_documents.append({
                "path": str(file_path),
                "name": file_path.name,
                "document_id": document.document_id,
                "element_count": len(document.elements)
            })
            
            return document
        except Exception as e:
            console.print(f"[bold red]Error loading document:[/] {e}")
            import traceback
            console.print(traceback.format_exc())
            return None
    
    def retrieve_context(self, query: str, n_results: int = 5) -> str:
        """Retrieve relevant context for a query."""
        if not self.loaded_documents:
            return "No documents have been loaded yet."
            
        # Query the vector store
        results = self.store.query(query, n_results=n_results)
        
        # Format context
        context = ""
        if results and len(results["documents"]) > 0 and len(results["documents"][0]) > 0:
            context_items = []
            for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                element_type = metadata.get("element_type", "unknown")
                page = metadata.get("page", "unknown")
                doc_id = metadata.get("document_id", "unknown")
                
                # Find source document path
                source = next((doc["path"] for doc in self.loaded_documents if doc["document_id"] == doc_id), "unknown")
                source_name = Path(source).name if source != "unknown" else "unknown"
                
                context_items.append(f"[{i+1}] From '{source_name}', page {page}, {element_type}:\n{doc}\n")
            
            context = "\n".join(context_items)
        
        return context
    
    async def generate_streaming_response(self, query: str) -> AsyncGenerator[str, None]:
        """Generate a streaming response for the query.
        
        Args:
            query: The user's question
            
        Yields:
            Incremental response chunks
        """
        # Resource monitoring setup for generation
        start_gen_time = time.time()
        process = psutil.Process(os.getpid())
        initial_available_memory = psutil.virtual_memory().available
        memory_limit_bytes = initial_available_memory * self.memory_limit_fraction
        # Note: Timeout for LLM generation is handled by httpx timeout, 
        # but we can add an overall timeout for the whole function if needed.
        # overall_timeout = self.timeout_seconds * 2 # Example: double the loading timeout

        console.print(f"Resource limits for generation: Memory Limit={memory_limit_bytes / (1024**2):.2f} MB")



        # Retrieve context
        with Progress(
            SpinnerColumn(),
            TextColumn("Retrieving context..."),
            console=console
        ) as progress:
            task = progress.add_task("Searching...", total=None)
            context = self.retrieve_context(query)
            progress.update(task, completed=True)

        # --- Resource Check after retrieval ---
        current_rss = process.memory_info().rss
        if current_rss > memory_limit_bytes:
            raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) after retrieval exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
        # elapsed_gen_time = time.time() - start_gen_time
        # if elapsed_gen_time > overall_timeout:
        #     raise ProcessingTimeoutError(f"Generation exceeded {overall_timeout} seconds limit.")
        
        if context == "No documents have been loaded yet.":
            yield "No documents have been loaded yet. Please load a document first."
            return
        
        # Construct the prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.
        
CONTEXT:
{context}

USER QUERY:
{query}

Respond to the user query based only on the information provided in the context. 
If the context doesn't contain relevant information to answer the query, politely state that you don't have that information.
"""
        
        # Prepare request for Ollama streaming
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "stream": True
        }
        
        # Initialize accumulated response
        accumulated_response = ""
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", f"{self.base_url}/generate", json=request_data) as response:
                    response.raise_for_status()
                    
                    async for chunk in response.aiter_text():
                        if not chunk.strip():
                            continue
                            
                        try:
                            # Process JSON chunk
                            data = json.loads(chunk)
                            
                            # Extract the response
                            response_piece = data.get("response", "")
                            accumulated_response += response_piece
                            
                            # Yield accumulated response so far
                            yield accumulated_response
                            
                            # --- Resource Check during streaming ---
                            current_rss = process.memory_info().rss
                            if current_rss > memory_limit_bytes:
                                raise MemoryLimitExceededError(f"Memory usage ({current_rss / (1024**2):.2f} MB) during generation exceeded limit ({memory_limit_bytes / (1024**2):.2f} MB).")
                            # elapsed_gen_time = time.time() - start_gen_time
                            # if elapsed_gen_time > overall_timeout:
                            #     raise ProcessingTimeoutError(f"Generation exceeded {overall_timeout} seconds limit.")

                            # If done, break
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            # Skip invalid JSON
                            continue
        except httpx.HTTPStatusError as e:
            yield f"\n\n[Error: HTTP {e.response.status_code}]"
        except httpx.RequestError as e:
            yield f"\n\n[Error: {str(e)}]"
        except asyncio.CancelledError:
            yield f"\n\n[Response generation cancelled]"
            raise
        except Exception as e:
            yield f"\n\n[Unexpected error: {str(e)}]"


@asynccontextmanager
async def handle_interrupt():
    """Context manager to handle keyboard interrupts gracefully."""
    # Set up interrupt handler
    loop = asyncio.get_running_loop()
    original_handler = loop.get_exception_handler()
    
    def custom_handler(loop, context):
        if "exception" in context and isinstance(context["exception"], KeyboardInterrupt):
            console.print("\n[bold yellow]Interrupted by user[/]")
        else:
            if original_handler is not None:
                original_handler(loop, context)
    
    try:
        loop.set_exception_handler(custom_handler)
        yield
    finally:
        loop.set_exception_handler(original_handler)


async def interactive_chat(chatbot: StreamingRAGChatbot) -> None:
    """Run an interactive chat session with streaming responses."""
    console.print(Panel.fit(
        "RAG Chatbot with Streaming Responses\n"
        "Type 'quit' to exit, 'load <path>' to load a document.",
        title="Streaming RAG Chatbot",
        border_style="blue"
    ))
    
    # Handle interruptions gracefully
    async with handle_interrupt():
        while True:
            try:
                # Get user input
                user_input = console.input("\n[bold cyan]You:[/] ")
                
                if user_input.lower() == "quit":
                    break
                    
                if user_input.lower().startswith("load "):
                    # Extract file path
                    file_path = user_input[5:].strip()
                    path = Path(file_path)
                    
                    if not path.exists():
                        console.print(f"[bold red]File not found:[/] {file_path}")
                        continue
                    
                    # Load document
                    console.print(f"[bold yellow]Loading document:[/] {file_path}")
                    start_time = time.time()
                    document = chatbot.load_document(path)
                    load_time = time.time() - start_time
                    
                    if document:
                        console.print(f"[bold green]Successfully loaded document in {load_time:.2f}s[/]")
                        console.print(f"Extracted {len(document.elements)} elements")
                    continue
                
                # Process query and generate streaming response
                console.print("\n[bold green]Assistant:[/]", end=" ")
                
                # Use Live display for streaming updates
                try:
                    with Live("", refresh_per_second=10, console=console) as live:
                        current_response = ""
                        async for response_so_far in chatbot.generate_streaming_response(user_input):
                            current_response = response_so_far
                            # Format as markdown for nicer display
                            live.update(Markdown(current_response))
                            # Small sleep to control refresh rate
                            await asyncio.sleep(0.05)
                except asyncio.CancelledError:
                    console.print("\n[bold yellow]Response generation cancelled[/]")
                except Exception as e:
                    console.print(f"\n[bold red]Error generating response:[/] {e}")
                    import traceback
                    console.print(traceback.format_exc())
            except KeyboardInterrupt:
                console.print("\n[bold yellow]Interrupted. Type 'quit' to exit.[/]")
            except Exception as e:
                console.print(f"\n[bold red]Unexpected error:[/] {e}")
                import traceback
                console.print(traceback.format_exc())


@app.command()
def run(
    load: Optional[Path] = typer.Option(None, "--load", "-l", help="Load a document at startup"),
    model: str = typer.Option("llama3.2:latest", "--model", "-m", help="LLM model to use"),
    api_url: str = typer.Option("http://localhost:11434/api", "--api", "-a", help="Ollama API URL"),
    timeout: int = typer.Option(30, "--timeout", help="Processing timeout for loading in seconds"),
    mem_limit: float = typer.Option(0.5, "--mem-limit", help="Memory limit as fraction of available memory (0.1-1.0)"),
):
    """Run the streaming RAG chatbot."""
    # Validate memory limit
    if not (0.1 <= mem_limit <= 1.0):
        console.print("[bold red]Error:[/] Memory limit must be between 0.1 and 1.0")
        raise typer.Exit(code=1)

    # Initialize chatbot
    chatbot = StreamingRAGChatbot(
        base_url=api_url,
        model=model,
        timeout_seconds=timeout,
        memory_limit_fraction=mem_limit,
    )
    
    # Load initial document if specified
    if load:
        if not load.exists():
            console.print(f"[bold red]Document not found:[/] {load}")
            raise typer.Exit(code=1)
            
        console.print(f"[bold yellow]Loading document:[/] {load}")
        document = chatbot.load_document(load)
        if document:
            console.print(f"[bold green]Successfully loaded document[/]")
            console.print(f"Extracted {len(document.elements)} elements")
    
    # Run interactive chat
    try:
        import platform
        
        if platform.system() == 'Windows':
            # Windows doesn't support signal.SIGINT handler with asyncio properly
            asyncio.run(interactive_chat(chatbot))
        else:
            # On Unix-like systems, we can handle signals better
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Add signal handlers for graceful shutdown
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda: None)
            
            try:
                loop.run_until_complete(interactive_chat(chatbot))
            finally:
                loop.close()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Chatbot stopped by user[/]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/] {e}")
        import traceback
        console.print(traceback.format_exc())


if __name__ == "__main__":
    app()