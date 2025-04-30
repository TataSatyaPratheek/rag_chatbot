"""Example of generating streaming responses with RAG."""

import argparse
import asyncio
import time
from pathlib import Path

import httpx
from mmrag.document_processing import PDFProcessor
from mmrag.vectordb import ChromaStore
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

console = Console()

class StreamingRAGChatbot:
    """A RAG chatbot that generates streaming responses."""
    
    def __init__(self, collection_name="streaming_rag"):
        """Initialize the chatbot."""
        # Set up vector store
        store_dir = Path("temp_streaming_rag")
        store_dir.mkdir(exist_ok=True)
        
        self.store = ChromaStore(
            persist_directory=store_dir,
            collection_name=collection_name
        )
        
        # Initialize Ollama client
        self.base_url = "http://localhost:11434/api"
        self.model = "llama3.2:latest"  # Can be configurable
        
        # Set up document processor
        self.processor = PDFProcessor(
            extract_tables=True,
            extract_images=True
        )
        
        # Track loaded documents
        self.loaded_documents = []
    
    def load_document(self, file_path):
        """Load a document into the system."""
        try:
            # Process the document
            document = self.processor.process(file_path)
            
            # Store in vector database
            self.store.add_document(document)
            
            # Track loaded document
            self.loaded_documents.append({
                "path": file_path,
                "document_id": document.document_id,
                "element_count": len(document.elements)
            })
            
            return document
        except Exception as e:
            console.print(f"[bold red]Error loading document:[/] {e}")
            return None
    
    def retrieve_context(self, query, n_results=5):
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
    
    async def generate_streaming_response(self, query):
        """Generate a streaming response for the query."""
        # Retrieve context
        context = self.retrieve_context(query)
        
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
                    async for chunk in response.aiter_text():
                        if not chunk.strip():
                            continue
                            
                        try:
                            # Process JSON chunk
                            import json
                            data = json.loads(chunk)
                            
                            # Extract the response
                            response_piece = data.get("response", "")
                            accumulated_response += response_piece
                            
                            # Yield accumulated response so far
                            yield accumulated_response
                            
                            # If done, break
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            # Skip invalid JSON
                            continue
        except Exception as e:
            yield f"\n\n[Error: {str(e)}]"

async def interactive_chat(chatbot):
    """Run an interactive chat session with streaming responses."""
    console.print("[bold]RAG Chatbot with Streaming Responses[/]")
    console.print("Type 'quit' to exit, 'load <path>' to load a document.")
    
    while True:
        # Get user input
        user_input = console.input("\n[bold cyan]You:[/] ")
        
        if user_input.lower() == "quit":
            break
            
        if user_input.lower().startswith("load "):
            # Extract file path
            file_path = user_input[5:].strip()
            
            # Load document
            console.print(f"[bold yellow]Loading document:[/] {file_path}")
            start_time = time.time()
            document = chatbot.load_document(file_path)
            load_time = time.time() - start_time
            
            if document:
                console.print(f"[bold green]Successfully loaded document in {load_time:.2f}s[/]")
                console.print(f"Extracted {len(document.elements)} elements")
            continue
        
        # Process query and generate streaming response
        console.print("\n[bold green]Assistant:[/]", end=" ")
        
        # Use Live display for streaming updates
        with Live("", refresh_per_second=10, console=console) as live:
            current_response = ""
            async for response_so_far in chatbot.generate_streaming_response(user_input):
                current_response = response_so_far
                # Format as markdown for nicer display
                live.update(Markdown(current_response))
                # Small sleep to control refresh rate
                await asyncio.sleep(0.05)

def main():
    """Run the streaming RAG example."""
    parser = argparse.ArgumentParser(description="Streaming RAG chatbot example.")
    parser.add_argument("--load", "-l", type=str, help="Load a document at startup")
    args = parser.parse_args()
    
    # Initialize chatbot
    chatbot = StreamingRAGChatbot()
    
    # Load initial document if specified
    if args.load:
        console.print(f"[bold yellow]Loading document:[/] {args.load}")
        document = chatbot.load_document(args.load)
        if document:
            console.print(f"[bold green]Successfully loaded document[/]")
            console.print(f"Extracted {len(document.elements)} elements")
    
    # Run interactive chat
    asyncio.run(interactive_chat(chatbot))

if __name__ == "__main__":
    main()
