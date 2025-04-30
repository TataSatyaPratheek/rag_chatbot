"""Basic RAG Chatbot implementation for PDF documents."""

import argparse
import os
from pathlib import Path

import httpx
from mmrag.document_processing import PDFProcessor
from mmrag.vectordb import ChromaStore
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

class RAGChatbot:
    """A simple RAG chatbot for PDF documents."""
    
    def __init__(self, persist_dir=None, collection_name="rag_chatbot"):
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
        
        # Set up LLM client (using Ollama by default)
        self.llm_base_url = "http://localhost:11434/api"
        self.llm_model = "llama3.2:latest"  # Default model
        
        # Track conversation history
        self.history = []
    
    def load_document(self, file_path):
        """Load and process a document."""
        try:
            console.print(f"Processing document: [bold blue]{file_path}[/]")
            document = self.processor.process(file_path)
            
            # Add to vector store
            self.store.add_document(document)
            
            console.print(f"[bold green]Successfully loaded document:[/] {Path(file_path).name}")
            console.print(f"Document ID: {document.document_id}")
            console.print(f"Extracted {len(document.elements)} elements")
            
            return document
        except Exception as e:
            console.print(f"[bold red]Error loading document:[/] {e}")
            return None
    
    def retrieve_context(self, query, n_results=5):
        """Retrieve relevant context for a query."""
        try:
            results = self.store.query(query, n_results=n_results)
            
            if results and len(results["documents"][0]) > 0:
                context_parts = []
                
                for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                    doc_id = metadata.get("document_id", "unknown")
                    element_type = metadata.get("element_type", "unknown")
                    page = metadata.get("page", "unknown")
                    
                    context_parts.append(f"[Excerpt {i+1} - {element_type} from page {page}]\n{doc}\n")
                
                return "\n".join(context_parts)
            else:
                return "No relevant context found in the documents."
        except Exception as e:
            console.print(f"[bold red]Error retrieving context:[/] {e}")
            return "Error retrieving context."
    
    def query_llm(self, prompt):
        """Query the LLM using Ollama."""
        try:
            # Prepare request data
            data = {
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False
            }
            
            # Send request
            response = httpx.post(
                f"{self.llm_base_url}/generate",
                json=data,
                timeout=60.0
            )
            
            # Parse response
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                return f"Error: HTTP {response.status_code}"
        except Exception as e:
            return f"Error querying LLM: {e}"
    
    def answer_query(self, query):
        """Answer a user query using RAG."""
        # Step 1: Retrieve relevant context
        context = self.retrieve_context(query)
        
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
        response = self.query_llm(prompt)
        
        # Step 4: Update conversation history
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response})
        
        return response
    
    def chat(self):
        """Run an interactive chat session."""
        console.print(Panel.fit(
            "RAG Chatbot - Ask questions about your documents\n"
            "Type 'quit' to exit, 'load <path>' to load a document, 'clear' to clear history",
            title="RAG Chatbot",
            border_style="blue"
        ))
        
        while True:
            # Get user input
            user_input = console.input("\n[bold cyan]You:[/] ")
            
            # Check for commands
            if user_input.lower() == "quit":
                break
                
            elif user_input.lower() == "clear":
                self.history = []
                console.print("[bold yellow]Conversation history cleared.[/]")
                continue
                
            elif user_input.lower().startswith("load "):
                file_path = user_input[5:].strip()
                self.load_document(file_path)
                continue
            
            # Process regular query
            console.print("\n[bold green]Assistant:[/]")
            response = self.answer_query(user_input)
            
            # Display formatted response
            console.print(Markdown(response))

def main():
    """Run the RAG chatbot."""
    parser = argparse.ArgumentParser(description="Basic RAG Chatbot for documents.")
    parser.add_argument("--documents", "-d", nargs="*", help="Documents to load at startup")
    parser.add_argument("--model", "-m", type=str, default="llama3.2:latest", help="LLM model to use")
    parser.add_argument("--persist-dir", "-p", type=str, help="Directory to persist vector database")
    
    args = parser.parse_args()
    
    # Initialize chatbot
    chatbot = RAGChatbot(persist_dir=args.persist_dir)
    
    # Set model if specified
    if args.model:
        chatbot.llm_model = args.model
    
    # Load documents if specified
    if args.documents:
        for doc_path in args.documents:
            chatbot.load_document(doc_path)
    
    # Start chat session
    chatbot.chat()

if __name__ == "__main__":
    main()
