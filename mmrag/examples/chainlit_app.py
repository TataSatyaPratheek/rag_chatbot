import os
import tempfile
import chainlit as cl
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
import logging

# Import from your existing codebase
from mmrag.document_processing.factory import get_processor
from mmrag.vectordb.chroma import ChromaStore # Using ChromaStore based on streamlit app
from mmrag.llm.client import OllamaClient

# --- Configuration (Reads from .env or uses defaults) ---
CHROMA_PATH = os.getenv("CHROMA_PATH", "/Users/vi/Documents/work/rag_chatbot/mmrag/examples/tourism_chroma_db") # Use absolute path
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tourism-data")
LLM_MODEL_DEFAULT = os.getenv("LLM_MODEL", "llama3.2:latest") # Updated default
EMBEDDING_MODEL_DEFAULT = os.getenv("TEXT_EMBEDDING_MODEL", "nomic-embed-text") # Updated default
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TEMP_UPLOAD_DIR = Path("/Users/vi/Documents/work/rag_chatbot/mmrag/examples/temp_uploads") # Absolute path
# Initialize components globally
vector_store: Optional[ChromaStore] = None
llm_client: Optional[OllamaClient] = None

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
llm_client: Optional[OllamaClient] = None

def ensure_directories():
    """Ensure required directories exist."""
    os.makedirs(CHROMA_PATH, exist_ok=True)
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session and welcome the user."""
    global vector_store, llm_client
    ensure_directories() # Ensure directories exist first

    # Clear ChromaDB cache as suggested for connection issues
    # chromadb.api.client.SharedSystemClient.clear_system_cache() # Keep commented unless specifically needed

    # Initialize clients (consider making model selection dynamic later)
    try:
        vector_store = ChromaStore(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PATH,
            embedding_model_name=EMBEDDING_MODEL_DEFAULT,
        )
        llm_client = OllamaClient(
            model=LLM_MODEL_DEFAULT,
            base_url=OLLAMA_BASE_URL
        )

        # --- Removed cl.ChatSettings and cl.Select ---
        # Inform user about models instead of using Select
        await cl.Message(
            content=f"Using LLM: `{LLM_MODEL_DEFAULT}` and Embedding Model: `{EMBEDDING_MODEL_DEFAULT}`. (These can be changed via environment variables)",
            author="System Info"
        ).send()
        llm_model_selected = LLM_MODEL_DEFAULT # Use default directly
        embedding_model_selected = EMBEDDING_MODEL_DEFAULT # Use default directly

        # Re-initialize clients with selected models
        vector_store = ChromaStore(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PATH,
            embedding_model_name=embedding_model_selected,
        )
        llm_client = OllamaClient(
            model=llm_model_selected,
            base_url=OLLAMA_BASE_URL
        )
        # Safely get count
        db_count = 0
        if vector_store and vector_store.collection:
            db_count = vector_store.collection.count()
        db_status = f"Database connected ({db_count} items, using '{embedding_model_selected}')."
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        logger.error(f"ChromaDB connection failed: {e}\n{error_detail}")
        db_status = f"⚠️ Database connection failed: {e}. Check logs for details. Please upload documents if needed."
        vector_store = None # Ensure it's None if connection failed

    try:
        # Check LLM connection - simple ping or model list
        # A simple check: try getting embeddings for a short text
        await asyncio.to_thread(llm_client.get_embeddings, ["test"])
        llm_status = f"LLM '{llm_model_selected}' ready."
    except Exception as e:
        llm_status = f"⚠️ LLM connection failed: {e}. Ensure Ollama is running."
        logger.error(f"LLM connection failed: {e}", exc_info=True)
        llm_client = None # Ensure it's None if connection failed


    # Welcome message with instructions
    init_message = f"""👋 Welcome to TourismInsight AI!
Status:
- {db_status}
- {llm_status}
---
Upload tourism documents or ask questions about travel trends for 2025.
"""
    await cl.Message(content=init_message, author="TourismInsight").send()

    # --- Removed cl.Avatar --- (Chainlit uses the author name automatically)

    # Show quick buttons for common tourism questions
    actions = [
        cl.Action(
            name="trends_2025",
            payload={"value": "What are the top travel trends for 2025?"}, # Use payload
            label="2025 Trends"
        ),
        cl.Action(
            name="adventure",
            payload={"value": "Tell me about adventure tourism growth"}, # Use payload
            label="Adventure Tourism"
        ),
        cl.Action(
            name="luxury",
            payload={"value": "What's happening in luxury travel?"}, # Use payload
            label="Luxury Travel"
        ),
    ]

    await cl.Message(
        content="Choose a topic or ask your own question:",
        author="TourismInsight",
        actions=actions
    ).send()

    # Ask for file uploads AFTER initial setup and message
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload tourism documents to process (PDF, PPTX, DOCX). You can skip this if data exists.",
            accept=["application/pdf", "application/vnd.openxmlformats-officedocument.presentationml.presentation", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
            max_size_mb=100,
            timeout=300, # 5 minutes timeout for upload
            raise_on_timeout=False # Don't raise error, just proceed
        ).send()
        # If user explicitly cancels or times out without uploading, files will be None or empty list
        if files is None or len(files) == 0:
            await cl.Message(content="No files uploaded. Proceeding with existing data if available.", author="TourismInsight").send()
            break # Exit loop if no files or timeout

    if files:
        await cl.Message(content=f"Received {len(files)} file(s). Processing...", author="TourismInsight").send()
        for file in files:
            await process_file(file)

async def process_file(file: cl.File):
    """Process an uploaded tourism document."""
    global vector_store, llm_client # Access global instances

    # Check if vector_store is initialized (should be by on_chat_start)
    if vector_store is None:
        await cl.Message(content=f"❌ Error: Database not initialized. Cannot process {file.name}.", author="TourismInsight").send()
        return

    # Create temp file path within the designated directory
    temp_file_path = TEMP_UPLOAD_DIR / file.name

    # Processing message
    processing_msg = cl.Message(content=f"⏳ Processing {file.name}...", author="TourismInsight")
    await processing_msg.send()

    try:
        # Write file to disk
        # Read content from the path provided by AskFileResponse
        with open(file.path, "rb") as f_in:
            with open(temp_file_path, "wb") as f_out:
                f_out.write(f_in.read())
        # Get appropriate processor (using OpenParse as suggested, ensure it's installed)
        # Consider making processor choice dynamic or configurable later
        processor = get_processor(
            temp_file_path,
            use_openparse=True, # Using OpenParse as per suggestion
            extract_tables=True,
            extract_images=False, # OpenParse image support is limited
        )

        # Process document asynchronously
        document = await processor.process(temp_file_path)

        # Add document to vector store (using the correct method)
        # Wrap the synchronous ChromaStore.add_document call
        await asyncio.to_thread(vector_store.add_document, document)

        # Update processing message
        processing_msg.content = f"✅ Successfully processed and stored: {file.name}"
        await processing_msg.update()

    except Exception as e:
        error_message = f"❌ Error processing {file.name}: {str(e)}"
        logger.error(error_message, exc_info=True) # Log detailed error
        processing_msg.content = error_message
        await processing_msg.update()

    finally:
        # Clean up temp file
        if temp_file_path.exists():
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.error(f"Failed to remove temporary file {temp_file_path}: {e}")

@cl.action_callback("trends_2025")
@cl.action_callback("adventure")
@cl.action_callback("luxury")
async def on_action(action: cl.Action):
    """Handle action button clicks."""
    action_value = action.payload.get("value", "") # Get value from payload
    # Simulate user message and trigger on_message
    await cl.Message(content=action_value, author="User").send()
    await on_message(cl.Message(content=action_value))

@cl.on_message
async def on_message(message: cl.Message):
    """Process user messages and generate responses."""
    query = message.content

    # Check if clients are initialized
    if vector_store is None or llm_client is None:
        await cl.Message(
            content="System not fully initialized. Please ensure the database is accessible and Ollama is running, or upload documents.",
            author="TourismInsight"
        ).send()
        return

    # Start a thinking message to show processing
    thinking_msg = cl.Message(content="", author="TourismInsight")
    await thinking_msg.stream_token("Searching tourism data... ")

    # Retrieve relevant documents
    try:
        results = await asyncio.to_thread(vector_store.query, query, n_results=5) # Wrap sync query
        contexts = []
        sources_for_display = set()
        if results and results.get("documents") and results["documents"][0]:
            for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
                source = metadata.get("filename", "Unknown source") # Use filename from metadata
                page = metadata.get("page")
                page_info = f" (Page {page})" if page else ""
                contexts.append(f"Info from {source}{page_info}:\n{doc}")
                sources_for_display.add(source)
        else:
            contexts.append("No relevant documents found in the database.")

        context_text = "\n\n---\n\n".join(contexts)

    except Exception as e:
        thinking_msg.content = f"Error retrieving documents: {str(e)}"
        await thinking_msg.update()
        return

    await thinking_msg.stream_token("Generating response... ")

    # Stream the response
    response_message = cl.Message(content="", author="TourismInsight")
    await response_message.send() # Send an empty message to update later

    system_prompt = """You are a tourism industry expert assistant. Answer the user's question based ONLY on the
    provided context information below. Be concise and factual. If the context doesn't contain the answer,
    state clearly that the information is not available in the provided documents."""

    rag_prompt = f"""
    Context information:
    {context_text}

    ---

    User question: {query}

    Answer:
    """

    full_response = ""
    try:
        # Use stream_chat which expects a list of messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": rag_prompt})

        response_stream = llm_client.stream_chat(
            messages=messages,
            temperature=0.5 # Lower temp for more factual answers
        )

        async for chunk in response_stream:
            full_response += chunk
            await response_message.stream_token(chunk)

        # Add sources after response
        if sources_for_display:
            sources_text = "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in sorted(list(sources_for_display))])
            await response_message.stream_token(sources_text)

        await response_message.update() # Final update to ensure everything is rendered
        thinking_msg.content = "Response generated." # Update thinking message
        await thinking_msg.update()

    except Exception as e:
        response_message.content = f"Error generating response: {str(e)}"
        await response_message.update()
        thinking_msg.content = "Error during generation."
        await thinking_msg.update()

# --- Removed @cl.on_file_upload ---
# @cl.on_file_upload(accept=["pdf", "pptx", "docx"], max_size_mb=100)
# async def on_file_upload(files: list[cl.File]):
#     # This function is now deprecated and its logic moved to on_message
#     pass