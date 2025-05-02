import streamlit as st
import os
import tempfile
import asyncio
from pathlib import Path

from mmrag.document_processing.factory import get_processor
from mmrag.vectordb.chroma import ChromaStore # Changed import
from mmrag.llm.client import OllamaClient

# Page config
st.set_page_config(
    page_title="Tourism Insights Explorer",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Configuration (Consider moving to a config file or env vars) ---
CHROMA_PATH = os.getenv("CHROMA_PATH", "./tourism_chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tourism-data")
LLM_MODEL_DEFAULT = os.getenv("LLM_MODEL", "llama3.2:latest") # Use updated default
EMBEDDING_MODEL_DEFAULT = os.getenv("EMBEDDING_MODEL", "nomic-embed-text") # Use config default name
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload tourism documents or ask questions about travel trends for 2025."}
    ]
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# --- Initialize database and Ollama client ---
@st.cache_resource
def get_vector_db(embedding_model):
    st.write(f"Initializing ChromaStore with embedding model: {embedding_model}")
    # Use ChromaStore and its expected arguments
    return ChromaStore(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_model_name=embedding_model,
    )

@st.cache_resource
def get_ollama_client(llm_model, embedding_model):
    st.write(f"Initializing OllamaClient with LLM: {llm_model}")
    return OllamaClient(
        model=llm_model, # Use 'model' argument for LLM
        base_url=OLLAMA_BASE_URL
    )

# --- Sidebar ---
with st.sidebar:
    st.title("🌴 Tourism Insights")

    st.subheader("LLM & Embedding Settings")
    llm_model_select = st.selectbox(
        "Select LLM Model",
        options=["llama3.2:latest", "mistral", "llama3", "phi3"], # Updated options
        index=0,
        help="Choose a local Ollama model for generation"
    )
    embedding_model_select = st.selectbox(
        "Select Embedding Model",
        options=["nomic-embed-text", "mxbai-embed-large"],
        index=0,
        help="Choose a local Ollama model for embeddings"
    )

    # Get clients based on selection
    vector_db = get_vector_db(embedding_model_select)
    ollama_client = get_ollama_client(llm_model_select, embedding_model_select)

    st.subheader("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload tourism documents",
        accept_multiple_files=True,
        type=["pdf", "pptx", "docx"] # Added docx
    )

    # Define an async function to handle processing
    async def process_files_async(files_to_process, temp_dir, vector_db):
        processed_count = 0
        for file in files_to_process:
            temp_path = os.path.join(temp_dir, file.name)
            try:
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())

                st.write(f"Processing {file.name}...")
                processor = get_processor(
                    Path(temp_path),
                    extract_tables=True,
                    extract_images=True, # Keep image extraction if needed later
                    advanced_table_detection=True,
                    use_openparse=True # Use OpenParse as suggested
                )
                # Await the async process method
                document = await processor.process(Path(temp_path))

                # Use the add_document method from ChromaStore
                if document and document.elements:
                    # ChromaStore's add_document handles element processing and embedding
                    vector_db.add_document(document)
                    # Assume success if no exception is raised
                    # (ChromaStore logs internally)
                    st.session_state.processed_files.add(file.name)
                    processed_count += 1
                else:
                    st.warning(f"No text content extracted from {file.name}")

            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path) # Clean up temp file
        return processed_count

    if uploaded_files:
        files_to_process = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        if files_to_process and st.button("Process New Documents"):
            with st.spinner("Processing documents... This might take a while."):
                temp_dir = tempfile.mkdtemp()
                # Run the async function using asyncio.run()
                processed_count = asyncio.run(
                    process_files_async(files_to_process, temp_dir, vector_db)
                )

            if processed_count > 0:
                st.success(f"Successfully processed and indexed {processed_count} new documents.")
            else:
                st.info("No new documents to process.")

    st.subheader("Processed Documents")
    if st.session_state.processed_files:
        for filename in st.session_state.processed_files:
            st.write(f"- {filename}")
    else:
        st.write("No documents processed yet.")

# --- Main chat area ---
st.title("Tourism Insights Explorer")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about travel trends or tourism insights..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        try:
            with st.spinner("Searching tourism data..."):
                results = vector_db.query(prompt, n_results=5)
                contexts = []
                if results and results.get("documents") and results["documents"][0]:
                    for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
                        source = metadata.get("source", "Unknown source")
                        page = metadata.get("page")
                        page_info = f" (Page {page})" if page else ""
                        contexts.append(f"Info from {source}{page_info}:\n{doc}")
                else:
                    st.write("No relevant documents found in the database.")
                    contexts.append("No relevant documents found.")

                context_text = "\n\n---\n\n".join(contexts)

            system_prompt = """You are a tourism industry expert assistant. Answer the user's question based ONLY on the
            provided context information below. Be concise and factual. If the context doesn't contain the answer,
            state clearly that the information is not available in the provided documents."""

            rag_prompt = f"""
            Context information:
            {context_text}

            ---

            User question: {prompt}

            Answer:
            """

            # Stream response using asyncio
            async def stream_response():
                # Use stream_chat which expects a list of messages
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": rag_prompt})

                response_stream = ollama_client.stream_chat(
                    messages=messages,
                    temperature=0.5 # Lower temp for more factual answers
                )
                res = ""
                async for chunk in response_stream:
                    res += chunk
                    response_container.markdown(res + "▌")
                return res

            full_response = asyncio.run(stream_response())
            response_container.markdown(full_response)

        except Exception as e:
            st.error(f"Error generating response: {e}")
            full_response = "I encountered an error. Please check if Ollama is running and the models are available."
            response_container.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})