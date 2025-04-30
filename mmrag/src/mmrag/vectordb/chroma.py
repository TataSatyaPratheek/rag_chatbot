"""ChromaDB integration for mmrag."""

import json
import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

import chromadb
import torch # Import torch
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from mmrag.config import config
from mmrag.document_processing.base import DocumentElement, ProcessedDocument

logger = logging.getLogger(__name__)

from mmrag.vectordb.cache import EmbeddingCache



class ChromaStore:
    """ChromaDB vector store for document elements."""
    
    def __init__(
        self,
        persist_directory: Optional[Union[str, Path]] = None,
        embedding_model: Optional[str] = None,
        collection_name: str = "document_elements",
    ):
        """Initialize the ChromaDB store.
        
        Args:
            persist_directory: Directory to persist the database.
                Defaults to config.chroma_persist_directory.
            embedding_model: Name of the sentence-transformers model to use.
                Defaults to config.embedding_model.
            collection_name: Name of the collection to use.
        """
        self.persist_directory = Path(persist_directory or config.chroma_persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model_name = embedding_model or config.embedding_model
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        
        # Create or get the collection
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Document elements for RAG"}
            )
            logger.info(f"Created new collection: {collection_name}")
        
        # Load the embedding model
        self.embedding_model = None  # Lazy load on first use
        
        # Add embedding cache
        self.embedding_cache = EmbeddingCache()
    
    def _get_embedding_for_text(self, text):
        """Get embedding for text with caching."""
        # Try to get from cache
        cached_embedding = self.embedding_cache.get_embedding(text, self.embedding_model_name)
        if cached_embedding is not None:
            return cached_embedding
            
        # Not in cache, compute embedding
        model = self._get_embedding_model()
        embedding = model.encode(text)
        
        # Save to cache
        self.embedding_cache.save_embedding(text, embedding, self.embedding_model_name)
        
        return embedding

        
        
    
    def _get_embedding_model(self):
        """Lazy-load the embedding model."""
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            # Explicitly set device to handle potential MPS/meta tensor issues
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.embedding_model.to(device) # Move model to device after loading
        return self.embedding_model
    
    def add_document(self, document: ProcessedDocument) -> None:
        """Add a processed document to the vector store.
        
        Args:
            document: Processed document to add.
        """
        elements = document.elements
        
        if not elements:
            logger.warning(f"No elements found in document: {document.document_id}")
            return
        
        # Generate embeddings for the elements
        texts = []
        metadatas = []
        ids = []
        
        for element in elements:
            # Skip elements without meaningful text
            text_content = self._get_element_text(element)
            if not text_content or len(text_content.strip()) < 10:
                continue
            
            # Add to the lists
            texts.append(text_content)
            metadatas.append({
                "document_id": document.document_id,
                "element_id": element.element_id,
                "element_type": element.element_type,
                "page": element.bbox.page if element.bbox else 0,
                "filename": document.filename,
                "doc_type": document.doc_type,
            })
            ids.append(f"{document.document_id}_{element.element_id}")
        
        if not texts:
            logger.warning(f"No valid elements to add for document: {document.document_id}")
            return
        
        # Generate embeddings
        model = self._get_embedding_model()
        embeddings = model.encode(texts)
        
        # Add to the collection
        self.collection.add(
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids,
            documents=texts,
        )
        
        logger.info(f"Added {len(texts)} elements from document: {document.document_id}")
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
    ) -> Dict:
        """Query the vector store.
        
        Args:
            query_text: Query text.
            n_results: Number of results to return.
            where: Filter on metadata.
            where_document: Filter on document content.
            
        Returns:
            Query results.
        """
        # Generate embedding for the query
        model = self._get_embedding_model()
        query_embedding = model.encode(query_text)
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
            where_document=where_document,
        )
        
        return results
    
    def delete_document(self, document_id: str) -> None:
        """Delete a document from the vector store.
        
        Args:
            document_id: Document ID to delete.
        """
        self.collection.delete(
            where={"document_id": document_id}
        )
        logger.info(f"Deleted document: {document_id}")
    
    def _get_element_text(self, element: DocumentElement) -> str:
        """Get text representation of an element for embedding.
        
        Args:
            element: Document element.
            
        Returns:
            Text representation.
        """
        if element.element_type == "text":
            return element.content
        
        elif element.element_type == "table":
            # Create a text representation of the table
            if isinstance(element.content, list):
                rows = []
                for row in element.content:
                    if isinstance(row, list):
                        rows.append(" | ".join(str(cell) for cell in row))
                return "\n".join(rows)
            return str(element.content)
        
        elif element.element_type == "image":
            # For images, use metadata and any alt text
            return f"Image: {element.metadata.get('description', 'No description')}"
        
        elif element.element_type == "chart":
            # For charts, use the description and any extracted data
            text = f"Chart: {element.content}"
            if element.data:
                if isinstance(element.data, dict):
                    data_str = ", ".join(f"{k}: {v}" for k, v in element.data.items())
                    text += f" Data: {data_str}"
                else:
                    text += f" Data: {element.data}"
            return text
        
        # Default case
        return str(element.content)
