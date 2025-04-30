# tests/unit/test_vectordb/test_chroma.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, ANY
from pathlib import Path

from mmrag.document_processing.base import ProcessedDocument, TextElement, TableElement, BoundingBox
from mmrag.vectordb.chroma import ChromaStore

class TestChromaStore:
    """Test suite for the ChromaDB vector store."""
    
    def test_initialization(self, temp_dir):
        """Test store initialization."""
        # Create a store with a specific directory
        store = ChromaStore(persist_directory=temp_dir, collection_name="test_collection")
        
        assert store.persist_directory == temp_dir
        assert store.embedding_model_name == "all-MiniLM-L6-v2"  # Default model
        assert hasattr(store, "collection")
        assert hasattr(store, "embedding_cache")
    
    @patch("mmrag.vectordb.chroma.SentenceTransformer") # Patch where it's looked up
    def test_get_embedding_model(self, mock_transformer, temp_dir):
        """Test lazy loading of embedding model."""
        store = ChromaStore(persist_directory=temp_dir)
        
        # Model should be None initially (lazy loading)
        assert store.embedding_model is None
        
        # Getting the embedding model should load it
        model = store._get_embedding_model()
        
        # Check that the model was loaded
        assert mock_transformer.called
        assert store.embedding_model is not None
    
    @patch("mmrag.vectordb.chroma.SentenceTransformer") # Patch where it's looked up
    def test_add_document(self, mock_transformer, temp_dir):
        """Test adding a document to the store."""
        # Create a mock embedding model
        mock_model = MagicMock() # Mock the model instance
        mock_model.encode.return_value = np.random.rand(384)
        mock_transformer.return_value = mock_model
        
        # Create a test document
        doc = ProcessedDocument(
            document_id="test-doc-1",
            filename="test.pdf",
            doc_type="pdf",
            elements=[
                TextElement(
                    element_id="text-1",
                    content="This is a test text element.",
                    bbox=BoundingBox(x0=0, y0=0, x1=100, y1=20, page=0)
                ),
                TableElement(
                    element_id="table-1",
                    content=[["Header 1", "Header 2"], ["Data 1", "Data 2"]],
                    bbox=BoundingBox(x0=0, y0=30, x1=100, y1=80, page=0)
                )
            ]
        )
        
        # Create a store with a mock collection
        store = ChromaStore(persist_directory=temp_dir)
        store.collection = MagicMock()
        
        # Add the document
        store.add_document(doc)
        
        # Check that the model's encode method was called correctly
        # It should be called once with a list of 2 text strings
        mock_model.encode.assert_called_once_with(ANY) # Check it was called
        assert len(mock_model.encode.call_args[0][0]) == 2 # Check it got 2 texts
        
        # Check that the collection.add method was called
        store.collection.add.assert_called_once()
        
        # Check that embeddings were created for both elements
        call_args = store.collection.add.call_args[1]
        assert len(call_args["documents"]) == 2
        assert len(call_args["metadatas"]) == 2
        assert np.array(call_args["embeddings"]).shape == (2, 384) # Check shape is (num_elements, embedding_dim)
        assert len(call_args["ids"]) == 2
        
        # Check that metadata was correctly set
        assert call_args["metadatas"][0]["document_id"] == "test-doc-1"
        assert call_args["metadatas"][0]["element_type"] == "text"
        assert call_args["metadatas"][1]["element_type"] == "table"
    
    @patch("mmrag.vectordb.chroma.SentenceTransformer") # Patch where it's looked up
    def test_query(self, mock_transformer, temp_dir):
        """Test querying the vector store."""
        # Create a mock embedding model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(384)
        mock_transformer.return_value = mock_model
        
        # Create a store with a mock collection
        store = ChromaStore(persist_directory=temp_dir)
        store.collection = MagicMock()
        
        # Mock the collection.query method
        mock_results = {
            "ids": [["doc_1_text_1", "doc_2_text_1"]],
            "documents": [["This is text from doc 1", "This is text from doc 2"]],
            "metadatas": [[
                {"document_id": "doc-1", "element_id": "text-1", "element_type": "text"},
                {"document_id": "doc-2", "element_id": "text-1", "element_type": "text"}
            ]],
            "distances": [[0.1, 0.2]]
        }
        store.collection.query.return_value = mock_results
        
        # Query the store
        results = store.query("test query", n_results=2)
        
        # Check that the model was used to encode the query
        mock_model.encode.assert_called_once()
        
        # Check that collection.query was called with correct parameters
        store.collection.query.assert_called_once()
        call_kwargs = store.collection.query.call_args[1]
        assert call_kwargs["n_results"] == 2
        
        # Check that results match the mock
        assert results == mock_results
    
    @patch("mmrag.vectordb.chroma.SentenceTransformer") # Patch where it's looked up
    def test_delete_document(self, mock_transformer, temp_dir):
        """Test deleting a document from the store."""
        # Create a store with a mock collection
        store = ChromaStore(persist_directory=temp_dir)
        store.collection = MagicMock()
        
        # Delete a document
        store.delete_document("test-doc-1")
        
        # Check that collection.delete was called with correct parameters
        store.collection.delete.assert_called_once()
        call_kwargs = store.collection.delete.call_args[1]
        assert call_kwargs["where"] == {"document_id": "test-doc-1"}
    
    def test_get_element_text(self, temp_dir):
        """Test text extraction from different element types."""
        store = ChromaStore(persist_directory=temp_dir)
        
        # Test text element
        text_element = TextElement(
            element_id="text-1",
            content="This is a test text element.",
            bbox=BoundingBox(x0=0, y0=0, x1=100, y1=20, page=0)
        )
        text_result = store._get_element_text(text_element)
        assert text_result == "This is a test text element."
        
        # Test table element
        table_element = TableElement(
            element_id="table-1",
            content=[["Header 1", "Header 2"], ["Data 1", "Data 2"]],
            bbox=BoundingBox(x0=0, y0=30, x1=100, y1=80, page=0)
        )
        table_result = store._get_element_text(table_element)
        assert "Header 1 | Header 2" in table_result
        assert "Data 1 | Data 2" in table_result
