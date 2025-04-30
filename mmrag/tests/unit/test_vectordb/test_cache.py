# tests/unit/test_vectordb/test_cache.py
import pytest
import os
import tempfile
import numpy as np
from pathlib import Path

from mmrag.vectordb.cache import EmbeddingCache

class TestEmbeddingCache:
    """Test suite for the embedding cache."""
    
    def test_cache_initialization(self):
        """Test that the cache initializes correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache(cache_dir=temp_dir)
            assert cache.cache_dir == Path(temp_dir)
            assert hasattr(cache, "get_embedding")
    
    def test_get_cache_key(self):
        """Test cache key generation."""
        cache = EmbeddingCache()
        
        # Same text should produce same key
        text = "This is a test text"
        key1 = cache._get_cache_key(text)
        key2 = cache._get_cache_key(text)
        assert key1 == key2
        
        # Different texts should produce different keys
        text2 = "This is a different text"
        key3 = cache._get_cache_key(text2)
        assert key1 != key3
    
    def test_save_and_retrieve_embedding(self):
        """Test saving and retrieving embeddings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache(cache_dir=temp_dir)
            
            # Create a test embedding
            text = "This is a test text"
            embedding = np.random.rand(384).tolist()  # Random 384-dim embedding
            model_name = "test-model"
            
            # Save the embedding
            cache.save_embedding(text, embedding, model_name)
            
            # Retrieve the embedding
            cached_embedding = cache.get_embedding(text, model_name)
            
            # Check that retrieved embedding matches saved embedding
            assert cached_embedding is not None
            assert len(cached_embedding) == len(embedding)
            assert all(abs(a - b) < 1e-6 for a, b in zip(cached_embedding, embedding))
            
            # Check for non-existent embedding
            nonexistent = cache.get_embedding("This text doesn't exist", model_name)
            assert nonexistent is None
    
    def test_lru_cache_behavior(self):
        """Test the LRU cache behavior."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a cache with small LRU size
            cache = EmbeddingCache(cache_dir=temp_dir)
            
            # Monkey-patch the LRU cache size for testing
            # This is normally set in __init__ with @functools.lru_cache(maxsize=1000)
            # but we can't easily test it without this patch
            cache.get_embedding.cache_clear()

            # Save some test embeddings
            model_name = "test-model"
            embeddings = {}

            for i in range(10):
                text = f"Text {i}"
                embedding = np.random.rand(384).tolist()
                embeddings[text] = embedding
                cache.save_embedding(text, embedding, model_name)
