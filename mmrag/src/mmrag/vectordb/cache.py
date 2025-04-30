# src/mmrag/vectordb/cache.py
import functools
import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Union

class EmbeddingCache:
    """Cache for text embeddings."""
    
    def __init__(self, cache_dir: Union[str, Path] = None):
        """Initialize the embedding cache."""
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache using functools.lru_cache
        self.get_embedding = functools.lru_cache(maxsize=1000)(self._get_embedding)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for the text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_embedding(self, text: str, model_name: str) -> List[float]:
        """Get embedding from persistent cache or compute it."""
        cache_key = self._get_cache_key(text)
        cache_path = self.cache_dir / f"{model_name}_{cache_key}.pkl"
        
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
        # Not in cache, return None
        return None
    
    def save_embedding(self, text: str, embedding: List[float], model_name: str) -> None:
        """Save embedding to persistent cache."""
        cache_key = self._get_cache_key(text)
        cache_path = self.cache_dir / f"{model_name}_{cache_key}.pkl"
        
        with open(cache_path, "wb") as f:
            pickle.dump(embedding, f)
