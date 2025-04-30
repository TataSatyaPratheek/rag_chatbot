"""Configuration management for mmrag."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


class Config(BaseModel):
    """Configuration for the mmrag package."""

    # Base paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")
    
    # ChromaDB settings
    chroma_persist_directory: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / ".chroma"
    )
    
    # LLM settings (using Ollama by default for local processing)
    llm_base_url: str = Field(default="http://localhost:11434/api")
    llm_model: str = Field(default="llama2")
    
    # Embedding settings
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    
    # Document processing settings
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create a config from environment variables."""
        return cls(
            chroma_persist_directory=os.getenv("CHROMA_PERSIST_DIR", cls.model_fields["chroma_persist_directory"].default),
            llm_base_url=os.getenv("LLM_BASE_URL", cls.model_fields["llm_base_url"].default),
            llm_model=os.getenv("LLM_MODEL", cls.model_fields["llm_model"].default),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.model_fields["embedding_model"].default),
            chunk_size=int(os.getenv("CHUNK_SIZE", cls.model_fields["chunk_size"].default)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", cls.model_fields["chunk_overlap"].default)),
        )


# Default configuration instance
config = Config.from_env()
