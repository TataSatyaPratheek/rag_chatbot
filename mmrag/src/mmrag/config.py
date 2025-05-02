"""Configuration management for mmrag."""

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Load environment variables from .env file
load_dotenv()

class ProcessorType(str, Enum):
    LEGACY = "legacy"
    OPENPARSE = "openparse"
    DOCLING = "docling"
    LLAMAPARSE = "llamaparse"

class MMRAGSettings(BaseSettings):
    """Configuration for the mmrag package, loaded from env vars or .env file."""
    
    # ChromaDB settings
    chroma_persist_directory: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / ".chroma"
    )
    
    # LLM settings (using Ollama by default for local processing)
    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:latest" # Updated default model
    
    # Embedding settings
    embedding_model: str = "nomic-embed-text" # Changed default embedding model
    
    # Document processing settings
    processor_type: ProcessorType = ProcessorType.OPENPARSE
    extract_tables: bool = True
    extract_images: bool = True
    advanced_table_detection: bool = False
    enable_llm_analysis: bool = True
    openai_api_key: Optional[str] = None # For OpenParse semantic or other OpenAI features
    llama_cloud_api_key: Optional[str] = None # For LlamaParse
    memory_limit_mb: float = 1000.0 # Default memory limit in MB
    timeout_seconds: int = 60 # Default processing timeout
    
    # Pydantic-settings configuration
    model_config = SettingsConfigDict(
        env_prefix='MMRAG_', # Prefix for environment variables, e.g., MMRAG_LLM_MODEL
        env_file='.env',     # Load variables from .env file
        extra='ignore'       # Ignore extra fields from environment/dotenv
    )

# Default configuration instance, automatically loads from environment/.env
config = MMRAGSettings()
