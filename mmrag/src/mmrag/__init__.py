"""Multimodal document processing for RAG applications."""

# Keep version in sync with pyproject.toml
__version__ = "0.3.0" 

from mmrag.config import config

# Expose only the config object and version at the top level
__all__ = ["config", "__version__"]
