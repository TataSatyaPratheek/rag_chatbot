"""Utility functions for LlamaParse integration."""

import os
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

def check_llamaparse_api_key() -> bool:
    """Check if the LlamaParse API key is available.
    
    Returns:
        True if API key is found, False otherwise
    """
    return "LLAMA_CLOUD_API_KEY" in os.environ

def get_llamaparse_cost_estimate(
    file_path: str, 
    use_multimodal: bool = True,
    multimodal_model: str = "anthropic-sonnet-3.5"
) -> Dict:
    """Estimate the cost of processing a document with LlamaParse.
    
    This is a rough estimate based on file size. The actual cost will vary
    based on the document's complexity, the number of pages, and whether
    multimodal processing is used.
    
    Args:
        file_path: Path to the document
        use_multimodal: Whether to use multimodal processing
        multimodal_model: Model to use for multimodal processing
        
    Returns:
        Dictionary with cost estimates
    """
    try:
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Base cost per MB (approximate)
        base_cost_per_mb = 0.002
        
        # Additional cost for multimodal processing
        multimodal_multiplier = 3.0 if use_multimodal else 1.0
        model_multiplier = 1.5 if multimodal_model == "openai-gpt4o" else 1.0
        
        estimated_cost = file_size_mb * base_cost_per_mb * multimodal_multiplier * model_multiplier
        
        return {
            "file_size_mb": round(file_size_mb, 2),
            "estimated_cost_usd": round(estimated_cost, 4),
            "use_multimodal": use_multimodal,
            "multimodal_model": multimodal_model if use_multimodal else None,
            "note": "This is a rough estimate. Actual costs may vary."
        }
    except Exception as e:
        logger.warning(f"Error estimating LlamaParse cost: {e}")
        return {
            "error": str(e),
            "note": "Could not estimate cost. Please check the file path."
        }

def get_optimal_llamaparse_settings(file_path: str) -> Dict:
    """Get optimal LlamaParse settings based on document type and size.
    
    Args:
        file_path: Path to the document
        
    Returns:
        Dictionary with recommended settings
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    # Default settings
    settings = {
        "result_type": "markdown",
        "use_multimodal": True,
        "multimodal_model": "anthropic-sonnet-3.5",
        "timeout_seconds": 60,
    }
    
    # Adjust based on file type
    if file_ext == ".pdf":
        # PDFs often have complex layouts
        settings["use_multimodal"] = True
    elif file_ext in [".pptx", ".ppt"]:
        # Presentations are highly visual
        settings["use_multimodal"] = True
        settings["multimodal_model"] = "openai-gpt4o"  # Better for visual content
    elif file_ext in [".docx", ".doc"]:
        # Word docs may have simpler layouts
        settings["use_multimodal"] = file_size_mb > 1.0  # Only use for larger docs
    elif file_ext == ".xlsx":
        # Excel files are mostly tables
        settings["use_multimodal"] = False
    
    # Adjust timeout based on file size
    if file_size_mb > 10:
        settings["timeout_seconds"] = 120
    if file_size_mb > 30:
        settings["timeout_seconds"] = 240
    
    return settings

def validate_llamaparse_output(llamaparse_documents: List) -> Dict:
    """Validate the output from LlamaParse for quality and completeness.
    
    Args:
        llamaparse_documents: List of documents returned by LlamaParse
        
    Returns:
        Dictionary with validation results
    """
    results = {
        "valid": True,
        "issues": [],
        "statistics": {
            "document_count": len(llamaparse_documents),
            "empty_documents": 0,
            "text_length": 0,
        }
    }
    
    for i, doc in enumerate(llamaparse_documents):
        # Check for empty documents
        if not hasattr(doc, "text") or not doc.text or not doc.text.strip():
            results["issues"].append(f"Document {i} has no text content")
            results["statistics"]["empty_documents"] += 1
            continue
            
        # Check text length
        text_length = len(doc.text)
        results["statistics"]["text_length"] += text_length
        
        if text_length < 100:
            results["issues"].append(f"Document {i} has suspiciously short text ({text_length} chars)")
        
        # Check for encoding issues
        if "�" in doc.text:
            results["issues"].append(f"Document {i} has encoding issues (replacement characters)")
    
    # Set valid flag based on issues
    if results["issues"]:
        results["valid"] = False
    
    return results