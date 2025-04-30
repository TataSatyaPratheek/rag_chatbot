# src/mmrag/document_processing/factory.py
"""Document processor factory."""

from pathlib import Path
from typing import Union

from mmrag.document_processing.base import DocumentProcessor
from mmrag.document_processing.pdf import PDFProcessor
from mmrag.document_processing.ppt import PowerPointProcessor

def get_processor(document_path: Union[str, Path]) -> DocumentProcessor:
    """Get the appropriate processor for a document.
    
    Args:
        document_path: Path to the document.
        
    Returns:
        Document processor for the document type.
        
    Raises:
        ValueError: If no processor is available for the document type.
    """
    document_path = Path(document_path)
    suffix = document_path.suffix.lower()
    
    if suffix == ".pdf":
        return PDFProcessor()
    elif suffix in [".pptx", ".ppt"]:
        return PowerPointProcessor()
    else:
        raise ValueError(f"Unsupported document type: {suffix}")
