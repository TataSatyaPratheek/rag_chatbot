# src/mmrag/document_processing/factory.py
"""Document processor factory."""

from pathlib import Path
from typing import Union, Optional

from mmrag.document_processing.base import DocumentProcessor
from mmrag.document_processing.pdf import PDFProcessor
from mmrag.document_processing.ppt import PowerPointProcessor

def get_processor(
    document_path: Union[str, Path],
    extract_tables: bool = True,
    extract_images: bool = True,
    advanced_table_detection: bool = False,
    enable_enhanced_visual: bool = False,
    enable_llm_analysis: bool = False,
) -> DocumentProcessor:
    """Get the appropriate processor for a document.
    
    Args:
        document_path: Path to the document.
        extract_tables: Whether to extract tables.
        extract_images: Whether to extract images.
        advanced_table_detection: Whether to use advanced table detection.
        enable_enhanced_visual: Whether to use enhanced visual processing.
        enable_llm_analysis: Whether to enable LLM analysis.
        
    Returns:
        Document processor for the document type.
        
    Raises:
        ValueError: If no processor is available for the document type.
    """
    document_path = Path(document_path)
    suffix = document_path.suffix.lower()
    
    if suffix == ".pdf":
        return PDFProcessor(
            extract_tables=extract_tables,
            extract_images=extract_images,
            advanced_table_detection=advanced_table_detection,
            enable_enhanced_visual=enable_enhanced_visual,
            enable_llm_analysis=enable_llm_analysis
        )
    elif suffix in [".pptx", ".ppt"]:
        return PowerPointProcessor(
            extract_tables=extract_tables,
            extract_images=extract_images
        )
    else:
        raise ValueError(f"Unsupported document type: {suffix}")
