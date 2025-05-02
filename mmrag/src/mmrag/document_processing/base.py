"""Base classes for document processing."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable

from pydantic import BaseModel, Field
class BoundingBox(BaseModel):
    """Bounding box for an element on a page."""
    
    x0: float = Field(..., description="Left coordinate")
    y0: float = Field(..., description="Top coordinate")
    x1: float = Field(..., description="Right coordinate")
    y1: float = Field(..., description="Bottom coordinate")
    page: int = Field(..., description="Page number (0-indexed)")
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "page": self.page,
        }


class DocumentElement(BaseModel):
    """Base class for document elements."""
    
    element_id: str = Field(..., description="Unique identifier for the element")
    element_type: str = Field(..., description="Type of element (text, table, image, etc.)")
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box for the element")
    content: Any = Field(..., description="Content of the element")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TextElement(DocumentElement):
    """Text element in a document."""
    
    element_type: str = "text"
    content: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "element_id": "text-1",
                "element_type": "text",
                "bbox": {
                    "x0": 0.0,
                    "y0": 0.0,
                    "x1": 100.0,
                    "y1": 20.0,
                    "page": 0,
                },
                "content": "This is a text element",
                "metadata": {
                    "font_size": 12,
                    "is_bold": False,
                },
            }
        }


class TableElement(DocumentElement):
    """Table element in a document."""
    
    element_type: str = "table"
    content: List[List[str]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "element_id": "table-1",
                "element_type": "table",
                "bbox": {
                    "x0": 0.0,
                    "y0": 0.0,
                    "x1": 100.0,
                    "y1": 50.0,
                    "page": 0,
                },
                "content": [
                    ["Header 1", "Header 2"],
                    ["Value 1", "Value 2"],
                ],
                "metadata": {
                    "num_rows": 2,
                    "num_cols": 2,
                },
            }
        }


class ImageElement(DocumentElement):
    """Image element in a document."""
    
    element_type: str = "image"
    content: str  # Base64 encoded image or path
    
    class Config:
        json_schema_extra = {
            "example": {
                "element_id": "image-1",
                "element_type": "image",
                "bbox": {
                    "x0": 0.0,
                    "y0": 0.0,
                    "x1": 100.0,
                    "y1": 100.0,
                    "page": 0,
                },
                "content": "base64encodedstring",
                "metadata": {
                    "width": 100,
                    "height": 100,
                    "image_type": "png",
                },
            }
        }


class ChartElement(DocumentElement):
    """Chart element in a document."""
    
    element_type: str = "chart"
    content: str  # Description of the chart
    data: Optional[Dict[str, Any]] = None  # Extracted data from the chart
    
    class Config:
        json_schema_extra = {
            "example": {
                "element_id": "chart-1",
                "element_type": "chart",
                "bbox": {
                    "x0": 0.0,
                    "y0": 0.0,
                    "x1": 100.0,
                    "y1": 100.0,
                    "page": 0,
                },
                "content": "Bar chart showing sales by region",
                "data": {
                    "type": "bar",
                    "x_axis": ["North", "South", "East", "West"],
                    "y_axis": [10, 20, 15, 25],
                },
                "metadata": {
                    "chart_type": "bar",
                },
            }
        }


class ProcessedDocument(BaseModel):
    """Processed document with extracted elements."""
    
    document_id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Original filename")
    doc_type: str = Field(..., description="Document type (pdf, ppt, etc.)")
    elements: List[DocumentElement] = Field(default_factory=list, description="Extracted elements")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    analysis: Optional[Dict[str, Any]] = Field(None, description="LLM analysis of the document")
        
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.model_dump(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ProcessedDocument":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls(**data)


class DocumentProcessor(ABC):
    """Base class for document processors."""
    
    # Make process asynchronous and add progress callback
    @abstractmethod
    async def process( # type: ignore
        self,
        document_path: Union[str, Path],
        progress_callback: Optional[Callable[[str, Optional[float]], Awaitable[None]]] = None
     ) -> ProcessedDocument:
        """Process a document and extract elements."""
        pass
    
    @abstractmethod
    def supports(self, document_path: Union[str, Path]) -> bool:
        """Check if the processor supports the given document."""
        pass
