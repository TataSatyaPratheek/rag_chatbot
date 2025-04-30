"""Table detection and extraction."""

import logging
import uuid
from typing import Dict, List, Optional, Tuple, Union

import fitz
import numpy as np
from pydantic import ValidationError

from mmrag.document_processing.base import BoundingBox, TableElement

logger = logging.getLogger(__name__)


class TableDetector:
    """Table detector for document pages."""
    
    def __init__(self, min_rows: int = 2, min_cols: int = 2):
        """Initialize the table detector.
        
        Args:
            min_rows: Minimum number of rows for a valid table.
            min_cols: Minimum number of columns for a valid table.
        """
        self.min_rows = min_rows
        self.min_cols = min_cols
    
    def detect_tables(self, page: fitz.Page, page_idx: int) -> List[TableElement]:
        """Detect tables on a page.
        
        This implementation uses a heuristic approach based on text alignment
        to detect potential tables. For production use, consider integrating
        with more sophisticated table detection libraries.
        
        Args:
            page: Page to process.
            page_idx: Index of the page.
            
        Returns:
            List of detected table elements.
        """
        table_elements = []
        
        # Get text spans with positions
        spans = page.get_text("dict")["blocks"]
        
        # Identify potential tabular structures based on alignment
        potential_tables = self._identify_potential_tables(spans)
        
        for i, table_data in enumerate(potential_tables):
            rows, bbox = table_data
            
            # Skip tables that are too small
            if len(rows) < self.min_rows or min(len(row) for row in rows) < self.min_cols:
                continue
            
            try:
                table_element = TableElement(
                    element_id=f"table-{page_idx}-{i}",
                    content=rows,
                    bbox=BoundingBox(
                        x0=bbox[0],
                        y0=bbox[1],
                        x1=bbox[2],
                        y1=bbox[3],
                        page=page_idx,
                    ),
                    metadata={
                        "num_rows": len(rows),
                        "num_cols": max(len(row) for row in rows),
                    },
                )
                table_elements.append(table_element)
            except ValidationError as e:
                logger.warning(f"Failed to create table element: {e}")
        
        return table_elements
    
    def _identify_potential_tables(self, blocks: List[Dict]) -> List[Tuple[List[List[str]], Tuple[float, float, float, float]]]:
        """Identify potential tables based on text alignment.
        
        This is a simplified approach for local-only processing.
        For production, consider using more sophisticated table detection methods.
        
        Args:
            blocks: Text blocks from PyMuPDF.
            
        Returns:
            List of (rows, bbox) tuples for potential tables.
        """
        potential_tables = []
        
        # Filter for blocks that might be part of tables
        text_blocks = []
        for block in blocks:
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text_blocks.append({
                            "text": span.get("text", "").strip(),
                            "bbox": span.get("bbox", (0, 0, 0, 0)),
                            "font_size": span.get("size", 0),
                        })
        
        # Skip if too few text blocks
        if len(text_blocks) < self.min_rows * self.min_cols:
            return potential_tables
        
        # Group text blocks by y-coordinate (rows)
        row_tolerance = 5  # pixels
        rows = {}
        
        for block in text_blocks:
            y_mid = (block["bbox"][1] + block["bbox"][3]) / 2
            y_bin = round(y_mid / row_tolerance) * row_tolerance
            
            if y_bin not in rows:
                rows[y_bin] = []
            
            rows[y_bin].append(block)
        
        # Sort rows by y-coordinate
        sorted_rows = sorted(rows.items(), key=lambda x: x[0])
        
        # Only consider areas with regular grid structure
        if len(sorted_rows) >= self.min_rows:
            # Check for column alignment
            for i in range(len(sorted_rows) - self.min_rows + 1):
                consecutive_rows = [row[1] for row in sorted_rows[i:i+self.min_rows]]
                
                # Check if all rows have similar number of cells
                row_lengths = [len(row) for row in consecutive_rows]
                if min(row_lengths) >= self.min_cols and max(row_lengths) <= min(row_lengths) * 1.5:
                    # Sort cells in each row by x-coordinate
                    for row in consecutive_rows:
                        row.sort(key=lambda x: x["bbox"][0])
                    
                    # Extract text content
                    table_rows = [[cell["text"] for cell in row] for row in consecutive_rows]
                    
                    # Calculate bounding box for the table
                    x0 = min(cell["bbox"][0] for row in consecutive_rows for cell in row)
                    y0 = min(cell["bbox"][1] for row in consecutive_rows for cell in row)
                    x1 = max(cell["bbox"][2] for row in consecutive_rows for cell in row)
                    y1 = max(cell["bbox"][3] for row in consecutive_rows for cell in row)
                    
                    potential_tables.append((table_rows, (x0, y0, x1, y1)))
        
        return potential_tables
