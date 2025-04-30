"""DSPy processors with guardrails for document processing."""

import json
import logging
import re
from typing import Dict, List, Optional, Union

import dspy

from mmrag.guardrails.signatures import (
    DocumentProcessingSignature,
    TableExtractionSignature,
    VisualElementSignature,
)

logger = logging.getLogger(__name__)


class DocumentProcessingModule(dspy.Module):
    """Document processing module with DSPy guardrails."""
    
    def __init__(self):
        super().__init__()
        # Use ChainOfThought for better reasoning and guardrails
        self.predictor = dspy.ChainOfThought(DocumentProcessingSignature)
    
    def forward(self, document_content: str, document_type: str, page_number: int = 0):
        """Process document content with guardrails."""
        # Process with guardrails through ChainOfThought
        result = self.predictor(
            document_content=document_content,
            document_type=document_type,
            page_number=page_number,
        )
        
        # Add validation guardrails
        validated_result = self.validate_output(result)
        
        return validated_result
    
    def validate_output(self, result):
        """Implement validation guardrails for the outputs."""
        validated = result
        
        # Validate potential tables
        try:
            if result.potential_tables:
                tables = json.loads(result.potential_tables)
                if not isinstance(tables, list):
                    # Fix format if not a list
                    validated.potential_tables = json.dumps([])
                    logger.warning("Invalid table format detected and corrected")
        except json.JSONDecodeError:
            # Fix if not valid JSON
            validated.potential_tables = json.dumps([])
            logger.warning("Invalid JSON in potential_tables fixed")
        
        # Validate visual elements
        try:
            if result.visual_elements:
                elements = json.loads(result.visual_elements)
                if not isinstance(elements, list):
                    # Fix format if not a list
                    validated.visual_elements = json.dumps([])
                    logger.warning("Invalid visual elements format detected and corrected")
        except json.JSONDecodeError:
            # Fix if not valid JSON
            validated.visual_elements = json.dumps([])
            logger.warning("Invalid JSON in visual_elements fixed")
        
        return validated


class TableExtractionModule(dspy.Module):
    """Table extraction module with DSPy guardrails."""
    
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(TableExtractionSignature)
    
    def forward(self, table_region: str, context: str = ""):
        """Extract table data with guardrails."""
        result = self.predictor(
            table_region=table_region,
            context=context,
        )
        
        validated_result = self.validate_output(result)
        
        return validated_result
    
    def validate_output(self, result):
        """Validate and fix table extraction output."""
        validated = result
        
        # Validate table data
        try:
            if result.table_data:
                table = json.loads(result.table_data)
                if not isinstance(table, list) or not all(isinstance(row, list) for row in table):
                    # Create empty table if invalid
                    validated.table_data = json.dumps([])
                    validated.row_count = "0"
                    validated.column_count = "0"
                    logger.warning("Invalid table data format detected and corrected")
                else:
                    # Ensure row and column counts match the data
                    validated.row_count = str(len(table))
                    max_cols = max(len(row) for row in table) if table else 0
                    validated.column_count = str(max_cols)
        except json.JSONDecodeError:
            # Fix if not valid JSON
            validated.table_data = json.dumps([])
            validated.row_count = "0"
            validated.column_count = "0"
            logger.warning("Invalid JSON in table_data fixed")
        
        return validated


class VisualElementModule(dspy.Module):
    """Visual element analysis module with DSPy guardrails."""
    
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(VisualElementSignature)
    
    def forward(self, element_type: str, surrounding_text: str):
        """Analyze visual element with guardrails."""
        result = self.predictor(
            element_type=element_type,
            surrounding_text=surrounding_text,
        )
        
        validated_result = self.validate_output(result)
        
        return validated_result
    
    def validate_output(self, result):
        """Validate and fix visual element output."""
        validated = result
        
        # Validate extracted data
        try:
            if result.extracted_data and result.extracted_data.strip() not in ["N/A", "None", ""]:
                # Try to parse as JSON
                json.loads(result.extracted_data)
                # If no exception, it's valid JSON
            else:
                # Set to empty object if not present or empty
                validated.extracted_data = "{}"
        except json.JSONDecodeError:
            # If not valid JSON, try to fix common issues
            fixed_json = self._attempt_json_repair(result.extracted_data)
            if fixed_json:
                validated.extracted_data = fixed_json
            else:
                validated.extracted_data = "{}"
                logger.warning("Invalid JSON in extracted_data fixed")
        
        return validated
    
    def _attempt_json_repair(self, json_str: str) -> Optional[str]:
        """Attempt to repair invalid JSON strings."""
        if not json_str:
            return "{}"
        
        # Try to fix common JSON errors
        try:
            # Fix missing quotes around keys
            fixed = re.sub(r'(\w+):', r'"\1":', json_str)
            # Fix single quotes
            fixed = fixed.replace("'", '"')
            # Validate
            json.loads(fixed)
            return fixed
        except:
            pass
        
        # If still not valid, return empty object
        return None
