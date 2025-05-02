"""DSPy processors with guardrails for document processing."""

import json
import logging
from typing import Dict, List, Optional, Union

import dspy
from mmrag.guardrails.validation import validate_json_dict_output, validate_json_list_output

from mmrag.guardrails.signatures import (
    SummarySignature,
    TopicsSignature,
    EntitiesSignature,
    TableExtractionSignature,
    TextAnalysisSignature,
    VisualElementSignature,
)

logger = logging.getLogger(__name__)

# --- Define Schemas for Validation ---

TOPICS_SCHEMA = {"type": "array", "items": {"type": "string"}}

ENTITIES_SCHEMA = {
    "type": "object",
    "patternProperties": {
        "^.*$": {"type": "array", "items": {"type": "string"}}
    },
    "additionalProperties": False # Disallow properties not matching the pattern
}

TEXT_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "A brief summary of the text."},
        "sentiment": {"type": "string", "description": "Overall sentiment (e.g., positive, negative, neutral)."},
        "purpose": {"type": "string", "description": "The likely purpose of this text segment (e.g., heading, paragraph, list item)."}
    },
    "required": ["summary", "sentiment", "purpose"]
}

TABLE_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string", "description": "A brief description of the table's content."},
        "insights": {"type": "array", "items": {"type": "string"}, "description": "Key insights derived from the table data."},
        "trends": {"type": "array", "items": {"type": "string"}, "description": "Notable trends observed in the table data."}
    },
    "required": ["description"] # Only description is strictly required
}

VISUAL_ANALYSIS_SCHEMA = TEXT_ANALYSIS_SCHEMA # Reuse text analysis schema for now


class DocumentProcessingModule(dspy.Module):
    """DEPRECATED: Use specific modules like SummaryModule, TopicsModule, etc."""
    def __init__(self):
        super().__init__()
        logger.warning("DocumentProcessingModule is deprecated. Use specific analysis modules.")
        # You might keep a simple predictor if needed for basic structure,
        # but the detailed analysis is moved to other modules.
        # self.predictor = dspy.Predict(SimpleStructureSignature) # Example

    def forward(self, *args, **kwargs):
        raise NotImplementedError("DocumentProcessingModule is deprecated.")


class SummaryModule(dspy.Module):
    """Generates document summary using DSPy."""
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(SummarySignature)

    def forward(self, text: str):
        result = self.predictor(text=text)
        # Basic validation: ensure summary is not empty
        if not result.summary or not result.summary.strip():
            logger.warning("LLM returned empty summary.")
            result.summary = "Summary could not be generated."
        return result


class TopicsModule(dspy.Module):
    """Extracts key topics using DSPy."""
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(TopicsSignature)

    def forward(self, text: str):
        result = self.predictor(text=text)
        # Validate the output string before returning
        validated_result = validate_json_list_output(
            result.topics_json,
            "topics_json"
            # Note: jsonschema currently not used for lists in this implementation
        )
        result.topics_json = validated_result
        return validated_result


class EntitiesModule(dspy.Module):
    """Extracts named entities using DSPy."""
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(EntitiesSignature)

    def forward(self, text: str):
        result = self.predictor(text=text)
        # Validate the output string before returning
        validated_result = validate_json_dict_output(
            result.entities_json,
            "entities_json",
            schema=ENTITIES_SCHEMA
        )
        result.entities_json = validated_result
        return validated_result


class TextAnalysisModule(dspy.Module):
    """Analyzes text elements using DSPy."""
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(TextAnalysisSignature)

    def forward(self, text: str, context: str = ""):
        result = self.predictor(text=text, context=context)
        # Validate the output string before returning
        validated_result = validate_json_dict_output(
            result.analysis_json,
            "analysis_json",
            schema=TEXT_ANALYSIS_SCHEMA
        )
        result.analysis_json = validated_result
        return validated_result


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
        
        # Validate the output string before returning
        validated_result = validate_json_dict_output(
            result.analysis_json,
            "analysis_json",
            schema=TABLE_ANALYSIS_SCHEMA
        )
        result.analysis_json = validated_result
        return validated_result
    


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
        
        # Validate the output string before returning
        validated_result = validate_json_dict_output(
            result.analysis_json,
            "analysis_json",
            schema=VISUAL_ANALYSIS_SCHEMA # Using text schema for now
        )
        result.analysis_json = validated_result
        return validated_result