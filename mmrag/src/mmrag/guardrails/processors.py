"""DSPy processors with guardrails for document processing."""

import json
import logging
import re
from typing import Dict, List, Optional, Union

import dspy

from mmrag.guardrails.signatures import (
    SummarySignature,
    TopicsSignature,
    EntitiesSignature,
    TableExtractionSignature,
    TextAnalysisSignature,
    VisualElementSignature,
)

logger = logging.getLogger(__name__)


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
        validated_result = self._validate_json_list_output(result, "topics_json")
        return validated_result


class EntitiesModule(dspy.Module):
    """Extracts named entities using DSPy."""
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(EntitiesSignature)

    def forward(self, text: str):
        result = self.predictor(text=text)
        validated_result = self._validate_json_dict_output(result, "entities_json")
        return validated_result


class TextAnalysisModule(dspy.Module):
    """Analyzes text elements using DSPy."""
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(TextAnalysisSignature)

    def forward(self, text: str, context: str = ""):
        result = self.predictor(text=text, context=context)
        validated_result = self._validate_json_dict_output(result, "analysis_json")
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
        
        validated_result = self._validate_json_dict_output(result, "analysis_json")
        
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
        
        validated_result = self._validate_json_dict_output(result, "analysis_json")
        
        return validated_result
    

# --- Helper methods for validation (can be part of a base class or utility) ---

def _validate_json_dict_output(self, result, field_name: str):
    """Validate and fix JSON dictionary output."""
    validated = result
    json_str = getattr(result, field_name, "{}")

    try:
        if json_str and json_str.strip():
            data = json.loads(json_str)
            if not isinstance(data, dict):
                setattr(validated, field_name, "{}")
                logger.warning(f"Invalid dict format in {field_name} fixed.")
        else:
            setattr(validated, field_name, "{}")
    except json.JSONDecodeError:
        fixed_json = self._attempt_json_repair(json_str)
        setattr(validated, field_name, fixed_json or "{}")
        if not fixed_json:
            logger.warning(f"Invalid JSON in {field_name} fixed.")
    return validated

def _validate_json_list_output(self, result, field_name: str):
    """Validate and fix JSON list output."""
    validated = result
    json_str = getattr(result, field_name, "[]")
    try:
        data = json.loads(json_str)
        if not isinstance(data, list):
            setattr(validated, field_name, "[]")
            logger.warning(f"Invalid list format in {field_name} fixed.")
    except json.JSONDecodeError:
        setattr(validated, field_name, "[]") # Fallback to empty list
        logger.warning(f"Invalid JSON in {field_name} fixed.")
    return validated

def _attempt_json_repair(self, json_str: str) -> Optional[str]:
    """Attempt to repair invalid JSON strings."""
    if not json_str:
        return "{}"

    # Try to find JSON within potential markdown code blocks
    match = re.search(r'```(json)?\s*(\{.*\}|\[.*\])\s*```', json_str, re.DOTALL)
    if match:
        json_str = match.group(2)

    # Try to fix common JSON errors
    try:
        # Fix missing quotes around keys
        fixed = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)
        # Fix single quotes to double quotes
        fixed = fixed.replace("'", '"')
        # Remove trailing commas (simple cases)
        fixed = re.sub(r',\s*([\}\]])', r'\1', fixed)
        # Validate
        json.loads(fixed)
        logger.info("Successfully repaired JSON string.")
        return fixed
    except Exception as e:
        logger.debug(f"JSON repair attempt failed: {e}")
        pass

    # If still not valid, return None
    return None

# Add validation methods to the modules
EntitiesModule._validate_json_dict_output = _validate_json_dict_output
EntitiesModule._attempt_json_repair = _attempt_json_repair
TopicsModule._validate_json_list_output = _validate_json_list_output
TopicsModule._attempt_json_repair = _attempt_json_repair # Needed if repair logic is complex
TextAnalysisModule._validate_json_dict_output = _validate_json_dict_output
TextAnalysisModule._attempt_json_repair = _attempt_json_repair
TableExtractionModule._validate_json_dict_output = _validate_json_dict_output
TableExtractionModule._attempt_json_repair = _attempt_json_repair
VisualElementModule._validate_json_dict_output = _validate_json_dict_output
VisualElementModule._attempt_json_repair = _attempt_json_repair