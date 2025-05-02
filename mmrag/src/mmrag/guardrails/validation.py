"""Validation and repair utilities for LLM JSON output."""

import json
import logging
import re
from typing import Dict, List, Optional, Union, Tuple, Any

logger = logging.getLogger(__name__)

try:
    from jsonschema import validate as jsonschema_validate
    from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
except ImportError:
    jsonschema_validate = None
    JSONSchemaValidationError = None


def attempt_json_repair(json_str: str) -> Optional[str]:
    """Attempt to repair invalid JSON strings."""
    if not json_str:
        return "{}"

    # Try to find JSON within potential markdown code blocks
    match = re.search(r'```(?:json)?\s*(\{.*\}|\[.*\])\s*```', json_str, re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1)

    # Try to fix common JSON errors
    try:
        # Fix missing quotes around keys (simple cases)
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


def validate_json_dict_output(
    json_str: str,
    field_name: str,
    schema: Optional[Dict[str, Any]] = None
) -> str:
    """Validate and fix JSON dictionary output string."""
    parsed_data = None
    try:
        if json_str and json_str.strip():
            parsed_data = json.loads(json_str)
        else:
            return "{}"
    except json.JSONDecodeError:
        fixed_json = attempt_json_repair(json_str)
        if not fixed_json:
            logger.warning(f"Invalid JSON in {field_name} could not be repaired.")
            return "{}"
        try:
            parsed_data = json.loads(fixed_json)
        except json.JSONDecodeError:
            logger.warning(f"Repaired JSON in {field_name} still invalid.")
            return "{}"

    if not isinstance(parsed_data, dict):
        logger.warning(f"LLM output for {field_name} is not a dictionary. Returning empty dict.")
        return "{}"

    # Schema validation if schema is provided and jsonschema is installed
    if schema and jsonschema_validate:
        try:
            jsonschema_validate(instance=parsed_data, schema=schema)
        except JSONSchemaValidationError as e:
            logger.warning(f"JSON schema validation failed for {field_name}: {e.message}. Returning parsed/repaired data anyway.")
            # Fall through to return the potentially invalid data

    # Return the validated (or repaired but potentially schema-invalid) JSON string
    return json.dumps(parsed_data)


def validate_json_list_output(json_str: str, field_name: str) -> str:
    """Validate and fix JSON list output string."""
    try:
        if json_str and json_str.strip():
            data = json.loads(json_str)
        else:
            return "[]"
    except json.JSONDecodeError:
        # Repair attempts for lists are less common/reliable, fallback to empty
        logger.warning(f"Invalid JSON list in {field_name}, attempting repair (basic).")
        # Basic repair: try adding brackets if missing? Very heuristic.
        if not json_str.strip().startswith('['): json_str = '[' + json_str
        if not json_str.strip().endswith(']'): json_str = json_str + ']'
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse/repair JSON list for {field_name}.")
            return "[]"

    if not isinstance(data, list):
        logger.warning(f"LLM output for {field_name} is not a list. Returning empty list.")
        return "[]"
    return json.dumps(data) # Return validated/repaired JSON string