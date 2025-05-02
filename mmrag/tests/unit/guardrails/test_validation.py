# tests/unit/guardrails/test_validation.py
import pytest
import json
from mmrag.guardrails.validation import validate_json_dict_output, attempt_json_repair

# --- Test attempt_json_repair ---

def test_attempt_json_repair_valid():
    valid_json = '{"key": "value"}'
    assert attempt_json_repair(valid_json) == valid_json

def test_attempt_json_repair_markdown():
    markdown_json = '```json\n{"key": "value"}\n```'
    expected_json = '{"key": "value"}'
    assert attempt_json_repair(markdown_json) == expected_json

def test_attempt_json_repair_missing_quotes():
    invalid_json = '{key: "value"}'
    expected_json = '{"key": "value"}'
    assert attempt_json_repair(invalid_json) == expected_json

def test_attempt_json_repair_single_quotes():
    invalid_json = "{'key': 'value'}"
    expected_json = '{"key": "value"}'
    assert attempt_json_repair(invalid_json) == expected_json

def test_attempt_json_repair_trailing_comma():
    invalid_json = '{"key": "value",}'
    expected_json = '{"key": "value"}'
    assert attempt_json_repair(invalid_json) == expected_json

def test_attempt_json_repair_unrepairable():
    unrepairable_json = '{"key": "value' # Missing closing brace
    assert attempt_json_repair(unrepairable_json) is None

# --- Test validate_json_dict_output (with schema) ---

SAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"}
    },
    "required": ["name"]
}

def test_validate_dict_valid_json_valid_schema():
    valid_json = '{"name": "Alice", "age": 30}'
    result = validate_json_dict_output(valid_json, "test_field", schema=SAMPLE_SCHEMA)
    assert json.loads(result) == {"name": "Alice", "age": 30}

def test_validate_dict_valid_json_invalid_schema_missing_required():
    invalid_json = '{"age": 30}' # Missing required 'name'
    # Should still return the JSON but log a warning (can't assert log here easily)
    result = validate_json_dict_output(invalid_json, "test_field", schema=SAMPLE_SCHEMA)
    assert json.loads(result) == {"age": 30}

def test_validate_dict_valid_json_invalid_schema_wrong_type():
    invalid_json = '{"name": "Alice", "age": "thirty"}' # age should be number
    result = validate_json_dict_output(invalid_json, "test_field", schema=SAMPLE_SCHEMA)
    assert json.loads(result) == {"name": "Alice", "age": "thirty"}

def test_validate_dict_repaired_json_valid_schema():
    repaired_json = "{name: 'Bob', age: 40,}"
    expected_data = {"name": "Bob", "age": 40}
    result = validate_json_dict_output(repaired_json, "test_field", schema=SAMPLE_SCHEMA)
    assert json.loads(result) == expected_data

def test_validate_dict_repaired_json_invalid_schema():
    repaired_json = "{age: 'forty',}" # Missing name, wrong type
    expected_data = {"age": "forty"}
    result = validate_json_dict_output(repaired_json, "test_field", schema=SAMPLE_SCHEMA)
    assert json.loads(result) == expected_data