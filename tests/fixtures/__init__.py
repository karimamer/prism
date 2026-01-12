"""
Test fixtures and utilities for regression testing.

This module provides:
- Golden fixture loading
- Output comparison with tolerances
- Test data generation
"""

import json
import math
from pathlib import Path
from typing import Any, Optional

# Fixtures directory
FIXTURES_DIR = Path(__file__).parent
GOLDEN_DIR = FIXTURES_DIR / "golden_outputs"


def load_golden_fixture(fixture_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load a golden test fixture by ID.

    Args:
        fixture_id: Fixture identifier (e.g., "sample_1")

    Returns:
        Tuple of (input_data, expected_output)

    Raises:
        FileNotFoundError: If fixture files don't exist
    """
    input_file = GOLDEN_DIR / f"{fixture_id}_input.json"
    output_file = GOLDEN_DIR / f"{fixture_id}_output.json"

    if not input_file.exists():
        raise FileNotFoundError(f"Input fixture not found: {input_file}")

    if not output_file.exists():
        raise FileNotFoundError(f"Output fixture not found: {output_file}")

    with open(input_file) as f:
        input_data = json.load(f)

    with open(output_file) as f:
        expected_output = json.load(f)

    return input_data, expected_output


def load_tolerance_config() -> dict[str, Any]:
    """
    Load tolerance configuration for comparisons.

    Returns:
        Tolerance configuration dictionary
    """
    config_file = GOLDEN_DIR / "tolerance_config.json"

    if not config_file.exists():
        # Return default config
        return {
            "tolerances": {
                "confidence": {
                    "type": "float",
                    "absolute_tolerance": 0.05,
                    "relative_tolerance": 0.1,
                },
                "processing_timestamp": {"type": "ignore"},
            }
        }

    with open(config_file) as f:
        return json.load(f)


def compare_values(
    actual: Any,
    expected: Any,
    field_name: str,
    tolerance_config: Optional[dict] = None,
) -> tuple[bool, str]:
    """
    Compare two values with tolerance configuration.

    Args:
        actual: Actual value
        expected: Expected value
        field_name: Name of the field being compared
        tolerance_config: Tolerance configuration

    Returns:
        Tuple of (match, error_message)
    """
    if tolerance_config is None:
        tolerance_config = load_tolerance_config()

    tolerances = tolerance_config.get("tolerances", {})

    # Check if field has specific tolerance
    field_tolerance = tolerances.get(field_name, {})
    tolerance_type = field_tolerance.get("type", "exact")

    # Handle ignore type
    if tolerance_type == "ignore":
        return True, ""

    # Handle exact match
    if tolerance_type == "exact":
        if actual == expected:
            return True, ""
        return False, f"{field_name}: expected {expected}, got {actual}"

    # Handle float tolerance
    if tolerance_type == "float":
        if not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
            return False, f"{field_name}: type mismatch"

        abs_tol = field_tolerance.get("absolute_tolerance", 0.0)
        rel_tol = field_tolerance.get("relative_tolerance", 0.0)

        # Check absolute tolerance
        if abs(actual - expected) <= abs_tol:
            return True, ""

        # Check relative tolerance
        if expected != 0 and abs((actual - expected) / expected) <= rel_tol:
            return True, ""

        return False, f"{field_name}: expected {expected}, got {actual} (tolerance exceeded)"

    # Handle integer tolerance
    if tolerance_type == "integer":
        if not isinstance(actual, int) or not isinstance(expected, int):
            return False, f"{field_name}: type mismatch"

        abs_tol = field_tolerance.get("absolute_tolerance", 0)

        if abs(actual - expected) <= abs_tol:
            return True, ""

        return False, f"{field_name}: expected {expected}, got {actual} (tolerance exceeded)"

    # Default: exact match
    if actual == expected:
        return True, ""

    return False, f"{field_name}: expected {expected}, got {actual}"


def compare_with_golden(
    actual_output: Any,
    expected_output: dict[str, Any],
    tolerance_config: Optional[dict] = None,
    mode: str = "strict",
) -> tuple[bool, list[str]]:
    """
    Compare actual output with golden fixture output.

    Args:
        actual_output: Actual output from system (dict or Pydantic model)
        expected_output: Expected output from golden fixture
        tolerance_config: Tolerance configuration
        mode: Comparison mode ("strict" or "lenient")

    Returns:
        Tuple of (matches, list_of_errors)
    """
    if tolerance_config is None:
        tolerance_config = load_tolerance_config()

    errors = []

    # Convert Pydantic model to dict if needed
    if hasattr(actual_output, "model_dump"):
        actual_dict = actual_output.model_dump()
    elif hasattr(actual_output, "dict"):
        actual_dict = actual_output.dict()
    else:
        actual_dict = actual_output

    # Compare key fields
    key_fields = ["text", "num_entities", "num_relations", "consensus_method"]

    for field in key_fields:
        if field in expected_output:
            if field not in actual_dict:
                errors.append(f"Missing field: {field}")
                continue

            matches, error = compare_values(
                actual_dict[field],
                expected_output[field],
                field,
                tolerance_config,
            )

            if not matches:
                errors.append(error)

    # Compare entities
    if "entities" in expected_output:
        if "entities" not in actual_dict:
            errors.append("Missing entities field")
        else:
            expected_entities = expected_output["entities"]
            actual_entities = actual_dict["entities"]

            # Check entity count (with tolerance)
            matches, error = compare_values(
                len(actual_entities),
                len(expected_entities),
                "num_entities",
                tolerance_config,
            )

            if not matches:
                errors.append(f"Entity count mismatch: {error}")

            # Compare each entity (order-independent)
            # This is simplified - in production you'd want fuzzy matching
            for i, expected_entity in enumerate(expected_entities):
                if i < len(actual_entities):
                    actual_entity = actual_entities[i]

                    # Compare entity fields
                    for efield in ["mention", "entity_id", "entity_type"]:
                        if efield in expected_entity:
                            exp_val = expected_entity[efield]
                            # Handle nested objects (like entity_type enum)
                            if isinstance(actual_entity, dict):
                                act_val = actual_entity.get(efield)
                            else:
                                act_val = getattr(actual_entity, efield, None)
                                # Handle Enum
                                if hasattr(act_val, "value"):
                                    act_val = act_val.value

                            matches, error = compare_values(
                                act_val,
                                exp_val,
                                f"entities[{i}].{efield}",
                                tolerance_config,
                            )

                            if not matches:
                                errors.append(error)

    return len(errors) == 0, errors


__all__ = [
    "load_golden_fixture",
    "load_tolerance_config",
    "compare_with_golden",
    "compare_values",
    "FIXTURES_DIR",
    "GOLDEN_DIR",
]
