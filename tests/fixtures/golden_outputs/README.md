# Golden Test Fixtures

This directory contains golden test fixtures for regression testing.

## Purpose

Golden fixtures provide expected outputs that tests can compare against to ensure:
1. Output format stability across versions
2. No unintended changes to core functionality
3. Backwards compatibility is maintained

## Structure

Each fixture consists of a pair:
- `sample_input_N.json` - Input document
- `sample_output_N.json` - Expected output

## Tolerance Configuration

Some fields may have acceptable variance (e.g., confidence scores).
See `tolerance_config.json` for field-specific tolerances.

## Usage

```python
from tests.fixtures import load_golden_fixture, compare_with_golden

# Load fixture
input_data, expected_output = load_golden_fixture("sample_1")

# Process
actual_output = system.process_text(input_data["text"])

# Compare with tolerances
assert compare_with_golden(actual_output, expected_output, tolerances)
```

## Updating Fixtures

When intentionally changing output format:
1. Update the corresponding `sample_output_N.json`
2. Document the change in CHANGELOG.md
3. Update version number if breaking change

## Fixtures List

| ID | Description | Entities | Relations | Difficulty |
|----|-------------|----------|-----------|------------|
| sample_1 | Basic entity linking | 3 | 1 | Medium |
| sample_2 | Unicode and emoji | 2 | 0 | Hard |
| sample_3 | Overlapping entities | 4 | 2 | Hard |
| sample_4 | NIL entities | 2 | 0 | Medium |
| sample_5 | Long document | 10+ | 5+ | Hard |
