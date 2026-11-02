# Day 28 — Data Quality and Schema Diagnostics

## Objective

Report row-level schema problems, missing values and duplicate records without changing user data.

## Executed results

```text
Project: IYB_INTERNSHIP

{'rows': 130, 'valid_rows': 130, 'invalid_rows': 0, 'duplicate_rows': 0, 'missing_cells': 0, 'issues': []}

{
  "rows": 3,
  "valid_rows": 1,
  "invalid_rows": 2,
  "duplicate_rows": 0,
  "missing_cells": 1,
  "issues": [
    {
      "row": 2,
      "issue": "Row 2, age: expected an integer between 15 and 22"
    },
    {
      "row": 3,
      "issue": "Row 3, school: expected one of ['GP', 'MS']"
    }
  ]
}

519
...                                                                      [100%]
3 passed in 1.43s


```

## Interpretation

The reference records conform to the serving schema. Explicit malformed test cases return row locations and correction guidance without echoing raw input values.

## Limitations

Completeness and domain validity cannot establish factual correctness or educational suitability. Duplicate students cannot be identified without identifiers, which this project does not collect.
