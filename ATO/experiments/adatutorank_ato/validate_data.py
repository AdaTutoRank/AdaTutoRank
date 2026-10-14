#!/usr/bin/env python3
"""Validate the parquet contract required by AdaTutoRank-ATO."""

from __future__ import annotations

import sys

import pyarrow.parquet as pq


REQUIRED_EXTRA = {"query", "initial_list", "rubrics", "scenario"}
FORBIDDEN_PRECOMPUTED = {"rubric_set", "rubric_set_reward"}


def validate(path: str) -> None:
    table = pq.read_table(path)
    required_columns = {"data_source", "prompt", "extra_info"}
    missing_columns = required_columns - set(table.column_names)
    if missing_columns:
        raise ValueError(f"{path}: missing columns {sorted(missing_columns)}")

    sources = set(table["data_source"].to_pylist())
    if sources != {"set_selection"}:
        raise ValueError(f"{path}: unexpected data_source values {sorted(sources)}")

    scenarios: dict[str, int] = {}
    for row_number, extra in enumerate(table["extra_info"].to_pylist()):
        extra = extra or {}
        missing = REQUIRED_EXTRA - set(extra)
        if missing:
            raise ValueError(f"{path}: row {row_number} missing extra_info {sorted(missing)}")
        forbidden = FORBIDDEN_PRECOMPUTED & set(extra)
        if forbidden:
            raise ValueError(
                f"{path}: row {row_number} contains precomputed fields {sorted(forbidden)}; "
                "ATO requires current-policy selection"
            )
        scenario = str(extra["scenario"]).lower()
        if scenario not in {"rag", "deepresearch"}:
            raise ValueError(f"{path}: row {row_number} has invalid scenario {scenario!r}")
        if scenario == "deepresearch" and not str(extra.get("thinking") or "").strip():
            raise ValueError(f"{path}: deepresearch row {row_number} has no query intent")
        scenarios[scenario] = scenarios.get(scenario, 0) + 1

    print(f"{path}: {table.num_rows} rows, scenarios={scenarios}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} PARQUET [PARQUET ...]")
    for parquet_path in sys.argv[1:]:
        validate(parquet_path)
