#!/usr/bin/env python3
"""
Regression test for the closed-form Weyl vector-block invariants.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import weyl_vector_block_formula as wvf


def test_closed_form_formula() -> dict[str, object]:
    rows = []
    for alpha_ratio in wvf.DEFAULT_ALPHAS:
        row = wvf.compare_formula(alpha_ratio)
        rows.append(row)
    max_diff = max(row["max_coeff_diff"] for row in rows)
    return {
        "test": "weyl_vector_closed_form",
        "rows": rows,
        "max_coeff_diff": max_diff,
        "pass": max_diff < 1.0e-10,
    }


def test_trace_dropped_closed_form_formula() -> dict[str, object]:
    rows = []
    for alpha_ratio in wvf.DEFAULT_ALPHAS:
        row = wvf.compare_trace_dropped_formula(alpha_ratio)
        rows.append(row)
    max_diff = max(row["max_coeff_diff"] for row in rows)
    return {
        "test": "weyl_vector_trace_dropped_closed_form",
        "rows": rows,
        "max_coeff_diff": max_diff,
        "pass": max_diff < 1.0e-10,
    }


def run_all_tests() -> dict[str, object]:
    result_full = test_closed_form_formula()
    result_drop = test_trace_dropped_closed_form_formula()
    status_full = "PASS" if result_full["pass"] else "FAIL"
    status_drop = "PASS" if result_drop["pass"] else "FAIL"
    print("Running Weyl vector-block closed-form test...")
    print(f"  full max coeff diff: {result_full['max_coeff_diff']:.3e} [{status_full}]")
    print(
        f"  trace-dropped max coeff diff: {result_drop['max_coeff_diff']:.3e} [{status_drop}]"
    )
    return {
        "weyl_vector_closed_form": result_full,
        "weyl_vector_trace_dropped_closed_form": result_drop,
    }


if __name__ == "__main__":
    run_all_tests()
