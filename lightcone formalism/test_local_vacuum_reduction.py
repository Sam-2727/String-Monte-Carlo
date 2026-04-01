#!/usr/bin/env python3
"""
Regression tests for the local Xi_loc nonzero-mode vacuum reduction.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import local_vacuum_reduction as lvr


def test_canonical_xi_two_point_scalar_formula() -> dict[str, object]:
    pairs = [(16, 24), (32, 48), (128, 192)]
    max_abs_error = 0.0
    rows = []
    for n1, n2 in pairs:
        numeric = lvr.canonical_xi_two_point_scalar(n1, n2)
        closed = lvr.canonical_xi_two_point_scalar_closed_form(n1, n2)
        error = abs(numeric - closed)
        max_abs_error = max(max_abs_error, error)
        rows.append(
            {
                "n1": n1,
                "n2": n2,
                "numeric": numeric,
                "closed_form": closed,
                "abs_error": error,
            }
        )
    return {
        "test": "canonical_xi_two_point_scalar_formula",
        "rows": rows,
        "max_abs_error": max_abs_error,
        "pass": max_abs_error < 1.0e-12,
    }


def test_benchmark_dilaton_quartic_sector_vanishes_after_contraction() -> dict[str, object]:
    benchmark = lvr.benchmark_vacuum_reduction_report()
    dilaton_rows = [
        row
        for row in benchmark["rows"]
        if row["channel"] == ["perp23", "perp23", "dilaton"]
    ]
    max_abs_value = max(abs(complex(row["contracted"])) for row in dilaton_rows)
    profiles = {row["lambda_ratio"]: row["profile"] for row in dilaton_rows}
    return {
        "test": "benchmark_dilaton_quartic_sector_vanishes_after_contraction",
        "profiles": profiles,
        "max_abs_value": float(max_abs_value),
        "pass": max_abs_value < 1.0e-12,
    }


def test_catalog_collapses_after_vacuum_contraction() -> dict[str, object]:
    qq_summary = lvr.contracted_catalog_summary(response_kind="qq", trace_dropped=True)
    delta_summary = lvr.contracted_catalog_summary(
        response_kind="delta",
        trace_dropped=True,
    )
    expected_qq_counts = {"reduced_only": 25, "vanishing": 100}
    expected_delta_counts = {"vanishing": 125}
    return {
        "test": "catalog_collapses_after_vacuum_contraction",
        "qq_counts": qq_summary["counts"],
        "delta_counts": delta_summary["counts"],
        "pass": (
            qq_summary["counts"] == expected_qq_counts
            and delta_summary["counts"] == expected_delta_counts
        ),
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "canonical_xi_two_point_scalar_formula": test_canonical_xi_two_point_scalar_formula(),
        "benchmark_dilaton_quartic_sector_vanishes_after_contraction": test_benchmark_dilaton_quartic_sector_vanishes_after_contraction(),
        "catalog_collapses_after_vacuum_contraction": test_catalog_collapses_after_vacuum_contraction(),
    }
    passed = sum(1 for item in results.values() if item.get("pass"))
    total = len(results)

    print("=" * 96)
    print("LOCAL XI VACUUM REDUCTION TESTS")
    print("=" * 96)
    for name, result in results.items():
        status = "PASS" if result.get("pass") else "FAIL"
        print(f"{name:64s} [{status}]")
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
