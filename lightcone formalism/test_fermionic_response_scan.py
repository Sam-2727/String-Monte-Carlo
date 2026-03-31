#!/usr/bin/env python3
"""
Regression tests for the pure fermionic response scan.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fermionic_response_scan as frs


def test_default_scan_relations() -> dict[str, object]:
    report = frs.run_scan()
    return {
        "test": "default_scan_relations",
        "max_abs_graviton_delta": report["max_abs_graviton_delta"],
        "max_abs_zero_channel": report["max_abs_zero_channel"],
        "max_mixed_ratio_error": report["max_mixed_ratio_error"],
        "max_lambda_sq_ratio_error": report["max_lambda_sq_ratio_error"],
        "max_diag_closed_form_error": report["max_diag_closed_form_error"],
        "monotone_decreasing_diag": report["monotone_decreasing_diag"],
        "pass": bool(report["all_checks_pass"]),
    }


def test_benchmark_grid_values() -> dict[str, object]:
    report = frs.run_scan(lambdas=[0.25, 0.4, 0.5])
    rows = {round(float(row["lambda"]), 12): row for row in report["rows"]}
    values = {
        0.25: 8.418729120241368,
        0.4: 5.3879866369544605,
        0.5: 3.7416573867739413,
    }
    max_abs_error = 0.0
    for lambda_ratio, target in values.items():
        diag = complex(rows[round(lambda_ratio, 12)]["diag_qq"])
        max_abs_error = max(
            max_abs_error,
            abs(diag.real - target) + abs(diag.imag),
        )

    return {
        "test": "benchmark_grid_values",
        "max_abs_error": float(max_abs_error),
        "pass": max_abs_error < 1.0e-12,
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "default_scan_relations": test_default_scan_relations(),
        "benchmark_grid_values": test_benchmark_grid_values(),
    }

    print("Running default response-scan relation test...")
    first = results["default_scan_relations"]
    print(
        f"  max |R_delta| = {first['max_abs_graviton_delta']:.3e}\n"
        f"  max zero channel = {first['max_abs_zero_channel']:.3e}\n"
        f"  max |mixed/diag - 1/2| = {first['max_mixed_ratio_error']:.3e}\n"
        f"  max |lambda^2 par23/diag - 1| = {first['max_lambda_sq_ratio_error']:.3e}\n"
        f"  max diag closed-form error = {first['max_diag_closed_form_error']:.3e}\n"
        f"  monotone decreasing diag = {first['monotone_decreasing_diag']} "
        f"[{'PASS' if first['pass'] else 'FAIL'}]"
    )

    print("\nRunning benchmark-grid value test...")
    second = results["benchmark_grid_values"]
    print(
        f"  max diag-qq error = {second['max_abs_error']:.3e} "
        f"[{'PASS' if second['pass'] else 'FAIL'}]"
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for result in results.values() if result.get("pass"))
    total = len(results)
    print(f"  {passed}/{total} tests passed")
    return results


if __name__ == "__main__":
    run_all_tests()
