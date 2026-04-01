#!/usr/bin/env python3
"""
Regression tests for the stable pre-GSO Bose-Fermi cancellation scan.
"""

from __future__ import annotations

import functools
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bose_fermi_cancellation_scan as bfcs
import single_cylinder_integrand as sci


@functools.lru_cache(maxsize=1)
def cached_scan() -> dict[str, object]:
    return bfcs.default_scan()


def test_log_polar_matches_safe_direct_ratio() -> dict[str, object]:
    row = bfcs.sector_ratio_log_polar(8, 0.7, 0.31, 1, 1)
    direct = sci.cylinder_trace_data(8, 0.7, 0.31, left_spin_sign=1, right_spin_sign=1)[
        "prototype_ratio"
    ]
    reconstructed = row["ratio_complex_if_safe"]
    abs_error = abs(reconstructed - direct)
    rel_error = abs_error / max(1.0, abs(direct))
    return {
        "test": "log_polar_matches_safe_direct_ratio",
        "abs_error": float(abs_error),
        "rel_error": float(rel_error),
        "pass": rel_error < 1.0e-12,
    }


def test_pre_gso_scan_does_not_cancel() -> dict[str, object]:
    summary = cached_scan()
    closest = summary["closest_to_cancellation"]
    return {
        "test": "pre_gso_scan_does_not_cancel",
        "closest": closest,
        "pass": closest["distance_to_one"] > 0.25,
    }


def test_large_ratio_log_remains_finite() -> dict[str, object]:
    row = bfcs.sector_ratio_log_polar(128, 2.0, 0.17, 1, 1)
    finite = math.isfinite(row["log_abs_ratio"]) and math.isfinite(row["phase_ratio"])
    return {
        "test": "large_ratio_log_remains_finite",
        "row": row,
        "pass": finite and abs(row["log_abs_ratio"]) > 10.0,
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "log_polar_matches_safe_direct_ratio": test_log_polar_matches_safe_direct_ratio(),
        "pre_gso_scan_does_not_cancel": test_pre_gso_scan_does_not_cancel(),
        "large_ratio_log_remains_finite": test_large_ratio_log_remains_finite(),
    }
    passed = sum(1 for result in results.values() if result.get("pass"))
    total = len(results)

    print("=" * 96)
    print("BOSE-FERMI CANCELLATION SCAN TESTS")
    print("=" * 96)
    for name, result in results.items():
        status = "PASS" if result.get("pass") else "FAIL"
        print(f"{name:40s} [{status}]")
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
