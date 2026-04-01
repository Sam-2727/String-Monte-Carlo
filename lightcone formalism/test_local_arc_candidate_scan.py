#!/usr/bin/env python3
"""
Regression tests for the simple local arc-admixture family.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import local_arc_candidate_scan as lacs


@functools.lru_cache(maxsize=1)
def cached_single_point() -> dict[str, object]:
    return lacs.single_point_candidate_scan()


@functools.lru_cache(maxsize=1)
def cached_family_scan() -> dict[str, object]:
    return lacs.family_scan()


def test_single_point_arc_family_preserves_mixed_zero_modes() -> dict[str, object]:
    report = cached_single_point()
    return {
        "test": "single_point_arc_family_preserves_mixed_zero_modes",
        "max_theta_cm_error": report["max_theta_cm_error"],
        "max_lambda_error": report["max_lambda_error"],
        "pass": (
            report["max_theta_cm_error"] < 1.0e-12
            and report["max_lambda_error"] < 1.0e-12
        ),
    }


def test_single_point_arc_family_is_benchmark_invariant() -> dict[str, object]:
    report = cached_single_point()
    return {
        "test": "single_point_arc_family_is_benchmark_invariant",
        "max_abs_error": report["max_abs_error"],
        "max_local_reduced_error": report["max_local_reduced_error"],
        "pass": (
            report["max_abs_error"] < 1.0e-12
            and report["max_local_reduced_error"] < 1.0e-12
        ),
    }


def test_family_arc_scan_is_benchmark_invariant() -> dict[str, object]:
    report = cached_family_scan()
    return {
        "test": "family_arc_scan_is_benchmark_invariant",
        "max_abs_error": report["max_abs_error"],
        "max_local_reduced_error": report["max_local_reduced_error"],
        "pass": (
            report["max_abs_error"] < 1.0e-12
            and report["max_local_reduced_error"] < 1.0e-12
        ),
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "single_point_arc_family_preserves_mixed_zero_modes": test_single_point_arc_family_preserves_mixed_zero_modes(),
        "single_point_arc_family_is_benchmark_invariant": test_single_point_arc_family_is_benchmark_invariant(),
        "family_arc_scan_is_benchmark_invariant": test_family_arc_scan_is_benchmark_invariant(),
    }
    passed = sum(1 for result in results.values() if result.get("pass"))
    total = len(results)

    print("=" * 96)
    print("LOCAL ARC CANDIDATE SCAN TESTS")
    print("=" * 96)
    for name, result in results.items():
        status = "PASS" if result.get("pass") else "FAIL"
        print(f"{name:48s} [{status}]")
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
