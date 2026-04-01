#!/usr/bin/env python3
"""
Regression tests for the broader nearest-neighbor local fermion family scan.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import local_nearest_neighbor_family_scan as lnnfs


@functools.lru_cache(maxsize=1)
def cached_single() -> dict[str, object]:
    return lnnfs.single_point_scan()


@functools.lru_cache(maxsize=1)
def cached_catalog() -> dict[str, object]:
    return lnnfs.full_catalog_scan()


def test_nearest_neighbor_family_is_benchmark_invariant() -> dict[str, object]:
    report = cached_single()
    return {
        "test": "nearest_neighbor_family_is_benchmark_invariant",
        "max_abs_error": report["max_abs_error"],
        "max_local_reduced_error": report["max_local_reduced_error"],
        "pass": (
            report["max_abs_error"] < 1.0e-12
            and report["max_local_reduced_error"] < 1.0e-12
        ),
    }


def test_nearest_neighbor_family_preserves_mixed_constraints() -> dict[str, object]:
    report = cached_single()
    return {
        "test": "nearest_neighbor_family_preserves_mixed_constraints",
        "max_theta_cm_error": report["max_theta_cm_error"],
        "max_lambda_error": report["max_lambda_error"],
        "pass": (
            report["max_theta_cm_error"] < 1.0e-12
            and report["max_lambda_error"] < 1.0e-12
        ),
    }


def test_nearest_neighbor_family_preserves_vacuum_catalog() -> dict[str, object]:
    report = cached_catalog()
    return {
        "test": "nearest_neighbor_family_preserves_vacuum_catalog",
        "max_qq_catalog_error": report["max_qq_catalog_error"],
        "max_delta_catalog_error": report["max_delta_catalog_error"],
        "pass": (
            report["max_qq_catalog_error"] < 1.0e-12
            and report["max_delta_catalog_error"] < 1.0e-12
        ),
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "nearest_neighbor_family_is_benchmark_invariant": test_nearest_neighbor_family_is_benchmark_invariant(),
        "nearest_neighbor_family_preserves_mixed_constraints": test_nearest_neighbor_family_preserves_mixed_constraints(),
        "nearest_neighbor_family_preserves_vacuum_catalog": test_nearest_neighbor_family_preserves_vacuum_catalog(),
    }
    passed = sum(1 for result in results.values() if result.get("pass"))
    total = len(results)

    print("=" * 104)
    print("LOCAL NEAREST-NEIGHBOR FAMILY SCAN TESTS")
    print("=" * 104)
    for name, result in results.items():
        status = "PASS" if result.get("pass") else "FAIL"
        print(f"{name:64s} [{status}]")
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
