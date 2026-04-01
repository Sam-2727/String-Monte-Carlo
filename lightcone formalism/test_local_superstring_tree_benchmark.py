#!/usr/bin/env python3
"""
Regression tests for the local three-point superstring benchmark.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import local_superstring_tree_benchmark as lstb


@functools.lru_cache(maxsize=1)
def cached_single_point() -> dict[str, object]:
    return lstb.compare_single_point(128, 192)


@functools.lru_cache(maxsize=1)
def cached_scan() -> dict[str, object]:
    return lstb.run_family_scan(
        alpha_prime=1.0,
        scales=[16, 32, 64, 128],
    )


def test_local_single_point_matches_analytic_target() -> dict[str, object]:
    report = cached_single_point()
    return {
        "test": "local_single_point_matches_analytic_target",
        "max_abs_error": report["max_abs_error"],
        "pass": report["max_abs_error"] < 1.0e-12,
    }


def test_local_single_point_matches_reduced_assembly() -> dict[str, object]:
    report = cached_single_point()
    return {
        "test": "local_single_point_matches_reduced_assembly",
        "max_local_reduced_error": report["max_local_reduced_error"],
        "pass": report["max_local_reduced_error"] < 1.0e-12,
    }


def test_local_family_scan_matches_analytic_target() -> dict[str, object]:
    report = cached_scan()
    return {
        "test": "local_family_scan_matches_analytic_target",
        "max_abs_error": report["max_abs_error"],
        "max_local_reduced_error": report["max_local_reduced_error"],
        "pass": (
            report["max_abs_error"] < 1.0e-12
            and report["max_local_reduced_error"] < 1.0e-12
        ),
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "local_single_point_matches_analytic_target": test_local_single_point_matches_analytic_target(),
        "local_single_point_matches_reduced_assembly": test_local_single_point_matches_reduced_assembly(),
        "local_family_scan_matches_analytic_target": test_local_family_scan_matches_analytic_target(),
    }
    passed = sum(1 for item in results.values() if item.get("pass"))
    total = len(results)

    print("=" * 96)
    print("LOCAL SUPERSTRING TREE BENCHMARK TESTS")
    print("=" * 96)
    for name, result in results.items():
        status = "PASS" if result.get("pass") else "FAIL"
        extra = (
            f"max_abs_error={result.get('max_abs_error', 0.0):.3e} "
            f"max_local_reduced_error={result.get('max_local_reduced_error', 0.0):.3e}"
        )
        print(f"{name:48s} [{status}] {extra}")
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
