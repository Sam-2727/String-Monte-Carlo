#!/usr/bin/env python3
"""
Regression tests for the bosonic tachyon prefactor remainder.
"""

from __future__ import annotations

import functools
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tachyon_prefactor_remainder as tpr


@functools.lru_cache(maxsize=1)
def cached_report() -> dict[str, object]:
    return tpr.full_report()


def test_family_prefactor_targets_match_extrapolations() -> dict[str, object]:
    report = cached_report()
    return {
        "test": "family_prefactor_targets_match_extrapolations",
        "max_abs_error": report["summary"]["max_abs_error"],
        "max_rel_error": report["summary"]["max_rel_error"],
        "pass": (
            report["summary"]["max_abs_error"] < 2.0e-5
            and report["summary"]["max_rel_error"] < 8.0e-7
        ),
    }


def test_largest_scale_rows_are_close_to_targets() -> dict[str, object]:
    report = cached_report()
    return {
        "test": "largest_scale_rows_are_close_to_targets",
        "max_abs_error": report["summary"]["max_final_row_abs_error"],
        "pass": report["summary"]["max_final_row_abs_error"] < 1.0e-4,
    }


def test_targets_reduce_to_tail_constant_plus_family_piece() -> dict[str, object]:
    tail_constant = tpr.invariant_tail_constant()
    family_errors = {}
    for row in cached_report()["families"]:
        reconstructed = (
            tail_constant
            + row["family_log_polynomial_piece"]
            - row["log_mu_squared"]
        )
        family_errors[row["family"]] = abs(reconstructed - row["prefactor_target"])
    return {
        "test": "targets_reduce_to_tail_constant_plus_family_piece",
        "family_errors": family_errors,
        "pass": max(family_errors.values()) < 1.0e-12,
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "family_prefactor_targets_match_extrapolations": test_family_prefactor_targets_match_extrapolations(),
        "largest_scale_rows_are_close_to_targets": test_largest_scale_rows_are_close_to_targets(),
        "targets_reduce_to_tail_constant_plus_family_piece": test_targets_reduce_to_tail_constant_plus_family_piece(),
    }
    passed = sum(1 for item in results.values() if item.get("pass"))
    total = len(results)
    print("=" * 96)
    print("TACHYON PREFACTOR REMAINDER TESTS")
    print("=" * 96)
    for name, result in results.items():
        status = "PASS" if result.get("pass") else "FAIL"
        print(f"{name:52s} [{status}]")
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
