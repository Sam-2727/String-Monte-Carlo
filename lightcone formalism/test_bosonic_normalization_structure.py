#!/usr/bin/env python3
"""
Regression tests for bosonic normalization-structure diagnostics.
"""

from __future__ import annotations

import functools
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bosonic_normalization_structure as bns


@functools.lru_cache(maxsize=1)
def cached_summary() -> dict[str, object]:
    return bns.default_summary()


def test_invariant_tail_constant() -> dict[str, object]:
    summary = cached_summary()["invariant_tail"]
    constant = summary["constant"]
    return {
        "test": "invariant_tail_constant",
        "constant": constant,
        "rmse": summary["rmse"],
        "max_abs": summary["max_abs"],
        "pass": summary["rmse"] < 5.0e-7 and (-22.6 < constant < -22.3),
    }


def test_factorized_tail_coefficients_match_expected() -> dict[str, object]:
    summary = cached_summary()["factorized_leg_tails"]
    incoming = summary["incoming_fit"]
    outgoing = summary["outgoing_fit"]
    return {
        "test": "factorized_tail_coefficients_match_expected",
        "incoming": incoming,
        "outgoing": outgoing,
        "pass": (
            abs(incoming["c_log"] - 7.0) < 1.0e-4
            and abs(incoming["c_1_over_n"] - math.pi) < 5.0e-4
            and abs(incoming["c_2_over_n2"] - bns.EXACT_C2) < 5.0e-4
            and abs(outgoing["c_log"] + 5.0) < 1.0e-4
            and abs(outgoing["c_1_over_n"] + math.pi) < 5.0e-4
            and abs(outgoing["c_2_over_n2"] - bns.EXACT_C2) < 5.0e-4
        ),
    }


def test_fixed_tail_residuals_are_tiny() -> dict[str, object]:
    summary = cached_summary()["factorized_leg_tails"]
    incoming = summary["incoming_fixed_tail"]
    outgoing = summary["outgoing_fixed_tail"]
    return {
        "test": "fixed_tail_residuals_are_tiny",
        "incoming_rmse": incoming["rmse"],
        "outgoing_rmse": outgoing["rmse"],
        "incoming_max_abs": incoming["max_abs"],
        "outgoing_max_abs": outgoing["max_abs"],
        "pass": incoming["rmse"] < 1.0e-10 and outgoing["rmse"] < 1.0e-10,
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "invariant_tail_constant": test_invariant_tail_constant(),
        "factorized_tail_coefficients_match_expected": test_factorized_tail_coefficients_match_expected(),
        "fixed_tail_residuals_are_tiny": test_fixed_tail_residuals_are_tiny(),
    }
    passed = sum(1 for result in results.values() if result.get("pass"))
    total = len(results)

    print("=" * 96)
    print("BOSONIC NORMALIZATION STRUCTURE TESTS")
    print("=" * 96)
    for name, result in results.items():
        status = "PASS" if result.get("pass") else "FAIL"
        print(f"{name:44s} [{status}]")
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
