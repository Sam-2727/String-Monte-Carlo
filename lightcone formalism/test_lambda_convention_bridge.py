#!/usr/bin/env python3
"""
Regression tests for the DM <-> PS/GSB Lambda convention bridge.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import lambda_convention_bridge as lcb


@functools.lru_cache(maxsize=1)
def cached_expected_comparison() -> dict[str, object]:
    return lcb.compare_dm_normalized_to_expected(0.7)


@functools.lru_cache(maxsize=1)
def cached_alpha_scan() -> dict[str, object]:
    return lcb.alpha_independence_scan((0.4, 0.7, 1.0, 1.7, 2.3))


def test_dm_expected_form_matches_rescaled_ps_coefficients() -> dict[str, object]:
    report = cached_expected_comparison()
    return {
        "test": "dm_expected_form_matches_rescaled_ps_coefficients",
        "max_error": report["max_error"],
        "pass": report["max_error"] < 1.0e-12,
    }


def test_dm_rescaled_coefficients_are_alpha_independent() -> dict[str, object]:
    report = cached_alpha_scan()
    return {
        "test": "dm_rescaled_coefficients_are_alpha_independent",
        "max_error": report["max_error"],
        "pass": report["max_error"] < 1.0e-12,
    }


def test_degree_rescaling_matches_dm_ps_relation() -> dict[str, object]:
    alpha_ratio = 1.7
    observed = {
        degree: lcb.degree_rescaling(alpha_ratio, degree)
        for degree in (0, 2, 4, 6, 8)
    }
    expected = {
        0: 1.0,
        2: alpha_ratio / 2.0,
        4: (alpha_ratio / 2.0) ** 2,
        6: (alpha_ratio / 2.0) ** 3,
        8: (alpha_ratio / 2.0) ** 4,
    }
    max_error = max(abs(observed[degree] - expected[degree]) for degree in observed)
    return {
        "test": "degree_rescaling_matches_dm_ps_relation",
        "max_error": float(max_error),
        "pass": max_error < 1.0e-15,
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "dm_expected_form_matches_rescaled_ps_coefficients": test_dm_expected_form_matches_rescaled_ps_coefficients(),
        "dm_rescaled_coefficients_are_alpha_independent": test_dm_rescaled_coefficients_are_alpha_independent(),
        "degree_rescaling_matches_dm_ps_relation": test_degree_rescaling_matches_dm_ps_relation(),
    }
    passed = sum(1 for result in results.values() if result.get("pass"))
    total = len(results)

    print("=" * 96)
    print("LAMBDA CONVENTION BRIDGE TESTS")
    print("=" * 96)
    for name, result in results.items():
        status = "PASS" if result.get("pass") else "FAIL"
        print(f"{name:56s} [{status}]")
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
