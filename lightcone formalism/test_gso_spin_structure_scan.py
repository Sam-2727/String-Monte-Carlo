#!/usr/bin/env python3
"""
Regression tests for the oscillator spin-structure GSO sign scan.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gso_spin_structure_scan as gsss


@functools.lru_cache(maxsize=1)
def cached_scan() -> dict[str, object]:
    return gsss.default_scan()


def test_log_polar_sum_matches_safe_direct_sample() -> dict[str, object]:
    report = gsss.direct_sample_check()
    return {
        "test": "log_polar_sum_matches_safe_direct_sample",
        "rel_error": report["rel_error"],
        "pass": report["rel_error"] < 1.0e-12,
    }


def test_best_sign_pattern_still_fails_to_cancel() -> dict[str, object]:
    report = cached_scan()["best_pattern_by_max_distance"]
    return {
        "test": "best_sign_pattern_still_fails_to_cancel",
        "max_distance_to_one": report["max_distance_to_one"],
        "closest_distance_to_one": report["closest_distance_to_one"],
        "pass": (
            report["max_distance_to_one"] > 10.0
            and report["closest_distance_to_one"] > 0.2
        ),
    }


def test_standard_pattern_is_not_an_accidental_cancellation() -> dict[str, object]:
    report = cached_scan()["standard_pattern"]
    return {
        "test": "standard_pattern_is_not_an_accidental_cancellation",
        "max_distance_to_one": report["max_distance_to_one"],
        "closest_distance_to_one": report["closest_distance_to_one"],
        "pass": (
            report["max_distance_to_one"] > 10.0
            and report["closest_distance_to_one"] > 0.2
        ),
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "log_polar_sum_matches_safe_direct_sample": test_log_polar_sum_matches_safe_direct_sample(),
        "best_sign_pattern_still_fails_to_cancel": test_best_sign_pattern_still_fails_to_cancel(),
        "standard_pattern_is_not_an_accidental_cancellation": test_standard_pattern_is_not_an_accidental_cancellation(),
    }

    print("=" * 96)
    print("GSO SPIN-STRUCTURE SCAN TESTS")
    print("=" * 96)
    for name, result in results.items():
        status = "PASS" if result.get("pass") else "FAIL"
        scalar_parts = []
        for key, value in result.items():
            if key in {"test", "pass"}:
                continue
            if isinstance(value, float):
                scalar_parts.append(f"{key}={value:.3e}")
            else:
                scalar_parts.append(f"{key}={value}")
        print(f"{name:46s} [{status}] " + ", ".join(scalar_parts))

    passed = sum(1 for result in results.values() if result.get("pass"))
    total = len(results)
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
