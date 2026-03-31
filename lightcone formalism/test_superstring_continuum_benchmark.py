#!/usr/bin/env python3
"""
Regression tests for the reduced-Lambda structural benchmark comparison.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import superstring_continuum_benchmark as scb


@functools.lru_cache(maxsize=1)
def cached_single_point() -> dict[str, object]:
    return scb.compare_single_point(128, 192)


@functools.lru_cache(maxsize=1)
def cached_scan() -> dict[str, object]:
    return scb.run_symmetric_family_scan(
        alpha_prime=1.0,
        scales=[16, 32, 64, 128],
        max_t=0.75,
        step=0.125,
        min_t=0.0,
    )


def test_second_order_sample_matches_continuum_target() -> dict[str, object]:
    report = cached_single_point()
    return {
        "test": "second_order_sample_matches_continuum_target",
        "max_abs_error": report["max_abs_error"],
        "pass": report["max_abs_error"] < 1.0e-12,
    }


def test_symmetric_family_scan_matches_continuum_target() -> dict[str, object]:
    report = cached_scan()
    return {
        "test": "symmetric_family_scan_matches_continuum_target",
        "max_abs_error": report["max_abs_error"],
        "pass": report["max_abs_error"] < 1.0e-12,
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "second_order_sample_matches_continuum_target": test_second_order_sample_matches_continuum_target(),
        "symmetric_family_scan_matches_continuum_target": test_symmetric_family_scan_matches_continuum_target(),
    }
    passed = sum(1 for item in results.values() if item.get("pass"))
    total = len(results)

    print("=" * 96)
    print("SUPERSTRING CONTINUUM BENCHMARK TESTS")
    print("=" * 96)
    for name, result in results.items():
        status = "PASS" if result.get("pass") else "FAIL"
        print(
            f"{name:44s} [{status}] "
            f"max_abs_error={result['max_abs_error']:.3e}"
        )
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
