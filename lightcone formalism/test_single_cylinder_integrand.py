#!/usr/bin/env python3
"""
Regression tests for the single-cylinder oscillator trace prototype.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import single_cylinder_integrand as sci


@functools.lru_cache(maxsize=1)
def cached_scan() -> dict[str, object]:
    return sci.default_scan()


def test_bosonic_trace_factor_closed_form() -> dict[str, object]:
    report = cached_scan()
    return {
        "test": "bosonic_trace_factor_closed_form",
        "max_rel_error": report["max_bosonic_rel_error"],
        "pass": report["max_bosonic_rel_error"] < 1.0e-12,
    }


def test_fermionic_trace_factor_closed_form() -> dict[str, object]:
    report = cached_scan()
    return {
        "test": "fermionic_trace_factor_closed_form",
        "max_rel_error": report["max_fermionic_rel_error"],
        "pass": report["max_fermionic_rel_error"] < 1.0e-12,
    }


def test_even_nyquist_sector_sample() -> dict[str, object]:
    report = sci.cylinder_trace_data(8, 0.7, 0.31, left_spin_sign=1, right_spin_sign=-1)
    max_error = max(
        report["bosonic_one_coordinate_rel_error"],
        report["fermion_left_component_rel_error"],
        report["fermion_right_component_rel_error"],
    )
    return {
        "test": "even_nyquist_sector_sample",
        "max_rel_error": max_error,
        "pass": max_error < 1.0e-12,
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "bosonic_trace_factor_closed_form": test_bosonic_trace_factor_closed_form(),
        "fermionic_trace_factor_closed_form": test_fermionic_trace_factor_closed_form(),
        "even_nyquist_sector_sample": test_even_nyquist_sector_sample(),
    }

    print("=" * 96)
    print("SINGLE CYLINDER INTEGRAND TESTS")
    print("=" * 96)
    for name, result in results.items():
        status = "PASS" if result.get("pass") else "FAIL"
        print(
            f"{name:44s} [{status}] "
            f"max_rel_error={result['max_rel_error']:.3e}"
        )
    passed = sum(1 for result in results.values() if result.get("pass"))
    total = len(results)
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
