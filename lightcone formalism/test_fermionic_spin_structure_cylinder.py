#!/usr/bin/env python3
"""
Regression tests for the fermionic spin-structure cylinder factors.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fermionic_spin_structure_cylinder as fssc


@functools.lru_cache(maxsize=1)
def cached_scan() -> dict[str, object]:
    return fssc.default_scan()


def test_periodic_sector_matches_existing_integrand() -> dict[str, object]:
    report = cached_scan()
    return {
        "test": "periodic_sector_matches_existing_integrand",
        "max_rel_error": report["max_periodic_compatibility_error"],
        "pass": report["max_periodic_compatibility_error"] < 1.0e-12,
    }


def test_antiperiodic_sector_closed_form() -> dict[str, object]:
    report = cached_scan()
    max_rel_error = max(report["max_rel_error"], report["max_log_polar_rel_error"])
    return {
        "test": "antiperiodic_sector_closed_form",
        "max_rel_error": max_rel_error,
        "pass": max_rel_error < 1.0e-12,
    }


def test_antiperiodic_sector_has_no_zero_mode() -> dict[str, object]:
    report = cached_scan()
    return {
        "test": "antiperiodic_sector_has_no_zero_mode",
        "min_frequency": report["min_antiperiodic_frequency"],
        "pass": report["min_antiperiodic_frequency"] > 0.0,
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "periodic_sector_matches_existing_integrand": test_periodic_sector_matches_existing_integrand(),
        "antiperiodic_sector_closed_form": test_antiperiodic_sector_closed_form(),
        "antiperiodic_sector_has_no_zero_mode": test_antiperiodic_sector_has_no_zero_mode(),
    }

    print("=" * 96)
    print("FERMIONIC SPIN-STRUCTURE CYLINDER TESTS")
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
        print(f"{name:44s} [{status}] " + ", ".join(scalar_parts))

    passed = sum(1 for result in results.values() if result.get("pass"))
    total = len(results)
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
