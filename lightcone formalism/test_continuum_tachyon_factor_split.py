#!/usr/bin/env python3
"""
Regression tests for the continuum tachyon factor split.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import continuum_tachyon_factor_split as ctfs


@functools.lru_cache(maxsize=1)
def cached_report() -> dict[str, object]:
    return ctfs.full_report()


def test_continuum_identity_is_exact() -> dict[str, object]:
    report = cached_report()
    return {
        "test": "continuum_identity_is_exact",
        "max_abs_error": report["summary"]["max_continuum_identity_error"],
        "pass": report["summary"]["max_continuum_identity_error"] < 1.0e-12,
    }


def test_discrete_exponent_matches_log_mu_squared() -> dict[str, object]:
    report = cached_report()
    return {
        "test": "discrete_exponent_matches_log_mu_squared",
        "max_abs_error": report["summary"]["max_discrete_vs_log_mu_abs_error"],
        "max_rel_error": report["summary"]["max_discrete_vs_log_mu_rel_error"],
        "pass": (
            report["summary"]["max_discrete_vs_log_mu_abs_error"] < 2.0e-5
            and report["summary"]["max_discrete_vs_log_mu_rel_error"] < 6.0e-5
        ),
    }


def test_family_rows_are_scale_stable() -> dict[str, object]:
    report = cached_report()
    family_errors = {
        row["family"]: abs(row["discrete_vs_log_mu_abs_error"])
        for row in report["families"]
    }
    return {
        "test": "family_rows_are_scale_stable",
        "family_errors": family_errors,
        "pass": max(family_errors.values()) < 2.0e-5,
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "continuum_identity_is_exact": test_continuum_identity_is_exact(),
        "discrete_exponent_matches_log_mu_squared": test_discrete_exponent_matches_log_mu_squared(),
        "family_rows_are_scale_stable": test_family_rows_are_scale_stable(),
    }
    passed = sum(1 for item in results.values() if item.get("pass"))
    total = len(results)
    print("=" * 96)
    print("CONTINUUM TACHYON FACTOR SPLIT TESTS")
    print("=" * 96)
    for name, result in results.items():
        status = "PASS" if result.get("pass") else "FAIL"
        print(f"{name:48s} [{status}]")
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
