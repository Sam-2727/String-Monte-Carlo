#!/usr/bin/env python3
"""
Regression tests for the endpoint-phase scan of the local fermion candidate.
"""

from __future__ import annotations

import functools
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import local_endpoint_phase_scan as leps


@functools.lru_cache(maxsize=1)
def cached_special_report() -> dict[str, object]:
    return leps.special_phase_report(128, 192)


@functools.lru_cache(maxsize=1)
def cached_family_report() -> dict[str, object]:
    return leps.family_stability_report()


def test_canonical_phase_is_cm_free() -> dict[str, object]:
    row = cached_special_report()["canonical"]
    return {
        "test": "canonical_phase_is_cm_free",
        "theta_cm_abs": row["theta_cm_abs"],
        "lambda_lat_coefficient": row["lambda_lat_coefficient"],
        "pass": row["theta_cm_abs"] < 1.0e-12
        and abs(row["lambda_lat_coefficient"] - 1.0) < 1.0e-12,
    }


def test_dm_phase_has_large_cm_contamination() -> dict[str, object]:
    row = cached_special_report()["dm_plus_i"]
    return {
        "test": "dm_phase_has_large_cm_contamination",
        "theta_cm_abs": row["theta_cm_abs"],
        "two_point_scalar": row["two_point_scalar"],
        "pass": row["theta_cm_abs"] > 1.0 and row["two_point_scalar"] > 100.0,
    }


def test_phase_family_selects_canonical_antiphase() -> dict[str, object]:
    report = cached_family_report()
    return {
        "test": "phase_family_selects_canonical_antiphase",
        "max_best_cm_phase_error": report["max_best_cm_phase_error"],
        "max_best_two_point_phase_error": report["max_best_two_point_phase_error"],
        "pass": report["max_best_cm_phase_error"] < 1.0e-12
        and report["max_best_two_point_phase_error"] < 1.0e-12,
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "canonical_phase_is_cm_free": test_canonical_phase_is_cm_free(),
        "dm_phase_has_large_cm_contamination": test_dm_phase_has_large_cm_contamination(),
        "phase_family_selects_canonical_antiphase": test_phase_family_selects_canonical_antiphase(),
    }

    print("=" * 96)
    print("LOCAL ENDPOINT PHASE SCAN TESTS")
    print("=" * 96)
    for name, result in results.items():
        status = "PASS" if result.get("pass") else "FAIL"
        print(f"{name:44s} [{status}]")
    passed = sum(1 for result in results.values() if result.get("pass"))
    total = len(results)
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
