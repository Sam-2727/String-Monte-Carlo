#!/usr/bin/env python3
"""
Regression tests for the vacuum-contracted arc-family catalog scan.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import local_arc_catalog_scan as lacs


@functools.lru_cache(maxsize=1)
def cached_scan() -> dict[str, object]:
    return lacs.sampled_arc_catalog_scan()


def test_arc_family_preserves_vacuum_catalog_values() -> dict[str, object]:
    report = cached_scan()
    return {
        "test": "arc_family_preserves_vacuum_catalog_values",
        "qq_max_abs_error": report["qq_max_abs_error"],
        "delta_max_abs_error": report["delta_max_abs_error"],
        "pass": (
            report["qq_max_abs_error"] < 1.0e-12
            and report["delta_max_abs_error"] < 1.0e-12
        ),
    }


def test_arc_family_preserves_vacuum_catalog_counts_qq() -> dict[str, object]:
    report = cached_scan()
    expected = {"reduced_only": 25, "vanishing": 100}
    counts_ok = all(row["counts"] == expected for row in report["qq_rows"])
    return {
        "test": "arc_family_preserves_vacuum_catalog_counts_qq",
        "rows": report["qq_rows"],
        "pass": counts_ok,
    }


def test_arc_family_preserves_vacuum_catalog_counts_delta() -> dict[str, object]:
    report = cached_scan()
    expected = {"vanishing": 125}
    counts_ok = all(row["counts"] == expected for row in report["delta_rows"])
    return {
        "test": "arc_family_preserves_vacuum_catalog_counts_delta",
        "rows": report["delta_rows"],
        "pass": counts_ok,
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "arc_family_preserves_vacuum_catalog_values": test_arc_family_preserves_vacuum_catalog_values(),
        "arc_family_preserves_vacuum_catalog_counts_qq": test_arc_family_preserves_vacuum_catalog_counts_qq(),
        "arc_family_preserves_vacuum_catalog_counts_delta": test_arc_family_preserves_vacuum_catalog_counts_delta(),
    }
    passed = sum(1 for result in results.values() if result.get("pass"))
    total = len(results)

    print("=" * 96)
    print("LOCAL ARC CATALOG SCAN TESTS")
    print("=" * 96)
    for name, result in results.items():
        status = "PASS" if result.get("pass") else "FAIL"
        print(f"{name:56s} [{status}]")
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
