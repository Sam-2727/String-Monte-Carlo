#!/usr/bin/env python3
"""
Regression tests for the comprehensive local-channel catalog.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import local_channel_catalog as lcc


@functools.lru_cache(maxsize=2)
def cached_qq_summary() -> dict[str, object]:
    return lcc.summarize_catalog(response_kind="qq")


@functools.lru_cache(maxsize=2)
def cached_delta_summary() -> dict[str, object]:
    return lcc.summarize_catalog(response_kind="delta")


def test_qq_catalog_class_counts() -> dict[str, object]:
    summary = cached_qq_summary()
    counts = summary["counts"]
    expected = {
        "pure_quadratic_local": 37,
        "pure_quartic_local": 16,
        "reduced_only": 16,
        "reduced_plus_quartic": 9,
        "vanishing": 47,
    }
    return {
        "test": "qq_catalog_class_counts",
        "counts": counts,
        "expected": expected,
        "pass": counts == expected,
    }


def test_delta_catalog_vanishes() -> dict[str, object]:
    summary = cached_delta_summary()
    counts = summary["counts"]
    return {
        "test": "delta_catalog_vanishes",
        "counts": counts,
        "pass": counts == {"vanishing": 125},
    }


def test_benchmark_channels_land_in_expected_classes() -> dict[str, object]:
    summary = lcc.channel_catalog(response_kind="qq")
    category_map = {
        tuple(row["channel"]): row["category"]
        for row in summary["rows"]
    }
    expected = {
        ("perp23", "perp23", "parallel"): "reduced_only",
        ("perp23", "perp24", "parallel"): "reduced_only",
        ("parallel", "perp23", "perp23"): "reduced_only",
        ("perp23", "perp23", "dilaton"): "pure_quartic_local",
        ("parallel", "parallel", "dilaton"): "pure_quartic_local",
        ("parallel", "parallel", "parallel"): "reduced_only",
        ("parallel", "parallel", "b23"): "pure_quadratic_local",
    }
    observed = {
        key: category_map[key]
        for key in expected
    }
    return {
        "test": "benchmark_channels_land_in_expected_classes",
        "observed": observed,
        "expected": expected,
        "pass": observed == expected,
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "qq_catalog_class_counts": test_qq_catalog_class_counts(),
        "delta_catalog_vanishes": test_delta_catalog_vanishes(),
        "benchmark_channels_land_in_expected_classes": test_benchmark_channels_land_in_expected_classes(),
    }
    passed = sum(1 for result in results.values() if result.get("pass"))
    total = len(results)

    print("=" * 96)
    print("LOCAL CHANNEL CATALOG TESTS")
    print("=" * 96)
    for name, result in results.items():
        status = "PASS" if result.get("pass") else "FAIL"
        print(f"{name:48s} [{status}]")
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
