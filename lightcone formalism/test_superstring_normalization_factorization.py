#!/usr/bin/env python3
"""
Regression tests for superstring_normalization_factorization.py.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import superstring_normalization_factorization as snf


@functools.lru_cache(maxsize=1)
def cached_report() -> dict[str, object]:
    return snf.analyze_factorization(
        alpha_prime=1.0,
        scales=[16, 32, 64, 128],
        max_t=0.75,
        step=0.125,
        min_positive_t=0.125,
        reference_lambda=0.4,
        reference_t=0.5,
    )


def test_positive_branch_is_rank_one() -> dict[str, object]:
    report = cached_report()
    return {
        "test": "positive_branch_is_rank_one",
        "rank1_rel_frob_error": report["rank1_rel_frob_error"],
        "sigma2_over_sigma1": report["sigma2_over_sigma1"],
        "pass": (
            report["rank1_rel_frob_error"] < 2.0e-4
            and report["sigma2_over_sigma1"] < 2.0e-4
        ),
    }


def test_reference_normalized_profiles_agree() -> dict[str, object]:
    report = cached_report()
    max_profile_diff = max(report["max_diff_from_reference_profile"])
    return {
        "test": "reference_normalized_profiles_agree",
        "max_profile_diff": max_profile_diff,
        "pass": max_profile_diff < 1.0e-3,
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "positive_branch_is_rank_one": test_positive_branch_is_rank_one(),
        "reference_normalized_profiles_agree": test_reference_normalized_profiles_agree(),
    }
    passed = sum(1 for item in results.values() if item.get("pass"))
    total = len(results)

    print("=" * 92)
    print("SUPERSTRING NORMALIZATION FACTORIZATION TESTS")
    print("=" * 92)
    for name, result in results.items():
        status = "PASS" if result.get("pass") else "FAIL"
        print(f"{name:36s} [{status}]")
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
