#!/usr/bin/env python3
"""
Regression tests for the analytic bosonic three-tachyon continuum benchmark.
"""

from __future__ import annotations

import functools
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import continuum_tachyon_benchmark as ctb


@functools.lru_cache(maxsize=1)
def cached_report() -> dict[str, object]:
    return ctb.full_report()


def test_continuum_targets_are_scale_invariant() -> dict[str, object]:
    gamma_1 = ctb.continuum_gamma_target(2.0, 3.0)
    gamma_2 = ctb.continuum_gamma_target(20.0, 30.0)
    mu_1 = ctb.continuum_mu_squared(2.0, 3.0)
    mu_2 = ctb.continuum_mu_squared(20.0, 30.0)
    return {
        "test": "continuum_targets_are_scale_invariant",
        "gamma_1": gamma_1,
        "gamma_2": gamma_2,
        "mu_1": mu_1,
        "mu_2": mu_2,
        "pass": abs(gamma_1 - gamma_2) < 1.0e-12 and abs(mu_1 - mu_2) < 1.0e-12,
    }


def test_fixed_ratio_gamma_matches_continuum_target() -> dict[str, object]:
    report = cached_report()
    summary = report["summary"]
    families = report["families"]
    within_envelopes = True
    for row in families:
        extrap = row["gamma_T_extrapolation"]
        target = row["gamma_T_continuum_target"]
        envelope = extrap["uncertainty"] + 5.0e-5
        if abs(extrap["estimate"] - target) > envelope:
            within_envelopes = False
            break
    return {
        "test": "fixed_ratio_gamma_matches_continuum_target",
        "max_abs_error": summary["max_abs_error"],
        "max_rel_error": summary["max_rel_error"],
        "pass": summary["max_abs_error"] < 5.0e-5 and within_envelopes,
    }


def test_family_errors_decrease_with_scale() -> dict[str, object]:
    monotone = {}
    for a, b in [(2, 3), (1, 2), (1, 1)]:
        row = ctb.family_gamma_report(a, b)
        target = row["gamma_T_continuum_target"]
        errors = [abs(entry["gamma_T"] - target) for entry in row["rows"]]
        monotone[(a, b)] = errors
    passes = []
    for errors in monotone.values():
        ok = all(errors[i + 1] < errors[i] for i in range(len(errors) - 1))
        passes.append(ok)
    return {
        "test": "family_errors_decrease_with_scale",
        "error_histories": {f"{a}:{b}": errors for (a, b), errors in monotone.items()},
        "pass": all(passes),
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "continuum_targets_are_scale_invariant": test_continuum_targets_are_scale_invariant(),
        "fixed_ratio_gamma_matches_continuum_target": test_fixed_ratio_gamma_matches_continuum_target(),
        "family_errors_decrease_with_scale": test_family_errors_decrease_with_scale(),
    }
    passed = sum(1 for result in results.values() if result.get("pass"))
    total = len(results)

    print("=" * 96)
    print("CONTINUUM TACHYON BENCHMARK TESTS")
    print("=" * 96)
    for name, result in results.items():
        status = "PASS" if result.get("pass") else "FAIL"
        print(f"{name:46s} [{status}]")
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
