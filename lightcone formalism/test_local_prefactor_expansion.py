#!/usr/bin/env python3
"""
Regression tests for the exact mixed-variable local-prefactor expansion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fermionic_graviton_contraction as fgc
import local_prefactor_expansion as lpe


def _max_poly_difference(
    left: dict[tuple[int, ...], complex],
    right: dict[tuple[int, ...], complex],
) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    return float(max(abs(left.get(key, 0.0j) - right.get(key, 0.0j)) for key in keys))


def test_degree_zero_matches_reduced_prefactor() -> dict[str, object]:
    alpha_ratio = 0.4
    pieces = lpe.prefactor_mixed_expansion(alpha_ratio, 0, 0, trace_dropped=True)
    reduced = fgc.v_prefactor_polynomial(alpha_ratio, 0, 0, trace_dropped=True)
    degree_zero = pieces.get(0, {})
    max_error = _max_poly_difference(degree_zero, reduced)
    return {
        "test": "degree_zero_matches_reduced_prefactor",
        "max_error": max_error,
        "pass": max_error < 1.0e-12,
    }


def test_split_recombines_shifted_prefactor() -> dict[str, object]:
    alpha_ratio = 0.4
    base = fgc.v_prefactor_polynomial(alpha_ratio, 0, 1, trace_dropped=False)
    shifted = lpe.substitute_lambda_plus_xi(base)
    pieces = lpe.split_by_xi_degree(shifted)
    recombined = lpe.recompose_split(pieces)
    max_error = _max_poly_difference(recombined, shifted)
    return {
        "test": "split_recombines_shifted_prefactor",
        "max_error": max_error,
        "pass": max_error < 1.0e-12,
    }


def test_local_prefactor_has_nontrivial_xi_corrections() -> dict[str, object]:
    pieces = lpe.prefactor_mixed_expansion(0.4, 0, 1, trace_dropped=False)
    profile = lpe.xi_degree_profile(pieces)
    odd_degrees = sorted(degree for degree in profile if degree % 2 == 1)
    even_correction_degrees = sorted(
        degree for degree in profile if degree % 2 == 0 and degree > 0
    )
    return {
        "test": "local_prefactor_has_nontrivial_xi_corrections",
        "profile": profile,
        "odd_degrees": odd_degrees,
        "even_correction_degrees": even_correction_degrees,
        "pass": bool(odd_degrees) and bool(even_correction_degrees),
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "degree_zero_matches_reduced_prefactor": test_degree_zero_matches_reduced_prefactor(),
        "split_recombines_shifted_prefactor": test_split_recombines_shifted_prefactor(),
        "local_prefactor_has_nontrivial_xi_corrections": test_local_prefactor_has_nontrivial_xi_corrections(),
    }
    passed = sum(1 for item in results.values() if item.get("pass"))
    total = len(results)

    print("=" * 84)
    print("LOCAL PREFACTOR EXPANSION TESTS")
    print("=" * 84)
    for name, result in results.items():
        status = "PASS" if result.get("pass") else "FAIL"
        print(f"{name:40s} [{status}]")
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
