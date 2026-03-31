#!/usr/bin/env python3
"""
Regression tests for the finite-N join-local fermionic scaffolding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import local_interaction_point_fermion as lif
import tachyon_check as tc


def test_site_decomposition_identity() -> dict[str, object]:
    errors = []
    for n_sites in (4, 5, 8):
        for site_index in (0, n_sites - 1):
            data = lif.site_fermion_decomposition(n_sites, site_index)
            errors.append(data.reconstruction_error)

    max_error = float(max(errors))
    return {
        "test": "site_decomposition_identity",
        "max_error": max_error,
        "pass": max_error < 1.0e-12,
    }


def test_join_arc_difference_rows() -> dict[str, object]:
    report = lif.join_local_fermion_data(6, 7)
    basis1, _ = tc.real_zero_sum_basis(report.n1)
    basis2, _ = tc.real_zero_sum_basis(report.n2)

    left_selector = report.nabla_plus_selector
    right_selector = report.nabla_minus_selector
    left_reconstructed = basis1 @ report.nabla_plus_oscillator_row
    right_reconstructed = basis2 @ report.nabla_minus_oscillator_row

    max_error = float(
        max(
            np.linalg.norm(left_reconstructed - left_selector, ord=np.inf),
            np.linalg.norm(right_reconstructed - right_selector, ord=np.inf),
        )
    )
    max_average_sum = float(
        max(abs(np.sum(left_selector)), abs(np.sum(right_selector)))
    )
    return {
        "test": "join_arc_difference_rows",
        "max_error": max_error,
        "max_average_sum": max_average_sum,
        "pass": max_error < 1.0e-12 and max_average_sum < 1.0e-12,
    }


def test_local_sites_are_not_leg_averages() -> dict[str, object]:
    report = lif.join_local_fermion_data(5, 8)
    average1 = lif.average_selector(report.n1)
    average2 = lif.average_selector(report.n2)
    deviation = float(
        max(
            np.linalg.norm(report.theta_i_plus.selector - average1),
            np.linalg.norm(report.theta_i_minus.selector - average2),
        )
    )
    return {
        "test": "local_sites_are_not_leg_averages",
        "deviation": deviation,
        "pass": deviation > 0.1,
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "site_decomposition_identity": test_site_decomposition_identity(),
        "join_arc_difference_rows": test_join_arc_difference_rows(),
        "local_sites_are_not_leg_averages": test_local_sites_are_not_leg_averages(),
    }
    passed = sum(1 for item in results.values() if item.get("pass"))
    total = len(results)

    print("=" * 84)
    print("LOCAL INTERACTION-POINT FERMION TESTS")
    print("=" * 84)
    for name, result in results.items():
        status = "PASS" if result.get("pass") else "FAIL"
        print(f"{name:36s} [{status}]")
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
