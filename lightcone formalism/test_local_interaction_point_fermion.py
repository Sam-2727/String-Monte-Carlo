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


def test_average_to_mixed_zero_mode_map() -> dict[str, object]:
    n1, n2 = 5, 8
    ((theta1_cm, theta1_lambda), (theta2_cm, theta2_lambda)) = (
        lif.average_to_mixed_zero_mode_map(n1, n2)
    )

    theta_cm = 1.7
    lambda_lat = -0.35
    theta1 = theta1_cm * theta_cm + theta1_lambda * lambda_lat
    theta2 = theta2_cm * theta_cm + theta2_lambda * lambda_lat

    n3 = n1 + n2
    theta_cm_back = (n1 * theta1 + n2 * theta2) / n3
    lambda_back = np.sqrt(n1 * n2 / n3) * (theta1 - theta2)

    max_error = float(max(abs(theta_cm_back - theta_cm), abs(lambda_back - lambda_lat)))
    return {
        "test": "average_to_mixed_zero_mode_map",
        "max_error": max_error,
        "pass": max_error < 1.0e-12,
    }


def test_canonical_local_difference_isolates_reduced_lambda() -> dict[str, object]:
    report = lif.canonical_local_difference_decomposition(5, 8)
    theta_cm_error = abs(report.theta_cm_coefficient)
    lambda_error = abs(report.lambda_lat_coefficient - 1.0)
    return {
        "test": "canonical_local_difference_isolates_reduced_lambda",
        "theta_cm_error": float(theta_cm_error),
        "lambda_error": float(lambda_error),
        "oscillator_norm_leg1": float(np.linalg.norm(report.oscillator_row_leg1)),
        "oscillator_norm_leg2": float(np.linalg.norm(report.oscillator_row_leg2)),
        "pass": theta_cm_error < 1.0e-12 and lambda_error < 1.0e-12,
    }


def test_endpoint_constraint_solution_matches_canonical() -> dict[str, object]:
    n1, n2 = 5, 8
    coeff_i_plus, coeff_i_minus = lif.endpoint_linear_coefficients_for_mixed_constraints(
        n1,
        n2,
        theta_cm_target=0.0,
        lambda_lat_target=1.0,
    )
    canonical = lif.canonical_local_difference_decomposition(n1, n2)
    diff = float(
        max(
            abs(coeff_i_plus - canonical.coeff_i_plus),
            abs(coeff_i_minus - canonical.coeff_i_minus),
        )
    )
    expected_scale = np.sqrt(n1 * n2 / (n1 + n2))
    expected_diff = float(
        max(abs(coeff_i_plus - expected_scale), abs(coeff_i_minus + expected_scale))
    )
    return {
        "test": "endpoint_constraint_solution_matches_canonical",
        "max_difference_from_canonical": diff,
        "max_difference_from_closed_form": expected_diff,
        "pass": diff < 1.0e-12 and expected_diff < 1.0e-12,
    }


def test_general_endpoint_linear_decomposition_constraints() -> dict[str, object]:
    n1, n2 = 7, 11
    target_theta_cm = 0.3 - 0.2j
    target_lambda = -1.4 + 0.5j
    coeff_i_plus, coeff_i_minus = lif.endpoint_linear_coefficients_for_mixed_constraints(
        n1,
        n2,
        theta_cm_target=target_theta_cm,
        lambda_lat_target=target_lambda,
    )
    decomposition = lif.decompose_join_linear_combination(
        n1,
        n2,
        coeff_i_plus,
        coeff_i_minus,
    )
    max_error = float(
        max(
            abs(decomposition.theta_cm_coefficient - target_theta_cm),
            abs(decomposition.lambda_lat_coefficient - target_lambda),
        )
    )
    return {
        "test": "general_endpoint_linear_decomposition_constraints",
        "max_error": max_error,
        "pass": max_error < 1.0e-12,
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "site_decomposition_identity": test_site_decomposition_identity(),
        "join_arc_difference_rows": test_join_arc_difference_rows(),
        "local_sites_are_not_leg_averages": test_local_sites_are_not_leg_averages(),
        "average_to_mixed_zero_mode_map": test_average_to_mixed_zero_mode_map(),
        "canonical_local_difference_isolates_reduced_lambda": test_canonical_local_difference_isolates_reduced_lambda(),
        "endpoint_constraint_solution_matches_canonical": test_endpoint_constraint_solution_matches_canonical(),
        "general_endpoint_linear_decomposition_constraints": test_general_endpoint_linear_decomposition_constraints(),
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
