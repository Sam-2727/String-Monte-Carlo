#!/usr/bin/env python3
"""
Unified numerical regression suite for the discrete-sigma lightcone program.

This runner does two things:

1. Collect the structured low-point continuum summaries from
   `low_point_validation.py`.
2. Run the current numerical regression tests for the bosonic tachyon sector,
   Neumann extraction, the superstring bosonic prefactor, and the graviton
   zero-mode / assembly diagnostics.

The goal is to make the low-point numerical program reproducible as a single
artifact with machine-readable output.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import low_point_validation as lpv
import test_bose_fermi_cancellation_scan as tbfcs
import test_bosonic_normalization_structure as tbns
import test_continuum_tachyon_benchmark as tctb
import test_continuum_tachyon_factor_split as tctfs
import test_fermionic_channel_responses as tfcr
import test_fermionic_graviton_contraction as tfgc
import test_fermionic_response_scan as tfrs
import test_graviton_assembly as tga
import test_graviton_prefactor as tgp
import test_lambda_convention_bridge as tlcb
import test_local_arc_catalog_scan as tlaccs
import test_local_arc_candidate_scan as tlacs
import test_local_channel_catalog as tlcc
import test_local_channel_response as tlcr
import test_local_endpoint_phase_scan as tleps
import test_local_interaction_point_fermion as tlipf
import test_local_prefactor_expansion as tlpe
import test_local_superstring_tree_benchmark as tlstb
import test_local_vacuum_reduction as tlvr
import test_neumann_extraction as tne
import test_projected_graviton_channels as tpgc
import test_single_cylinder_integrand as tsci
import test_superstring_decisive_test as tsdt
import test_superstring_continuum_benchmark as tscb
import test_superstring_normalization_factorization as tsnf
import test_tachyon_amplitude as tta
import test_twisted_cylinder as ttc
import test_weyl_vector_formula as twvf


def json_safe(value: Any) -> Any:
    """Recursively convert report data into JSON-safe Python objects."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    return value


def summarize_passes(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total = sum(1 for item in results.values() if isinstance(item, dict) and "pass" in item)
    passed = sum(1 for item in results.values() if isinstance(item, dict) and item.get("pass"))
    return {
        "passed": passed,
        "total": total,
        "all_passed": passed == total,
    }


def run_tachyon_tests() -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    for n1, n2 in [(8, 12), (16, 24), (32, 48)]:
        results[f"overlap_{n1}_{n2}"] = tta.test_overlap_identities(n1, n2)
    for a, b in [(2, 3), (1, 1), (1, 2)]:
        results[f"gamma_{a}_{b}"] = tta.test_gamma_convergence(a, b)
    results["critical_dimension"] = tta.test_critical_dimension()
    results["large_n_asymptotics"] = tta.test_large_n_asymptotics()
    results["ratio_independence"] = tta.test_amplitude_ratio_independence()
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_bosonic_normalization_structure_tests() -> dict[str, Any]:
    results = {
        "invariant_tail_constant": tbns.test_invariant_tail_constant(),
        "factorized_tail_coefficients_match_expected": tbns.test_factorized_tail_coefficients_match_expected(),
        "fixed_tail_residuals_are_tiny": tbns.test_fixed_tail_residuals_are_tiny(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_continuum_tachyon_benchmark_tests() -> dict[str, Any]:
    results = {
        "continuum_targets_are_scale_invariant": tctb.test_continuum_targets_are_scale_invariant(),
        "fixed_ratio_gamma_matches_continuum_target": tctb.test_fixed_ratio_gamma_matches_continuum_target(),
        "family_errors_decrease_with_scale": tctb.test_family_errors_decrease_with_scale(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_continuum_tachyon_factor_split_tests() -> dict[str, Any]:
    results = {
        "continuum_identity_is_exact": tctfs.test_continuum_identity_is_exact(),
        "discrete_exponent_matches_log_mu_squared": tctfs.test_discrete_exponent_matches_log_mu_squared(),
        "family_rows_are_scale_stable": tctfs.test_family_rows_are_scale_stable(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_neumann_tests() -> dict[str, Any]:
    results = {
        "symplectic_obstruction": tne.test_symplectic_obstruction(),
        "neumann_symmetry": tne.test_neumann_symmetry(),
        "neumann_vs_reduced_gaussian": tne.test_neumann_vs_reduced_gaussian(),
        "massless_covariance": tne.test_massless_covariance(),
        "ttm_trace_suppression": tne.test_ttm_trace_suppression(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_graviton_prefactor_tests() -> dict[str, Any]:
    results = {
        "parity_obstruction": tgp.test_parity_obstruction(),
        "parity_scan": tgp.test_parity_scan(30),
        "second_order_stencil": tgp.test_second_order_stencil(),
        "prefactor_convergence": tgp.test_prefactor_convergence(),
        "ratio_scan": tgp.test_ratio_scan(),
        "weyl_tensor_structure": tgp.test_weyl_tensor_structure(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_graviton_assembly_tests() -> dict[str, Any]:
    results = {
        "bosonic_tensor_structure": tga.test_bosonic_tensor_structure(),
        "fermionic_zeromodes": tga.test_fermionic_zeromodes(),
        "graviton_v_matrix_elements": tga.test_graviton_v_operator_matrix_elements(),
        "graviton_ratio_dependence": tga.test_full_graviton_ratio_dependence(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_lambda_convention_bridge_tests() -> dict[str, Any]:
    results = {
        "dm_expected_form_matches_rescaled_ps_coefficients": tlcb.test_dm_expected_form_matches_rescaled_ps_coefficients(),
        "dm_rescaled_coefficients_are_alpha_independent": tlcb.test_dm_rescaled_coefficients_are_alpha_independent(),
        "degree_rescaling_matches_dm_ps_relation": tlcb.test_degree_rescaling_matches_dm_ps_relation(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_fermionic_graviton_contraction_tests() -> dict[str, Any]:
    results = {
        "trace_dropped_benchmark_responses": tfcr.test_trace_dropped_benchmark_responses(),
        "trace_dropped_zero_response_channels": tfcr.test_trace_dropped_zero_response_channels(),
        "benchmark_response_value": tfcr.test_benchmark_response_value(),
        "response_scan_relations": tfrs.test_default_scan_relations(),
        "response_scan_benchmark_grid": tfrs.test_benchmark_grid_values(),
        "response_scan_offgrid_closed_form": tfrs.test_offgrid_closed_form_values(),
        "delta_reduction_identity": tfgc.test_delta_reduction_identity(),
        "trace_dropped_zero_channels": tfgc.test_trace_dropped_zero_channels(),
        "trace_dropped_graviton_channels": tfgc.test_trace_dropped_graviton_channels(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_local_interaction_point_fermion_tests() -> dict[str, Any]:
    results = {
        "site_decomposition_identity": tlipf.test_site_decomposition_identity(),
        "join_arc_difference_rows": tlipf.test_join_arc_difference_rows(),
        "local_sites_are_not_leg_averages": tlipf.test_local_sites_are_not_leg_averages(),
        "average_to_mixed_zero_mode_map": tlipf.test_average_to_mixed_zero_mode_map(),
        "canonical_local_difference_isolates_reduced_lambda": tlipf.test_canonical_local_difference_isolates_reduced_lambda(),
        "endpoint_constraint_solution_matches_canonical": tlipf.test_endpoint_constraint_solution_matches_canonical(),
        "general_endpoint_linear_decomposition_constraints": tlipf.test_general_endpoint_linear_decomposition_constraints(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_local_prefactor_expansion_tests() -> dict[str, Any]:
    results = {
        "degree_zero_matches_reduced_prefactor": tlpe.test_degree_zero_matches_reduced_prefactor(),
        "split_recombines_shifted_prefactor": tlpe.test_split_recombines_shifted_prefactor(),
        "local_prefactor_has_nontrivial_xi_corrections": tlpe.test_local_prefactor_has_nontrivial_xi_corrections(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_local_channel_response_tests() -> dict[str, Any]:
    results = {
        "benchmark_graviton_channels_are_xi_independent": tlcr.test_benchmark_graviton_channels_are_xi_independent(),
        "benchmark_dilaton_channel_is_pure_quartic": tlcr.test_benchmark_dilaton_channel_is_pure_quartic(),
        "trace_dropped_delta_benchmark_channel_vanishes": tlcr.test_trace_dropped_delta_benchmark_channel_vanishes(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_local_channel_catalog_tests() -> dict[str, Any]:
    results = {
        "qq_catalog_class_counts": tlcc.test_qq_catalog_class_counts(),
        "delta_catalog_vanishes": tlcc.test_delta_catalog_vanishes(),
        "benchmark_channels_land_in_expected_classes": tlcc.test_benchmark_channels_land_in_expected_classes(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_local_arc_candidate_scan_tests() -> dict[str, Any]:
    results = {
        "single_point_arc_family_preserves_mixed_zero_modes": tlacs.test_single_point_arc_family_preserves_mixed_zero_modes(),
        "single_point_arc_family_is_benchmark_invariant": tlacs.test_single_point_arc_family_is_benchmark_invariant(),
        "family_arc_scan_is_benchmark_invariant": tlacs.test_family_arc_scan_is_benchmark_invariant(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_local_arc_catalog_scan_tests() -> dict[str, Any]:
    results = {
        "arc_family_preserves_vacuum_catalog_values": tlaccs.test_arc_family_preserves_vacuum_catalog_values(),
        "arc_family_preserves_vacuum_catalog_counts_qq": tlaccs.test_arc_family_preserves_vacuum_catalog_counts_qq(),
        "arc_family_preserves_vacuum_catalog_counts_delta": tlaccs.test_arc_family_preserves_vacuum_catalog_counts_delta(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_local_endpoint_phase_scan_tests() -> dict[str, Any]:
    results = {
        "canonical_phase_is_cm_free": tleps.test_canonical_phase_is_cm_free(),
        "dm_phase_has_large_cm_contamination": tleps.test_dm_phase_has_large_cm_contamination(),
        "phase_family_selects_canonical_antiphase": tleps.test_phase_family_selects_canonical_antiphase(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_local_vacuum_reduction_tests() -> dict[str, Any]:
    results = {
        "canonical_xi_two_point_scalar_formula": tlvr.test_canonical_xi_two_point_scalar_formula(),
        "benchmark_dilaton_quartic_sector_vanishes_after_contraction": tlvr.test_benchmark_dilaton_quartic_sector_vanishes_after_contraction(),
        "catalog_collapses_after_vacuum_contraction": tlvr.test_catalog_collapses_after_vacuum_contraction(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_local_superstring_tree_benchmark_tests() -> dict[str, Any]:
    results = {
        "local_single_point_matches_analytic_target": tlstb.test_local_single_point_matches_analytic_target(),
        "local_single_point_matches_reduced_assembly": tlstb.test_local_single_point_matches_reduced_assembly(),
        "local_family_scan_matches_analytic_target": tlstb.test_local_family_scan_matches_analytic_target(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_weyl_formula_tests() -> dict[str, Any]:
    results = {
        "weyl_vector_closed_form": twvf.test_closed_form_formula(),
        "weyl_vector_trace_dropped_closed_form": twvf.test_trace_dropped_closed_form_formula(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_projected_graviton_channel_tests() -> dict[str, Any]:
    results = {
        "polarization_tensors": tpgc.test_polarization_tensors(),
        "projection_formulas": tpgc.test_projection_formulas(),
        "live_second_order_scan": tpgc.test_live_second_order_scan(),
        "trace_dropped_second_order_scan": tpgc.test_trace_dropped_second_order_scan(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_superstring_decisive_tests() -> dict[str, Any]:
    results = {
        "minimal_stencil_is_only_blocked": tsdt.test_minimal_stencil_is_only_blocked(),
        "unblocked_trace_dropped_relations": tsdt.test_unblocked_trace_dropped_relations(),
        "unblocked_trace_dropped_closed_forms": tsdt.test_unblocked_trace_dropped_closed_forms(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_superstring_continuum_benchmark_tests() -> dict[str, Any]:
    results = {
        "second_order_sample_matches_continuum_target": tscb.test_second_order_sample_matches_continuum_target(),
        "symmetric_family_scan_matches_continuum_target": tscb.test_symmetric_family_scan_matches_continuum_target(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_superstring_normalization_tests() -> dict[str, Any]:
    results = {
        "positive_branch_is_rank_one": tsnf.test_positive_branch_is_rank_one(),
        "reference_normalized_profiles_agree": tsnf.test_reference_normalized_profiles_agree(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_twisted_cylinder_tests() -> dict[str, Any]:
    results = {
        "exact_shift_recovery": ttc.test_exact_shift_recovery(),
        "generic_twist_reality_pattern": ttc.test_generic_twist_reality_pattern(),
        "oscillator_trace_positivity_and_closed_form": ttc.test_oscillator_trace_positivity_and_closed_form(),
        "fermionic_transport_spectrum": ttc.test_fermionic_transport_spectrum(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_single_cylinder_integrand_tests() -> dict[str, Any]:
    results = {
        "bosonic_trace_factor_closed_form": tsci.test_bosonic_trace_factor_closed_form(),
        "fermionic_trace_factor_closed_form": tsci.test_fermionic_trace_factor_closed_form(),
        "even_nyquist_sector_sample": tsci.test_even_nyquist_sector_sample(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def run_bose_fermi_cancellation_tests() -> dict[str, Any]:
    results = {
        "log_polar_matches_safe_direct_ratio": tbfcs.test_log_polar_matches_safe_direct_ratio(),
        "pre_gso_scan_does_not_cancel": tbfcs.test_pre_gso_scan_does_not_cancel(),
        "large_ratio_log_remains_finite": tbfcs.test_large_ratio_log_remains_finite(),
    }
    return {
        "summary": summarize_passes(results),
        "results": results,
    }


def extract_key_benchmarks(report: dict[str, Any]) -> dict[str, Any]:
    low_point = report["low_point_validation"]
    tachyon = low_point["tachyon"]
    massless = low_point["massless"]
    prefactor = low_point["superstring_prefactor"]
    second_second = next(
        row for row in prefactor["families"] if row["label"] == "second/second"
    )
    graviton = report["tests"]["graviton_assembly"]["results"]["graviton_v_matrix_elements"]
    decisive = report["tests"]["superstring_decisive"]["results"][
        "unblocked_trace_dropped_relations"
    ]
    decisive_closed = report["tests"]["superstring_decisive"]["results"][
        "unblocked_trace_dropped_closed_forms"
    ]
    continuum_benchmark = report["tests"]["superstring_continuum_benchmark"]["results"][
        "symmetric_family_scan_matches_continuum_target"
    ]
    fermionic = report["tests"]["fermionic_graviton_contraction"]["results"]
    factorization = report["tests"]["superstring_normalization"]["results"][
        "positive_branch_is_rank_one"
    ]
    factorization_profile = report["tests"]["superstring_normalization"]["results"][
        "reference_normalized_profiles_agree"
    ]
    twisted = report["tests"]["twisted_cylinder"]["results"]
    cylinder = report["tests"]["single_cylinder_integrand"]["results"]
    bosonic_tail = report["tests"]["bosonic_normalization_structure"]["results"]
    local_catalog = report["tests"]["local_channel_catalog"]["results"]
    local_arc = report["tests"]["local_arc_candidate_scan"]["results"]
    local_arc_catalog = report["tests"]["local_arc_catalog_scan"]["results"]
    local_endpoint_phase = report["tests"]["local_endpoint_phase_scan"]["results"]
    local_vacuum = report["tests"]["local_vacuum_reduction"]["results"]
    local_tree = report["tests"]["local_superstring_tree_benchmark"]["results"]
    bose_fermi = report["tests"]["bose_fermi_cancellation"]["results"]
    continuum_tachyon = report["tests"]["continuum_tachyon_benchmark"]["results"]
    continuum_split = report["tests"]["continuum_tachyon_factor_split"]["results"]
    critical_scan = tachyon["d_perp_scan"]
    best_d = min(critical_scan, key=lambda row: row["rmse"])
    return {
        "tachyon_factorization_rmse": tachyon["factorization"]["rmse"],
        "tachyon_critical_dimension": best_d["d_perp"],
        "tachyon_tail_constant": tachyon["invariant_tail"]["constant"],
        "ttm_a_tr_limit": massless["A_tr_extrapolation"]["estimate"],
        "ttm_a_tr_uncertainty": massless["A_tr_extrapolation"]["uncertainty"],
        "ttm_b_rel_limit": massless["B_rel_extrapolation"]["estimate"],
        "ttm_b_rel_uncertainty": massless["B_rel_extrapolation"]["uncertainty"],
        "prefactor_second_second_N1Bqq_limit": second_second["extrapolations"]["N1Bqq"]["estimate"],
        "prefactor_second_second_N1Bqq_uncertainty": second_second["extrapolations"]["N1Bqq"]["uncertainty"],
        "prefactor_second_second_A_delta_limit": second_second["extrapolations"]["A_delta"]["estimate"],
        "prefactor_second_second_A_delta_uncertainty": second_second["extrapolations"]["A_delta"]["uncertainty"],
        "weyl_vector_block_A": graviton["A"],
        "weyl_vector_block_B": graviton["B"],
        "weyl_vector_block_C": graviton["C"],
        "weyl_vector_block_residual": graviton["fit_max_residual"],
        "fermionic_delta_reduction_error": fermionic["delta_reduction_identity"]["abs_error"],
        "fermionic_response_diag_qq": fermionic["trace_dropped_benchmark_responses"]["diag_qq"],
        "fermionic_response_mixed_ratio_error": fermionic["trace_dropped_benchmark_responses"]["mixed_ratio_error"],
        "fermionic_response_lambda_sq_ratio_error": fermionic["trace_dropped_benchmark_responses"]["lambda_sq_ratio_error"],
        "fermionic_response_zero_max": fermionic["trace_dropped_zero_response_channels"]["max_abs_response"],
        "fermionic_response_scan_monotone": fermionic["response_scan_relations"]["monotone_decreasing_diag"],
        "fermionic_response_scan_closed_form_error": fermionic["response_scan_relations"]["max_diag_closed_form_error"],
        "fermionic_response_scan_grid_error": fermionic["response_scan_benchmark_grid"]["max_abs_error"],
        "fermionic_response_scan_offgrid_error": fermionic["response_scan_offgrid_closed_form"]["max_abs_error"],
        "fermionic_zero_channel_max": fermionic["trace_dropped_zero_channels"]["max_abs_value"],
        "fermionic_perp_perp_parallel": fermionic["trace_dropped_graviton_channels"]["perp_perp_parallel"],
        "fermionic_parallel_parallel_parallel": fermionic["trace_dropped_graviton_channels"]["parallel_parallel_parallel"],
        "trace_dropped_mixed_ratio": 0.5,
        "trace_dropped_max_mixed_ratio_error": decisive["max_mixed_ratio_error"],
        "trace_dropped_max_parallel_perp_lambda_sq_error": decisive["max_parallel_perp_lambda_sq_error"],
        "trace_dropped_zero_channel_max": decisive["max_zero_channel"],
        "trace_dropped_max_closed_form_error": decisive_closed["max_benchmark_closed_form_error"],
        "continuum_benchmark_max_abs_error": continuum_benchmark["max_abs_error"],
        "normalization_rank1_rel_error": factorization["rank1_rel_frob_error"],
        "normalization_sigma2_over_sigma1": factorization["sigma2_over_sigma1"],
        "normalization_max_profile_diff": factorization_profile["max_profile_diff"],
        "twisted_shift_max_error": twisted["exact_shift_recovery"]["max_shift_error"],
        "twisted_cross_max_error": twisted["exact_shift_recovery"]["max_cross_error"],
        "twisted_trace_logdet_error": twisted["oscillator_trace_positivity_and_closed_form"]["max_logdet_error"],
        "twisted_trace_factor_rel_error": twisted["oscillator_trace_positivity_and_closed_form"]["max_trace_factor_rel_error"],
        "twisted_trace_min_real_eigenvalue": twisted["oscillator_trace_positivity_and_closed_form"]["min_real_eigenvalue"],
        "twisted_fermion_transport_error": twisted["fermionic_transport_spectrum"]["max_abs_error"],
        "single_cylinder_bosonic_rel_error": cylinder["bosonic_trace_factor_closed_form"]["max_rel_error"],
        "single_cylinder_fermionic_rel_error": cylinder["fermionic_trace_factor_closed_form"]["max_rel_error"],
        "bosonic_invariant_tail_constant": bosonic_tail["invariant_tail_constant"]["constant"],
        "bosonic_invariant_tail_rmse": bosonic_tail["invariant_tail_constant"]["rmse"],
        "bosonic_incoming_tail_rmse": bosonic_tail["fixed_tail_residuals_are_tiny"]["incoming_rmse"],
        "bosonic_outgoing_tail_rmse": bosonic_tail["fixed_tail_residuals_are_tiny"]["outgoing_rmse"],
        "continuum_gamma_max_abs_error": continuum_tachyon["fixed_ratio_gamma_matches_continuum_target"]["max_abs_error"],
        "continuum_gamma_max_rel_error": continuum_tachyon["fixed_ratio_gamma_matches_continuum_target"]["max_rel_error"],
        "continuum_mu_identity_max_error": continuum_split["continuum_identity_is_exact"]["max_abs_error"],
        "continuum_mu_discrete_max_abs_error": continuum_split["discrete_exponent_matches_log_mu_squared"]["max_abs_error"],
        "continuum_mu_discrete_max_rel_error": continuum_split["discrete_exponent_matches_log_mu_squared"]["max_rel_error"],
        "local_qq_catalog_counts": local_catalog["qq_catalog_class_counts"]["counts"],
        "local_delta_catalog_counts": local_catalog["delta_catalog_vanishes"]["counts"],
        "local_arc_single_point_error": local_arc["single_point_arc_family_is_benchmark_invariant"]["max_abs_error"],
        "local_arc_single_point_reduced_error": local_arc["single_point_arc_family_is_benchmark_invariant"]["max_local_reduced_error"],
        "local_arc_family_error": local_arc["family_arc_scan_is_benchmark_invariant"]["max_abs_error"],
        "local_arc_family_reduced_error": local_arc["family_arc_scan_is_benchmark_invariant"]["max_local_reduced_error"],
        "local_arc_catalog_qq_error": local_arc_catalog["arc_family_preserves_vacuum_catalog_values"]["qq_max_abs_error"],
        "local_arc_catalog_delta_error": local_arc_catalog["arc_family_preserves_vacuum_catalog_values"]["delta_max_abs_error"],
        "local_endpoint_phase_dm_theta_cm_abs": local_endpoint_phase["dm_phase_has_large_cm_contamination"]["theta_cm_abs"],
        "local_endpoint_phase_dm_two_point_scalar": local_endpoint_phase["dm_phase_has_large_cm_contamination"]["two_point_scalar"],
        "local_endpoint_phase_best_cm_phase_error": local_endpoint_phase["phase_family_selects_canonical_antiphase"]["max_best_cm_phase_error"],
        "local_endpoint_phase_best_two_point_phase_error": local_endpoint_phase["phase_family_selects_canonical_antiphase"]["max_best_two_point_phase_error"],
        "local_vacuum_qq_counts": local_vacuum["catalog_collapses_after_vacuum_contraction"]["qq_counts"],
        "local_vacuum_delta_counts": local_vacuum["catalog_collapses_after_vacuum_contraction"]["delta_counts"],
        "local_vacuum_dilaton_max": local_vacuum["benchmark_dilaton_quartic_sector_vanishes_after_contraction"]["max_abs_value"],
        "local_tree_single_point_error": local_tree["local_single_point_matches_analytic_target"]["max_abs_error"],
        "local_tree_single_point_reduced_error": local_tree["local_single_point_matches_reduced_assembly"]["max_local_reduced_error"],
        "local_tree_scan_error": local_tree["local_family_scan_matches_analytic_target"]["max_abs_error"],
        "local_tree_scan_reduced_error": local_tree["local_family_scan_matches_analytic_target"]["max_local_reduced_error"],
        "pre_gso_closest_distance_to_one": bose_fermi["pre_gso_scan_does_not_cancel"]["closest"]["distance_to_one"],
        "pre_gso_closest_log_abs_ratio": bose_fermi["pre_gso_scan_does_not_cancel"]["closest"]["log_abs_ratio"],
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    low_point = {
        "tachyon": lpv.tachyon_summary(
            args.alpha_prime,
            args.d_perp,
            args.factorized_min_n,
            args.factorized_max_n,
            args.factorized_max_n3,
        ),
        "massless": lpv.massless_summary(args.alpha_prime, args.ratio_scale),
        "superstring_prefactor": lpv.superstring_prefactor_summary(args.ratio_scale),
    }

    tests = {
        "tachyon_amplitude": run_tachyon_tests(),
        "bosonic_normalization_structure": run_bosonic_normalization_structure_tests(),
        "continuum_tachyon_benchmark": run_continuum_tachyon_benchmark_tests(),
        "continuum_tachyon_factor_split": run_continuum_tachyon_factor_split_tests(),
        "neumann_extraction": run_neumann_tests(),
        "graviton_prefactor": run_graviton_prefactor_tests(),
        "graviton_assembly": run_graviton_assembly_tests(),
        "lambda_convention_bridge": run_lambda_convention_bridge_tests(),
        "fermionic_graviton_contraction": run_fermionic_graviton_contraction_tests(),
        "local_interaction_point_fermion": run_local_interaction_point_fermion_tests(),
        "local_prefactor_expansion": run_local_prefactor_expansion_tests(),
        "local_channel_response": run_local_channel_response_tests(),
        "local_channel_catalog": run_local_channel_catalog_tests(),
        "local_arc_candidate_scan": run_local_arc_candidate_scan_tests(),
        "local_arc_catalog_scan": run_local_arc_catalog_scan_tests(),
        "local_endpoint_phase_scan": run_local_endpoint_phase_scan_tests(),
        "local_vacuum_reduction": run_local_vacuum_reduction_tests(),
        "local_superstring_tree_benchmark": run_local_superstring_tree_benchmark_tests(),
        "weyl_formula": run_weyl_formula_tests(),
        "projected_graviton_channels": run_projected_graviton_channel_tests(),
        "superstring_decisive": run_superstring_decisive_tests(),
        "superstring_continuum_benchmark": run_superstring_continuum_benchmark_tests(),
        "superstring_normalization": run_superstring_normalization_tests(),
        "twisted_cylinder": run_twisted_cylinder_tests(),
        "single_cylinder_integrand": run_single_cylinder_integrand_tests(),
        "bose_fermi_cancellation": run_bose_fermi_cancellation_tests(),
    }

    module_summaries = {
        name: module["summary"] for name, module in tests.items()
    }
    total = sum(summary["total"] for summary in module_summaries.values())
    passed = sum(summary["passed"] for summary in module_summaries.values())

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "parameters": {
            "alpha_prime": args.alpha_prime,
            "d_perp": args.d_perp,
            "factorized_min_n": args.factorized_min_n,
            "factorized_max_n": args.factorized_max_n,
            "factorized_max_n3": args.factorized_max_n3,
            "ratio_scale": args.ratio_scale,
        },
        "low_point_validation": low_point,
        "tests": tests,
        "suite_summary": {
            "passed": passed,
            "total": total,
            "all_passed": passed == total,
            "module_summaries": module_summaries,
        },
    }
    report["key_benchmarks"] = extract_key_benchmarks(report)
    return report


def format_complex_pair(value: Any) -> str:
    if isinstance(value, complex):
        return f"{value.real:.6f}{value.imag:+.6f}i"
    if isinstance(value, dict) and set(value.keys()) == {"real", "imag"}:
        return f"{value['real']:.6f}{value['imag']:+.6f}i"
    return str(value)


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["suite_summary"]
    benchmarks = report["key_benchmarks"]
    tachyon_module = report["tests"]["tachyon_amplitude"]["summary"]
    bosonic_tail_module = report["tests"]["bosonic_normalization_structure"]["summary"]
    continuum_tachyon_module = report["tests"]["continuum_tachyon_benchmark"]["summary"]
    continuum_split_module = report["tests"]["continuum_tachyon_factor_split"]["summary"]
    neumann_module = report["tests"]["neumann_extraction"]["summary"]
    prefactor_module = report["tests"]["graviton_prefactor"]["summary"]
    graviton_module = report["tests"]["graviton_assembly"]["summary"]
    lambda_bridge_module = report["tests"]["lambda_convention_bridge"]["summary"]
    local_fermion_module = report["tests"]["local_interaction_point_fermion"]["summary"]
    local_catalog_module = report["tests"]["local_channel_catalog"]["summary"]
    local_arc_module = report["tests"]["local_arc_candidate_scan"]["summary"]
    local_arc_catalog_module = report["tests"]["local_arc_catalog_scan"]["summary"]
    local_endpoint_phase_module = report["tests"]["local_endpoint_phase_scan"]["summary"]
    local_vacuum_module = report["tests"]["local_vacuum_reduction"]["summary"]
    local_tree_module = report["tests"]["local_superstring_tree_benchmark"]["summary"]
    projected_module = report["tests"]["projected_graviton_channels"]["summary"]
    decisive_module = report["tests"]["superstring_decisive"]["summary"]
    normalization_module = report["tests"]["superstring_normalization"]["summary"]
    twisted_module = report["tests"]["twisted_cylinder"]["summary"]
    cylinder_module = report["tests"]["single_cylinder_integrand"]["summary"]
    bose_fermi_module = report["tests"]["bose_fermi_cancellation"]["summary"]

    lines = [
        "# Numerical Suite Report",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Overview",
        "",
        f"- Overall status: `{summary['passed']}/{summary['total']}` tests passed",
        f"- Bosonic TTT factorization rmse: `{benchmarks['tachyon_factorization_rmse']:.12e}`",
        f"- Critical dimension from factorization scan: `D_perp = {benchmarks['tachyon_critical_dimension']}`",
        f"- Large-N tachyon tail constant: `{benchmarks['tachyon_tail_constant']:.9f}`",
        (
            "- TTM continuum limits: "
            f"`A_tr -> {benchmarks['ttm_a_tr_limit']:.9f} +/- {benchmarks['ttm_a_tr_uncertainty']:.3e}`, "
            f"`B_rel -> {benchmarks['ttm_b_rel_limit']:.9f} +/- {benchmarks['ttm_b_rel_uncertainty']:.3e}`"
        ),
        (
            "- Superstring second/second continuum limits: "
            f"`N1*B_qq -> {benchmarks['prefactor_second_second_N1Bqq_limit']:.9f} +/- {benchmarks['prefactor_second_second_N1Bqq_uncertainty']:.3e}`, "
            f"`A_delta -> {benchmarks['prefactor_second_second_A_delta_limit']:.9f} +/- {benchmarks['prefactor_second_second_A_delta_uncertainty']:.3e}`"
        ),
        (
            "- Weyl vector-block coefficients: "
            f"`A = {format_complex_pair(benchmarks['weyl_vector_block_A'])}`, "
            f"`B = {format_complex_pair(benchmarks['weyl_vector_block_B'])}`, "
            f"`C = {format_complex_pair(benchmarks['weyl_vector_block_C'])}`"
        ),
        f"- Weyl vector-block fit residual: `{benchmarks['weyl_vector_block_residual']:.12e}`",
        (
            "- Pure fermionic benchmark responses at `lambda = 2/5`: "
            f"`R_qq(23,23,||) = {format_complex_pair(benchmarks['fermionic_response_diag_qq'])}`, "
            f"`max |R_delta| / zero-response error = {benchmarks['fermionic_response_zero_max']:.3e}`, "
            f"`|mixed/diag - 1/2| = {benchmarks['fermionic_response_mixed_ratio_error']:.3e}`, "
            f"`|lambda^2 par23/diag - 1| = {benchmarks['fermionic_response_lambda_sq_ratio_error']:.3e}`, "
            f"`scan monotone = {benchmarks['fermionic_response_scan_monotone']}`, "
            f"`closed-form error = {benchmarks['fermionic_response_scan_closed_form_error']:.3e}`, "
            f"`grid benchmark error = {benchmarks['fermionic_response_scan_grid_error']:.3e}`, "
            f"`off-grid error = {benchmarks['fermionic_response_scan_offgrid_error']:.3e}`"
        ),
        (
            "- Trace-dropped fermionic channel test: "
            f"`A_mix / A_diag = {benchmarks['trace_dropped_mixed_ratio']:.1f}` "
            f"with `max error = {benchmarks['trace_dropped_max_mixed_ratio_error']:.3e}`, "
            f"`max |lambda^2 A_par23 / A_diag - 1| = {benchmarks['trace_dropped_max_parallel_perp_lambda_sq_error']:.3e}`, "
            f"`max |zero channel| = {benchmarks['trace_dropped_zero_channel_max']:.3e}`"
        ),
        (
            "- Positive-branch N1*A_diag factorization: "
            f"`rank1 rel error = {benchmarks['normalization_rank1_rel_error']:.3e}`, "
            f"`sigma2/sigma1 = {benchmarks['normalization_sigma2_over_sigma1']:.3e}`, "
            f"`max normalized profile diff = {benchmarks['normalization_max_profile_diff']:.3e}`"
        ),
        (
            "- Twisted-cylinder checks: "
            f"`max |R-P| = {benchmarks['twisted_shift_max_error']:.3e}`, "
            f"`max |B(T,m/N)-B(T,0)P| = {benchmarks['twisted_cross_max_error']:.3e}`, "
            f"`trace logdet error = {benchmarks['twisted_trace_logdet_error']:.3e}`, "
            f"`trace-factor rel error = {benchmarks['twisted_trace_factor_rel_error']:.3e}`, "
            f"`min Re-eigenvalue = {benchmarks['twisted_trace_min_real_eigenvalue']:.3e}`, "
            f"`fermion transport error = {benchmarks['twisted_fermion_transport_error']:.3e}`"
        ),
        (
            "- Single-cylinder oscillator trace prototype: "
            f"`bosonic closed-form rel error = {benchmarks['single_cylinder_bosonic_rel_error']:.3e}`, "
            f"`fermionic closed-form rel error = {benchmarks['single_cylinder_fermionic_rel_error']:.3e}`"
        ),
        (
            "- Bosonic normalization structure: "
            f"`C_tail = {benchmarks['bosonic_invariant_tail_constant']:.9f}`, "
            f"`invariant-tail rmse = {benchmarks['bosonic_invariant_tail_rmse']:.3e}`, "
            f"`incoming/outgoing fixed-tail rmse = {benchmarks['bosonic_incoming_tail_rmse']:.3e} / {benchmarks['bosonic_outgoing_tail_rmse']:.3e}`"
        ),
        (
            "- Continuum tachyon Schur benchmark: "
            f"`max |gamma_inf - gamma_cont| = {benchmarks['continuum_gamma_max_abs_error']:.3e}`, "
            f"`max relative error = {benchmarks['continuum_gamma_max_rel_error']:.3e}`"
        ),
        (
            "- Continuum tachyon factor split: "
            f"`max |q_rel^2/(2 gamma_cont) - log mu^2| = {benchmarks['continuum_mu_identity_max_error']:.3e}`, "
            f"`max |exp_inf - log mu^2| = {benchmarks['continuum_mu_discrete_max_abs_error']:.3e}`, "
            f"`max relative error = {benchmarks['continuum_mu_discrete_max_rel_error']:.3e}`"
        ),
        "- DM/PS Lambda convention bridge: degree-rescaled prefactor tensors are alpha-independent and match the expected DM-normalized coefficients to numerical precision.",
        (
            "- Local channel catalog (trace-dropped): "
            f"`qq counts = {benchmarks['local_qq_catalog_counts']}`, "
            f"`delta counts = {benchmarks['local_delta_catalog_counts']}`"
        ),
        (
            "- Local arc-admixture scan: "
            f"`single-point analytic error = {benchmarks['local_arc_single_point_error']:.3e}`, "
            f"`single-point local-reduced error = {benchmarks['local_arc_single_point_reduced_error']:.3e}`, "
            f"`family-scan analytic error = {benchmarks['local_arc_family_error']:.3e}`, "
            f"`family-scan local-reduced error = {benchmarks['local_arc_family_reduced_error']:.3e}`"
        ),
        (
            "- Local arc-admixture vacuum-catalog scan: "
            f"`qq catalog error = {benchmarks['local_arc_catalog_qq_error']:.3e}`, "
            f"`delta catalog error = {benchmarks['local_arc_catalog_delta_error']:.3e}`"
        ),
        (
            "- Local endpoint-phase scan: "
            f"`|Theta_cm|` for the naive `+i` endpoint sum after unit-Lambda normalization"
            f" = {benchmarks['local_endpoint_phase_dm_theta_cm_abs']:.3e}, "
            f"`two-point scalar = {benchmarks['local_endpoint_phase_dm_two_point_scalar']:.6f}`, "
            f"`best CM/two-point phase errors = "
            f"{benchmarks['local_endpoint_phase_best_cm_phase_error']:.3e} / "
            f"{benchmarks['local_endpoint_phase_best_two_point_phase_error']:.3e}`"
        ),
        (
            "- Local vacuum reduction (canonical endpoint-difference candidate): "
            f"`qq counts after contraction = {benchmarks['local_vacuum_qq_counts']}`, "
            f"`delta counts after contraction = {benchmarks['local_vacuum_delta_counts']}`, "
            f"`max |dilaton quartic contraction| = {benchmarks['local_vacuum_dilaton_max']:.3e}`"
        ),
        (
            "- Local superstring tree benchmark: "
            f"`single-point analytic error = {benchmarks['local_tree_single_point_error']:.3e}`, "
            f"`single-point local-reduced error = {benchmarks['local_tree_single_point_reduced_error']:.3e}`, "
            f"`family-scan analytic error = {benchmarks['local_tree_scan_error']:.3e}`, "
            f"`family-scan local-reduced error = {benchmarks['local_tree_scan_reduced_error']:.3e}`"
        ),
        (
            "- Pre-GSO one-cylinder ratio scan: "
            f"`closest distance to 1 = {benchmarks['pre_gso_closest_distance_to_one']:.6f}`, "
            f"`closest log|R| = {benchmarks['pre_gso_closest_log_abs_ratio']:.6f}`"
        ),
        "",
        "## Module Status",
        "",
        f"- `tachyon_amplitude`: `{tachyon_module['passed']}/{tachyon_module['total']}` passed",
        f"- `bosonic_normalization_structure`: `{bosonic_tail_module['passed']}/{bosonic_tail_module['total']}` passed",
        f"- `continuum_tachyon_benchmark`: `{continuum_tachyon_module['passed']}/{continuum_tachyon_module['total']}` passed",
        f"- `continuum_tachyon_factor_split`: `{continuum_split_module['passed']}/{continuum_split_module['total']}` passed",
        f"- `neumann_extraction`: `{neumann_module['passed']}/{neumann_module['total']}` passed",
        f"- `graviton_prefactor`: `{prefactor_module['passed']}/{prefactor_module['total']}` passed",
        f"- `graviton_assembly`: `{graviton_module['passed']}/{graviton_module['total']}` passed",
        f"- `lambda_convention_bridge`: `{lambda_bridge_module['passed']}/{lambda_bridge_module['total']}` passed",
        f"- `local_interaction_point_fermion`: `{local_fermion_module['passed']}/{local_fermion_module['total']}` passed",
        f"- `local_channel_catalog`: `{local_catalog_module['passed']}/{local_catalog_module['total']}` passed",
        f"- `local_arc_candidate_scan`: `{local_arc_module['passed']}/{local_arc_module['total']}` passed",
        f"- `local_arc_catalog_scan`: `{local_arc_catalog_module['passed']}/{local_arc_catalog_module['total']}` passed",
        f"- `local_endpoint_phase_scan`: `{local_endpoint_phase_module['passed']}/{local_endpoint_phase_module['total']}` passed",
        f"- `local_vacuum_reduction`: `{local_vacuum_module['passed']}/{local_vacuum_module['total']}` passed",
        f"- `local_superstring_tree_benchmark`: `{local_tree_module['passed']}/{local_tree_module['total']}` passed",
        f"- `projected_graviton_channels`: `{projected_module['passed']}/{projected_module['total']}` passed",
        f"- `superstring_decisive`: `{decisive_module['passed']}/{decisive_module['total']}` passed",
        f"- `superstring_normalization`: `{normalization_module['passed']}/{normalization_module['total']}` passed",
        f"- `twisted_cylinder`: `{twisted_module['passed']}/{twisted_module['total']}` passed",
        f"- `single_cylinder_integrand`: `{cylinder_module['passed']}/{cylinder_module['total']}` passed",
        f"- `bose_fermi_cancellation`: `{bose_fermi_module['passed']}/{bose_fermi_module['total']}` passed",
        "",
        "## Notes",
        "",
        "- This report aggregates the structured low-point continuum fits with the current numerical regression tests.",
        "- It is intended as a reproducible checkpoint for the bosonic three-tachyon, bosonic 2T+1M, and superstring graviton-prefactor program.",
    ]
    return "\n".join(lines) + "\n"


def print_console_summary(report: dict[str, Any]) -> None:
    summary = report["suite_summary"]
    benchmarks = report["key_benchmarks"]
    print("=" * 92)
    print("DISCRETE-SIGMA NUMERICAL SUITE")
    print("=" * 92)
    print(
        f"Overall test status: {summary['passed']}/{summary['total']} passed "
        f"(all_passed={summary['all_passed']})"
    )
    print()
    print("Key benchmarks:")
    print(f"  TTT factorization rmse       = {benchmarks['tachyon_factorization_rmse']:.12e}")
    print(f"  Critical dimension           = D_perp {benchmarks['tachyon_critical_dimension']}")
    print(f"  Tachyon tail constant        = {benchmarks['tachyon_tail_constant']:.9f}")
    print(
        "  TTM limit A_tr               = "
        f"{benchmarks['ttm_a_tr_limit']:.9f} +/- {benchmarks['ttm_a_tr_uncertainty']:.3e}"
    )
    print(
        "  TTM limit B_rel              = "
        f"{benchmarks['ttm_b_rel_limit']:.9f} +/- {benchmarks['ttm_b_rel_uncertainty']:.3e}"
    )
    print(
        "  Prefactor second/second      = "
        f"N1*B_qq -> {benchmarks['prefactor_second_second_N1Bqq_limit']:.9f} +/- "
        f"{benchmarks['prefactor_second_second_N1Bqq_uncertainty']:.3e}, "
        f"A_delta -> {benchmarks['prefactor_second_second_A_delta_limit']:.9f} +/- "
        f"{benchmarks['prefactor_second_second_A_delta_uncertainty']:.3e}"
    )
    print(
        "  Weyl vector-block coefficients = "
        f"A {format_complex_pair(benchmarks['weyl_vector_block_A'])}, "
        f"B {format_complex_pair(benchmarks['weyl_vector_block_B'])}, "
        f"C {format_complex_pair(benchmarks['weyl_vector_block_C'])}"
    )
    print(
        "  Weyl vector-block residual   = "
        f"{benchmarks['weyl_vector_block_residual']:.12e}"
    )
    print(
        "  Pure fermionic response test  = "
        f"Rqq_diag {format_complex_pair(benchmarks['fermionic_response_diag_qq'])}, "
        f"max zero {benchmarks['fermionic_response_zero_max']:.3e}, "
        f"mixed ratio err {benchmarks['fermionic_response_mixed_ratio_error']:.3e}, "
        f"lambda^2 ratio err {benchmarks['fermionic_response_lambda_sq_ratio_error']:.3e}, "
        f"scan monotone {benchmarks['fermionic_response_scan_monotone']}, "
        f"closed-form err {benchmarks['fermionic_response_scan_closed_form_error']:.3e}, "
        f"grid err {benchmarks['fermionic_response_scan_grid_error']:.3e}, "
        f"off-grid err {benchmarks['fermionic_response_scan_offgrid_error']:.3e}"
    )
    print(
        "  Trace-dropped fermion test   = "
        f"A_mix/A_diag -> {benchmarks['trace_dropped_mixed_ratio']:.1f}, "
        f"max error {benchmarks['trace_dropped_max_mixed_ratio_error']:.3e}, "
        f"max |lambda^2 A_par23/A_diag - 1| {benchmarks['trace_dropped_max_parallel_perp_lambda_sq_error']:.3e}, "
        f"max |zero channel| {benchmarks['trace_dropped_zero_channel_max']:.3e}"
    )
    print(
        "  N1*A_diag factorization      = "
        f"rank1 rel error {benchmarks['normalization_rank1_rel_error']:.3e}, "
        f"sigma2/sigma1 {benchmarks['normalization_sigma2_over_sigma1']:.3e}, "
        f"max profile diff {benchmarks['normalization_max_profile_diff']:.3e}"
    )
    print(
        "  Twisted cylinder checks      = "
        f"max |R-P| {benchmarks['twisted_shift_max_error']:.3e}, "
        f"max |B-B0P| {benchmarks['twisted_cross_max_error']:.3e}, "
        f"trace logdet err {benchmarks['twisted_trace_logdet_error']:.3e}, "
        f"trace-factor rel err {benchmarks['twisted_trace_factor_rel_error']:.3e}, "
        f"min Re-eig {benchmarks['twisted_trace_min_real_eigenvalue']:.3e}, "
        f"fermion transport err {benchmarks['twisted_fermion_transport_error']:.3e}"
    )
    print(
        "  Local arc-admixture scan     = "
        f"single-point err {benchmarks['local_arc_single_point_error']:.3e}, "
        f"single-point local-reduced err {benchmarks['local_arc_single_point_reduced_error']:.3e}, "
        f"family err {benchmarks['local_arc_family_error']:.3e}, "
        f"family local-reduced err {benchmarks['local_arc_family_reduced_error']:.3e}"
    )
    print(
        "  Local arc catalog scan       = "
        f"qq err {benchmarks['local_arc_catalog_qq_error']:.3e}, "
        f"delta err {benchmarks['local_arc_catalog_delta_error']:.3e}"
    )
    print(
        "  Local vacuum reduction       = "
        f"qq counts {benchmarks['local_vacuum_qq_counts']}, "
        f"delta counts {benchmarks['local_vacuum_delta_counts']}, "
        f"max |dil quartic| {benchmarks['local_vacuum_dilaton_max']:.3e}"
    )
    print(
        "  Local tree benchmark         = "
        f"single-point err {benchmarks['local_tree_single_point_error']:.3e}, "
        f"single-point local-reduced err {benchmarks['local_tree_single_point_reduced_error']:.3e}, "
        f"family err {benchmarks['local_tree_scan_error']:.3e}, "
        f"family local-reduced err {benchmarks['local_tree_scan_reduced_error']:.3e}"
    )
    print(
        "  Single-cylinder prototype    = "
        f"bosonic rel err {benchmarks['single_cylinder_bosonic_rel_error']:.3e}, "
        f"fermionic rel err {benchmarks['single_cylinder_fermionic_rel_error']:.3e}"
    )
    print(
        "  Bosonic normalization tail   = "
        f"C_tail {benchmarks['bosonic_invariant_tail_constant']:.9f}, "
        f"invariant rmse {benchmarks['bosonic_invariant_tail_rmse']:.3e}, "
        f"in/out fixed-tail rmse {benchmarks['bosonic_incoming_tail_rmse']:.3e} / "
        f"{benchmarks['bosonic_outgoing_tail_rmse']:.3e}"
    )
    print(
        "  Continuum gamma benchmark    = "
        f"max abs err {benchmarks['continuum_gamma_max_abs_error']:.3e}, "
        f"max rel err {benchmarks['continuum_gamma_max_rel_error']:.3e}"
    )
    print(
        "  Continuum mu split           = "
        f"max |exp_cont-logmu| {benchmarks['continuum_mu_identity_max_error']:.3e}, "
        f"max |exp_inf-logmu| {benchmarks['continuum_mu_discrete_max_abs_error']:.3e}, "
        f"max rel err {benchmarks['continuum_mu_discrete_max_rel_error']:.3e}"
    )
    print(
        "  Local channel catalog        = "
        f"qq {benchmarks['local_qq_catalog_counts']}, "
        f"delta {benchmarks['local_delta_catalog_counts']}"
    )
    print(
        "  Pre-GSO BF ratio scan        = "
        f"closest distance {benchmarks['pre_gso_closest_distance_to_one']:.6f}, "
        f"closest log|R| {benchmarks['pre_gso_closest_log_abs_ratio']:.6f}"
    )
    print()
    print("Per-module summary:")
    for name, module in report["suite_summary"]["module_summaries"].items():
        print(
            f"  {name:20s} {module['passed']:2d}/{module['total']:2d} "
            f"(all_passed={module['all_passed']})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-prime", type=float, default=1.0)
    parser.add_argument("--d-perp", type=int, default=24)
    parser.add_argument("--factorized-min-n", type=int, default=4)
    parser.add_argument("--factorized-max-n", type=int, default=40)
    parser.add_argument("--factorized-max-n3", type=int, default=80)
    parser.add_argument("--ratio-scale", type=int, default=128)
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="optional path to write the full numerical suite as JSON",
    )
    parser.add_argument(
        "--markdown-out",
        type=str,
        default=None,
        help="optional path to write a concise markdown summary",
    )
    args = parser.parse_args()

    report = build_report(args)
    print_console_summary(report)

    if args.json_out is not None:
        json_path = Path(args.json_out)
        json_path.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n")
        print(f"\nWrote JSON report to {json_path}")
    if args.markdown_out is not None:
        md_path = Path(args.markdown_out)
        md_path.write_text(markdown_report(json_safe(report)))
        print(f"Wrote markdown report to {md_path}")


if __name__ == "__main__":
    main()
