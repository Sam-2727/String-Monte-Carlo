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
import test_fermionic_graviton_contraction as tfgc
import test_graviton_assembly as tga
import test_graviton_prefactor as tgp
import test_neumann_extraction as tne
import test_projected_graviton_channels as tpgc
import test_superstring_decisive_test as tsdt
import test_superstring_normalization_factorization as tsnf
import test_tachyon_amplitude as tta
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


def run_fermionic_graviton_contraction_tests() -> dict[str, Any]:
    results = {
        "delta_reduction_identity": tfgc.test_delta_reduction_identity(),
        "trace_dropped_zero_channels": tfgc.test_trace_dropped_zero_channels(),
        "trace_dropped_graviton_channels": tfgc.test_trace_dropped_graviton_channels(),
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
    fermionic = report["tests"]["fermionic_graviton_contraction"]["results"]
    factorization = report["tests"]["superstring_normalization"]["results"][
        "positive_branch_is_rank_one"
    ]
    factorization_profile = report["tests"]["superstring_normalization"]["results"][
        "reference_normalized_profiles_agree"
    ]
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
        "fermionic_zero_channel_max": fermionic["trace_dropped_zero_channels"]["max_abs_value"],
        "fermionic_perp_perp_parallel": fermionic["trace_dropped_graviton_channels"]["perp_perp_parallel"],
        "fermionic_parallel_parallel_parallel": fermionic["trace_dropped_graviton_channels"]["parallel_parallel_parallel"],
        "trace_dropped_mixed_ratio": 0.5,
        "trace_dropped_max_mixed_ratio_error": decisive["max_mixed_ratio_error"],
        "trace_dropped_max_parallel_perp_lambda_sq_error": decisive["max_parallel_perp_lambda_sq_error"],
        "trace_dropped_zero_channel_max": decisive["max_zero_channel"],
        "normalization_rank1_rel_error": factorization["rank1_rel_frob_error"],
        "normalization_sigma2_over_sigma1": factorization["sigma2_over_sigma1"],
        "normalization_max_profile_diff": factorization_profile["max_profile_diff"],
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
        "neumann_extraction": run_neumann_tests(),
        "graviton_prefactor": run_graviton_prefactor_tests(),
        "graviton_assembly": run_graviton_assembly_tests(),
        "fermionic_graviton_contraction": run_fermionic_graviton_contraction_tests(),
        "weyl_formula": run_weyl_formula_tests(),
        "projected_graviton_channels": run_projected_graviton_channel_tests(),
        "superstring_decisive": run_superstring_decisive_tests(),
        "superstring_normalization": run_superstring_normalization_tests(),
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
    neumann_module = report["tests"]["neumann_extraction"]["summary"]
    prefactor_module = report["tests"]["graviton_prefactor"]["summary"]
    graviton_module = report["tests"]["graviton_assembly"]["summary"]
    projected_module = report["tests"]["projected_graviton_channels"]["summary"]
    decisive_module = report["tests"]["superstring_decisive"]["summary"]
    normalization_module = report["tests"]["superstring_normalization"]["summary"]

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
        "",
        "## Module Status",
        "",
        f"- `tachyon_amplitude`: `{tachyon_module['passed']}/{tachyon_module['total']}` passed",
        f"- `neumann_extraction`: `{neumann_module['passed']}/{neumann_module['total']}` passed",
        f"- `graviton_prefactor`: `{prefactor_module['passed']}/{prefactor_module['total']}` passed",
        f"- `graviton_assembly`: `{graviton_module['passed']}/{graviton_module['total']}` passed",
        f"- `projected_graviton_channels`: `{projected_module['passed']}/{projected_module['total']}` passed",
        f"- `superstring_decisive`: `{decisive_module['passed']}/{decisive_module['total']}` passed",
        f"- `superstring_normalization`: `{normalization_module['passed']}/{normalization_module['total']}` passed",
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
