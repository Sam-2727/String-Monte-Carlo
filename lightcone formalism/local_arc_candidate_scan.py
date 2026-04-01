#!/usr/bin/env python3
"""
Three-point benchmark scan over simple local arc admixtures.

This module extends the canonical endpoint-difference local fermion by the
one-sided local arc differences already packaged in
`local_interaction_point_fermion.py`:

    Lambda_trial
      = Lambda_join
      + c_+^(nabla) nabla_+ theta
      + c_-^(nabla) nabla_- theta.

These arc terms are purely nonzero-mode and therefore preserve the mixed
zero-mode coefficients (Theta_cm, Lambda_lat). The point of the scan is to see
whether the current three-point vacuum benchmark can distinguish them.

For the sampled candidate family below, it cannot: the benchmark channels remain
identical to the reduced/lightcone target because the explicit local benchmark
polynomials already collapse to Xi-degree 0 before the vacuum average.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import fermionic_graviton_contraction as fgc
import local_interaction_point_fermion as lif
import local_superstring_tree_benchmark as lstb
import local_vacuum_reduction as lvr
import prefactor_family_ranking as pfr
import superstring_prefactor_check as spc


DEFAULT_ARC_CANDIDATES: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.5, 0.0),
    (0.0, 0.5),
    (0.5, -0.5),
    (0.5, 0.5),
    (1.0, -1.0),
)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


def candidate_label(coeff_nabla_plus: float, coeff_nabla_minus: float) -> str:
    return f"({coeff_nabla_plus:g},{coeff_nabla_minus:g})"


def candidate_mixed_report(
    n1: int,
    n2: int,
    coeff_nabla_plus: float,
    coeff_nabla_minus: float,
) -> dict[str, Any]:
    decomposition = lif.canonical_local_candidate_with_arc_admixtures(
        n1,
        n2,
        coeff_nabla_plus=coeff_nabla_plus,
        coeff_nabla_minus=coeff_nabla_minus,
    )
    return {
        "candidate": candidate_label(coeff_nabla_plus, coeff_nabla_minus),
        "theta_cm_coefficient": decomposition.theta_cm_coefficient,
        "lambda_lat_coefficient": decomposition.lambda_lat_coefficient,
        "two_point_scalar": lvr.local_candidate_two_point_scalar_from_decomposition(
            decomposition
        ),
    }


def single_point_candidate_scan(
    n1: int = 128,
    n2: int = 192,
    *,
    arc_candidates: tuple[tuple[float, float], ...] = DEFAULT_ARC_CANDIDATES,
    alpha_prime: float = 1.0,
) -> dict[str, Any]:
    prefactor = spc.prefactor_data(
        n1,
        n2,
        alpha_prime,
        left_variant="second_order",
        right_variant="second_order",
    )
    lambda_ratio = n1 / (n1 + n2)
    rows = []
    max_abs_error = 0.0
    max_local_reduced_error = 0.0
    max_theta_cm_error = 0.0
    max_lambda_error = 0.0

    for coeff_nabla_plus, coeff_nabla_minus in arc_candidates:
        mixed = candidate_mixed_report(n1, n2, coeff_nabla_plus, coeff_nabla_minus)
        _, point_abs_error, point_local_reduced_error = lstb._benchmark_channel_rows(
            a_delta=prefactor.a_delta_reduced,
            b_qq=prefactor.b_qq_reduced,
            lambda_ratio=lambda_ratio,
            n1=n1,
            n2=n2,
            trace_dropped=True,
            two_point_scalar=mixed["two_point_scalar"],
        )
        theta_cm_error = abs(mixed["theta_cm_coefficient"])
        lambda_error = abs(mixed["lambda_lat_coefficient"] - 1.0)
        max_abs_error = max(max_abs_error, point_abs_error)
        max_local_reduced_error = max(max_local_reduced_error, point_local_reduced_error)
        max_theta_cm_error = max(max_theta_cm_error, float(theta_cm_error))
        max_lambda_error = max(max_lambda_error, float(lambda_error))
        rows.append(
            {
                **mixed,
                "max_abs_error": float(point_abs_error),
                "max_local_reduced_error": float(point_local_reduced_error),
                "theta_cm_error": float(theta_cm_error),
                "lambda_error": float(lambda_error),
            }
        )

    return {
        "parameters": {
            "n1": n1,
            "n2": n2,
            "lambda_ratio": float(lambda_ratio),
            "alpha_prime": float(alpha_prime),
            "arc_candidates": [list(item) for item in arc_candidates],
        },
        "rows": rows,
        "max_abs_error": float(max_abs_error),
        "max_local_reduced_error": float(max_local_reduced_error),
        "max_theta_cm_error": float(max_theta_cm_error),
        "max_lambda_error": float(max_lambda_error),
        "pass": (
            max_abs_error < 1.0e-12
            and max_local_reduced_error < 1.0e-12
            and max_theta_cm_error < 1.0e-12
            and max_lambda_error < 1.0e-12
        ),
    }


def family_scan(
    *,
    arc_candidates: tuple[tuple[float, float], ...] = DEFAULT_ARC_CANDIDATES,
    families: tuple[tuple[int, int], ...] = pfr.DEFAULT_RATIO_FAMILIES,
    scales: tuple[int, ...] = (16, 32, 64),
    alpha_prime: float = 1.0,
) -> dict[str, Any]:
    rows = []
    max_abs_error = 0.0
    max_local_reduced_error = 0.0
    for coeff_nabla_plus, coeff_nabla_minus in arc_candidates:
        for a, b in families:
            for scale in scales:
                n1 = a * scale
                n2 = b * scale
                prefactor = spc.prefactor_data(
                    n1,
                    n2,
                    alpha_prime,
                    left_variant="second_order",
                    right_variant="second_order",
                )
                lambda_ratio = n1 / (n1 + n2)
                two_point_scalar = lvr.local_candidate_two_point_scalar(
                    n1,
                    n2,
                    coeff_nabla_plus=coeff_nabla_plus,
                    coeff_nabla_minus=coeff_nabla_minus,
                )
                _, point_abs_error, point_local_reduced_error = lstb._benchmark_channel_rows(
                    a_delta=prefactor.a_delta_reduced,
                    b_qq=prefactor.b_qq_reduced,
                    lambda_ratio=lambda_ratio,
                    n1=n1,
                    n2=n2,
                    trace_dropped=True,
                    two_point_scalar=two_point_scalar,
                )
                max_abs_error = max(max_abs_error, point_abs_error)
                max_local_reduced_error = max(
                    max_local_reduced_error,
                    point_local_reduced_error,
                )
                rows.append(
                    {
                        "candidate": candidate_label(coeff_nabla_plus, coeff_nabla_minus),
                        "family": f"{a}:{b}",
                        "scale": scale,
                        "n1": n1,
                        "n2": n2,
                        "max_abs_error": float(point_abs_error),
                        "max_local_reduced_error": float(point_local_reduced_error),
                    }
                )
    return {
        "parameters": {
            "arc_candidates": [list(item) for item in arc_candidates],
            "families": [f"{a}:{b}" for a, b in families],
            "scales": list(scales),
            "alpha_prime": float(alpha_prime),
        },
        "rows": rows,
        "max_abs_error": float(max_abs_error),
        "max_local_reduced_error": float(max_local_reduced_error),
        "pass": max_abs_error < 1.0e-12 and max_local_reduced_error < 1.0e-12,
    }


def print_report(single: dict[str, Any], family: dict[str, Any]) -> None:
    print("=" * 112)
    print("LOCAL ARC CANDIDATE SCAN")
    print("=" * 112)
    print(
        f"single-point max_abs_error={single['max_abs_error']:.3e}, "
        f"single-point max_local_reduced_error={single['max_local_reduced_error']:.3e}"
    )
    print(
        f"single-point max_theta_cm_error={single['max_theta_cm_error']:.3e}, "
        f"single-point max_lambda_error={single['max_lambda_error']:.3e}"
    )
    print(
        f"family max_abs_error={family['max_abs_error']:.3e}, "
        f"family max_local_reduced_error={family['max_local_reduced_error']:.3e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    single = single_point_candidate_scan()
    family = family_scan()
    report = {"single_point": single, "family_scan": family}
    print_report(single, family)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n")
        print(f"Wrote JSON report to {args.json_out}")


if __name__ == "__main__":
    main()
