#!/usr/bin/env python3
"""
Three-point scan over the full nearest-neighbor join-local fermion family.

This module extends the earlier arc-admixture checks to the full zero-average
nearest-neighbor basis supported in the resolved interaction region:

    Lambda_trial = Lambda_join
                 + c_{1,+} (theta_1^(1) - theta_0^(1))
                 + c_{1,-} (theta_0^(1) - theta_{N_1-1}^(1))
                 + c_{2,+} (theta_1^(2) - theta_0^(2))
                 + c_{2,-} (theta_0^(2) - theta_{N_2-1}^(2)).

The mixed zero-mode constraints remain fixed at

    Theta_cm = 0,   Lambda_lat = 1,

for every candidate in the family. The purpose of the scan is to test whether
the present three-point vacuum data are sensitive to this larger local family.
For this linear family, once the mixed zero-mode data are fixed, the
candidate-dependence of the diagonal vacuum reduction enters only through the
scalar two-point invariant

    C_Xi = <Xi_loc^a Xi_loc^b> delta_ab.

At the current stage, the question is not whether this family is the final
branch-point operator, but whether it already produces dangerous O(1)
ambiguities in the continuum benchmark.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import fermionic_graviton_contraction as fgc
import local_channel_response as lcr
import local_interaction_point_fermion as lif
import local_superstring_tree_benchmark as lstb
import local_vacuum_reduction as lvr


DEFAULT_CANDIDATES: tuple[tuple[float, float, float, float], ...] = (
    (0.0, 0.0, 0.0, 0.0),
    (0.5, 0.0, 0.0, 0.0),
    (0.0, 0.5, 0.0, 0.0),
    (0.0, 0.0, 0.5, 0.0),
    (0.0, 0.0, 0.0, 0.5),
    (0.5, -0.5, 0.0, 0.0),
    (0.0, 0.0, 0.5, -0.5),
    (0.5, 0.5, -0.5, -0.5),
    (1.0, -1.0, 1.0, -1.0),
)

DEFAULT_LAMBDA_GRID = lvr.DEFAULT_LAMBDA_GRID


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


def candidate_label(
    coeff_leg1_forward: float,
    coeff_leg1_backward: float,
    coeff_leg2_forward: float,
    coeff_leg2_backward: float,
) -> str:
    return (
        f"({coeff_leg1_forward:g},{coeff_leg1_backward:g},"
        f"{coeff_leg2_forward:g},{coeff_leg2_backward:g})"
    )


def candidate_decomposition_report(
    n1: int,
    n2: int,
    coeff_leg1_forward: float,
    coeff_leg1_backward: float,
    coeff_leg2_forward: float,
    coeff_leg2_backward: float,
) -> dict[str, Any]:
    decomposition = lif.canonical_nearest_neighbor_local_candidate(
        n1,
        n2,
        coeff_leg1_forward=coeff_leg1_forward,
        coeff_leg1_backward=coeff_leg1_backward,
        coeff_leg2_forward=coeff_leg2_forward,
        coeff_leg2_backward=coeff_leg2_backward,
    )
    return {
        "candidate": candidate_label(
            coeff_leg1_forward,
            coeff_leg1_backward,
            coeff_leg2_forward,
            coeff_leg2_backward,
        ),
        "theta_cm_coefficient": decomposition.theta_cm_coefficient,
        "lambda_lat_coefficient": decomposition.lambda_lat_coefficient,
        "two_point_scalar": lvr.local_candidate_two_point_scalar_from_decomposition(
            decomposition
        ),
    }


@lru_cache(maxsize=None)
def _cached_local_polynomial(
    key1: str,
    key2: str,
    key3: str,
    lambda_ratio: float,
    response_kind: str,
    trace_dropped: bool,
):
    polarizations = fgc.polarization_tensors()
    return lcr.local_channel_response_polynomial(
        polarizations[key1],
        polarizations[key2],
        polarizations[key3],
        lambda_ratio,
        response_kind=response_kind,
        trace_dropped=trace_dropped,
    )


def _contracted_response_from_labels(
    key1: str,
    key2: str,
    key3: str,
    lambda_ratio: float,
    *,
    response_kind: str,
    trace_dropped: bool,
    two_point_scalar: complex,
) -> complex:
    poly = _cached_local_polynomial(
        key1,
        key2,
        key3,
        float(lambda_ratio),
        response_kind,
        bool(trace_dropped),
    )
    return lvr.vacuum_contract_local_polynomial(poly, two_point_scalar=two_point_scalar)


def single_point_scan(
    n1: int = 128,
    n2: int = 192,
    *,
    candidates: tuple[tuple[float, float, float, float], ...] = DEFAULT_CANDIDATES,
    alpha_prime: float = 1.0,
) -> dict[str, Any]:
    prefactor = lstb.spc.prefactor_data(  # type: ignore[attr-defined]
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

    for coeffs in candidates:
        mixed = candidate_decomposition_report(n1, n2, *coeffs)
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
            "candidates": [list(item) for item in candidates],
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


def full_catalog_scan(
    *,
    n1: int = 128,
    n2: int = 192,
    candidates: tuple[tuple[float, float, float, float], ...] = DEFAULT_CANDIDATES,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    trace_dropped: bool = True,
) -> dict[str, Any]:
    keys = tuple(sorted(fgc.polarization_tensors().keys()))
    canonical_scalar = lvr.canonical_xi_two_point_scalar(n1, n2)

    rows = []
    max_qq_error = 0.0
    max_delta_error = 0.0

    for coeffs in candidates:
        mixed = candidate_decomposition_report(n1, n2, *coeffs)
        two_point_scalar = mixed["two_point_scalar"]
        candidate_max_qq = 0.0
        candidate_max_delta = 0.0
        for lambda_ratio in lambda_grid:
            for key1 in keys:
                for key2 in keys:
                    for key3 in keys:
                        qq_value = _contracted_response_from_labels(
                            key1,
                            key2,
                            key3,
                            lambda_ratio,
                            response_kind="qq",
                            trace_dropped=trace_dropped,
                            two_point_scalar=two_point_scalar,
                        )
                        qq_canonical = _contracted_response_from_labels(
                            key1,
                            key2,
                            key3,
                            lambda_ratio,
                            response_kind="qq",
                            trace_dropped=trace_dropped,
                            two_point_scalar=canonical_scalar,
                        )
                        delta_value = _contracted_response_from_labels(
                            key1,
                            key2,
                            key3,
                            lambda_ratio,
                            response_kind="delta",
                            trace_dropped=trace_dropped,
                            two_point_scalar=two_point_scalar,
                        )
                        delta_canonical = _contracted_response_from_labels(
                            key1,
                            key2,
                            key3,
                            lambda_ratio,
                            response_kind="delta",
                            trace_dropped=trace_dropped,
                            two_point_scalar=canonical_scalar,
                        )
                        candidate_max_qq = max(candidate_max_qq, abs(qq_value - qq_canonical))
                        candidate_max_delta = max(
                            candidate_max_delta,
                            abs(delta_value - delta_canonical),
                        )
        max_qq_error = max(max_qq_error, float(candidate_max_qq))
        max_delta_error = max(max_delta_error, float(candidate_max_delta))
        rows.append(
            {
                **mixed,
                "max_qq_catalog_error": float(candidate_max_qq),
                "max_delta_catalog_error": float(candidate_max_delta),
            }
        )

    return {
        "parameters": {
            "n1": int(n1),
            "n2": int(n2),
            "lambda_grid": list(lambda_grid),
            "trace_dropped": bool(trace_dropped),
            "candidates": [list(item) for item in candidates],
        },
        "rows": rows,
        "max_qq_catalog_error": float(max_qq_error),
        "max_delta_catalog_error": float(max_delta_error),
        "pass": max_qq_error < 1.0e-12 and max_delta_error < 1.0e-12,
    }


def print_report(single: dict[str, Any], catalog: dict[str, Any]) -> None:
    print("=" * 112)
    print("LOCAL NEAREST-NEIGHBOR FAMILY SCAN")
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
        f"catalog max_qq_error={catalog['max_qq_catalog_error']:.3e}, "
        f"catalog max_delta_error={catalog['max_delta_catalog_error']:.3e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    single = single_point_scan()
    catalog = full_catalog_scan()
    report = {"single_point": single, "catalog_scan": catalog}
    print_report(single, catalog)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n")
        print(f"Wrote JSON report to {args.json_out}")


if __name__ == "__main__":
    main()
