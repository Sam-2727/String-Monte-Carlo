#!/usr/bin/env python3
"""
Tree-level nonzero-mode vacuum reduction of the local Xi_loc sectors.

This module evaluates the first explicit nonzero-mode contraction for the
current canonical local interaction-point fermion candidate

    Lambda_join = sqrt(N1 N2 / N3) (theta_{I_+} - theta_{I_-})
                = Lambda_lat + Xi_loc.

At three points with all nonzero-mode oscillators in their external vacuum, the
incoming legs remain independent and spinor-diagonal. In the present canonical
endpoint-difference candidate, Xi_loc has the form

    Xi_loc^a = sum_i r_i Gamma_i^a,

with the same real row coefficients for each SO(8) spinor component `a`, and
with `Gamma_i^a` drawn from a product quasifree vacuum that is diagonal in the
spinor index. Consequently

    <Xi_loc^a Xi_loc^b>_vac = C_Xi(N1,N2) delta^{ab},

for one scalar `C_Xi`, and every exact local channel polynomial built from
distinct Xi indices has vanishing positive-degree vacuum expectation.

The practical consequence is simple but important: for the current canonical
local candidate, the explicit local three-point channel polynomials produced by
`local_channel_response.py` reduce under the nonzero-mode vacuum contraction to
their Xi-degree-0 pieces. In particular, the benchmark dilaton quartic sector
vanishes exactly after this contraction.

This is still conditional on the current endpoint-difference local candidate and
does not fix the missing branch-point normalization of the true local GS
interaction-point fermion. What it does show is that, for the present candidate
and for three-point vacuum external states, the unresolved Xi_loc sectors do not
modify the sampled local channel catalog.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import fermionic_graviton_contraction as fgc
import local_channel_catalog as lcc
import local_channel_response as lcr
import local_interaction_point_fermion as lif


DEFAULT_LAMBDA_GRID = (0.25, 0.4, 0.5)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    return value


def canonical_xi_two_point_scalar(n1: int, n2: int) -> float:
    """
    Diagonal Xi two-point scalar for the canonical endpoint-difference candidate.

    In the product oscillator vacuum, each real fermionic oscillator contributes
    a two-point value `1/2`. Therefore

        <Xi_loc^a Xi_loc^b> = (1/2) delta^{ab} (||r_1||^2 + ||r_2||^2),

    where `r_1`, `r_2` are the leg-1 and leg-2 oscillator rows appearing in the
    exact decomposition of `Lambda_join`.
    """
    decomposition = lif.canonical_local_difference_decomposition(n1, n2)
    return local_candidate_two_point_scalar_from_decomposition(decomposition)


def local_candidate_two_point_scalar_from_decomposition(
    decomposition: lif.MixedLinearFermionDecomposition,
) -> float:
    """Diagonal Xi two-point scalar for a generic linear local candidate."""
    return 0.5 * float(
        np.vdot(decomposition.oscillator_row_leg1, decomposition.oscillator_row_leg1).real
        + np.vdot(decomposition.oscillator_row_leg2, decomposition.oscillator_row_leg2).real
    )


def local_candidate_two_point_scalar(
    n1: int,
    n2: int,
    *,
    coeff_nabla_plus: complex = 0.0,
    coeff_nabla_minus: complex = 0.0,
) -> float:
    """
    Diagonal Xi two-point scalar for the canonical endpoint family with arc admixtures.
    """
    decomposition = lif.canonical_local_candidate_with_arc_admixtures(
        n1,
        n2,
        coeff_nabla_plus=coeff_nabla_plus,
        coeff_nabla_minus=coeff_nabla_minus,
    )
    return local_candidate_two_point_scalar_from_decomposition(decomposition)


def canonical_xi_two_point_scalar_closed_form(n1: int, n2: int) -> float:
    """
    Closed form of the same scalar using ||site_row||^2 = 1 - 1/N.
    """
    n3 = n1 + n2
    return float((2 * n1 * n2 - n1 - n2) / (2 * n3))


def diagonal_spinor_wick(
    monomial: tuple[int, ...],
    two_point_scalar: complex,
) -> complex:
    """
    Wick contraction with spinor-diagonal two-point data.

    The current local channel polynomials are sparse Grassmann monomials in the
    eight Xi spinor labels. Because the contraction is diagonal in the SO(8)
    index, any monomial with all indices distinct has zero expectation unless it
    is the degree-0 monomial.
    """
    if len(monomial) % 2 == 1:
        return 0.0j
    if not monomial:
        return 1.0 + 0.0j

    first = monomial[0]
    total = 0.0j
    for position in range(1, len(monomial)):
        if monomial[position] != first:
            continue
        sign = -1.0 if (position - 1) % 2 else 1.0
        remainder = monomial[1:position] + monomial[position + 1 :]
        total += sign * two_point_scalar * diagonal_spinor_wick(
            remainder,
            two_point_scalar,
        )
    return total


def vacuum_contract_local_polynomial(
    poly: dict[tuple[int, ...], complex],
    *,
    two_point_scalar: complex = 1.0,
) -> complex:
    """
    Contract the Xi_loc polynomial against the spinor-diagonal vacuum.
    """
    total = 0.0j
    for monomial, coefficient in poly.items():
        total += complex(coefficient) * diagonal_spinor_wick(monomial, two_point_scalar)
    return total


def contracted_local_channel_response(
    epsilon_1: np.ndarray,
    epsilon_2: np.ndarray,
    epsilon_3: np.ndarray,
    lambda_ratio: float,
    *,
    response_kind: str = "qq",
    trace_dropped: bool = True,
    two_point_scalar: complex = 1.0,
) -> complex:
    poly = lcr.local_channel_response_polynomial(
        epsilon_1,
        epsilon_2,
        epsilon_3,
        lambda_ratio,
        response_kind=response_kind,
        trace_dropped=trace_dropped,
    )
    return vacuum_contract_local_polynomial(poly, two_point_scalar=two_point_scalar)


def contracted_catalog_summary(
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    *,
    response_kind: str = "qq",
    trace_dropped: bool = True,
) -> dict[str, Any]:
    """
    Collapse the local channel catalog after the Xi_loc vacuum contraction.

    Because the current vacuum contraction kills every positive-degree Xi
    monomial, the contracted response on the sampled grid is exactly the stored
    Xi-degree-0 value for each channel.
    """
    catalog = lcc.channel_catalog(
        lambda_grid,
        response_kind=response_kind,
        trace_dropped=trace_dropped,
    )

    rows = []
    counts: dict[str, int] = {}
    for row in catalog["rows"]:
        contracted_values = [complex(value) for value in row["xi_zero_values"]]
        max_abs = max(abs(value) for value in contracted_values) if contracted_values else 0.0
        category = "vanishing" if max_abs < 1.0e-12 else "reduced_only"
        counts[category] = counts.get(category, 0) + 1
        rows.append(
            {
                "channel": row["channel"],
                "category": category,
                "contracted_values": contracted_values,
                "max_abs_value": float(max_abs),
            }
        )

    return {
        "lambda_grid": list(lambda_grid),
        "response_kind": response_kind,
        "trace_dropped": trace_dropped,
        "counts": counts,
        "rows": rows,
    }


def benchmark_vacuum_reduction_report(
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
) -> dict[str, Any]:
    polarizations = fgc.polarization_tensors()
    channels = (
        ("perp23", "perp23", "parallel"),
        ("perp23", "perp24", "parallel"),
        ("parallel", "perp23", "perp23"),
        ("perp23", "perp23", "dilaton"),
    )
    rows = []
    max_abs_error = 0.0
    for lambda_ratio in lambda_grid:
        for channel in channels:
            poly = lcr.local_channel_response_polynomial(
                polarizations[channel[0]],
                polarizations[channel[1]],
                polarizations[channel[2]],
                lambda_ratio,
                response_kind="qq",
                trace_dropped=True,
            )
            contracted = vacuum_contract_local_polynomial(poly)
            reduced = complex(lcr.xi_zero_component(poly))
            max_abs_error = max(max_abs_error, float(abs(contracted - reduced)))
            rows.append(
                {
                    "lambda_ratio": float(lambda_ratio),
                    "channel": list(channel),
                    "contracted": contracted,
                    "reduced": reduced,
                    "abs_error": float(abs(contracted - reduced)),
                    "profile": lcr.xi_degree_profile(poly),
                }
            )
    return {
        "lambda_grid": list(lambda_grid),
        "rows": rows,
        "max_abs_error": max_abs_error,
    }


def print_summary() -> None:
    qq_summary = contracted_catalog_summary(response_kind="qq", trace_dropped=True)
    delta_summary = contracted_catalog_summary(response_kind="delta", trace_dropped=True)
    benchmark = benchmark_vacuum_reduction_report()
    print("=" * 96)
    print("LOCAL XI VACUUM REDUCTION")
    print("=" * 96)
    print(
        "For the current canonical endpoint-difference local candidate, the "
        "spinor-diagonal nonzero-mode vacuum contraction kills every positive-"
        "degree Xi_loc sector."
    )
    print()
    print(f"Trace-dropped qq counts after contraction: {qq_summary['counts']}")
    print(f"Trace-dropped delta counts after contraction: {delta_summary['counts']}")
    print(
        "Benchmark local-to-reduced collapse max error: "
        f"{benchmark['max_abs_error']:.3e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    qq_summary = contracted_catalog_summary(response_kind="qq", trace_dropped=True)
    delta_summary = contracted_catalog_summary(response_kind="delta", trace_dropped=True)
    benchmark = benchmark_vacuum_reduction_report()
    report = {
        "qq_summary": qq_summary,
        "delta_summary": delta_summary,
        "benchmark": benchmark,
        "canonical_two_point_examples": {
            "16_24": canonical_xi_two_point_scalar(16, 24),
            "16_24_closed_form": canonical_xi_two_point_scalar_closed_form(16, 24),
            "128_192": canonical_xi_two_point_scalar(128, 192),
            "128_192_closed_form": canonical_xi_two_point_scalar_closed_form(128, 192),
        },
    }

    print_summary()
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n")
        print(f"Wrote JSON report to {args.json_out}")


if __name__ == "__main__":
    main()
