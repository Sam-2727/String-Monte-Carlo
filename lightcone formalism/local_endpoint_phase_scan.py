#!/usr/bin/env python3
"""
Finite-N endpoint-phase scan for the local superstring interaction-point fermion.

This module studies the simplest unresolved branch-point question that remains
inside the endpoint-linear family. We normalize the general phase family

    Lambda_trial(phi) = c(phi) [theta_{I_+} + exp(i phi) theta_{I_-}],

so that the reduced overlap-constrained zero-mode coefficient is fixed to
Lambda_lat = 1.  The resulting decomposition is then measured in the exact
finite-N mixed basis:

    Lambda_trial(phi) = Theta_cm * C_cm(phi) + Lambda_lat + Xi_loc(phi).

This is useful because the canonical endpoint difference corresponds to
phi = pi, while the most naive Dijkgraaf-Motl-style endpoint sum would point
instead toward phi = pi/2. The scan makes the finite-N consequences explicit:

1. phi = pi is the unique CM-free point in the equal-magnitude endpoint family,
2. it also minimizes the nonzero-mode two-point scalar after unit-Lambda
   normalization,
3. the naive DM-inspired endpoint sum is not a harmless rephasing of the
   canonical candidate; it carries substantial center-of-mass contamination and
   a much larger Xi_loc size.

This does not yet derive the true DM branch-point normalization. What it does
show is that a literal endpoint sum with relative phase `i` cannot be used as
the finite-N local candidate without additional local completion.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import local_interaction_point_fermion as lif
import local_vacuum_reduction as lvr


DEFAULT_PHASE_GRID = tuple(float(value) for value in np.linspace(-math.pi, math.pi, 129))
DEFAULT_SPECIAL_PHASES = (
    ("same_phase", 0.0),
    ("dm_plus_i", 0.5 * math.pi),
    ("canonical", math.pi),
    ("dm_minus_i", -0.5 * math.pi),
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
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    return value


def normalize_endpoint_phase_candidate(
    n1: int,
    n2: int,
    phase: float,
) -> dict[str, Any] | None:
    """
    Normalize theta_{I_+} + e^{i phi} theta_{I_-} to unit Lambda_lat coefficient.
    """
    coeff_i_plus = 1.0 + 0.0j
    coeff_i_minus = cmath.exp(1j * float(phase))
    raw = lif.decompose_join_linear_combination(n1, n2, coeff_i_plus, coeff_i_minus)
    if abs(raw.lambda_lat_coefficient) < 1.0e-12:
        return None
    overall_scale = 1.0 / raw.lambda_lat_coefficient
    normalized = lif.decompose_join_linear_combination(
        n1,
        n2,
        overall_scale * coeff_i_plus,
        overall_scale * coeff_i_minus,
    )
    two_point_scalar = lvr.local_candidate_two_point_scalar_from_decomposition(normalized)
    return {
        "phase": float(phase),
        "raw_lambda_lat_coefficient": raw.lambda_lat_coefficient,
        "overall_scale": overall_scale,
        "normalized_coeff_i_plus": normalized.coeff_i_plus,
        "normalized_coeff_i_minus": normalized.coeff_i_minus,
        "theta_cm_coefficient": normalized.theta_cm_coefficient,
        "lambda_lat_coefficient": normalized.lambda_lat_coefficient,
        "theta_cm_abs": float(abs(normalized.theta_cm_coefficient)),
        "leg1_oscillator_norm": float(np.linalg.norm(normalized.oscillator_row_leg1)),
        "leg2_oscillator_norm": float(np.linalg.norm(normalized.oscillator_row_leg2)),
        "two_point_scalar": float(two_point_scalar),
    }


def phase_family_scan(
    n1: int,
    n2: int,
    *,
    phases: tuple[float, ...] = DEFAULT_PHASE_GRID,
) -> dict[str, Any]:
    rows = []
    skipped_phases = []
    for phase in phases:
        row = normalize_endpoint_phase_candidate(n1, n2, phase)
        if row is None:
            skipped_phases.append(float(phase))
            continue
        rows.append(row)
    best_cm = min(rows, key=lambda row: row["theta_cm_abs"])
    best_two_point = min(rows, key=lambda row: row["two_point_scalar"])
    canonical_plus, canonical_minus = lif.canonical_endpoint_difference_coefficients(n1, n2)
    return {
        "parameters": {
            "n1": int(n1),
            "n2": int(n2),
            "num_phases": len(phases),
            "num_skipped_phases": len(skipped_phases),
        },
        "canonical_endpoint_coefficients": {
            "i_plus": canonical_plus,
            "i_minus": canonical_minus,
        },
        "best_cm_phase": best_cm,
        "best_two_point_phase": best_two_point,
        "skipped_phases": skipped_phases,
        "rows": rows,
    }


def special_phase_report(n1: int, n2: int) -> dict[str, Any]:
    return {
        label: normalize_endpoint_phase_candidate(n1, n2, phase)
        for label, phase in DEFAULT_SPECIAL_PHASES
    }


def family_stability_report(
    *,
    families: tuple[tuple[int, int], ...] = ((1, 3), (1, 2), (2, 3), (1, 1)),
    scales: tuple[int, ...] = (16, 32, 64),
    phases: tuple[float, ...] = DEFAULT_PHASE_GRID,
) -> dict[str, Any]:
    rows = []
    max_best_cm_error = 0.0
    max_best_two_point_error = 0.0
    for a, b in families:
        for scale in scales:
            n1 = a * scale
            n2 = b * scale
            scan = phase_family_scan(n1, n2, phases=phases)
            best_cm_phase = float(scan["best_cm_phase"]["phase"])
            best_two_point_phase = float(scan["best_two_point_phase"]["phase"])
            cm_error = min(abs(best_cm_phase - math.pi), abs(best_cm_phase + math.pi))
            two_point_error = min(
                abs(best_two_point_phase - math.pi),
                abs(best_two_point_phase + math.pi),
            )
            max_best_cm_error = max(max_best_cm_error, cm_error)
            max_best_two_point_error = max(max_best_two_point_error, two_point_error)
            rows.append(
                {
                    "family": f"{a}:{b}",
                    "scale": int(scale),
                    "n1": int(n1),
                    "n2": int(n2),
                    "best_cm_phase": best_cm_phase,
                    "best_two_point_phase": best_two_point_phase,
                    "best_cm_abs": float(scan["best_cm_phase"]["theta_cm_abs"]),
                    "best_two_point_scalar": float(scan["best_two_point_phase"]["two_point_scalar"]),
                }
            )
    return {
        "parameters": {
            "families": [f"{a}:{b}" for a, b in families],
            "scales": list(scales),
            "num_phases": len(phases),
        },
        "rows": rows,
        "max_best_cm_phase_error": float(max_best_cm_error),
        "max_best_two_point_phase_error": float(max_best_two_point_error),
    }


def print_report(report: dict[str, Any]) -> None:
    special = report["special"]
    family = report["family_scan"]
    print("=" * 112)
    print("LOCAL ENDPOINT PHASE SCAN")
    print("=" * 112)
    print(
        "Special candidates after unit-Lambda normalization:"
        f" canonical |Theta_cm|={special['canonical']['theta_cm_abs']:.3e},"
        f" dm_plus_i |Theta_cm|={special['dm_plus_i']['theta_cm_abs']:.3e},"
        f" dm_minus_i |Theta_cm|={special['dm_minus_i']['theta_cm_abs']:.3e},"
        f" same_phase |Theta_cm|={special['same_phase']['theta_cm_abs']:.3e}"
    )
    print(
        "Canonical vs DM two-point scalar:"
        f" canonical={special['canonical']['two_point_scalar']:.6f},"
        f" dm_plus_i={special['dm_plus_i']['two_point_scalar']:.6f},"
        f" dm_minus_i={special['dm_minus_i']['two_point_scalar']:.6f}"
    )
    print(
        f"Family stability: max best-CM phase error={family['max_best_cm_phase_error']:.3e}, "
        f"max best-two-point phase error={family['max_best_two_point_phase_error']:.3e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n1", type=int, default=128)
    parser.add_argument("--n2", type=int, default=192)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report = {
        "special": special_phase_report(args.n1, args.n2),
        "phase_scan": phase_family_scan(args.n1, args.n2),
        "family_scan": family_stability_report(),
    }
    print_report(report)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n")
        print(f"Wrote JSON report to {args.json_out}")


if __name__ == "__main__":
    main()
