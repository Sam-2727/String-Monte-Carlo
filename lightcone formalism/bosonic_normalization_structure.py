#!/usr/bin/env python3
"""
Machine-readable bosonic normalization-structure diagnostics.

This module packages the strongest current finite-N evidence on the bosonic
normalization problem without pretending to solve the remaining continuum match.
It makes two already-observed structures reproducible in one place:

1. the gauge-invariant full three-leg tail

       C_tail + 7 log N1 + 7 log N2 - 5 log N3
       + pi (1/N1 + 1/N2 - 1/N3)
       + (pi^2/72) (1/N1^2 + 1/N2^2 + 1/N3^2),

2. the corresponding gauge-fixed incoming/outgoing one-string tail fits.
"""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

import tachyon_check as tc


EXACT_C2 = math.pi * math.pi / 72.0


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    return value


def grid_rows(
    *,
    min_n: int = 4,
    max_n: int = 60,
    max_n3: int = 120,
    alpha_prime: float = 1.0,
    d_perp: int = 24,
) -> list[tuple[int, int, int, float]]:
    rows = []
    for n1 in range(min_n, max_n + 1):
        for n2 in range(min_n, max_n + 1):
            if n1 + n2 > max_n3:
                continue
            data = tc.compute_tachyon_data(n1, n2, alpha_prime, d_perp)
            rows.append((n1, n2, n1 + n2, data.log_required_norm_noext))
    return rows


def invariant_tail_summary(
    rows: list[tuple[int, int, int, float]],
) -> dict[str, float]:
    shifted = []
    for n1, n2, n3, log_creq in rows:
        tail = (
            7.0 * math.log(n1)
            + 7.0 * math.log(n2)
            - 5.0 * math.log(n3)
            + math.pi * (1.0 / n1 + 1.0 / n2 - 1.0 / n3)
            + EXACT_C2 * (1.0 / (n1 * n1) + 1.0 / (n2 * n2) + 1.0 / (n3 * n3))
        )
        shifted.append(log_creq - tail)
    shifted_arr = np.array(shifted)
    constant = float(np.mean(shifted_arr))
    residuals = shifted_arr - constant
    return {
        "constant": constant,
        "rmse": float(math.sqrt(np.mean(residuals * residuals))),
        "max_abs": float(np.max(np.abs(residuals))),
        "num_rows": len(rows),
    }


def _fit_leg_functions(
    rows: list[tuple[int, int, int, float]],
) -> dict[str, Any]:
    solution = tc.solve_exact_leg_factorization_from_rows(rows)
    return {
        "incoming_functions": solution["incoming_functions"],
        "outgoing_functions": solution["outgoing_functions"],
        "incoming_list": solution["incoming_list"],
        "outgoing_list": solution["outgoing_list"],
        "constant_term": solution["constant_term"],
    }


def _fit_tail_model(
    functions: dict[int, float],
    labels: list[int],
    start: int,
) -> dict[str, Any]:
    ns = [n for n in labels if n >= start]
    design = np.array(
        [[math.log(n), 1.0 / n, 1.0 / (n * n), 1.0, float(n)] for n in ns]
    )
    target = np.array([functions[n] for n in ns])
    coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ coeffs
    return {
        "fit_start": start,
        "num_points": len(ns),
        "c_log": float(coeffs[0]),
        "c_1_over_n": float(coeffs[1]),
        "c_2_over_n2": float(coeffs[2]),
        "c_0": float(coeffs[3]),
        "c_linear": float(coeffs[4]),
        "rmse": float(math.sqrt(np.mean(residuals * residuals))),
        "max_abs": float(np.max(np.abs(residuals))),
    }


def _fixed_tail_residual(
    functions: dict[int, float],
    labels: list[int],
    start: int,
    *,
    c_log: float,
    c_1: float,
    c_2: float,
) -> dict[str, Any]:
    ns = [n for n in labels if n >= start]
    design = np.array([[1.0, float(n)] for n in ns])
    target = np.array(
        [
            functions[n]
            - c_log * math.log(n)
            - c_1 / n
            - c_2 / (n * n)
            for n in ns
        ]
    )
    coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ coeffs
    return {
        "fit_start": start,
        "num_points": len(ns),
        "c_0": float(coeffs[0]),
        "c_linear": float(coeffs[1]),
        "rmse": float(math.sqrt(np.mean(residuals * residuals))),
        "max_abs": float(np.max(np.abs(residuals))),
    }


def factorized_leg_tail_summary(
    rows: list[tuple[int, int, int, float]],
    *,
    incoming_start: int = 30,
    outgoing_start: int = 60,
) -> dict[str, Any]:
    leg_data = _fit_leg_functions(rows)
    incoming_fit = _fit_tail_model(
        leg_data["incoming_functions"],
        leg_data["incoming_list"],
        incoming_start,
    )
    outgoing_fit = _fit_tail_model(
        leg_data["outgoing_functions"],
        leg_data["outgoing_list"],
        outgoing_start,
    )
    incoming_fixed = _fixed_tail_residual(
        leg_data["incoming_functions"],
        leg_data["incoming_list"],
        incoming_start,
        c_log=7.0,
        c_1=math.pi,
        c_2=EXACT_C2,
    )
    outgoing_fixed = _fixed_tail_residual(
        leg_data["outgoing_functions"],
        leg_data["outgoing_list"],
        outgoing_start,
        c_log=-5.0,
        c_1=-math.pi,
        c_2=EXACT_C2,
    )
    return {
        "incoming_fit": incoming_fit,
        "outgoing_fit": outgoing_fit,
        "incoming_fixed_tail": incoming_fixed,
        "outgoing_fixed_tail": outgoing_fixed,
    }


@lru_cache(maxsize=None)
def default_summary() -> dict[str, Any]:
    rows = grid_rows()
    return {
        "grid": {"min_n": 4, "max_n": 60, "max_n3": 120, "num_rows": len(rows)},
        "invariant_tail": invariant_tail_summary(rows),
        "factorized_leg_tails": factorized_leg_tail_summary(rows),
    }


def print_summary(summary: dict[str, Any]) -> None:
    invariant = summary["invariant_tail"]
    incoming = summary["factorized_leg_tails"]["incoming_fit"]
    outgoing = summary["factorized_leg_tails"]["outgoing_fit"]
    incoming_fixed = summary["factorized_leg_tails"]["incoming_fixed_tail"]
    outgoing_fixed = summary["factorized_leg_tails"]["outgoing_fixed_tail"]
    print("=" * 96)
    print("BOSONIC NORMALIZATION STRUCTURE")
    print("=" * 96)
    print(f"Grid: {summary['grid']}")
    print(
        "Invariant tail: "
        f"C_tail={invariant['constant']:.9f}, "
        f"rmse={invariant['rmse']:.3e}, "
        f"max_abs={invariant['max_abs']:.3e}"
    )
    print(
        "Incoming tail fit: "
        f"c_log={incoming['c_log']:.9f}, "
        f"c_1={incoming['c_1_over_n']:.9f}, "
        f"c_2={incoming['c_2_over_n2']:.9f}, "
        f"rmse={incoming['rmse']:.3e}"
    )
    print(
        "Outgoing tail fit: "
        f"c_log={outgoing['c_log']:.9f}, "
        f"c_1={outgoing['c_1_over_n']:.9f}, "
        f"c_2={outgoing['c_2_over_n2']:.9f}, "
        f"rmse={outgoing['rmse']:.3e}"
    )
    print(
        "Fixed incoming tail residual: "
        f"rmse={incoming_fixed['rmse']:.3e}, max_abs={incoming_fixed['max_abs']:.3e}"
    )
    print(
        "Fixed outgoing tail residual: "
        f"rmse={outgoing_fixed['rmse']:.3e}, max_abs={outgoing_fixed['max_abs']:.3e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="optional path for a JSON dump",
    )
    args = parser.parse_args()
    summary = default_summary()
    print_summary(summary)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(json_safe(summary), indent=2))
        print()
        print(f"Wrote JSON summary to {args.json_out}")


if __name__ == "__main__":
    main()
