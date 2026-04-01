#!/usr/bin/env python3
"""
Continuum benchmark for the bosonic three-tachyon Schur complement.

This module packages the cleanest current analytic comparison for the bosonic
three-tachyon sector. In the standard closed lightcone-SFT normalization
reviewed in the bundled source references, the cubic tachyon exponent is
controlled by

    tau0 = alpha1 log alpha1 + alpha2 log alpha2 - alpha3 log alpha3

in the positive-width outgoing convention alpha3 = alpha1 + alpha2 > 0. With

    P = alpha1 p2 - alpha2 p1 = - alpha3 q_rel ,

the continuum cubic vertex predicts

    exp[ tau0 * P^2 / (2 alpha1 alpha2 alpha3) ]
    = exp[ - q_rel^2 / (2 gamma_cont) ],

so that

    gamma_cont = - alpha1 alpha2 / (alpha3 tau0).

The overall cubic normalization is still left open. This benchmark only fixes
the kinematic Schur-complement sector and records the accompanying HIKKO
mu(alpha)^2 factor for later normalization work.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import continuum_extrapolation as ce
import tachyon_check as tc


DEFAULT_RATIO_FAMILIES: tuple[tuple[int, int], ...] = (
    (1, 3),
    (1, 2),
    (3, 5),
    (2, 3),
    (1, 1),
)

DEFAULT_SCALES: tuple[int, ...] = (8, 16, 32, 64, 128, 256)


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


def continuum_tau0(alpha1: float, alpha2: float, alpha3: float | None = None) -> float:
    """
    Positive-width outgoing convention for the closed cubic vertex.

    The reference formulas are usually written with alpha_1 + alpha_2 + alpha_3 = 0
    and alpha_3 < 0. In the note/code we instead use alpha_3 = alpha_1 + alpha_2 > 0.
    The translated tau0 is therefore

        tau0 = alpha1 log alpha1 + alpha2 log alpha2 - alpha3 log alpha3.
    """
    if alpha3 is None:
        alpha3 = alpha1 + alpha2
    return (
        alpha1 * math.log(alpha1)
        + alpha2 * math.log(alpha2)
        - alpha3 * math.log(alpha3)
    )


def continuum_mu_squared(alpha1: float, alpha2: float, alpha3: float | None = None) -> float:
    """
    HIKKO/lightcone three-string prefactor [mu(alpha_1,alpha_2,alpha_3)]^2
    translated to the positive-width outgoing convention.
    """
    if alpha3 is None:
        alpha3 = alpha1 + alpha2
    tau0 = continuum_tau0(alpha1, alpha2, alpha3)
    log_mu_sq = -2.0 * tau0 * (1.0 / alpha1 + 1.0 / alpha2 - 1.0 / alpha3)
    return math.exp(log_mu_sq)


def continuum_gamma_target(alpha1: float, alpha2: float, alpha3: float | None = None) -> float:
    """Continuum Schur-complement target gamma_T(alpha_1,alpha_2,alpha_3)."""
    if alpha3 is None:
        alpha3 = alpha1 + alpha2
    tau0 = continuum_tau0(alpha1, alpha2, alpha3)
    return -alpha1 * alpha2 / (alpha3 * tau0)


def parse_family(text: str) -> tuple[int, int]:
    pieces = [piece.strip() for piece in text.split(",") if piece.strip()]
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError(f"expected a,b family, got {text!r}")
    a, b = (int(piece) for piece in pieces)
    if a <= 0 or b <= 0:
        raise argparse.ArgumentTypeError("require a,b > 0")
    return a, b


def parse_scales(text: str) -> list[int]:
    pieces = [piece.strip() for piece in text.split(",") if piece.strip()]
    scales = [int(piece) for piece in pieces]
    if not scales:
        raise argparse.ArgumentTypeError("need at least one scale")
    if any(scale <= 0 for scale in scales):
        raise argparse.ArgumentTypeError("all scales must be positive")
    return scales


def family_label(a: int, b: int) -> str:
    return f"{a}:{b}"


def family_gamma_report(
    a: int,
    b: int,
    scales: list[int] | tuple[int, ...] = DEFAULT_SCALES,
    alpha_prime: float = 1.0,
) -> dict[str, object]:
    ns: list[int] = []
    gamma_values: list[float] = []
    rows: list[dict[str, float]] = []
    for scale in scales:
        n1 = a * scale
        n2 = b * scale
        data = tc.compute_tachyon_data(n1, n2, alpha_prime, d_perp=24)
        ns.append(n1)
        gamma_values.append(data.gamma_t)
        rows.append(
            {
                "scale": scale,
                "N1": n1,
                "N2": n2,
                "lambda": n1 / (n1 + n2),
                "gamma_T": data.gamma_t,
            }
        )

    alpha1 = float(a)
    alpha2 = float(b)
    alpha3 = float(a + b)
    target = continuum_gamma_target(alpha1, alpha2, alpha3)
    mu_sq = continuum_mu_squared(alpha1, alpha2, alpha3)
    extrap = ce.summary_to_dict(ce.summarize_extrapolation(ns, gamma_values))
    estimate = extrap["estimate"]
    abs_error = estimate - target
    rel_error = abs_error / target

    return {
        "family": family_label(a, b),
        "a": a,
        "b": b,
        "lambda": a / (a + b),
        "alpha1": alpha1,
        "alpha2": alpha2,
        "alpha3": alpha3,
        "tau0": continuum_tau0(alpha1, alpha2, alpha3),
        "mu_squared": mu_sq,
        "rows": rows,
        "gamma_T_extrapolation": extrap,
        "gamma_T_continuum_target": target,
        "gamma_T_abs_error": abs_error,
        "gamma_T_rel_error": rel_error,
    }


def full_report(
    families: list[tuple[int, int]] | tuple[tuple[int, int], ...] = DEFAULT_RATIO_FAMILIES,
    scales: list[int] | tuple[int, ...] = DEFAULT_SCALES,
    alpha_prime: float = 1.0,
) -> dict[str, object]:
    rows = [family_gamma_report(a, b, scales, alpha_prime) for a, b in families]
    max_abs_error = max(abs(row["gamma_T_abs_error"]) for row in rows)
    max_rel_error = max(abs(row["gamma_T_rel_error"]) for row in rows)
    return {
        "parameters": {
            "alpha_prime": alpha_prime,
            "families": [family_label(a, b) for a, b in families],
            "scales": list(scales),
        },
        "families": rows,
        "summary": {
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
        },
    }


def print_report(report: dict[str, object]) -> None:
    print("=" * 104)
    print("CONTINUUM BOSONIC THREE-TACHYON BENCHMARK")
    print("=" * 104)
    print(
        f"Families: {report['parameters']['families']} | "
        f"scales: {report['parameters']['scales']}"
    )
    print(
        f"Max abs error = {report['summary']['max_abs_error']:.6e}, "
        f"max rel error = {report['summary']['max_rel_error']:.6e}"
    )
    print()
    print(
        " family    lambda      gamma_inf         gamma_cont         abs err        "
        "rel err        mu^2"
    )
    for row in report["families"]:
        extrap = row["gamma_T_extrapolation"]
        print(
            f" {row['family']:>5s}  "
            f"{row['lambda']:8.5f}  "
            f"{extrap['estimate']:14.9f}  "
            f"{row['gamma_T_continuum_target']:14.9f}  "
            f"{row['gamma_T_abs_error']:12.3e}  "
            f"{row['gamma_T_rel_error']:12.3e}  "
            f"{row['mu_squared']:10.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        nargs="*",
        type=parse_family,
        default=list(DEFAULT_RATIO_FAMILIES),
        help="optional fixed-ratio families a,b",
    )
    parser.add_argument(
        "--scales",
        type=parse_scales,
        default=list(DEFAULT_SCALES),
        help="comma-separated scale list",
    )
    parser.add_argument("--alpha-prime", type=float, default=1.0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report = full_report(args.families, args.scales, args.alpha_prime)
    print_report(report)

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n")
        print(f"\nWrote JSON report to {args.json_out}")


if __name__ == "__main__":
    main()
