#!/usr/bin/env python3
"""
On-shell split of the continuum bosonic three-tachyon cubic factor.

The continuum three-string vertex contains both:

    [mu(alpha_1,alpha_2,alpha_3)]^2

and the momentum-dependent tachyon exponent

    exp[- q_rel^2 / (2 gamma_cont)].

In the positive-width outgoing convention used in this repo and at alpha' = 1,
the on-shell three-tachyon kinematics imply an exact identity

    log mu(alpha)^2 = q_rel^2 / (2 gamma_cont).

This helper packages that identity and compares the discrete Richardson limit of
the Schur-complement exponent directly to the continuum Mandelstam mu-factor.
It does not solve the remaining overall cubic normalization problem; rather, it
isolates that problem by showing that the already-validated gamma_T benchmark
also reproduces the full continuum on-shell exponential.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import continuum_extrapolation as ce
import continuum_tachyon_benchmark as ctb
import tachyon_check as tc


DEFAULT_RATIO_FAMILIES: tuple[tuple[int, int], ...] = ctb.DEFAULT_RATIO_FAMILIES
DEFAULT_SCALES: tuple[int, ...] = ctb.DEFAULT_SCALES


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


def continuum_on_shell_exponent(alpha1: float, alpha2: float, alpha_prime: float = 1.0) -> float:
    alpha3 = alpha1 + alpha2
    gamma = ctb.continuum_gamma_target(alpha1, alpha2, alpha3)
    q_rel_sq = tc.on_shell_q_rel_sq(int(alpha1), int(alpha2), alpha_prime)
    return q_rel_sq / (2.0 * gamma)


def family_report(
    a: int,
    b: int,
    scales: list[int] | tuple[int, ...] = DEFAULT_SCALES,
    alpha_prime: float = 1.0,
) -> dict[str, object]:
    ns: list[int] = []
    discrete_exponents: list[float] = []
    rows: list[dict[str, float]] = []
    for scale in scales:
        n1 = a * scale
        n2 = b * scale
        data = tc.compute_tachyon_data(n1, n2, alpha_prime, d_perp=24)
        ns.append(n1)
        discrete_exponents.append(data.exponent)
        rows.append(
            {
                "scale": scale,
                "N1": n1,
                "N2": n2,
                "lambda": n1 / (n1 + n2),
                "discrete_exponent": data.exponent,
            }
        )

    extrap = ce.summary_to_dict(ce.summarize_extrapolation(ns, discrete_exponents))
    alpha1 = float(a)
    alpha2 = float(b)
    log_mu_sq = math.log(ctb.continuum_mu_squared(alpha1, alpha2, alpha1 + alpha2))
    continuum_exp = continuum_on_shell_exponent(alpha1, alpha2, alpha_prime)
    return {
        "family": f"{a}:{b}",
        "a": a,
        "b": b,
        "lambda": a / (a + b),
        "rows": rows,
        "discrete_exponent_extrapolation": extrap,
        "continuum_log_mu_squared": log_mu_sq,
        "continuum_on_shell_exponent": continuum_exp,
        "continuum_identity_error": continuum_exp - log_mu_sq,
        "discrete_vs_log_mu_abs_error": extrap["estimate"] - log_mu_sq,
        "discrete_vs_log_mu_rel_error": (extrap["estimate"] - log_mu_sq) / log_mu_sq,
    }


def full_report(
    families: list[tuple[int, int]] | tuple[tuple[int, int], ...] = DEFAULT_RATIO_FAMILIES,
    scales: list[int] | tuple[int, ...] = DEFAULT_SCALES,
    alpha_prime: float = 1.0,
) -> dict[str, object]:
    rows = [family_report(a, b, scales, alpha_prime) for a, b in families]
    return {
        "parameters": {
            "families": [f"{a}:{b}" for a, b in families],
            "scales": list(scales),
            "alpha_prime": alpha_prime,
        },
        "families": rows,
        "summary": {
            "max_continuum_identity_error": max(abs(row["continuum_identity_error"]) for row in rows),
            "max_discrete_vs_log_mu_abs_error": max(abs(row["discrete_vs_log_mu_abs_error"]) for row in rows),
            "max_discrete_vs_log_mu_rel_error": max(abs(row["discrete_vs_log_mu_rel_error"]) for row in rows),
        },
    }


def parse_scales(text: str) -> list[int]:
    return ctb.parse_scales(text)


def print_report(report: dict[str, object]) -> None:
    print("=" * 104)
    print("CONTINUUM BOSONIC TACHYON FACTOR SPLIT")
    print("=" * 104)
    print(
        f"max continuum identity error = {report['summary']['max_continuum_identity_error']:.3e}, "
        f"max discrete-vs-logmu abs error = {report['summary']['max_discrete_vs_log_mu_abs_error']:.3e}, "
        f"max discrete-vs-logmu rel error = {report['summary']['max_discrete_vs_log_mu_rel_error']:.3e}"
    )
    print()
    print(" family    exp_inf          log_mu^2         abs err        rel err")
    for row in report["families"]:
        extrap = row["discrete_exponent_extrapolation"]
        print(
            f" {row['family']:>5s}  "
            f"{extrap['estimate']:14.9f}  "
            f"{row['continuum_log_mu_squared']:14.9f}  "
            f"{row['discrete_vs_log_mu_abs_error']:12.3e}  "
            f"{row['discrete_vs_log_mu_rel_error']:12.3e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scales",
        type=parse_scales,
        default=list(DEFAULT_SCALES),
        help="comma-separated scale list",
    )
    parser.add_argument("--alpha-prime", type=float, default=1.0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report = full_report(scales=args.scales, alpha_prime=args.alpha_prime)
    print_report(report)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
