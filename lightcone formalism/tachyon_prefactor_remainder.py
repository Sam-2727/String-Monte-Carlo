#!/usr/bin/env python3
"""
Fixed-ratio report for the remaining bosonic three-tachyon scalar prefactor.

The discrete three-tachyon normalization diagnostic

    log C_req^(B)(N1,N2)

already contains the full on-shell exponential. The helpers
`continuum_tachyon_benchmark.py` and `continuum_tachyon_factor_split.py`
showed that this exponential matches the continuum Schur-complement / HIKKO
factor `mu(alpha)^2`. What remains unresolved is the scalar prefactor after
that known continuum piece is removed.

For a fixed-ratio family `N1 = a s`, `N2 = b s`, `N3 = (a+b) s`, define

    R_pref(N1,N2) = log C_req^(B)(N1,N2) - log mu(a,b,a+b)^2.

Using the already-fitted invariant large-N tail,

    C_tail + 7 log N1 + 7 log N2 - 5 log N3
    + pi (1/N1 + 1/N2 - 1/N3)
    + (pi^2/72)(1/N1^2 + 1/N2^2 + 1/N3^2),

this module packages the family-dependent target

    R_pref(a s, b s)
      = 9 log s + R_target(a,b) + O(s^{-1}),

where `R_target(a,b)` is fully explicit in terms of `C_tail`, the family
polynomial logs, and `log mu(a,b,a+b)^2`.

This is still not the final Mandelstam/HIKKO coupling match. It is the
machine-readable object that any such normalization match must reproduce.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import bosonic_normalization_structure as bns
import continuum_extrapolation as ce
import continuum_tachyon_benchmark as ctb
import tachyon_check as tc


DEFAULT_RATIO_FAMILIES: tuple[tuple[int, int], ...] = ctb.DEFAULT_RATIO_FAMILIES
DEFAULT_SCALES: tuple[int, ...] = ctb.DEFAULT_SCALES
EXACT_C2 = bns.EXACT_C2


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


def family_label(a: int, b: int) -> str:
    return f"{a}:{b}"


def family_log_mu_squared(a: int, b: int) -> float:
    return math.log(ctb.continuum_mu_squared(float(a), float(b), float(a + b)))


def family_log_polynomial_piece(a: int, b: int) -> float:
    return 7.0 * math.log(a) + 7.0 * math.log(b) - 5.0 * math.log(a + b)


def scale_tail_piece(a: int, b: int, scale: int) -> float:
    n1 = a * scale
    n2 = b * scale
    n3 = n1 + n2
    return (
        9.0 * math.log(scale)
        + math.pi * (1.0 / n1 + 1.0 / n2 - 1.0 / n3)
        + EXACT_C2 * (1.0 / (n1 * n1) + 1.0 / (n2 * n2) + 1.0 / (n3 * n3))
    )


def invariant_tail_constant() -> float:
    return float(bns.default_summary()["invariant_tail"]["constant"])


def family_prefactor_target(a: int, b: int) -> float:
    return (
        invariant_tail_constant()
        + family_log_polynomial_piece(a, b)
        - family_log_mu_squared(a, b)
    )


def family_prefactor_report(
    a: int,
    b: int,
    scales: list[int] | tuple[int, ...] = DEFAULT_SCALES,
    alpha_prime: float = 1.0,
) -> dict[str, object]:
    target = family_prefactor_target(a, b)
    log_mu_sq = family_log_mu_squared(a, b)

    ns: list[int] = []
    reduced_rows: list[float] = []
    rows: list[dict[str, float]] = []
    for scale in scales:
        n1 = a * scale
        n2 = b * scale
        data = tc.compute_tachyon_data(n1, n2, alpha_prime, d_perp=24)
        tail_scale = scale_tail_piece(a, b, scale)
        prefactor_remainder = data.log_required_norm_noext - log_mu_sq
        reduced_value = prefactor_remainder - tail_scale
        ns.append(n1)
        reduced_rows.append(reduced_value)
        rows.append(
            {
                "scale": scale,
                "N1": n1,
                "N2": n2,
                "lambda": n1 / (n1 + n2),
                "log_required_norm_noext": data.log_required_norm_noext,
                "continuum_log_mu_squared": log_mu_sq,
                "prefactor_remainder": prefactor_remainder,
                "scale_tail_piece": tail_scale,
                "reduced_prefactor_value": reduced_value,
                "target_abs_error": reduced_value - target,
            }
        )

    extrap = ce.summary_to_dict(ce.summarize_extrapolation(ns, reduced_rows))
    estimate = extrap["estimate"]
    abs_error = estimate - target
    rel_error = abs_error / abs(target)
    return {
        "family": family_label(a, b),
        "a": a,
        "b": b,
        "lambda": a / (a + b),
        "log_mu_squared": log_mu_sq,
        "family_log_polynomial_piece": family_log_polynomial_piece(a, b),
        "prefactor_target": target,
        "rows": rows,
        "reduced_prefactor_extrapolation": extrap,
        "prefactor_target_abs_error": abs_error,
        "prefactor_target_rel_error": rel_error,
    }


def full_report(
    families: list[tuple[int, int]] | tuple[tuple[int, int], ...] = DEFAULT_RATIO_FAMILIES,
    scales: list[int] | tuple[int, ...] = DEFAULT_SCALES,
    alpha_prime: float = 1.0,
) -> dict[str, object]:
    family_rows = [family_prefactor_report(a, b, scales, alpha_prime) for a, b in families]
    return {
        "parameters": {
            "families": [family_label(a, b) for a, b in families],
            "scales": list(scales),
            "alpha_prime": alpha_prime,
            "invariant_tail_constant": invariant_tail_constant(),
        },
        "families": family_rows,
        "summary": {
            "max_abs_error": max(abs(row["prefactor_target_abs_error"]) for row in family_rows),
            "max_rel_error": max(abs(row["prefactor_target_rel_error"]) for row in family_rows),
            "max_final_row_abs_error": max(
                abs(row["rows"][-1]["target_abs_error"]) for row in family_rows
            ),
        },
    }


def parse_scales(text: str) -> list[int]:
    return ctb.parse_scales(text)


def print_report(report: dict[str, object]) -> None:
    print("=" * 104)
    print("BOSONIC TACHYON PREFACTOR REMAINDER")
    print("=" * 104)
    print(
        f"max abs error = {report['summary']['max_abs_error']:.3e}, "
        f"max rel error = {report['summary']['max_rel_error']:.3e}, "
        f"max largest-scale row error = {report['summary']['max_final_row_abs_error']:.3e}"
    )
    print()
    print(" family   target            extrapolated       abs err        rel err")
    for row in report["families"]:
        extrap = row["reduced_prefactor_extrapolation"]
        print(
            f" {row['family']:>5s}  "
            f"{row['prefactor_target']:16.9f}  "
            f"{extrap['estimate']:16.9f}  "
            f"{row['prefactor_target_abs_error']:12.3e}  "
            f"{row['prefactor_target_rel_error']:12.3e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        nargs="*",
        type=ctb.parse_family,
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


if __name__ == "__main__":
    main()
