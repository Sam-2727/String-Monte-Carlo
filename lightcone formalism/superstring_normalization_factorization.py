#!/usr/bin/env python3
"""
Check whether the surviving reduced-Lambda graviton branch differs only by normalization.

This is the explicit-fermionic counterpart of the earlier proxy test. We use
the continuum-extrapolated benchmark channel

    A_diag(lambda, t) = A_F(e23, e23, e_parallel)

on the positive symmetric branch t > 0 and ask whether

    A_diag(lambda, t) ≈ N(t) F(lambda)

is approximately rank one. With the explicit trace-dropped response reduction,

    A_diag(lambda, t) = 4 sqrt(14) (1-lambda)^2 B_qq(lambda, t),

so this is equivalently a factorization check on the remaining bosonic
coefficient B_qq along the unblocked branch. As with the underlying channel
data, this is a statement about the reduced Lambda ansatz, not yet about a
derived local finite-N fermionic interaction-point operator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import superstring_decisive_test as sdt


REFERENCE_LAMBDA = 0.4


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


def build_matrix(
    alpha_prime: float,
    scales: list[int],
    max_t: float,
    step: float,
    min_positive_t: float,
) -> dict[str, object]:
    report = sdt.run_decisive_scan(
        alpha_prime=alpha_prime,
        scales=scales,
        max_t=max_t,
        step=step,
        min_t=min_positive_t,
    )

    rows = []
    lambdas = None
    for summary in report["summaries"]:
        if summary["blocked"]:
            continue
        diag = []
        diag_unc = []
        local_lambdas = []
        for row in summary["rows"]:
            local_lambdas.append(float(row["lambda"]))
            diag.append(row["N1_scaled_diag_summary"]["estimate"])
            diag_unc.append(row["N1_scaled_diag_summary"]["uncertainty"])
        if lambdas is None:
            lambdas = local_lambdas
        rows.append(
            {
                "t": float(summary["t"]),
                "diag": diag,
                "diag_uncertainty": diag_unc,
            }
        )

    if not rows:
        raise ValueError("no positive-t candidates found")
    if lambdas is None:
        raise ValueError("no lambda data found")

    matrix = np.array([row["diag"] for row in rows], dtype=float)
    uncertainty = np.array([row["diag_uncertainty"] for row in rows], dtype=float)
    return {
        "lambdas": lambdas,
        "rows": rows,
        "matrix": matrix,
        "uncertainty": uncertainty,
    }


def analyze_factorization(
    alpha_prime: float,
    scales: list[int],
    max_t: float,
    step: float,
    min_positive_t: float,
    reference_lambda: float = REFERENCE_LAMBDA,
    reference_t: float = 0.5,
) -> dict[str, object]:
    data = build_matrix(alpha_prime, scales, max_t, step, min_positive_t)
    lambdas = list(data["lambdas"])
    matrix = data["matrix"]
    uncertainty = data["uncertainty"]
    rows = data["rows"]

    try:
        ref_lambda_index = lambdas.index(reference_lambda)
    except ValueError as exc:
        raise ValueError(f"reference lambda {reference_lambda} not present") from exc

    t_values = [row["t"] for row in rows]
    try:
        ref_t_index = t_values.index(reference_t)
    except ValueError as exc:
        raise ValueError(f"reference t {reference_t} not present") from exc

    normalized = matrix / matrix[:, [ref_lambda_index]]
    normalized_unc = uncertainty / np.abs(matrix[:, [ref_lambda_index]])

    ref_profile = normalized[ref_t_index]
    max_diff_from_ref = np.max(np.abs(normalized - ref_profile), axis=1)
    pointwise_spread = np.ptp(normalized, axis=0)

    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    rank1 = singular_values[0] * np.outer(u[:, 0], vt[0, :])
    rank1_rel_frob_error = float(np.linalg.norm(matrix - rank1) / np.linalg.norm(matrix))
    sigma2_over_sigma1 = float(singular_values[1] / singular_values[0])

    return {
        "parameters": {
            "alpha_prime": alpha_prime,
            "scales": scales,
            "max_t": max_t,
            "step": step,
            "min_positive_t": min_positive_t,
            "reference_lambda": reference_lambda,
            "reference_t": reference_t,
        },
        "lambdas": lambdas,
        "t_values": t_values,
        "reference_profile": ref_profile.tolist(),
        "normalized_profiles": normalized.tolist(),
        "normalized_profile_uncertainty": normalized_unc.tolist(),
        "max_diff_from_reference_profile": max_diff_from_ref.tolist(),
        "pointwise_spread": pointwise_spread.tolist(),
        "singular_values": singular_values.tolist(),
        "rank1_rel_frob_error": rank1_rel_frob_error,
        "sigma2_over_sigma1": sigma2_over_sigma1,
        "rows": rows,
        "factorization_pass": bool(
            rank1_rel_frob_error < 2.0e-4
            and sigma2_over_sigma1 < 2.0e-4
            and float(np.max(max_diff_from_ref)) < 1.0e-3
        ),
    }


def print_report(report: dict[str, object]) -> None:
    print("=" * 114)
    print("SUPERSTRING FERMIONIC NORMALIZATION FACTORIZATION")
    print("=" * 114)
    print(
        "Factorization status: "
        f"{report['factorization_pass']} "
        f"(rank1 rel Frobenius error = {report['rank1_rel_frob_error']:.3e}, "
        f"sigma2/sigma1 = {report['sigma2_over_sigma1']:.3e}, "
        f"max normalized profile spread = {max(report['pointwise_spread']):.3e})"
    )
    print(
        f"Reference calibration: lambda = {report['parameters']['reference_lambda']}, "
        f"t = {report['parameters']['reference_t']}"
    )
    print()
    print(
        f"{'t':>7s} {'max diff vs ref':>16s} "
        f"{'Adiag(1:3)':>14s} {'Adiag(2:3)':>14s} {'Adiag(1:1)':>14s}"
    )
    print("-" * 78)
    for t, row, max_diff in zip(
        report["t_values"],
        report["rows"],
        report["max_diff_from_reference_profile"],
    ):
        print(
            f"{t:7.3f} {max_diff:16.3e} "
            f"{row['diag'][0]:14.6f} {row['diag'][3]:14.6f} {row['diag'][4]:14.6f}"
        )
    print()
    print("Reference-normalized lambda profile:")
    print("  lambdas = " + ", ".join(f"{lam:.5f}" for lam in report["lambdas"]))
    print("  profile = " + ", ".join(f"{value:.9f}" for value in report["reference_profile"]))


def markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# Superstring Fermionic Normalization Factorization",
        "",
        f"- factorization_pass: `{report['factorization_pass']}`",
        f"- rank1 rel Frobenius error: `{report['rank1_rel_frob_error']:.3e}`",
        f"- sigma2/sigma1: `{report['sigma2_over_sigma1']:.3e}`",
        f"- max normalized profile spread: `{max(report['pointwise_spread']):.3e}`",
        "",
        "| t | max diff vs ref | Adiag(1:3) | Adiag(2:3) | Adiag(1:1) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for t, row, max_diff in zip(
        report["t_values"],
        report["rows"],
        report["max_diff_from_reference_profile"],
    ):
        lines.append(
            "| "
            f"{t:.3f} | {max_diff:.3e} | "
            f"{row['diag'][0]:.6f} | {row['diag'][3]:.6f} | {row['diag'][4]:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_scales(text: str) -> list[int]:
    return sdt.parse_scales(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-prime", type=float, default=1.0)
    parser.add_argument("--scales", type=parse_scales, default=[16, 32, 64, 128])
    parser.add_argument("--max-t", type=float, default=0.75)
    parser.add_argument("--step", type=float, default=0.125)
    parser.add_argument("--min-positive-t", type=float, default=0.125)
    parser.add_argument("--reference-lambda", type=float, default=REFERENCE_LAMBDA)
    parser.add_argument("--reference-t", type=float, default=0.5)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--markdown-out", type=str, default=None)
    args = parser.parse_args()

    report = analyze_factorization(
        alpha_prime=args.alpha_prime,
        scales=args.scales,
        max_t=args.max_t,
        step=args.step,
        min_positive_t=args.min_positive_t,
        reference_lambda=args.reference_lambda,
        reference_t=args.reference_t,
    )
    print_report(report)

    if args.json_out is not None:
        json_path = Path(args.json_out)
        json_path.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n")
        print(f"\nWrote JSON report to {json_path}")
    if args.markdown_out is not None:
        md_path = Path(args.markdown_out)
        md_path.write_text(markdown_report(json_safe(report)) + "\n")
        print(f"Wrote markdown report to {md_path}")


if __name__ == "__main__":
    main()
