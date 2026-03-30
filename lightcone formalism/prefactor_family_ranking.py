#!/usr/bin/env python3
"""
Rank support-three local prefactor families by continuum behavior across lambda.

This script scans the one-sided three-point family

  left  = (-1 + t_+, 1 - 2 t_+, t_+)
  right = (1 + t_-, -1 - 2 t_-, t_-)

and evaluates each candidate by:

- continuum extrapolations at several fixed ratios lambda = N1 / (N1 + N2),
- smoothness of the extrapolated lambda-dependence,
- flatness of A_delta(lambda),
- sign and monotonicity diagnostics,
- and extrapolation-uncertainty envelopes.

The point is not to prove a unique stencil, but to identify which candidates
are numerically the most coherent across the low-point tests.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import continuum_extrapolation as ce
import superstring_prefactor_check as spc


DEFAULT_RATIO_FAMILIES = [(1, 3), (1, 2), (3, 5), (2, 3), (1, 1)]
DEFAULT_SCALES = [8, 16, 32, 64, 128, 256]


def parse_scales(text: str) -> list[int]:
    pieces = [piece.strip() for piece in text.split(",") if piece.strip()]
    scales = [int(piece) for piece in pieces]
    if not scales:
        raise argparse.ArgumentTypeError("need at least one scale")
    if any(scale <= 0 for scale in scales):
        raise argparse.ArgumentTypeError("all scales must be positive")
    return scales


def quadratic_rms(xs: list[float], ys: list[float]) -> float:
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    design = np.column_stack([np.ones_like(x), x, x * x])
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coeffs
    return float(math.sqrt(np.mean(residual * residual)))


def family_extrapolations(
    left_t: float,
    right_t: float,
    scales: list[int],
    families: list[tuple[int, int]],
    alpha_prime: float,
) -> list[dict[str, object]]:
    rows = []
    for a, b in families:
        ns = []
        n1_bqq = []
        sqrt_eta_plus = []
        sqrt_eta_minus = []
        a_delta_vals = []
        lam = a / (a + b)
        for scale in scales:
            n1 = a * scale
            n2 = b * scale
            data = spc.prefactor_data_three_point_family(
                n1, n2, left_t, right_t, alpha_prime
            )
            ns.append(n1)
            n1_bqq.append(n1 * data.b_qq_reduced)
            sqrt_eta_plus.append(math.sqrt(n1) * data.eta_plus)
            sqrt_eta_minus.append(math.sqrt(n1) * data.eta_minus)
            a_delta_vals.append(data.a_delta_reduced)
        rows.append(
            {
                "family": f"{a}:{b}",
                "a": a,
                "b": b,
                "lambda": lam,
                "N1_Bqq": ce.summary_to_dict(ce.summarize_extrapolation(ns, n1_bqq)),
                "sqrtN_eta_plus": ce.summary_to_dict(
                    ce.summarize_extrapolation(ns, sqrt_eta_plus)
                ),
                "sqrtN_eta_minus": ce.summary_to_dict(
                    ce.summarize_extrapolation(ns, sqrt_eta_minus)
                ),
                "A_delta": ce.summary_to_dict(
                    ce.summarize_extrapolation(ns, a_delta_vals)
                ),
            }
        )
    rows.sort(key=lambda row: row["lambda"])
    return rows


def candidate_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    lambdas = [row["lambda"] for row in rows]
    bqq_vals = [row["N1_Bqq"]["estimate"] for row in rows]
    bqq_unc = [row["N1_Bqq"]["uncertainty"] for row in rows]
    eta_vals = [row["sqrtN_eta_minus"]["estimate"] for row in rows]
    eta_unc = [row["sqrtN_eta_minus"]["uncertainty"] for row in rows]
    a_vals = [row["A_delta"]["estimate"] for row in rows]
    a_unc = [row["A_delta"]["uncertainty"] for row in rows]

    mean_abs_bqq = float(np.mean(np.abs(bqq_vals)))
    mean_abs_eta = float(np.mean(np.abs(eta_vals)))
    a_mean = float(np.mean(a_vals))

    bqq_positive = all(value > 0.0 for value in bqq_vals)
    eta_negative = all(value < 0.0 for value in eta_vals)
    bqq_monotone = all(
        bqq_vals[index] < bqq_vals[index + 1]
        for index in range(len(bqq_vals) - 1)
    )
    eta_monotone = all(
        eta_vals[index] > eta_vals[index + 1]
        for index in range(len(eta_vals) - 1)
    )

    blocked = max(abs(value) for value in bqq_vals) < 1.0e-6
    a_flat_rel = float(
        max(abs(value - a_mean) for value in a_vals) / max(abs(a_mean), 1.0e-12)
    )
    bqq_smooth_rel = float(
        quadratic_rms(lambdas, bqq_vals) / max(mean_abs_bqq, 1.0e-12)
    )
    eta_smooth_rel = float(
        quadratic_rms(lambdas, eta_vals) / max(mean_abs_eta, 1.0e-12)
    )
    a_smooth_rel = float(
        quadratic_rms(lambdas, a_vals) / max(abs(a_mean), 1.0e-12)
    )
    bqq_unc_rel = float(
        np.mean([unc / max(abs(val), 1.0e-12) for unc, val in zip(bqq_unc, bqq_vals)])
    )
    eta_unc_rel = float(
        np.mean([unc / max(abs(val), 1.0e-12) for unc, val in zip(eta_unc, eta_vals)])
    )
    a_unc_rel = float(
        np.mean([unc / max(abs(a_mean), 1.0e-12) for unc in a_unc])
    )

    return {
        "mean_abs_N1_Bqq": mean_abs_bqq,
        "mean_abs_sqrtN_eta_minus": mean_abs_eta,
        "A_delta_mean": a_mean,
        "A_delta_flat_rel": a_flat_rel,
        "N1_Bqq_quadratic_rms_rel": bqq_smooth_rel,
        "sqrtN_eta_minus_quadratic_rms_rel": eta_smooth_rel,
        "A_delta_quadratic_rms_rel": a_smooth_rel,
        "N1_Bqq_unc_rel": bqq_unc_rel,
        "sqrtN_eta_minus_unc_rel": eta_unc_rel,
        "A_delta_unc_rel": a_unc_rel,
        "blocked": blocked,
        "N1_Bqq_positive": bqq_positive,
        "sqrtN_eta_minus_negative": eta_negative,
        "N1_Bqq_monotone": bqq_monotone,
        "sqrtN_eta_minus_monotone": eta_monotone,
    }


def ranking_key(candidate: dict[str, object]) -> tuple[object, ...]:
    metrics = candidate["metrics"]
    return (
        metrics["blocked"],
        not metrics["N1_Bqq_positive"],
        not metrics["sqrtN_eta_minus_negative"],
        not metrics["N1_Bqq_monotone"],
        not metrics["sqrtN_eta_minus_monotone"],
        metrics["A_delta_flat_rel"],
        metrics["A_delta_unc_rel"],
        metrics["N1_Bqq_quadratic_rms_rel"],
        metrics["sqrtN_eta_minus_quadratic_rms_rel"],
        metrics["N1_Bqq_unc_rel"],
        -metrics["mean_abs_N1_Bqq"],
    )


def scan_candidates(
    scales: list[int],
    families: list[tuple[int, int]],
    alpha_prime: float,
    max_abs_t: float,
    step: float,
    symmetric_only: bool,
) -> list[dict[str, object]]:
    if step <= 0.0:
        raise ValueError("step must be positive")
    n_steps = int(round(2.0 * max_abs_t / step))
    ts = [-max_abs_t + index * step for index in range(n_steps + 1)]

    candidates = []
    for left_t in ts:
        for right_t in ts:
            if symmetric_only and abs(left_t + right_t) > 1.0e-12:
                continue
            rows = family_extrapolations(left_t, right_t, scales, families, alpha_prime)
            candidates.append(
                {
                    "left_t": left_t,
                    "right_t": right_t,
                    "rows": rows,
                    "metrics": candidate_metrics(rows),
                }
            )
    candidates.sort(key=ranking_key)
    for index, candidate in enumerate(candidates, start=1):
        candidate["rank"] = index
    return candidates


def print_top_candidates(candidates: list[dict[str, object]], top_k: int) -> None:
    print("=" * 116)
    print("SUPPORT-THREE PREFACTOR FAMILY RANKING")
    print("=" * 116)
    print(
        f"{'rank':>4s} {'t_+':>8s} {'t_-':>8s} {'blocked':>8s} "
        f"{'A_flat':>10s} {'Bqq_smooth':>12s} {'eta_smooth':>12s} "
        f"{'Bqq_unc':>10s} {'mean|Bqq|':>12s}"
    )
    print("-" * 96)
    for candidate in candidates[:top_k]:
        metrics = candidate["metrics"]
        print(
            f"{candidate['rank']:4d} "
            f"{candidate['left_t']:8.3f} {candidate['right_t']:8.3f} "
            f"{str(metrics['blocked']):>8s} "
            f"{metrics['A_delta_flat_rel']:10.3e} "
            f"{metrics['N1_Bqq_quadratic_rms_rel']:12.3e} "
            f"{metrics['sqrtN_eta_minus_quadratic_rms_rel']:12.3e} "
            f"{metrics['N1_Bqq_unc_rel']:10.3e} "
            f"{metrics['mean_abs_N1_Bqq']:12.6f}"
        )
    print()

    if candidates:
        best = candidates[0]
        metrics = best["metrics"]
        print("Best candidate detail:")
        print(
            f"  t_+ = {best['left_t']:.3f}, t_- = {best['right_t']:.3f}, "
            f"blocked = {metrics['blocked']}"
        )
        print(
            f"  monotone: Bqq = {metrics['N1_Bqq_monotone']}, "
            f"eta_- = {metrics['sqrtN_eta_minus_monotone']}"
        )
        print(
            f"  A_delta mean = {metrics['A_delta_mean']:.9f}, "
            f"relative flatness = {metrics['A_delta_flat_rel']:.3e}"
        )
        print("  continuum values by lambda:")
        print(
            f"  {'family':>8s} {'lambda':>8s} {'N1*Bqq':>12s} "
            f"{'sqrtN eta_-':>14s} {'A_delta':>12s}"
        )
        for row in best["rows"]:
            print(
                f"  {row['family']:>8s} {row['lambda']:8.5f} "
                f"{row['N1_Bqq']['estimate']:12.9f} "
                f"{row['sqrtN_eta_minus']['estimate']:14.9f} "
                f"{row['A_delta']['estimate']:12.9f}"
            )
        print()


def markdown_report(
    candidates: list[dict[str, object]],
    top_k: int,
    params: dict[str, object],
) -> str:
    lines = [
        "# Support-Three Prefactor Family Ranking",
        "",
        (
            f"Grid: `|t| <= {params['max_abs_t']}`, step `{params['step']}`, "
            f"symmetric_only = `{params['symmetric_only']}`"
        ),
        f"Scales: `{params['scales']}`",
        "",
        "## Top Candidates",
        "",
        "| rank | t_+ | t_- | blocked | A_flat | Bqq_smooth | eta_smooth | Bqq_unc | mean|Bqq| |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in candidates[:top_k]:
        metrics = candidate["metrics"]
        lines.append(
            "| "
            f"{candidate['rank']} | {candidate['left_t']:.3f} | {candidate['right_t']:.3f} | "
            f"{metrics['blocked']} | "
            f"{metrics['A_delta_flat_rel']:.3e} | "
            f"{metrics['N1_Bqq_quadratic_rms_rel']:.3e} | "
            f"{metrics['sqrtN_eta_minus_quadratic_rms_rel']:.3e} | "
            f"{metrics['N1_Bqq_unc_rel']:.3e} | "
            f"{metrics['mean_abs_N1_Bqq']:.6f} |"
        )
    if candidates:
        best = candidates[0]
        lines.extend(
            [
                "",
                "## Best Candidate Detail",
                "",
                f"Best candidate: `t_+ = {best['left_t']:.3f}`, `t_- = {best['right_t']:.3f}`",
                "",
                "| family | lambda | N1*Bqq(∞) | sqrtN eta_-(∞) | A_delta(∞) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in best["rows"]:
            lines.append(
                "| "
                f"{row['family']} | {row['lambda']:.5f} | "
                f"{row['N1_Bqq']['estimate']:.9f} | "
                f"{row['sqrtN_eta_minus']['estimate']:.9f} | "
                f"{row['A_delta']['estimate']:.9f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-prime", type=float, default=1.0)
    parser.add_argument("--scales", type=parse_scales, default=DEFAULT_SCALES)
    parser.add_argument("--max-abs-t", type=float, default=0.75)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--symmetric-only", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--markdown-out", type=str, default=None)
    args = parser.parse_args()

    candidates = scan_candidates(
        scales=args.scales,
        families=DEFAULT_RATIO_FAMILIES,
        alpha_prime=args.alpha_prime,
        max_abs_t=args.max_abs_t,
        step=args.step,
        symmetric_only=args.symmetric_only,
    )
    print_top_candidates(candidates, args.top_k)

    params = {
        "alpha_prime": args.alpha_prime,
        "scales": args.scales,
        "max_abs_t": args.max_abs_t,
        "step": args.step,
        "symmetric_only": args.symmetric_only,
    }
    report = {"parameters": params, "candidates": candidates}

    if args.json_out is not None:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Wrote JSON report to {out_path}")
    if args.markdown_out is not None:
        md_path = Path(args.markdown_out)
        md_path.write_text(markdown_report(candidates, args.top_k, params))
        print(f"Wrote markdown report to {md_path}")


if __name__ == "__main__":
    main()
