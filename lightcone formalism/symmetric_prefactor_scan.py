#!/usr/bin/env python3
"""
Dedicated scan of the symmetric support-three prefactor family.

We parameterize the family by

  t = t_- = -t_+

so that:

  left  = (-1 - t, 1 + 2t, -t)
  right = (1 + t, -1 - 2t, t).

This is the clean slice containing:
- the minimal stencil at t = 0,
- the standard one-sided second-order choice at t = 1/2,
- and the stronger nearby local stencils seen in the coarse ranking scans.

For each t we compute continuum extrapolations across a fixed set of ratios and
report:
- A_delta mean and flatness in lambda,
- N1*Bqq and sqrt(N1) eta_- smoothness and uncertainty,
- the rank by the existing family-ranking key,
- and a Pareto front over the shape/uncertainty metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import prefactor_family_ranking as pfr


def dominated_by(metrics_a: dict[str, object], metrics_b: dict[str, object]) -> bool:
    keys = [
        "A_delta_flat_rel",
        "A_delta_unc_rel",
        "N1_Bqq_quadratic_rms_rel",
        "N1_Bqq_unc_rel",
        "sqrtN_eta_minus_quadratic_rms_rel",
        "sqrtN_eta_minus_unc_rel",
    ]
    weakly_better = all(metrics_b[key] <= metrics_a[key] for key in keys)
    strictly_better = any(metrics_b[key] < metrics_a[key] for key in keys)
    return weakly_better and strictly_better


def scan_symmetric_family(
    scales: list[int],
    families: list[tuple[int, int]],
    alpha_prime: float,
    max_t: float,
    step: float,
) -> list[dict[str, object]]:
    candidates = pfr.scan_candidates(
        scales=scales,
        families=families,
        alpha_prime=alpha_prime,
        max_abs_t=max_t,
        step=step,
        symmetric_only=True,
    )
    filtered = []
    for candidate in candidates:
        t = candidate["right_t"]
        filtered.append(
            {
                "t": t,
                "left_t": candidate["left_t"],
                "right_t": candidate["right_t"],
                "rank": candidate["rank"],
                "rows": candidate["rows"],
                "metrics": candidate["metrics"],
            }
        )
    filtered.sort(key=lambda item: item["t"])

    for item in filtered:
        metrics = item["metrics"]
        item["pareto"] = not any(
            dominated_by(metrics, other["metrics"])
            for other in filtered
            if other is not item
        )
    return filtered


def print_report(candidates: list[dict[str, object]]) -> None:
    print("=" * 118)
    print("SYMMETRIC SUPPORT-THREE PREFACTOR SCAN")
    print("=" * 118)
    print(
        f"{'t':>7s} {'rank':>6s} {'pareto':>8s} {'blocked':>8s} "
        f"{'A_mean':>12s} {'A_flat':>10s} {'Bqq_smooth':>12s} "
        f"{'Bqq_unc':>10s} {'mean|Bqq|':>12s}"
    )
    print("-" * 96)
    for item in candidates:
        m = item["metrics"]
        print(
            f"{item['t']:7.3f} {item['rank']:6d} {str(item['pareto']):>8s} "
            f"{str(m['blocked']):>8s} {m['A_delta_mean']:12.9f} "
            f"{m['A_delta_flat_rel']:10.3e} {m['N1_Bqq_quadratic_rms_rel']:12.3e} "
            f"{m['N1_Bqq_unc_rel']:10.3e} {m['mean_abs_N1_Bqq']:12.6f}"
        )
    print()

    pareto = [item for item in candidates if item["pareto"]]
    if pareto:
        print("Pareto front:")
        for item in pareto:
            m = item["metrics"]
            print(
                f"  t = {item['t']:.3f}: rank {item['rank']}, "
                f"A_flat = {m['A_delta_flat_rel']:.3e}, "
                f"Bqq_smooth = {m['N1_Bqq_quadratic_rms_rel']:.3e}, "
                f"Bqq_unc = {m['N1_Bqq_unc_rel']:.3e}"
            )
        print()

    second_order = next((item for item in candidates if abs(item["t"] - 0.5) < 1.0e-12), None)
    if second_order is not None:
        m = second_order["metrics"]
        print("Second-order reference point:")
        print(
            f"  t = 0.500, rank = {second_order['rank']}, pareto = {second_order['pareto']}, "
            f"A_mean = {m['A_delta_mean']:.9f}, A_flat = {m['A_delta_flat_rel']:.3e}"
        )
        print(
            f"  mean|N1*Bqq| = {m['mean_abs_N1_Bqq']:.9f}, "
            f"Bqq_unc = {m['N1_Bqq_unc_rel']:.3e}, "
            f"eta_smooth = {m['sqrtN_eta_minus_quadratic_rms_rel']:.3e}"
        )
        print()


def markdown_report(candidates: list[dict[str, object]], params: dict[str, object]) -> str:
    lines = [
        "# Symmetric Support-Three Prefactor Scan",
        "",
        f"Grid: `0 <= t <= {params['max_t']}`, step `{params['step']}`",
        f"Scales: `{params['scales']}`",
        "",
        "| t | rank | pareto | blocked | A_mean | A_flat | Bqq_smooth | Bqq_unc | mean|Bqq| |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in candidates:
        m = item["metrics"]
        lines.append(
            "| "
            f"{item['t']:.3f} | {item['rank']} | {item['pareto']} | {m['blocked']} | "
            f"{m['A_delta_mean']:.9f} | {m['A_delta_flat_rel']:.3e} | "
            f"{m['N1_Bqq_quadratic_rms_rel']:.3e} | {m['N1_Bqq_unc_rel']:.3e} | "
            f"{m['mean_abs_N1_Bqq']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_scales(text: str) -> list[int]:
    return pfr.parse_scales(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-prime", type=float, default=1.0)
    parser.add_argument("--scales", type=parse_scales, default=pfr.DEFAULT_SCALES)
    parser.add_argument("--max-t", type=float, default=0.75)
    parser.add_argument("--step", type=float, default=0.125)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--markdown-out", type=str, default=None)
    args = parser.parse_args()

    candidates = scan_symmetric_family(
        scales=args.scales,
        families=pfr.DEFAULT_RATIO_FAMILIES,
        alpha_prime=args.alpha_prime,
        max_t=args.max_t,
        step=args.step,
    )
    print_report(candidates)

    report = {
        "parameters": {
            "alpha_prime": args.alpha_prime,
            "scales": args.scales,
            "max_t": args.max_t,
            "step": args.step,
        },
        "candidates": candidates,
    }
    if args.json_out is not None:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Wrote JSON report to {out_path}")
    if args.markdown_out is not None:
        md_path = Path(args.markdown_out)
        md_path.write_text(markdown_report(candidates, report["parameters"]))
        print(f"Wrote markdown report to {md_path}")


if __name__ == "__main__":
    main()
