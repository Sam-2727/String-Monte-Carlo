#!/usr/bin/env python3
"""
Positive-branch optimizer for the symmetric graviton proxy scan.

This is a lightweight wrapper around `graviton_proxy_scan.py` that answers a
practical question:

  given the current vector-block proxy, does the positive symmetric branch
  have an interior optimum, or does the ranking simply drift toward larger t?

The output is meant to guide numerical interpretation, not to declare a final
physical choice of stencil.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import graviton_proxy_scan as gps


def parse_scales(text: str) -> list[int]:
    return gps.sps.parse_scales(text)


def monotone_decreasing(values: list[float]) -> bool:
    return all(values[index] >= values[index + 1] for index in range(len(values) - 1))


def monotone_increasing(values: list[float]) -> bool:
    return all(values[index] <= values[index + 1] for index in range(len(values) - 1))


def build_report(
    alpha_prime: float,
    scales: list[int],
    t_min: float,
    t_max: float,
    step: float,
) -> dict[str, object]:
    candidates = gps.scan_proxies(
        alpha_prime=alpha_prime,
        scales=scales,
        max_t=t_max,
        step=step,
        min_t=t_min,
    )
    candidates.sort(key=lambda item: item["t"])

    t_values = [candidate["t"] for candidate in candidates]
    ratio_smooth = [candidate["proxy_metrics"]["M_ratio_smooth_rel"] for candidate in candidates]
    mqq_unc = [candidate["proxy_metrics"]["M_qq_unc_rel"] for candidate in candidates]
    mean_abs_mqq = [candidate["proxy_metrics"]["mean_abs_M_qq"] for candidate in candidates]

    best_by_rank = min(candidates, key=lambda item: item["rank_proxy"])
    best_by_ratio = min(candidates, key=lambda item: item["proxy_metrics"]["M_ratio_smooth_rel"])
    best_by_unc = min(candidates, key=lambda item: item["proxy_metrics"]["M_qq_unc_rel"])
    best_by_size = max(candidates, key=lambda item: item["proxy_metrics"]["mean_abs_M_qq"])
    second_order = next((item for item in candidates if abs(item["t"] - 0.5) < 1.0e-12), None)

    return {
        "parameters": {
            "alpha_prime": alpha_prime,
            "scales": scales,
            "t_min": t_min,
            "t_max": t_max,
            "step": step,
        },
        "trend": {
            "t_values": t_values,
            "ratio_smooth_monotone_decreasing": monotone_decreasing(ratio_smooth),
            "mqq_unc_monotone_decreasing": monotone_decreasing(mqq_unc),
            "mean_abs_mqq_monotone_increasing": monotone_increasing(mean_abs_mqq),
        },
        "best_by_rank": best_by_rank,
        "best_by_ratio_smooth": best_by_ratio,
        "best_by_mqq_unc": best_by_unc,
        "best_by_mean_abs_mqq": best_by_size,
        "second_order": second_order,
        "candidates": candidates,
    }


def print_report(report: dict[str, object]) -> None:
    print("=" * 118)
    print("GRAVITON PROXY OPTIMIZER")
    print("=" * 118)
    params = report["parameters"]
    trend = report["trend"]
    print(
        f"t-range: [{params['t_min']}, {params['t_max']}], step {params['step']} | "
        f"scales: {params['scales']}"
    )
    print(
        "Trend checks: "
        f"ratio_smooth decreasing = {trend['ratio_smooth_monotone_decreasing']}, "
        f"Mqq_unc decreasing = {trend['mqq_unc_monotone_decreasing']}, "
        f"mean|Mqq| increasing = {trend['mean_abs_mqq_monotone_increasing']}"
    )
    print()

    def one_line(label: str, candidate: dict[str, object]) -> None:
        metrics = candidate["proxy_metrics"]
        print(
            f"{label:20s} t = {candidate['t']:.3f}, "
            f"proxy rank = {candidate['rank_proxy']}, "
            f"ratio_smooth = {metrics['M_ratio_smooth_rel']:.3e}, "
            f"Mqq_unc = {metrics['M_qq_unc_rel']:.3e}, "
            f"mean|Mqq| = {metrics['mean_abs_M_qq']:.6f}"
        )

    one_line("best by rank", report["best_by_rank"])
    one_line("best by ratio_smooth", report["best_by_ratio_smooth"])
    one_line("best by Mqq_unc", report["best_by_mqq_unc"])
    one_line("best by mean|Mqq|", report["best_by_mean_abs_mqq"])
    if report["second_order"] is not None:
        one_line("second-order ref", report["second_order"])
    print()
    print("Positive-branch sample:")
    print(
        f"{'t':>7s} {'proxy rank':>10s} {'ratio_smooth':>14s} "
        f"{'Mqq_unc':>12s} {'mean|Mqq|':>12s}"
    )
    print("-" * 66)
    for candidate in report["candidates"]:
        metrics = candidate["proxy_metrics"]
        print(
            f"{candidate['t']:7.3f} {candidate['rank_proxy']:10d} "
            f"{metrics['M_ratio_smooth_rel']:14.3e} "
            f"{metrics['M_qq_unc_rel']:12.3e} "
            f"{metrics['mean_abs_M_qq']:12.6f}"
        )
    print()


def markdown_report(report: dict[str, object]) -> str:
    trend = report["trend"]
    lines = [
        "# Graviton Proxy Optimizer",
        "",
        (
            f"t-range: `{report['parameters']['t_min']}` to `{report['parameters']['t_max']}`, "
            f"step `{report['parameters']['step']}`"
        ),
        f"Scales: `{report['parameters']['scales']}`",
        "",
        "## Trend Checks",
        "",
        f"- `ratio_smooth` monotone decreasing: `{trend['ratio_smooth_monotone_decreasing']}`",
        f"- `Mqq_unc` monotone decreasing: `{trend['mqq_unc_monotone_decreasing']}`",
        f"- `mean|Mqq|` monotone increasing: `{trend['mean_abs_mqq_monotone_increasing']}`",
        "",
        "## Positive Branch",
        "",
        "| t | proxy rank | ratio_smooth | Mqq_unc | mean|Mqq| |",
        "|---:|---:|---:|---:|---:|",
    ]
    for candidate in report["candidates"]:
        metrics = candidate["proxy_metrics"]
        lines.append(
            "| "
            f"{candidate['t']:.3f} | {candidate['rank_proxy']} | "
            f"{metrics['M_ratio_smooth_rel']:.3e} | "
            f"{metrics['M_qq_unc_rel']:.3e} | "
            f"{metrics['mean_abs_M_qq']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-prime", type=float, default=1.0)
    parser.add_argument("--scales", type=parse_scales, default=[16, 32, 64, 128])
    parser.add_argument("--t-min", type=float, default=0.0)
    parser.add_argument("--t-max", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--markdown-out", type=str, default=None)
    args = parser.parse_args()

    report = build_report(
        alpha_prime=args.alpha_prime,
        scales=args.scales,
        t_min=args.t_min,
        t_max=args.t_max,
        step=args.step,
    )
    print_report(report)

    if args.json_out is not None:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n")
        print(f"Wrote JSON report to {out_path}")
    if args.markdown_out is not None:
        md_path = Path(args.markdown_out)
        md_path.write_text(markdown_report(report))
        print(f"Wrote markdown report to {md_path}")


if __name__ == "__main__":
    main()
