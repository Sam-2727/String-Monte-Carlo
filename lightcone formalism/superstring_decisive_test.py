#!/usr/bin/env python3
"""
Superstring channel test inside the reduced Lambda ansatz.

This module no longer uses the Weyl vector-block proxy as its primary object.
Instead, it extrapolates the trace-dropped tree-level fermionic channel
amplitudes obtained from `fermionic_graviton_contraction.py` across fixed-ratio
families and the symmetric support-three branch

    t = t_- = -t_+.

The main questions are:

1. Is the minimal stencil t = 0 genuinely blocked in the full contraction?
2. Do the unblocked channels satisfy the observed universal relations
   - A(23,24,||) / A(23,23,||) = 1/2
   - A(||,23,23) / A(23,23,||) = 1 / lambda^2
   - dilaton and B-field benchmark channels vanish?

All conclusions from this module are conditional on that reduced ansatz.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import continuum_extrapolation as ce
import fermionic_graviton_contraction as fgc
import prefactor_family_ranking as pfr
import superstring_prefactor_check as spc


CHANNELS = {
    "diag": ("perp23", "perp23", "parallel"),
    "mixed": ("perp23", "perp24", "parallel"),
    "parallel_parallel_parallel": ("parallel", "parallel", "parallel"),
    "parallel_perp_perp": ("parallel", "perp23", "perp23"),
    "perp_dilaton": ("perp23", "perp23", "dilaton"),
    "parallel_dilaton": ("parallel", "parallel", "dilaton"),
    "perp_b23": ("perp23", "perp23", "b23"),
    "parallel_b23": ("parallel", "parallel", "b23"),
}


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


def _channel_series_for_family(
    a: int,
    b: int,
    scales: list[int],
    t: float,
    alpha_prime: float,
) -> dict[str, object]:
    polarizations = fgc.polarization_tensors()
    lambda_ratio = a / (a + b)

    ns: list[int] = []
    series = {name: [] for name in CHANNELS}
    max_imag = {name: 0.0 for name in CHANNELS}
    max_benchmark_closed_form_error = 0.0

    for scale in scales:
        n1 = a * scale
        n2 = b * scale
        prefactor = spc.prefactor_data_three_point_family(
            n1,
            n2,
            -t,
            t,
            alpha_prime,
        )
        ns.append(n1)
        closed_forms = fgc.benchmark_trace_dropped_amplitude_closed_forms(
            lambda_ratio,
            prefactor.b_qq_reduced,
        )

        for name, (eps1, eps2, eps3) in CHANNELS.items():
            amplitude = fgc.fermionic_channel_amplitude_from_ab(
                polarizations[eps1],
                polarizations[eps2],
                polarizations[eps3],
                prefactor.a_delta_reduced,
                prefactor.b_qq_reduced,
                lambda_ratio,
                trace_dropped=True,
            )
            series[name].append(float(amplitude.real))
            max_imag[name] = max(max_imag[name], float(abs(amplitude.imag)))
            benchmark_key = (eps1, eps2, eps3)
            if benchmark_key in closed_forms:
                max_benchmark_closed_form_error = max(
                    max_benchmark_closed_form_error,
                    float(abs(amplitude - closed_forms[benchmark_key])),
                )

    channel_summaries = {
        name: ce.summary_to_dict(ce.summarize_extrapolation(ns, values))
        for name, values in series.items()
    }
    scaled_diag = [n * value for n, value in zip(ns, series["diag"])]
    n1_scaled_diag_summary = ce.summary_to_dict(
        ce.summarize_extrapolation(ns, scaled_diag)
    )

    diag = channel_summaries["diag"]["estimate"]
    mixed = channel_summaries["mixed"]["estimate"]
    parallel_perp = channel_summaries["parallel_perp_perp"]["estimate"]
    parallel_parallel = channel_summaries["parallel_parallel_parallel"]["estimate"]
    zero_channel_max = max(
        abs(channel_summaries["perp_dilaton"]["estimate"]),
        abs(channel_summaries["parallel_dilaton"]["estimate"]),
        abs(channel_summaries["perp_b23"]["estimate"]),
        abs(channel_summaries["parallel_b23"]["estimate"]),
    )

    ratio_mixed = math.nan
    ratio_parallel_perp = math.nan
    ratio_parallel_parallel = math.nan
    if abs(diag) > 1.0e-14:
        ratio_mixed = mixed / diag
        ratio_parallel_perp = parallel_perp / diag
        ratio_parallel_parallel = parallel_parallel / diag

    return {
        "family": f"{a}:{b}",
        "a": a,
        "b": b,
        "lambda": lambda_ratio,
        "channel_summaries": channel_summaries,
        "N1_scaled_diag_summary": n1_scaled_diag_summary,
        "max_abs_imag": max_imag,
        "ratio_mixed_over_diag": ratio_mixed,
        "ratio_parallel_perp_over_diag": ratio_parallel_perp,
        "ratio_parallel_parallel_over_diag": ratio_parallel_parallel,
        "zero_channel_max": zero_channel_max,
        "max_benchmark_closed_form_error": max_benchmark_closed_form_error,
    }


def _summarize_candidate(
    t: float,
    families: list[tuple[int, int]],
    scales: list[int],
    alpha_prime: float,
) -> dict[str, object]:
    rows = [
        _channel_series_for_family(a, b, scales, t, alpha_prime)
        for a, b in families
    ]
    rows.sort(key=lambda row: row["lambda"])

    diag_vals = [abs(row["channel_summaries"]["diag"]["estimate"]) for row in rows]
    blocked = max(diag_vals) < 1.0e-9

    max_mixed_ratio_error = 0.0
    max_parallel_perp_lambda_sq_error = 0.0
    max_zero_channel = 0.0
    max_abs_imag = 0.0
    max_benchmark_closed_form_error = 0.0
    for row in rows:
        max_zero_channel = max(max_zero_channel, float(row["zero_channel_max"]))
        max_abs_imag = max(max_abs_imag, max(row["max_abs_imag"].values()))
        max_benchmark_closed_form_error = max(
            max_benchmark_closed_form_error,
            float(row["max_benchmark_closed_form_error"]),
        )
        if blocked:
            continue
        lam = float(row["lambda"])
        max_mixed_ratio_error = max(
            max_mixed_ratio_error,
            abs(float(row["ratio_mixed_over_diag"]) - 0.5),
        )
        max_parallel_perp_lambda_sq_error = max(
            max_parallel_perp_lambda_sq_error,
            abs(float(row["ratio_parallel_perp_over_diag"]) * lam * lam - 1.0),
        )

    return {
        "t": t,
        "rows": rows,
        "blocked": blocked,
        "mean_abs_diag": float(np.mean(diag_vals)),
        "max_mixed_ratio_error": max_mixed_ratio_error,
        "max_parallel_perp_lambda_sq_error": max_parallel_perp_lambda_sq_error,
        "max_zero_channel": max_zero_channel,
        "max_abs_imag": max_abs_imag,
        "max_benchmark_closed_form_error": max_benchmark_closed_form_error,
    }


def run_decisive_scan(
    alpha_prime: float,
    scales: list[int],
    max_t: float,
    step: float,
    min_t: float,
    families: list[tuple[int, int]] | None = None,
) -> dict[str, object]:
    if families is None:
        families = pfr.DEFAULT_RATIO_FAMILIES

    n_steps = int(round((max_t - min_t) / step))
    t_values = [min_t + index * step for index in range(n_steps + 1)]
    summaries = [
        _summarize_candidate(t, families, scales, alpha_prime)
        for t in t_values
    ]
    blocked = [summary for summary in summaries if summary["blocked"]]
    unblocked = [summary for summary in summaries if not summary["blocked"]]

    max_mixed_ratio_error = max(
        (summary["max_mixed_ratio_error"] for summary in unblocked),
        default=math.nan,
    )
    max_parallel_perp_lambda_sq_error = max(
        (summary["max_parallel_perp_lambda_sq_error"] for summary in unblocked),
        default=math.nan,
    )
    max_zero_channel = max(
        (summary["max_zero_channel"] for summary in summaries),
        default=math.nan,
    )
    max_abs_imag = max(
        (summary["max_abs_imag"] for summary in summaries),
        default=math.nan,
    )
    max_benchmark_closed_form_error = max(
        (summary["max_benchmark_closed_form_error"] for summary in summaries),
        default=math.nan,
    )

    return {
        "parameters": {
            "alpha_prime": alpha_prime,
            "scales": scales,
            "max_t": max_t,
            "step": step,
            "min_t": min_t,
            "families": families,
        },
        "summaries": summaries,
        "blocked_t_values": [summary["t"] for summary in blocked],
        "unblocked_t_values": [summary["t"] for summary in unblocked],
        "universal_relations": {
            "expected_mixed_ratio": 0.5,
            "expected_parallel_perp_lambda_sq": 1.0,
            "max_mixed_ratio_error": float(max_mixed_ratio_error),
            "max_parallel_perp_lambda_sq_error": float(max_parallel_perp_lambda_sq_error),
            "max_zero_channel": float(max_zero_channel),
            "max_abs_imag": float(max_abs_imag),
            "max_benchmark_closed_form_error": float(max_benchmark_closed_form_error),
            "all_unblocked_universal": bool(
                max_mixed_ratio_error < 1.0e-12
                and max_parallel_perp_lambda_sq_error < 1.0e-12
                and max_zero_channel < 1.0e-12
                and max_abs_imag < 1.0e-12
                and max_benchmark_closed_form_error < 1.0e-12
            ),
        },
    }


def print_report(report: dict[str, object]) -> None:
    print("=" * 118)
    print("DECISIVE SUPERSTRING TEST FROM EXPLICIT FERMIONIC CONTRACTION")
    print("=" * 118)
    universal = report["universal_relations"]
    print(
        "Unblocked-channel universality: "
        f"{universal['all_unblocked_universal']} "
        f"(max |A_mix/A_diag - 1/2| = {universal['max_mixed_ratio_error']:.3e}, "
        f"max |lambda^2 A_par23/A_diag - 1| = {universal['max_parallel_perp_lambda_sq_error']:.3e}, "
        f"max |zero channel| = {universal['max_zero_channel']:.3e}, "
        f"max |imag part| = {universal['max_abs_imag']:.3e}, "
        f"max |A - A_closed| = {universal['max_benchmark_closed_form_error']:.3e})"
    )
    print(f"Blocked t values   : {report['blocked_t_values']}")
    print(f"Unblocked t values : {report['unblocked_t_values']}")
    print()
    print(
        f"{'t':>7s} {'blocked':>8s} {'mean|diag|':>14s} "
        f"{'max|mix-1/2|':>14s} {'max|lam^2 par-1|':>18s} {'max|zero|':>12s}"
    )
    print("-" * 92)
    for summary in report["summaries"]:
        print(
            f"{summary['t']:7.3f} {str(summary['blocked']):>8s} "
            f"{summary['mean_abs_diag']:14.6f} "
            f"{summary['max_mixed_ratio_error']:14.3e} "
            f"{summary['max_parallel_perp_lambda_sq_error']:18.3e} "
            f"{summary['max_zero_channel']:12.3e}"
        )


def markdown_report(report: dict[str, object]) -> str:
    universal = report["universal_relations"]
    lines = [
        "# Superstring Channel Test In Reduced Lambda Ansatz",
        "",
        f"- Blocked t values: `{report['blocked_t_values']}`",
        f"- Unblocked t values: `{report['unblocked_t_values']}`",
        (
            "- Universal unblocked-channel relations: "
            f"`all_unblocked_universal = {universal['all_unblocked_universal']}`, "
            f"`max |A_mix/A_diag - 1/2| = {universal['max_mixed_ratio_error']:.3e}`, "
            f"`max |lambda^2 A_par23/A_diag - 1| = {universal['max_parallel_perp_lambda_sq_error']:.3e}`, "
            f"`max |zero channel| = {universal['max_zero_channel']:.3e}`, "
            f"`max |imag part| = {universal['max_abs_imag']:.3e}`, "
            f"`max |A - A_closed| = {universal['max_benchmark_closed_form_error']:.3e}`"
        ),
        "",
        "| t | blocked | mean|diag| | max|mix-1/2| | max|lambda^2 par-1| | max|zero| |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in report["summaries"]:
        lines.append(
            "| "
            f"{summary['t']:.3f} | {summary['blocked']} | {summary['mean_abs_diag']:.6f} | "
            f"{summary['max_mixed_ratio_error']:.3e} | "
            f"{summary['max_parallel_perp_lambda_sq_error']:.3e} | "
            f"{summary['max_zero_channel']:.3e} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_scales(text: str) -> list[int]:
    return pfr.parse_scales(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-prime", type=float, default=1.0)
    parser.add_argument("--scales", type=parse_scales, default=[16, 32, 64, 128])
    parser.add_argument("--max-t", type=float, default=0.75)
    parser.add_argument("--step", type=float, default=0.125)
    parser.add_argument("--min-t", type=float, default=0.0)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--markdown-out", type=str, default=None)
    args = parser.parse_args()

    report = run_decisive_scan(
        alpha_prime=args.alpha_prime,
        scales=args.scales,
        max_t=args.max_t,
        step=args.step,
        min_t=args.min_t,
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
