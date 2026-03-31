#!/usr/bin/env python3
"""
Structural comparison with the known flat-space GS lightcone cubic prefactor.

In the flat-space Green-Schwarz lightcone cubic vertex, the local prefactor is

    H_3 ~ P^I P^J v_{IJ}(Lambda),

with

    P^I = alpha_1 p_{(2)}^I - alpha_2 p_{(1)}^I = -alpha_3 q_rel^I.

In the same reduced Lambda ansatz used throughout the current superstring
helpers, the trace-dropped benchmark channel targets are therefore the explicit continuum
lightcone formulas obtained by contracting qhat^I qhat^J v_{IJ}(Lambda) against
the benchmark polarization states. Those formulas are encoded by
`benchmark_trace_dropped_amplitude_closed_forms` and can be compared directly
to the explicit discrete finite-N amplitudes. This is a structural comparison
inside the reduced ansatz, not an independent validation of the genuinely local
finite-N fermionic interaction-point operator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fermionic_graviton_contraction as fgc
import prefactor_family_ranking as pfr
import superstring_prefactor_check as spc


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


def _benchmark_channel_map(report: dict[str, Any]) -> dict[tuple[str, str, str], complex]:
    return {
        tuple(row["channels"]): complex(row["amplitude"])
        for row in report["rows"]
        if tuple(row["channels"]) in fgc.BENCHMARK_CHANNELS
    }


def compare_single_point(
    n1: int,
    n2: int,
    alpha_prime: float = 1.0,
    left_variant: str = "second_order",
    right_variant: str = "second_order",
    trace_dropped: bool = True,
) -> dict[str, Any]:
    report = fgc.channel_report(
        n1,
        n2,
        alpha_prime=alpha_prime,
        left_variant=left_variant,
        right_variant=right_variant,
        trace_dropped=trace_dropped,
    )
    lambda_ratio = float(report["parameters"]["lambda_ratio"])
    b_qq = complex(report["bosonic_prefactor"]["B_qq"])
    targets = fgc.benchmark_trace_dropped_amplitude_closed_forms(lambda_ratio, b_qq)
    actual = _benchmark_channel_map(report)

    rows = []
    max_abs_error = 0.0
    for channel in fgc.BENCHMARK_CHANNELS:
        value = actual[channel]
        target = targets[channel]
        error = abs(value - target)
        max_abs_error = max(max_abs_error, error)
        rows.append(
            {
                "channel": list(channel),
                "actual": value,
                "target": target,
                "abs_error": float(error),
            }
        )

    return {
        "parameters": report["parameters"],
        "bosonic_prefactor": report["bosonic_prefactor"],
        "rows": rows,
        "max_abs_error": float(max_abs_error),
        "pass": max_abs_error < 1.0e-12,
    }


def run_symmetric_family_scan(
    alpha_prime: float = 1.0,
    scales: list[int] | None = None,
    max_t: float = 0.75,
    step: float = 0.125,
    min_t: float = 0.0,
    families: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    if scales is None:
        scales = [16, 32, 64, 128]
    if families is None:
        families = pfr.DEFAULT_RATIO_FAMILIES

    polarizations = fgc.polarization_tensors()
    t_values = [min_t + index * step for index in range(int(round((max_t - min_t) / step)) + 1)]
    rows = []
    max_abs_error = 0.0

    for t in t_values:
        for a, b in families:
            lambda_ratio = a / (a + b)
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
                targets = fgc.benchmark_trace_dropped_amplitude_closed_forms(
                    lambda_ratio,
                    prefactor.b_qq_reduced,
                )

                point_max = 0.0
                for channel in fgc.BENCHMARK_CHANNELS:
                    eps1, eps2, eps3 = channel
                    value = fgc.fermionic_channel_amplitude_from_ab(
                        polarizations[eps1],
                        polarizations[eps2],
                        polarizations[eps3],
                        prefactor.a_delta_reduced,
                        prefactor.b_qq_reduced,
                        lambda_ratio,
                        trace_dropped=True,
                    )
                    error = abs(value - targets[channel])
                    max_abs_error = max(max_abs_error, error)
                    point_max = max(point_max, error)

                rows.append(
                    {
                        "t": float(t),
                        "family": f"{a}:{b}",
                        "scale": int(scale),
                        "lambda": float(lambda_ratio),
                        "max_abs_error": float(point_max),
                    }
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
        "rows": rows,
        "max_abs_error": float(max_abs_error),
        "pass": max_abs_error < 1.0e-12,
    }


def print_report(report: dict[str, Any]) -> None:
    if "rows" in report and report["rows"] and "channel" in report["rows"][0]:
        print("=" * 112)
        print("SUPERSTRING CONTINUUM BENCHMARK: SINGLE POINT")
        print("=" * 112)
        print(
            f"lambda={report['parameters']['lambda_ratio']:.6f} "
            f"max_abs_error={report['max_abs_error']:.3e} "
            f"pass={report['pass']}"
        )
        for row in report["rows"]:
            print(
                f"{tuple(row['channel'])}: "
                f"actual={complex(row['actual']).real:.12g} "
                f"target={complex(row['target']).real:.12g} "
                f"err={row['abs_error']:.3e}"
            )
        return

    print("=" * 112)
    print("SUPERSTRING CONTINUUM BENCHMARK: SYMMETRIC FAMILY SCAN")
    print("=" * 112)
    print(
        f"max_abs_error={report['max_abs_error']:.3e} "
        f"pass={report['pass']}"
    )


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Superstring Continuum Benchmark",
        "",
        f"- pass: `{report['pass']}`",
        f"- max_abs_error: `{report['max_abs_error']:.3e}`",
        "",
    ]
    return "\n".join(lines)


def parse_scales(text: str) -> list[int]:
    return pfr.parse_scales(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["single", "scan"], default="scan")
    parser.add_argument("--n1", type=int, default=128)
    parser.add_argument("--n2", type=int, default=192)
    parser.add_argument("--alpha-prime", type=float, default=1.0)
    parser.add_argument("--left-variant", type=str, default="second_order")
    parser.add_argument("--right-variant", type=str, default="second_order")
    parser.add_argument("--scales", type=parse_scales, default=[16, 32, 64, 128])
    parser.add_argument("--max-t", type=float, default=0.75)
    parser.add_argument("--step", type=float, default=0.125)
    parser.add_argument("--min-t", type=float, default=0.0)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--markdown-out", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "single":
        report = compare_single_point(
            n1=args.n1,
            n2=args.n2,
            alpha_prime=args.alpha_prime,
            left_variant=args.left_variant,
            right_variant=args.right_variant,
        )
    else:
        report = run_symmetric_family_scan(
            alpha_prime=args.alpha_prime,
            scales=args.scales,
            max_t=args.max_t,
            step=args.step,
            min_t=args.min_t,
        )

    print_report(report)

    if args.json_out is not None:
        path = Path(args.json_out)
        path.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n")
        print(f"\nWrote JSON report to {path}")
    if args.markdown_out is not None:
        path = Path(args.markdown_out)
        path.write_text(markdown_report(json_safe(report)) + "\n")
        print(f"Wrote markdown report to {path}")


if __name__ == "__main__":
    main()
