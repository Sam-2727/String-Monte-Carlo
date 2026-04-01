#!/usr/bin/env python3
"""
End-to-end benchmark of the local three-point superstring tree amplitude.

This module takes the current canonical endpoint-difference local candidate

    Lambda_join = sqrt(N1 N2 / N3) (theta_{I_+} - theta_{I_-})
                = Lambda_lat + Xi_loc,

contracts its Xi_loc sectors against the explicit three-point nonzero-mode
vacuum reduction from `local_vacuum_reduction.py`, assembles the benchmark
channels with the bosonic prefactor coefficients `(A_delta, B_qq)`, and
compares the result to the known trace-dropped GS lightcone benchmark channel
formulas.

The comparison is still conditional on the current canonical local candidate
and on three-point vacuum external states. What it adds beyond the older
reduced-Lambda benchmark is that the amplitude is now built from the explicit
vacuum-contracted local channel polynomials rather than inserted directly from
the reduced ansatz.
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
import local_vacuum_reduction as lvr
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


def local_channel_amplitude_from_ab(
    epsilon_1,
    epsilon_2,
    epsilon_3,
    a_delta: complex,
    b_qq: complex,
    lambda_ratio: float,
    n1: int,
    n2: int,
    *,
    trace_dropped: bool = True,
    two_point_scalar: complex | None = None,
) -> complex:
    """
    Assemble one channel from the explicit vacuum-contracted local responses.
    """
    if two_point_scalar is None:
        two_point_scalar = lvr.canonical_xi_two_point_scalar(n1, n2)
    delta_response = lvr.contracted_local_channel_response(
        epsilon_1,
        epsilon_2,
        epsilon_3,
        lambda_ratio,
        response_kind="delta",
        trace_dropped=trace_dropped,
        two_point_scalar=two_point_scalar,
    )
    qq_response = lvr.contracted_local_channel_response(
        epsilon_1,
        epsilon_2,
        epsilon_3,
        lambda_ratio,
        response_kind="qq",
        trace_dropped=trace_dropped,
        two_point_scalar=two_point_scalar,
    )
    return complex(a_delta) * delta_response + complex(b_qq) * qq_response


def _benchmark_channel_rows(
    *,
    a_delta: complex,
    b_qq: complex,
    lambda_ratio: float,
    n1: int,
    n2: int,
    trace_dropped: bool = True,
    two_point_scalar: complex | None = None,
) -> tuple[list[dict[str, Any]], float, float]:
    polarizations = fgc.polarization_tensors()
    targets = fgc.benchmark_trace_dropped_amplitude_closed_forms(lambda_ratio, b_qq)
    rows = []
    max_abs_error = 0.0
    max_local_reduced_error = 0.0

    for channel in fgc.BENCHMARK_CHANNELS:
        eps1, eps2, eps3 = channel
        local_value = local_channel_amplitude_from_ab(
            polarizations[eps1],
            polarizations[eps2],
            polarizations[eps3],
            a_delta,
            b_qq,
            lambda_ratio,
            n1,
            n2,
            trace_dropped=trace_dropped,
            two_point_scalar=two_point_scalar,
        )
        reduced_value = fgc.fermionic_channel_amplitude_from_ab(
            polarizations[eps1],
            polarizations[eps2],
            polarizations[eps3],
            a_delta,
            b_qq,
            lambda_ratio,
            trace_dropped=trace_dropped,
        )
        target = targets[channel]
        abs_error = abs(local_value - target)
        local_reduced_error = abs(local_value - reduced_value)
        max_abs_error = max(max_abs_error, float(abs_error))
        max_local_reduced_error = max(max_local_reduced_error, float(local_reduced_error))
        rows.append(
            {
                "channel": list(channel),
                "local_value": local_value,
                "reduced_value": reduced_value,
                "target": target,
                "abs_error": float(abs_error),
                "local_reduced_error": float(local_reduced_error),
            }
        )

    return rows, max_abs_error, max_local_reduced_error


def compare_single_point(
    n1: int,
    n2: int,
    alpha_prime: float = 1.0,
    left_variant: str = "second_order",
    right_variant: str = "second_order",
    trace_dropped: bool = True,
) -> dict[str, Any]:
    prefactor = spc.prefactor_data(
        n1,
        n2,
        alpha_prime,
        left_variant=left_variant,
        right_variant=right_variant,
    )
    lambda_ratio = n1 / (n1 + n2)
    rows, max_abs_error, max_local_reduced_error = _benchmark_channel_rows(
        a_delta=prefactor.a_delta_reduced,
        b_qq=prefactor.b_qq_reduced,
        lambda_ratio=lambda_ratio,
        n1=n1,
        n2=n2,
        trace_dropped=trace_dropped,
    )
    return {
        "parameters": {
            "n1": int(n1),
            "n2": int(n2),
            "lambda_ratio": float(lambda_ratio),
            "alpha_prime": float(alpha_prime),
            "left_variant": left_variant,
            "right_variant": right_variant,
            "trace_dropped": bool(trace_dropped),
        },
        "bosonic_prefactor": {
            "A_delta": prefactor.a_delta_reduced,
            "B_qq": prefactor.b_qq_reduced,
        },
        "rows": rows,
        "max_abs_error": float(max_abs_error),
        "max_local_reduced_error": float(max_local_reduced_error),
        "pass": max_abs_error < 1.0e-12 and max_local_reduced_error < 1.0e-12,
    }


def run_family_scan(
    alpha_prime: float = 1.0,
    scales: list[int] | None = None,
    families: list[tuple[int, int]] | None = None,
    trace_dropped: bool = True,
) -> dict[str, Any]:
    if scales is None:
        scales = [16, 32, 64, 128]
    if families is None:
        families = pfr.DEFAULT_RATIO_FAMILIES

    rows = []
    max_abs_error = 0.0
    max_local_reduced_error = 0.0
    for a, b in families:
        for scale in scales:
            n1 = a * scale
            n2 = b * scale
            prefactor = spc.prefactor_data(
                n1,
                n2,
                alpha_prime,
                left_variant="second_order",
                right_variant="second_order",
            )
            lambda_ratio = n1 / (n1 + n2)
            _, point_abs_error, point_local_reduced_error = _benchmark_channel_rows(
                a_delta=prefactor.a_delta_reduced,
                b_qq=prefactor.b_qq_reduced,
                lambda_ratio=lambda_ratio,
                n1=n1,
                n2=n2,
                trace_dropped=trace_dropped,
            )
            max_abs_error = max(max_abs_error, point_abs_error)
            max_local_reduced_error = max(max_local_reduced_error, point_local_reduced_error)
            rows.append(
                {
                    "family": f"{a}:{b}",
                    "scale": int(scale),
                    "n1": int(n1),
                    "n2": int(n2),
                    "lambda_ratio": float(lambda_ratio),
                    "max_abs_error": float(point_abs_error),
                    "max_local_reduced_error": float(point_local_reduced_error),
                }
            )

    return {
        "parameters": {
            "alpha_prime": float(alpha_prime),
            "scales": scales,
            "families": families,
            "trace_dropped": bool(trace_dropped),
        },
        "rows": rows,
        "max_abs_error": float(max_abs_error),
        "max_local_reduced_error": float(max_local_reduced_error),
        "pass": max_abs_error < 1.0e-12 and max_local_reduced_error < 1.0e-12,
    }


def print_report(report: dict[str, Any]) -> None:
    if report["rows"] and "channel" in report["rows"][0]:
        print("=" * 112)
        print("LOCAL SUPERSTRING TREE BENCHMARK: SINGLE POINT")
        print("=" * 112)
        print(
            f"lambda={report['parameters']['lambda_ratio']:.6f} "
            f"max_abs_error={report['max_abs_error']:.3e} "
            f"max_local_reduced_error={report['max_local_reduced_error']:.3e} "
            f"pass={report['pass']}"
        )
        for row in report["rows"]:
            print(
                f"{tuple(row['channel'])}: "
                f"local={complex(row['local_value']).real:.12g} "
                f"target={complex(row['target']).real:.12g} "
                f"reduced={complex(row['reduced_value']).real:.12g} "
                f"err={row['abs_error']:.3e}"
            )
        return

    print("=" * 112)
    print("LOCAL SUPERSTRING TREE BENCHMARK: FAMILY SCAN")
    print("=" * 112)
    print(
        f"max_abs_error={report['max_abs_error']:.3e} "
        f"max_local_reduced_error={report['max_local_reduced_error']:.3e} "
        f"pass={report['pass']}"
    )


def markdown_report(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Local Superstring Tree Benchmark",
            "",
            f"- pass: `{report['pass']}`",
            f"- max_abs_error: `{report['max_abs_error']:.3e}`",
            f"- max_local_reduced_error: `{report['max_local_reduced_error']:.3e}`",
            "",
        ]
    )


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
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
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
        report = run_family_scan(
            alpha_prime=args.alpha_prime,
            scales=args.scales,
        )

    print_report(report)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n")
    if args.markdown_out is not None:
        args.markdown_out.write_text(markdown_report(report))


if __name__ == "__main__":
    main()
