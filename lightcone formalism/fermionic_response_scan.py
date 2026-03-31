#!/usr/bin/env python3
"""
Scan the pure fermionic zero-mode response basis across a lambda grid.

This isolates the Grassmann part of the three-graviton contraction before the
bosonic prefactor coefficients A_delta and B_qq are folded back in.
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

import fermionic_graviton_contraction as fgc


DEFAULT_LAMBDAS = [0.25, 1.0 / 3.0, 0.375, 0.4, 0.5]


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


def _row_map(report: dict[str, object]) -> dict[tuple[str, str, str], dict[str, object]]:
    return {
        tuple(row["channels"]): row
        for row in report["rows"]
    }


def run_scan(
    lambdas: list[float] | None = None,
    trace_dropped: bool = True,
) -> dict[str, object]:
    if lambdas is None:
        lambdas = list(DEFAULT_LAMBDAS)

    rows = []
    max_abs_delta = 0.0
    max_abs_zero = 0.0
    max_mixed_ratio_error = 0.0
    max_lambda_sq_ratio_error = 0.0
    max_diag_closed_form_error = 0.0

    for lambda_ratio in lambdas:
        report = fgc.channel_response_report(lambda_ratio, trace_dropped=trace_dropped)
        row_map = _row_map(report)

        diag = row_map[("perp23", "perp23", "parallel")]
        mixed = row_map[("perp23", "perp24", "parallel")]
        parallel_perp = row_map[("parallel", "perp23", "perp23")]
        perp_dilaton = row_map[("perp23", "perp23", "dilaton")]
        parallel_dilaton = row_map[("parallel", "parallel", "dilaton")]
        perp_b23 = row_map[("perp23", "perp23", "b23")]
        parallel_b23 = row_map[("parallel", "parallel", "b23")]

        diag_delta = complex(diag["delta_response"])
        diag_qq = complex(diag["qq_response"])
        mixed_delta = complex(mixed["delta_response"])
        mixed_qq = complex(mixed["qq_response"])
        parallel_perp_delta = complex(parallel_perp["delta_response"])
        parallel_perp_qq = complex(parallel_perp["qq_response"])

        mixed_ratio = mixed_qq / diag_qq
        lambda_sq_ratio = (lambda_ratio**2) * parallel_perp_qq / diag_qq
        closed_forms = fgc.benchmark_response_closed_forms(lambda_ratio)
        diag_target = closed_forms[("perp23", "perp23", "parallel")].real
        diag_closed_form_error = abs(diag_qq.real - diag_target) + abs(diag_qq.imag)

        zero_channels = [
            complex(perp_dilaton["delta_response"]),
            complex(perp_dilaton["qq_response"]),
            complex(parallel_dilaton["delta_response"]),
            complex(parallel_dilaton["qq_response"]),
            complex(perp_b23["delta_response"]),
            complex(perp_b23["qq_response"]),
            complex(parallel_b23["delta_response"]),
            complex(parallel_b23["qq_response"]),
        ]
        max_abs_zero = max(max_abs_zero, *(abs(value) for value in zero_channels))

        graviton_deltas = [diag_delta, mixed_delta, parallel_perp_delta]
        max_abs_delta = max(max_abs_delta, *(abs(value) for value in graviton_deltas))
        max_mixed_ratio_error = max(max_mixed_ratio_error, abs(mixed_ratio - 0.5))
        max_lambda_sq_ratio_error = max(
            max_lambda_sq_ratio_error,
            abs(lambda_sq_ratio - 1.0),
        )
        max_diag_closed_form_error = max(
            max_diag_closed_form_error,
            diag_closed_form_error,
        )

        rows.append(
            {
                "lambda": float(lambda_ratio),
                "diag_delta": diag_delta,
                "diag_qq": diag_qq,
                "diag_closed_form": float(diag_target),
                "diag_closed_form_error": float(diag_closed_form_error),
                "mixed_delta": mixed_delta,
                "mixed_qq": mixed_qq,
                "parallel_perp_delta": parallel_perp_delta,
                "parallel_perp_qq": parallel_perp_qq,
                "mixed_over_diag": mixed_ratio,
                "lambda_sq_parallel_perp_over_diag": lambda_sq_ratio,
            }
        )

    diag_values = [complex(row["diag_qq"]).real for row in rows]
    normalized_profile = [value / diag_values[-1] for value in diag_values]
    monotone_decreasing = all(
        diag_values[index + 1] < diag_values[index]
        for index in range(len(diag_values) - 1)
    )

    return {
        "parameters": {
            "lambdas": [float(value) for value in lambdas],
            "trace_dropped": bool(trace_dropped),
        },
        "rows": rows,
        "normalized_diag_profile": normalized_profile,
        "monotone_decreasing_diag": bool(monotone_decreasing),
        "max_abs_graviton_delta": float(max_abs_delta),
        "max_abs_zero_channel": float(max_abs_zero),
        "max_mixed_ratio_error": float(max_mixed_ratio_error),
        "max_lambda_sq_ratio_error": float(max_lambda_sq_ratio_error),
        "max_diag_closed_form_error": float(max_diag_closed_form_error),
        "all_checks_pass": bool(
            max_abs_delta < 1.0e-12
            and max_abs_zero < 1.0e-12
            and max_mixed_ratio_error < 1.0e-12
            and max_lambda_sq_ratio_error < 1.0e-12
            and max_diag_closed_form_error < 1.0e-12
            and monotone_decreasing
        ),
    }


def print_report(report: dict[str, object]) -> None:
    print("=" * 116)
    print("PURE FERMIONIC RESPONSE SCAN")
    print("=" * 116)
    print(
        f"all_checks_pass={report['all_checks_pass']} "
        f"(max |R_delta| = {report['max_abs_graviton_delta']:.3e}, "
        f"max zero = {report['max_abs_zero_channel']:.3e}, "
        f"max |mixed/diag - 1/2| = {report['max_mixed_ratio_error']:.3e}, "
        f"max |lambda^2 par23/diag - 1| = {report['max_lambda_sq_ratio_error']:.3e}, "
        f"max diag closed-form error = {report['max_diag_closed_form_error']:.3e})"
    )
    print(f"monotone_decreasing_diag={report['monotone_decreasing_diag']}")
    print()
    print(
        f"{'lambda':>8s} {'Rqq_diag':>16s} {'closed form':>16s} {'Rqq_mixed':>16s} "
        f"{'Rqq_par23':>16s} {'mixed/diag':>16s} {'lambda^2 par23/diag':>24s}"
    )
    print("-" * 126)
    for row in report["rows"]:
        diag = complex(row["diag_qq"])
        mixed = complex(row["mixed_qq"])
        parallel_perp = complex(row["parallel_perp_qq"])
        mixed_ratio = complex(row["mixed_over_diag"])
        lambda_sq_ratio = complex(row["lambda_sq_parallel_perp_over_diag"])
        print(
            f"{row['lambda']:8.5f} "
            f"{diag.real:16.9f} "
            f"{row['diag_closed_form']:16.9f} "
            f"{mixed.real:16.9f} "
            f"{parallel_perp.real:16.9f} "
            f"{mixed_ratio.real:16.9f} "
            f"{lambda_sq_ratio.real:24.9f}"
        )
    print()
    print(
        "Normalized diagonal profile (relative to the last lambda): "
        + ", ".join(f"{value:.9f}" for value in report["normalized_diag_profile"])
    )


def markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# Pure Fermionic Response Scan",
        "",
        f"- all_checks_pass: `{report['all_checks_pass']}`",
        f"- monotone_decreasing_diag: `{report['monotone_decreasing_diag']}`",
        f"- max |R_delta|: `{report['max_abs_graviton_delta']:.3e}`",
        f"- max zero channel: `{report['max_abs_zero_channel']:.3e}`",
        f"- max |mixed/diag - 1/2|: `{report['max_mixed_ratio_error']:.3e}`",
        f"- max |lambda^2 par23/diag - 1|: `{report['max_lambda_sq_ratio_error']:.3e}`",
        f"- max diag closed-form error: `{report['max_diag_closed_form_error']:.3e}`",
        "",
        "| lambda | Rqq_diag | closed form | Rqq_mixed | Rqq_par23 | mixed/diag | lambda^2 par23/diag |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        diag = row["diag_qq"]
        mixed = row["mixed_qq"]
        parallel = row["parallel_perp_qq"]
        mixed_ratio = row["mixed_over_diag"]
        lambda_sq_ratio = row["lambda_sq_parallel_perp_over_diag"]
        if isinstance(diag, dict):
            diag = complex(diag["real"], diag["imag"])
            mixed = complex(mixed["real"], mixed["imag"])
            parallel = complex(parallel["real"], parallel["imag"])
            mixed_ratio = complex(mixed_ratio["real"], mixed_ratio["imag"])
            lambda_sq_ratio = complex(
                lambda_sq_ratio["real"],
                lambda_sq_ratio["imag"],
            )
        else:
            diag = complex(diag)
            mixed = complex(mixed)
            parallel = complex(parallel)
            mixed_ratio = complex(mixed_ratio)
            lambda_sq_ratio = complex(lambda_sq_ratio)
        lines.append(
            "| "
            f"{row['lambda']:.5f} | "
            f"{diag.real:.9f} | "
            f"{row['diag_closed_form']:.9f} | "
            f"{mixed.real:.9f} | "
            f"{parallel.real:.9f} | "
            f"{mixed_ratio.real:.9f} | "
            f"{lambda_sq_ratio.real:.9f} |"
        )
    lines.append("")
    lines.append(
        "Normalized diagonal profile: "
        + ", ".join(f"`{value:.9f}`" for value in report["normalized_diag_profile"])
    )
    lines.append("")
    return "\n".join(lines)


def parse_lambdas(text: str) -> list[float]:
    return [float(piece) for piece in text.split(",") if piece.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambdas", type=parse_lambdas, default=DEFAULT_LAMBDAS)
    parser.add_argument("--keep-trace", action="store_true")
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--markdown-out", type=str, default=None)
    args = parser.parse_args()

    report = run_scan(
        lambdas=list(args.lambdas),
        trace_dropped=not args.keep_trace,
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
