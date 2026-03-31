#!/usr/bin/env python3
"""
Regression tests for the pure fermionic zero-mode channel responses.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fermionic_graviton_contraction as fgc


BENCHMARK_LAMBDA = 0.4


def _row_map(report: dict[str, object]) -> dict[tuple[str, str, str], dict[str, object]]:
    return {
        tuple(row["channels"]): row
        for row in report["rows"]
    }


def test_trace_dropped_benchmark_responses() -> dict[str, object]:
    report = fgc.channel_response_report(BENCHMARK_LAMBDA, trace_dropped=True)
    rows = _row_map(report)

    diag = rows[("perp23", "perp23", "parallel")]
    mixed = rows[("perp23", "perp24", "parallel")]
    parallel_perp = rows[("parallel", "perp23", "perp23")]

    diag_delta = complex(diag["delta_response"])
    diag_qq = complex(diag["qq_response"])
    mixed_delta = complex(mixed["delta_response"])
    mixed_qq = complex(mixed["qq_response"])
    parallel_perp_delta = complex(parallel_perp["delta_response"])
    parallel_perp_qq = complex(parallel_perp["qq_response"])

    mixed_ratio_error = abs(mixed_qq / diag_qq - 0.5)
    lambda_sq_ratio_error = abs(
        (BENCHMARK_LAMBDA**2) * parallel_perp_qq / diag_qq - 1.0
    )

    return {
        "test": "trace_dropped_benchmark_responses",
        "lambda_ratio": BENCHMARK_LAMBDA,
        "diag_delta": diag_delta,
        "diag_qq": diag_qq,
        "mixed_delta": mixed_delta,
        "mixed_qq": mixed_qq,
        "parallel_perp_delta": parallel_perp_delta,
        "parallel_perp_qq": parallel_perp_qq,
        "mixed_ratio_error": float(mixed_ratio_error),
        "lambda_sq_ratio_error": float(lambda_sq_ratio_error),
        "pass": (
            abs(diag_delta) < 1.0e-12
            and abs(mixed_delta) < 1.0e-12
            and abs(parallel_perp_delta) < 1.0e-12
            and abs(mixed_ratio_error) < 1.0e-12
            and abs(lambda_sq_ratio_error) < 1.0e-12
        ),
    }


def test_trace_dropped_zero_response_channels() -> dict[str, object]:
    report = fgc.channel_response_report(BENCHMARK_LAMBDA, trace_dropped=True)
    rows = _row_map(report)

    zero_channels = [
        ("perp23", "perp23", "dilaton"),
        ("parallel", "parallel", "dilaton"),
        ("perp23", "perp23", "b23"),
        ("parallel", "parallel", "b23"),
    ]
    max_zero = 0.0
    for channel in zero_channels:
        row = rows[channel]
        max_zero = max(max_zero, abs(complex(row["delta_response"])))
        max_zero = max(max_zero, abs(complex(row["qq_response"])))

    return {
        "test": "trace_dropped_zero_response_channels",
        "lambda_ratio": BENCHMARK_LAMBDA,
        "max_abs_response": float(max_zero),
        "pass": max_zero < 1.0e-12,
    }


def test_benchmark_response_value() -> dict[str, object]:
    report = fgc.channel_response_report(BENCHMARK_LAMBDA, trace_dropped=True)
    rows = _row_map(report)
    diag_qq = complex(rows[("perp23", "perp23", "parallel")]["qq_response"])
    target = 5.3879866369544605
    error = abs(diag_qq.real - target) + abs(diag_qq.imag)
    return {
        "test": "benchmark_response_value",
        "lambda_ratio": BENCHMARK_LAMBDA,
        "diag_qq": diag_qq,
        "target": target,
        "abs_error": float(error),
        "pass": error < 1.0e-12,
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "trace_dropped_benchmark_responses": test_trace_dropped_benchmark_responses(),
        "trace_dropped_zero_response_channels": test_trace_dropped_zero_response_channels(),
        "benchmark_response_value": test_benchmark_response_value(),
    }

    print("Running trace-dropped benchmark-response test...")
    benchmark = results["trace_dropped_benchmark_responses"]
    print(
        f"  diag qq = {benchmark['diag_qq']:.12g}\n"
        f"  mixed qq = {benchmark['mixed_qq']:.12g}\n"
        f"  parallel-perp qq = {benchmark['parallel_perp_qq']:.12g}\n"
        f"  |mixed/diag - 1/2| = {benchmark['mixed_ratio_error']:.3e}\n"
        f"  |lambda^2 par23/diag - 1| = {benchmark['lambda_sq_ratio_error']:.3e} "
        f"[{'PASS' if benchmark['pass'] else 'FAIL'}]"
    )

    print("\nRunning zero-response-channel test...")
    zero = results["trace_dropped_zero_response_channels"]
    print(
        f"  max |response| = {zero['max_abs_response']:.3e} "
        f"[{'PASS' if zero['pass'] else 'FAIL'}]"
    )

    print("\nRunning benchmark response-value test...")
    value = results["benchmark_response_value"]
    print(
        f"  diag qq = {value['diag_qq']:.12g} "
        f"(target {value['target']:.12g}) "
        f"error = {value['abs_error']:.3e} "
        f"[{'PASS' if value['pass'] else 'FAIL'}]"
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for result in results.values() if result.get("pass"))
    total = len(results)
    print(f"  {passed}/{total} tests passed")
    return results


if __name__ == "__main__":
    run_all_tests()
