#!/usr/bin/env python3
"""
Regression tests for the trace-dropped superstring decisive test.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import superstring_decisive_test as sdt


@functools.lru_cache(maxsize=1)
def cached_report() -> dict[str, object]:
    return sdt.run_decisive_scan(
        alpha_prime=1.0,
        scales=[16, 32, 64, 128],
        max_t=0.75,
        step=0.125,
        min_t=0.0,
    )


def test_minimal_stencil_is_only_blocked() -> dict[str, object]:
    report = cached_report()
    blocked = report["blocked_t_values"]
    return {
        "test": "minimal_stencil_is_only_blocked",
        "blocked_t_values": blocked,
        "pass": blocked == [0.0],
    }


def test_unblocked_trace_dropped_relations() -> dict[str, object]:
    report = cached_report()
    universal = report["universal_relations"]
    return {
        "test": "unblocked_trace_dropped_relations",
        "max_mixed_ratio_error": universal["max_mixed_ratio_error"],
        "max_parallel_perp_lambda_sq_error": universal["max_parallel_perp_lambda_sq_error"],
        "max_zero_channel": universal["max_zero_channel"],
        "max_abs_imag": universal["max_abs_imag"],
        "pass": bool(universal["all_unblocked_universal"]),
    }


def test_unblocked_trace_dropped_closed_forms() -> dict[str, object]:
    report = cached_report()
    universal = report["universal_relations"]
    return {
        "test": "unblocked_trace_dropped_closed_forms",
        "max_benchmark_closed_form_error": universal["max_benchmark_closed_form_error"],
        "pass": universal["max_benchmark_closed_form_error"] < 1.0e-12,
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "minimal_stencil_is_only_blocked": test_minimal_stencil_is_only_blocked(),
        "unblocked_trace_dropped_relations": test_unblocked_trace_dropped_relations(),
        "unblocked_trace_dropped_closed_forms": test_unblocked_trace_dropped_closed_forms(),
    }
    passed = sum(1 for item in results.values() if item.get("pass"))
    total = len(results)

    print("=" * 84)
    print("SUPERSTRING DECISIVE TESTS")
    print("=" * 84)
    for name, result in results.items():
        status = "PASS" if result.get("pass") else "FAIL"
        print(f"{name:36s} [{status}]")
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
