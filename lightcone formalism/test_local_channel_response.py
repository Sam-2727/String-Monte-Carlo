#!/usr/bin/env python3
"""
Regression tests for unreduced local fermionic channel polynomials.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fermionic_graviton_contraction as fgc
import local_channel_response as lcr


SAMPLED_LAMBDA_GRID = (0.25, 0.4, 0.5)


def test_benchmark_graviton_channels_are_xi_independent() -> dict[str, object]:
    polarizations = fgc.polarization_tensors()
    channels = [
        ("diag", "perp23", "perp23", "parallel"),
        ("mixed", "perp23", "perp24", "parallel"),
        ("parallel", "parallel", "perp23", "perp23"),
    ]
    max_abs_error = 0.0
    bad_profiles: list[dict[str, object]] = []

    for lambda_ratio in SAMPLED_LAMBDA_GRID:
        for label, key1, key2, key3 in channels:
            local_poly = lcr.local_channel_response_polynomial(
                polarizations[key1],
                polarizations[key2],
                polarizations[key3],
                lambda_ratio,
                response_kind="qq",
                trace_dropped=True,
            )
            _, reduced_qq = fgc.fermionic_channel_responses(
                polarizations[key1],
                polarizations[key2],
                polarizations[key3],
                lambda_ratio,
                trace_dropped=True,
            )
            profile = lcr.xi_degree_profile(local_poly)
            xi_zero = lcr.xi_zero_component(local_poly)
            max_abs_error = max(max_abs_error, float(abs(xi_zero - reduced_qq)))
            if profile != {0: 1}:
                bad_profiles.append(
                    {
                        "lambda_ratio": lambda_ratio,
                        "channel": label,
                        "profile": profile,
                    }
                )

    return {
        "test": "benchmark_graviton_channels_are_xi_independent",
        "sampled_lambda_grid": list(SAMPLED_LAMBDA_GRID),
        "max_abs_error": max_abs_error,
        "bad_profiles": bad_profiles,
        "pass": max_abs_error < 1.0e-12 and not bad_profiles,
    }


def test_benchmark_dilaton_channel_is_pure_quartic() -> dict[str, object]:
    polarizations = fgc.polarization_tensors()
    bad_profiles: list[dict[str, object]] = []
    max_abs_zero = 0.0

    for lambda_ratio in SAMPLED_LAMBDA_GRID:
        local_poly = lcr.local_channel_response_polynomial(
            polarizations["perp23"],
            polarizations["perp23"],
            polarizations["dilaton"],
            lambda_ratio,
            response_kind="qq",
            trace_dropped=True,
        )
        profile = lcr.xi_degree_profile(local_poly)
        xi_zero = lcr.xi_zero_component(local_poly)
        max_abs_zero = max(max_abs_zero, float(abs(xi_zero)))
        if profile != {4: 14}:
            bad_profiles.append(
                {
                    "lambda_ratio": lambda_ratio,
                    "profile": profile,
                }
            )

    return {
        "test": "benchmark_dilaton_channel_is_pure_quartic",
        "sampled_lambda_grid": list(SAMPLED_LAMBDA_GRID),
        "max_abs_zero": max_abs_zero,
        "bad_profiles": bad_profiles,
        "pass": max_abs_zero < 1.0e-12 and not bad_profiles,
    }


def test_trace_dropped_delta_benchmark_channel_vanishes() -> dict[str, object]:
    polarizations = fgc.polarization_tensors()
    bad_profiles: list[dict[str, object]] = []

    for lambda_ratio in SAMPLED_LAMBDA_GRID:
        local_poly = lcr.local_channel_response_polynomial(
            polarizations["perp23"],
            polarizations["perp23"],
            polarizations["parallel"],
            lambda_ratio,
            response_kind="delta",
            trace_dropped=True,
        )
        profile = lcr.xi_degree_profile(local_poly)
        if profile:
            bad_profiles.append(
                {
                    "lambda_ratio": lambda_ratio,
                    "profile": profile,
                    "xi_zero": lcr.xi_zero_component(local_poly),
                }
            )

    return {
        "test": "trace_dropped_delta_benchmark_channel_vanishes",
        "sampled_lambda_grid": list(SAMPLED_LAMBDA_GRID),
        "bad_profiles": bad_profiles,
        "pass": not bad_profiles,
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "benchmark_graviton_channels_are_xi_independent": test_benchmark_graviton_channels_are_xi_independent(),
        "benchmark_dilaton_channel_is_pure_quartic": test_benchmark_dilaton_channel_is_pure_quartic(),
        "trace_dropped_delta_benchmark_channel_vanishes": test_trace_dropped_delta_benchmark_channel_vanishes(),
    }
    passed = sum(1 for item in results.values() if item.get("pass"))
    total = len(results)

    print("=" * 84)
    print("LOCAL CHANNEL RESPONSE TESTS")
    print("=" * 84)
    for name, result in results.items():
        status = "PASS" if result.get("pass") else "FAIL"
        print(f"{name:44s} [{status}]")
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
