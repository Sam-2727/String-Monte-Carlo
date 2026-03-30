#!/usr/bin/env python3
"""
Regression tests for projected_graviton_channels.py.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import projected_graviton_channels as pgc


def test_polarization_tensors():
    tensors = pgc.polarization_tensors()
    grav_parallel = tensors["graviton_parallel"]
    grav_perp = tensors["graviton_perpendicular"]
    dilaton = tensors["dilaton"]
    bfield = tensors["bfield_23"]

    return {
        "test": "polarization_tensors",
        "trace_grav_parallel": float(np.trace(grav_parallel)),
        "trace_grav_perpendicular": float(np.trace(grav_perp)),
        "fro_grav_parallel": float(np.linalg.norm(grav_parallel)),
        "fro_grav_perpendicular": float(np.linalg.norm(grav_perp)),
        "fro_dilaton": float(np.linalg.norm(dilaton)),
        "bfield_symmetry_error": float(np.max(np.abs(bfield + bfield.T))),
        "pass": (
            abs(np.trace(grav_parallel)) < 1.0e-12
            and abs(np.trace(grav_perp)) < 1.0e-12
            and abs(np.linalg.norm(grav_parallel) - 1.0) < 1.0e-12
            and abs(np.linalg.norm(grav_perp) - 1.0) < 1.0e-12
            and abs(np.linalg.norm(dilaton) - 1.0) < 1.0e-12
            and np.max(np.abs(bfield + bfield.T)) < 1.0e-12
        ),
    }


def test_projection_formulas():
    synthetic_row = {
        "family": "synthetic",
        "lambda": 0.4,
        "M_delta": complex(2.0, 0.0),
        "M_qq": complex(3.5, 0.0),
        "M_delta_uncertainty": 0.1,
        "M_qq_uncertainty": 0.2,
    }
    projected = pgc.projected_channels_from_row(synthetic_row)
    channels = projected["channels"]

    expected_vector_parallel = 5.5
    expected_vector_transverse = 2.0
    expected_grav_parallel = math.sqrt(7.0 / 8.0) * 3.5
    expected_dilaton = math.sqrt(8.0) * 2.0 + 3.5 / math.sqrt(8.0)

    return {
        "test": "projection_formulas",
        "vector_parallel_error": float(abs(channels["vector_parallel"] - expected_vector_parallel)),
        "vector_transverse_error": float(abs(channels["vector_transverse"] - expected_vector_transverse)),
        "graviton_parallel_error": float(abs(channels["graviton_parallel"] - expected_grav_parallel)),
        "dilaton_error": float(abs(channels["dilaton"] - expected_dilaton)),
        "zero_perpendicular": float(abs(channels["graviton_perpendicular"])),
        "zero_bfield": float(abs(channels["bfield_23"])),
        "pass": (
            abs(channels["vector_parallel"] - expected_vector_parallel) < 1.0e-12
            and abs(channels["vector_transverse"] - expected_vector_transverse) < 1.0e-12
            and abs(channels["graviton_parallel"] - expected_grav_parallel) < 1.0e-12
            and abs(channels["dilaton"] - expected_dilaton) < 1.0e-12
            and abs(channels["graviton_perpendicular"]) < 1.0e-12
            and abs(channels["bfield_23"]) < 1.0e-12
        ),
    }


def test_live_second_order_scan():
    candidates = pgc.scan_projected_channels(
        alpha_prime=1.0,
        scales=[16, 32, 64, 128],
        max_t=0.5,
        step=0.5,
        min_t=0.5,
    )
    if len(candidates) != 1:
        return {
            "test": "live_second_order_scan",
            "error": f"expected one candidate, got {len(candidates)}",
            "pass": False,
        }

    candidate = candidates[0]
    metrics = candidate["channel_metrics"]
    return {
        "test": "live_second_order_scan",
        "rank": candidate["rank_channels"],
        "vector_ratio_smooth_rel": metrics["vector_ratio_smooth_rel"],
        "graviton_parallel_smooth_rel": metrics["graviton_parallel_smooth_rel"],
        "graviton_parallel_unc_rel": metrics["graviton_parallel_unc_rel"],
        "mean_abs_graviton_parallel": metrics["mean_abs_graviton_parallel"],
        "zero_graviton_perpendicular": metrics["zero_graviton_perpendicular"],
        "zero_bfield_23": metrics["zero_bfield_23"],
        "pass": (
            metrics["zero_graviton_perpendicular"] < 1.0e-12
            and metrics["zero_bfield_23"] < 1.0e-12
            and metrics["mean_abs_graviton_parallel"] > 0.1
            and metrics["vector_ratio_smooth_rel"] < 5.0e-3
            and metrics["graviton_parallel_unc_rel"] < 5.0e-3
        ),
    }


def test_trace_dropped_second_order_scan():
    candidates = pgc.scan_projected_channels(
        alpha_prime=1.0,
        scales=[16, 32, 64, 128],
        max_t=0.5,
        step=0.5,
        min_t=0.5,
        trace_dropped=True,
    )
    if len(candidates) != 1:
        return {
            "test": "trace_dropped_second_order_scan",
            "error": f"expected one candidate, got {len(candidates)}",
            "pass": False,
        }

    candidate = candidates[0]
    rows = candidate["channel_rows"]
    ratios = [float(np.real(row["vector_ratio"])) for row in rows]
    max_ratio_error = max(abs(value + 7.0) for value in ratios)
    metrics = candidate["channel_metrics"]
    return {
        "test": "trace_dropped_second_order_scan",
        "rank": candidate["rank_channels"],
        "ratios": ratios,
        "max_ratio_error": max_ratio_error,
        "graviton_parallel_smooth_rel": metrics["graviton_parallel_smooth_rel"],
        "zero_graviton_perpendicular": metrics["zero_graviton_perpendicular"],
        "zero_bfield_23": metrics["zero_bfield_23"],
        "pass": (
            max_ratio_error < 1.0e-12
            and metrics["zero_graviton_perpendicular"] < 1.0e-12
            and metrics["zero_bfield_23"] < 1.0e-12
            and metrics["mean_abs_graviton_parallel"] > 0.1
        ),
    }


def run_all_tests():
    results = {
        "polarization_tensors": test_polarization_tensors(),
        "projection_formulas": test_projection_formulas(),
        "live_second_order_scan": test_live_second_order_scan(),
        "trace_dropped_second_order_scan": test_trace_dropped_second_order_scan(),
    }
    passed = sum(1 for item in results.values() if item.get("pass"))
    total = len(results)

    print("=" * 78)
    print("PROJECTED GRAVITON CHANNEL TESTS")
    print("=" * 78)
    for name, result in results.items():
        status = "PASS" if result.get("pass") else "FAIL"
        print(f"{name:28s} [{status}]")
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
