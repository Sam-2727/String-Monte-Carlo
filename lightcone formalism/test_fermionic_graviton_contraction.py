#!/usr/bin/env python3
"""
Regression tests for fermionic_graviton_contraction.py.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fermionic_graviton_contraction as fgc
import gs_zero_mode_prefactor as gp
import superstring_prefactor_check as sp


FULL_TOP_24 = tuple(range(24))


def _delta_24() -> dict[tuple[int, ...], complex]:
    poly: dict[tuple[int, ...], complex] = {(): 1.0}
    for index in range(8):
        factor = {
            (index,): 1.0,
            (8 + index,): 1.0,
            (16 + index,): 1.0,
        }
        poly = fgc.multiply_sparse(poly, factor)
    return poly


def _integrate_24(poly: dict[tuple[int, ...], complex]) -> complex:
    return complex(poly.get(FULL_TOP_24, 0.0j))


def _embed_external_state(i: int, j: int, leg: int) -> dict[tuple[int, ...], complex]:
    base = fgc.graviton_wavefunction(np.eye(8, dtype=complex)[[i]].T @ np.eye(8, dtype=complex)[[j]], 1.0)
    return fgc.embed_leg(base, leg)


def _substitute_prefactor(
    alpha_ratio: float,
    i: int,
    j: int,
) -> dict[tuple[int, ...], complex]:
    return fgc.substitute_two_leg(
        fgc.v_prefactor_polynomial(alpha_ratio, i, j, False),
        -(1.0 - alpha_ratio),
        alpha_ratio,
    )


def _brute_force_reference(alpha_ratio: float) -> complex:
    delta = _delta_24()
    state_1 = _embed_external_state(0, 0, 0)
    state_2 = _embed_external_state(0, 0, 1)
    state_3 = _embed_external_state(0, 0, 2)
    prefactor = _substitute_prefactor(alpha_ratio, 0, 0)
    full = fgc.multiply_sparse(
        delta,
        fgc.multiply_sparse(
            state_1,
            fgc.multiply_sparse(
                state_2,
                fgc.multiply_sparse(state_3, prefactor),
            ),
        ),
    )
    return _integrate_24(full)


def test_delta_reduction_identity() -> dict[str, object]:
    alpha_ratio = 0.4
    brute_force = _brute_force_reference(alpha_ratio)

    epsilon = np.zeros((8, 8), dtype=complex)
    epsilon[0, 0] = 1.0
    bosonic_tensor = np.zeros((8, 8), dtype=complex)
    bosonic_tensor[0, 0] = 1.0
    reduced = fgc.fermionic_channel_amplitude(
        epsilon,
        epsilon,
        epsilon,
        bosonic_tensor,
        alpha_ratio,
        trace_dropped=False,
    )

    return {
        "test": "delta_reduction_identity",
        "alpha_ratio": alpha_ratio,
        "brute_force": brute_force,
        "reduced": reduced,
        "abs_error": float(abs(brute_force - reduced)),
        "pass": abs(brute_force - reduced) < 1.0e-12,
    }


def test_trace_dropped_zero_channels() -> dict[str, object]:
    polarizations = fgc.polarization_tensors()
    data = sp.prefactor_data(
        128,
        192,
        1.0,
        left_variant="second_order",
        right_variant="second_order",
    )
    bosonic_tensor = fgc.bosonic_tensor_from_prefactor_data(data)
    lambda_ratio = 128 / (128 + 192)

    samples = {
        "perp_perp_dilaton": fgc.fermionic_channel_amplitude(
            polarizations["perp23"],
            polarizations["perp23"],
            polarizations["dilaton"],
            bosonic_tensor,
            lambda_ratio,
            trace_dropped=True,
        ),
        "parallel_parallel_dilaton": fgc.fermionic_channel_amplitude(
            polarizations["parallel"],
            polarizations["parallel"],
            polarizations["dilaton"],
            bosonic_tensor,
            lambda_ratio,
            trace_dropped=True,
        ),
        "perp_perp_b23": fgc.fermionic_channel_amplitude(
            polarizations["perp23"],
            polarizations["perp23"],
            polarizations["b23"],
            bosonic_tensor,
            lambda_ratio,
            trace_dropped=True,
        ),
        "parallel_parallel_b23": fgc.fermionic_channel_amplitude(
            polarizations["parallel"],
            polarizations["parallel"],
            polarizations["b23"],
            bosonic_tensor,
            lambda_ratio,
            trace_dropped=True,
        ),
    }

    max_abs_value = max(abs(value) for value in samples.values())
    return {
        "test": "trace_dropped_zero_channels",
        "samples": samples,
        "max_abs_value": float(max_abs_value),
        "pass": max_abs_value < 1.0e-12,
    }


def test_trace_dropped_graviton_channels() -> dict[str, object]:
    polarizations = fgc.polarization_tensors()
    data = sp.prefactor_data(
        128,
        192,
        1.0,
        left_variant="second_order",
        right_variant="second_order",
    )
    bosonic_tensor = fgc.bosonic_tensor_from_prefactor_data(data)
    lambda_ratio = 128 / (128 + 192)

    a_perp_perp_parallel = fgc.fermionic_channel_amplitude(
        polarizations["perp23"],
        polarizations["perp23"],
        polarizations["parallel"],
        bosonic_tensor,
        lambda_ratio,
        trace_dropped=True,
    )
    a_perp_mixed_parallel = fgc.fermionic_channel_amplitude(
        polarizations["perp23"],
        polarizations["perp24"],
        polarizations["parallel"],
        bosonic_tensor,
        lambda_ratio,
        trace_dropped=True,
    )
    a_parallel_parallel_parallel = fgc.fermionic_channel_amplitude(
        polarizations["parallel"],
        polarizations["parallel"],
        polarizations["parallel"],
        bosonic_tensor,
        lambda_ratio,
        trace_dropped=True,
    )

    half_ratio_error = abs(
        a_perp_mixed_parallel / a_perp_perp_parallel - 0.5
    )
    return {
        "test": "trace_dropped_graviton_channels",
        "perp_perp_parallel": a_perp_perp_parallel,
        "perp_mixed_parallel": a_perp_mixed_parallel,
        "parallel_parallel_parallel": a_parallel_parallel_parallel,
        "half_ratio_error": float(abs(half_ratio_error)),
        "pass": (
            abs(a_perp_perp_parallel) > 1.0e-3
            and abs(a_parallel_parallel_parallel) > abs(a_perp_perp_parallel)
            and abs(half_ratio_error) < 1.0e-12
        ),
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "delta_reduction_identity": test_delta_reduction_identity(),
        "trace_dropped_zero_channels": test_trace_dropped_zero_channels(),
        "trace_dropped_graviton_channels": test_trace_dropped_graviton_channels(),
    }

    print("Running delta-reduction identity test...")
    reduction = results["delta_reduction_identity"]
    print(
        f"  brute={reduction['brute_force']:.12g} "
        f"reduced={reduction['reduced']:.12g} "
        f"error={reduction['abs_error']:.3e} "
        f"[{'PASS' if reduction['pass'] else 'FAIL'}]"
    )

    print("\nRunning trace-dropped zero-channel test...")
    zeros = results["trace_dropped_zero_channels"]
    print(
        f"  max |zero-channel amplitude| = {zeros['max_abs_value']:.3e} "
        f"[{'PASS' if zeros['pass'] else 'FAIL'}]"
    )

    print("\nRunning trace-dropped graviton-channel test...")
    graviton = results["trace_dropped_graviton_channels"]
    print(
        f"  A(perp23,perp23,parallel) = {graviton['perp_perp_parallel']:.12g}\n"
        f"  A(perp23,perp24,parallel) = {graviton['perp_mixed_parallel']:.12g}\n"
        f"  A(parallel,parallel,parallel) = {graviton['parallel_parallel_parallel']:.12g}\n"
        f"  |ratio - 1/2| = {graviton['half_ratio_error']:.3e} "
        f"[{'PASS' if graviton['pass'] else 'FAIL'}]"
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
