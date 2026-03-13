#!/usr/bin/env python3
r"""
Explicit GS lightcone zero-mode Clifford module in one fixed SO(8) convention.

This module uses the SO(8) gamma convention from `so8_gamma.py` to build the
16-dimensional massless ground-state module

    8_v \oplus 8_c

on which the GS lightcone fermionic zero modes act. In the basis

    |I>,   I = 1,...,8,      (vector states)
    |\dot a>,  dot a = 1,...,8,   (conjugate-spinor states)

the zero modes satisfy

    {S_0^a, S_0^b} = delta^{ab},
    sqrt(2) S_0^a |I>      = gamma^I_{a \dot a} |\dot a>,
    sqrt(2) S_0^a |\dot a> = gamma^I_{a \dot a} |I>.

This makes the external-state side of the three-graviton problem explicit in a
fixed convention. What it does NOT yet fix is the separate symbol/operator map
that turns the Grassmann polynomial v_{IJ}(Y) into an operator on this module.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import so8_gamma


@dataclass(frozen=True)
class GSZeroModeModule:
    vector_to_c: tuple[np.ndarray, ...]
    c_to_vector: tuple[np.ndarray, ...]
    sigma: tuple[np.ndarray, ...]
    s_zero: tuple[np.ndarray, ...]
    vector_basis: np.ndarray
    c_basis: np.ndarray
    max_clifford_error: float
    max_action_error: float


def vector_to_c_matrices(
    gamma_data: so8_gamma.SO8GammaData,
) -> tuple[np.ndarray, ...]:
    """
    For each chiral spinor index a, build the 8_c x 8_v matrix

        (G_a)_{dot a, I} = gamma^I_{a dot a}.
    """
    matrices = []
    for a in range(8):
        matrix = np.zeros((8, 8), dtype=complex)
        for i in range(8):
            for dot_a in range(8):
                matrix[dot_a, i] = gamma_data.gamma_s_to_c[i][a, dot_a]
        matrices.append(matrix)
    return tuple(matrices)


def build_zero_mode_module() -> GSZeroModeModule:
    gamma_data = so8_gamma.so8_gamma_data()
    v_to_c = vector_to_c_matrices(gamma_data)
    c_to_v = tuple(-matrix.T for matrix in v_to_c)

    sigma = []
    s_zero = []
    max_clifford_error = 0.0
    max_action_error = 0.0
    identity16 = np.eye(16, dtype=complex)

    vector_basis = np.eye(16, dtype=complex)[:, :8]
    c_basis = np.eye(16, dtype=complex)[:, 8:]

    for a in range(8):
        sigma_a = np.block(
            [
                [np.zeros((8, 8), dtype=complex), c_to_v[a]],
                [v_to_c[a], np.zeros((8, 8), dtype=complex)],
            ]
        )
        sigma.append(sigma_a)
        s_zero.append(sigma_a / np.sqrt(2.0))

        for i in range(8):
            vector_state = vector_basis[:, i]
            lhs = np.sqrt(2.0) * (s_zero[a] @ vector_state)
            rhs = np.concatenate(
                [
                    np.zeros(8, dtype=complex),
                    v_to_c[a][:, i],
                ]
            )
            max_action_error = max(
                max_action_error, float(np.max(np.abs(lhs - rhs)))
            )

        for dot_a in range(8):
            spinor_state = c_basis[:, dot_a]
            lhs = np.sqrt(2.0) * (s_zero[a] @ spinor_state)
            rhs = np.concatenate(
                [
                    c_to_v[a][:, dot_a],
                    np.zeros(8, dtype=complex),
                ]
            )
            max_action_error = max(
                max_action_error, float(np.max(np.abs(lhs - rhs)))
            )

    for a in range(8):
        for b in range(8):
            anticommutator = s_zero[a] @ s_zero[b] + s_zero[b] @ s_zero[a]
            max_clifford_error = max(
                max_clifford_error,
                float(
                    np.max(
                        np.abs(anticommutator - (a == b) * identity16)
                    )
                ),
            )

    return GSZeroModeModule(
        vector_to_c=v_to_c,
        c_to_vector=c_to_v,
        sigma=tuple(sigma),
        s_zero=tuple(s_zero),
        vector_basis=vector_basis,
        c_basis=c_basis,
        max_clifford_error=max_clifford_error,
        max_action_error=max_action_error,
    )


def print_summary(show_sample: bool) -> None:
    module = build_zero_mode_module()
    print("=" * 78)
    print("GS LIGHTCONE ZERO-MODE MODULE")
    print("=" * 78)
    print(f"max zero-mode Clifford error : {module.max_clifford_error:.3e}")
    print(f"max action-formula error     : {module.max_action_error:.3e}")
    print()
    print("Basis ordering:")
    print("  1. |I>, I = 1,...,8")
    print("  2. |dot a>, dot a = 1,...,8")
    print()
    print(
        "Conventions: sqrt(2) S_0^a |I> = gamma^I_{a dot a} |dot a>, "
        "sqrt(2) S_0^a |dot a> = gamma^I_{a dot a} |I>."
    )
    print(
        "This fixes the external-state Clifford module. A separate symbol/operator "
        "map is still needed to turn v_IJ(Y) into an operator on this module."
    )

    if show_sample:
        print()
        sample = module.s_zero[0]
        print("Sample S_0^1 matrix:")
        with np.printoptions(precision=3, suppress=True):
            print(sample)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-sample",
        action="store_true",
        help="also print one sample 16x16 zero-mode operator",
    )
    args = parser.parse_args()
    print_summary(args.show_sample)


if __name__ == "__main__":
    main()
