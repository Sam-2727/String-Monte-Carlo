#!/usr/bin/env python3
"""
Closed-form vector-block invariants for the Weyl-quantized GS prefactor.

Numerically, the 8_v -> 8_v block of v_{IJ}(Lambda) under the candidate Weyl
map fits the minimal SO(8)-covariant tensor basis

    A(alpha) delta_{IJ} delta_{KL}
  + B(alpha) delta_{IK} delta_{JL}
  + C(alpha) delta_{IL} delta_{JK}.

The coefficients are simple rational functions of alpha = alpha_1 / alpha_3:

    A(alpha) = (1 + 4/alpha^2)^2
    B(alpha) = -32/alpha^2 + i (16/alpha^3 - 4/alpha)
    C(alpha) = -32/alpha^2 - i (16/alpha^3 - 4/alpha)

This module packages those formulas and compares them against the explicit
numeric fit from `gs_weyl_symbol_diagnostic.py`.

It also exposes the physically motivated "trace-dropped" variant suggested by
the flat-space on-shell three-particle discussion of Spradlin-Volovich:
drop the pieces of v_{IJ} proportional to delta_{IJ}. In the present SO(8)
coefficient decomposition that means removing the degree-0 term w0 and the
degree-8 term w8, while keeping y2, w4, and y6. The resulting vector-block
invariants are

    A_on(alpha) = 8/alpha^2
    B_on(alpha) = -32/alpha^2 + i (16/alpha^3 - 4/alpha)
    C_on(alpha) = -32/alpha^2 - i (16/alpha^3 - 4/alpha).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gs_weyl_symbol_diagnostic as gw
import gs_zero_mode_module as gm
import gs_zero_mode_prefactor as gp


DEFAULT_ALPHAS = [0.25, 1.0 / 3.0, 0.4, 0.5, 0.625, 0.75, 1.0, 1.5, 2.0, 3.0]


def analytic_vector_block_invariants(alpha_ratio: float) -> tuple[complex, complex, complex]:
    if alpha_ratio == 0.0:
        raise ValueError("alpha_ratio must be nonzero")
    alpha = float(alpha_ratio)
    inv_alpha = 1.0 / alpha
    inv_alpha2 = inv_alpha * inv_alpha
    inv_alpha3 = inv_alpha2 * inv_alpha

    a_coeff = (1.0 + 4.0 * inv_alpha2) ** 2
    imag_coeff = 16.0 * inv_alpha3 - 4.0 * inv_alpha
    b_coeff = complex(-32.0 * inv_alpha2, imag_coeff)
    c_coeff = complex(-32.0 * inv_alpha2, -imag_coeff)
    return complex(a_coeff, 0.0), b_coeff, c_coeff


def analytic_trace_dropped_vector_block_invariants(
    alpha_ratio: float,
) -> tuple[complex, complex, complex]:
    if alpha_ratio == 0.0:
        raise ValueError("alpha_ratio must be nonzero")
    alpha = float(alpha_ratio)
    inv_alpha = 1.0 / alpha
    inv_alpha2 = inv_alpha * inv_alpha
    inv_alpha3 = inv_alpha2 * inv_alpha

    a_coeff = 8.0 * inv_alpha2
    imag_coeff = 16.0 * inv_alpha3 - 4.0 * inv_alpha
    b_coeff = complex(-32.0 * inv_alpha2, imag_coeff)
    c_coeff = complex(-32.0 * inv_alpha2, -imag_coeff)
    return complex(a_coeff, 0.0), b_coeff, c_coeff


def compare_formula(alpha_ratio: float) -> dict[str, object]:
    numeric_coeffs, max_fit_residual, rms_fit_residual = gw.fit_vector_block_invariants(alpha_ratio)
    numeric_a, numeric_b, numeric_c = (complex(value) for value in numeric_coeffs)
    analytic_a, analytic_b, analytic_c = analytic_vector_block_invariants(alpha_ratio)
    diff_a = abs(numeric_a - analytic_a)
    diff_b = abs(numeric_b - analytic_b)
    diff_c = abs(numeric_c - analytic_c)
    return {
        "alpha_ratio": alpha_ratio,
        "numeric_A": numeric_a,
        "numeric_B": numeric_b,
        "numeric_C": numeric_c,
        "analytic_A": analytic_a,
        "analytic_B": analytic_b,
        "analytic_C": analytic_c,
        "max_coeff_diff": max(diff_a, diff_b, diff_c),
        "A_diff": diff_a,
        "B_diff": diff_b,
        "C_diff": diff_c,
        "fit_max_residual": max_fit_residual,
        "fit_rms_residual": rms_fit_residual,
    }


def _trace_dropped_vector_block_invariants_numeric(
    alpha_ratio: float,
) -> tuple[np.ndarray, float, float]:
    rows: list[list[float]] = []
    values: list[complex] = []

    module = gm.build_zero_mode_module()
    prefactor = gp.build_v_prefactor(alpha_ratio)
    antisym = gw.build_antisymmetrized_products(module.sigma)

    for i in range(8):
        for j in range(8):
            operator = np.zeros((16, 16), dtype=complex)
            for key, value in prefactor.y2.items():
                operator += value[i, j] * antisym(key)
            for key, value in prefactor.w4.items():
                operator += value[i, j] * antisym(key)
            for key, value in prefactor.y6.items():
                operator += value[i, j] * antisym(key)

            vector_block = operator[:8, :8]
            for k in range(8):
                for ell in range(8):
                    rows.append(
                        [
                            1.0 if (i == j and k == ell) else 0.0,
                            1.0 if (i == k and j == ell) else 0.0,
                            1.0 if (i == ell and j == k) else 0.0,
                        ]
                    )
                    values.append(vector_block[k, ell])

    design = np.array(rows, dtype=float)
    target = np.array(values, dtype=complex)
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    residual = design @ coefficients - target
    max_error = float(np.max(np.abs(residual)))
    rms_error = float(np.sqrt(np.mean(np.abs(residual) ** 2)))
    return coefficients, max_error, rms_error


def compare_trace_dropped_formula(alpha_ratio: float) -> dict[str, object]:
    numeric_coeffs, max_fit_residual, rms_fit_residual = (
        _trace_dropped_vector_block_invariants_numeric(alpha_ratio)
    )
    numeric_a, numeric_b, numeric_c = (complex(value) for value in numeric_coeffs)
    analytic_a, analytic_b, analytic_c = (
        analytic_trace_dropped_vector_block_invariants(alpha_ratio)
    )
    diff_a = abs(numeric_a - analytic_a)
    diff_b = abs(numeric_b - analytic_b)
    diff_c = abs(numeric_c - analytic_c)
    return {
        "alpha_ratio": alpha_ratio,
        "numeric_A": numeric_a,
        "numeric_B": numeric_b,
        "numeric_C": numeric_c,
        "analytic_A": analytic_a,
        "analytic_B": analytic_b,
        "analytic_C": analytic_c,
        "max_coeff_diff": max(diff_a, diff_b, diff_c),
        "A_diff": diff_a,
        "B_diff": diff_b,
        "C_diff": diff_c,
        "fit_max_residual": max_fit_residual,
        "fit_rms_residual": rms_fit_residual,
    }


def print_report(alphas: list[float]) -> None:
    print("=" * 106)
    print("WEYL VECTOR-BLOCK CLOSED-FORM CHECK")
    print("=" * 106)
    print(
        f"{'alpha':>8s} {'max coeff diff':>18s} {'A diff':>12s} "
        f"{'B diff':>12s} {'fit residual':>14s}"
    )
    print("-" * 74)
    for alpha in alphas:
        row = compare_formula(alpha)
        print(
            f"{alpha:8.5f} {row['max_coeff_diff']:18.9e} "
            f"{row['A_diff']:12.3e} {row['B_diff']:12.3e} "
            f"{row['fit_max_residual']:14.3e}"
        )
    print()


def parse_alphas(text: str) -> list[float]:
    pieces = [piece.strip() for piece in text.split(",") if piece.strip()]
    if not pieces:
        raise argparse.ArgumentTypeError("need at least one alpha ratio")
    return [float(piece) for piece in pieces]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alphas",
        type=parse_alphas,
        default=DEFAULT_ALPHAS,
        help="comma-separated alpha ratios to compare",
    )
    args = parser.parse_args()
    print_report(args.alphas)


if __name__ == "__main__":
    main()
