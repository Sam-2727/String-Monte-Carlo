#!/usr/bin/env python3
r"""
Diagnostic Weyl map from the GS Grassmann symbol v_{IJ}(Y) to the zero-mode module.

This file is intentionally labeled as a diagnostic rather than part of the main
derivation. It combines

- the explicit continuum coefficient set v_{IJ}(Y) from `gs_zero_mode_prefactor.py`,
- the explicit GS ground-state Clifford module from `gs_zero_mode_module.py`,

using the standard Weyl-style map

    Y^{a1} ... Y^{ak}  ->  Sigma^{[a1 ... ak]},

where Sigma^a = sqrt(2) S_0^a are the 16x16 Clifford generators on
8_v \oplus 8_c.

This is a natural candidate symbol/operator correspondence, but the note does
NOT yet claim that it is the physically correct lattice/continuum dictionary for
the cubic prefactor. The goal here is only to test structural consequences of
that candidate map.
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gs_zero_mode_module
import gs_zero_mode_prefactor


def v_operator_from_data(
    prefactor: gs_zero_mode_prefactor.VPrefactorData,
    antisym,
    i: int,
    j: int,
) -> np.ndarray:
    operator = (
        prefactor.w0[i, j] * np.eye(16, dtype=complex)
        + prefactor.w8[i, j] * antisym(tuple(range(8)))
    )
    for key, value in prefactor.y2.items():
        operator += value[i, j] * antisym(key)
    for key, value in prefactor.w4.items():
        operator += value[i, j] * antisym(key)
    for key, value in prefactor.y6.items():
        operator += value[i, j] * antisym(key)
    return operator


def build_antisymmetrized_products(sigma: tuple[np.ndarray, ...]):
    @functools.lru_cache(maxsize=None)
    def antisym(key: tuple[int, ...]) -> np.ndarray:
        if len(key) == 0:
            return np.eye(16, dtype=complex)
        if len(key) == 1:
            return sigma[key[0]]

        last = key[-1]
        previous = antisym(key[:-1])
        out = previous @ sigma[last]
        for position, index in enumerate(key[:-1]):
            shorter = key[:-1][:position] + key[:-1][position + 1 :]
            out += ((-1) ** (len(key) - 1 - position)) * antisym(
                shorter + (last,)
            ) @ sigma[index]
        return out / len(key)

    return antisym


def v_operator(alpha_ratio: float, i: int, j: int) -> np.ndarray:
    module = gs_zero_mode_module.build_zero_mode_module()
    prefactor = gs_zero_mode_prefactor.build_v_prefactor(alpha_ratio)
    antisym = build_antisymmetrized_products(module.sigma)
    return v_operator_from_data(prefactor, antisym, i, j)


def fit_vector_block_invariants(
    alpha_ratio: float,
) -> tuple[np.ndarray, float, float]:
    """
    Fit the 8_v -> 8_v block of v_{IJ} to the minimal SO(8)-covariant tensor basis

        A delta_{IJ} delta_{KL}
      + B delta_{IK} delta_{JL}
      + C delta_{IL} delta_{JK}.

    This does not assume the Weyl map is physical. It only asks whether the
    resulting vector block introduces any tensor structures beyond the obvious
    invariant ones.
    """
    rows: list[list[float]] = []
    values: list[complex] = []
    module = gs_zero_mode_module.build_zero_mode_module()
    prefactor = gs_zero_mode_prefactor.build_v_prefactor(alpha_ratio)
    antisym = build_antisymmetrized_products(module.sigma)

    for i in range(8):
        for j in range(8):
            vector_block = v_operator_from_data(prefactor, antisym, i, j)[:8, :8]
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


def print_summary(
    alpha_ratio: float,
    ij_pair: tuple[int, int],
    report_vector_fit: bool,
) -> None:
    i, j = ij_pair
    if not (1 <= i <= 8 and 1 <= j <= 8):
        raise ValueError("I,J must lie in {1,...,8}")

    operator = v_operator(alpha_ratio, i - 1, j - 1)
    vv = operator[:8, :8]
    vc = operator[:8, 8:]
    cv = operator[8:, :8]
    cc = operator[8:, 8:]

    print("=" * 78)
    print("GS WEYL-SYMBOL DIAGNOSTIC")
    print("=" * 78)
    print(f"alpha ratio           : {alpha_ratio:.6g}")
    print(f"sample pair           : ({i},{j})")
    print()
    print(
        "Using the candidate Weyl map Y^{a1}...Y^{ak} -> Sigma^{[a1...ak]}, "
        "the even prefactor preserves the 8_v and 8_c sectors:"
    )
    print(f"  max |vector->spinor block| : {np.max(np.abs(vc)):.3e}")
    print(f"  max |spinor->vector block| : {np.max(np.abs(cv)):.3e}")
    print(f"  max |vector block|         : {np.max(np.abs(vv)):.3e}")
    print(f"  max |spinor block|         : {np.max(np.abs(cc)):.3e}")
    if report_vector_fit:
        coefficients, max_error, rms_error = fit_vector_block_invariants(
            alpha_ratio
        )
        print()
        print("Vector-sector invariant fit:")
        print(
            "  v_{IJ}|_{8_v} = A delta_{IJ} delta_{KL}"
            " + B delta_{IK} delta_{JL} + C delta_{IL} delta_{JK}"
        )
        print(f"  A : {coefficients[0]}")
        print(f"  B : {coefficients[1]}")
        print(f"  C : {coefficients[2]}")
        print(f"  max fit residual : {max_error:.3e}")
        print(f"  rms fit residual : {rms_error:.3e}")
    print()
    print(
        "This is only a structural diagnostic. The physically correct symbol/operator "
        "map still has to be fixed independently before these blocks can be used in "
        "the three-graviton amplitude."
    )


def parse_ij_pair(text: str) -> tuple[int, int]:
    pieces = [piece.strip() for piece in text.split(",") if piece.strip()]
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("expected I,J")
    return int(pieces[0]), int(pieces[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alpha-ratio",
        type=float,
        default=1.0,
        help="the continuum alpha appearing in v_{IJ}(Y)",
    )
    parser.add_argument(
        "--sample-ij",
        type=parse_ij_pair,
        default=(1, 2),
        help="sample I,J pair to inspect (default: 1,2)",
    )
    parser.add_argument(
        "--report-vector-fit",
        action="store_true",
        help="also fit the vector block to the minimal SO(8)-invariant tensor basis",
    )
    args = parser.parse_args()
    print_summary(
        args.alpha_ratio, args.sample_ij, report_vector_fit=args.report_vector_fit
    )


if __name__ == "__main__":
    main()
