#!/usr/bin/env python3
r"""
Explicit SO(8) coefficient builder for the continuum GS cubic prefactor.

This helper fixes one concrete SO(8) convention for the Grassmann polynomial
v_{IJ}(Y) that appears in the continuum lightcone Green-Schwarz cubic vertex.
It uses the local gamma-matrix convention from `so8_gamma.py` and the imported
continuum formulas quoted in Pankiewicz-Stefanski, hep-th/0301015, eqs.
(4.26)-(4.31):

    v_{IJ}(Y) = w_{IJ}(Y) + y_{IJ}(Y),
    w_{IJ}(Y) = delta_{IJ}
              + (1 / (6 alpha^2)) t_{IJ}^{abcd} Y^a Y^b Y^c Y^d
              + (16 / (8! alpha^4)) delta_{IJ} eps^{abcdefgh} Y^a...Y^h,
    i y_{IJ}(Y) = (1 / (2 alpha)) gamma_{ab}^{IJ} Y^a Y^b
                + (2 / (6! alpha^3)) gamma_{ab}^{IJ}
                  eps^{ab cdefgh} Y^c...Y^h,
    t_{IJ}^{abcd} = gamma_{[ab}^{IK} gamma_{cd]}^{JK}.

The script makes the polynomial coefficients explicit in a canonical monomial
basis. It does not yet supply the external-state / Clifford-module dictionary
needed to turn these coefficients into the full graviton matrix element.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import so8_gamma


PairKey = tuple[int, int]
QuadKey = tuple[int, int, int, int]
HexKey = tuple[int, int, int, int, int, int]


def inversion_sign(indices: tuple[int, ...]) -> int:
    sign = 1
    for i, value_i in enumerate(indices):
        for value_j in indices[i + 1 :]:
            if value_i > value_j:
                sign *= -1
    return sign


def canonicalize(indices: tuple[int, ...]) -> tuple[int, tuple[int, ...]] | None:
    if len(set(indices)) != len(indices):
        return None
    return inversion_sign(indices), tuple(sorted(indices))


def complement(indices: tuple[int, ...], size: int = 8) -> tuple[int, ...]:
    taken = set(indices)
    return tuple(index for index in range(size) if index not in taken)


def signed_gamma_ss(
    gamma_data: so8_gamma.SO8GammaData, i: int, j: int
) -> np.ndarray:
    if i == j:
        return np.zeros((8, 8), dtype=complex)
    if i < j:
        return gamma_data.gamma_ss[(i + 1, j + 1)]
    return -gamma_data.gamma_ss[(j + 1, i + 1)]


@dataclass(frozen=True)
class VPrefactorData:
    alpha_ratio: float
    w0: np.ndarray
    w4: dict[QuadKey, np.ndarray]
    w8: np.ndarray
    y2: dict[PairKey, np.ndarray]
    y6: dict[HexKey, np.ndarray]
    t4: dict[QuadKey, np.ndarray]
    max_w_symmetry_error: float
    max_y_antisymmetry_error: float
    max_t_trace_error: float
    max_trace_y2_error: float
    max_trace_y6_error: float


def build_t_four_form(gamma_data: so8_gamma.SO8GammaData) -> dict[QuadKey, np.ndarray]:
    coeffs = {
        key: np.zeros((8, 8), dtype=complex) for key in combinations(range(8), 4)
    }

    for i in range(8):
        for j in range(8):
            for k in range(8):
                gamma_ik = signed_gamma_ss(gamma_data, i, k)
                gamma_jk = signed_gamma_ss(gamma_data, j, k)
                for a, b, c, d in product(range(8), repeat=4):
                    canonical = canonicalize((a, b, c, d))
                    if canonical is None:
                        continue
                    sign, key = canonical
                    coeffs[key][i, j] += (
                        sign * gamma_ik[a, b] * gamma_jk[c, d] / 24.0
                    )
    return coeffs


def build_v_prefactor(alpha_ratio: float = 1.0) -> VPrefactorData:
    gamma_data = so8_gamma.so8_gamma_data()
    if alpha_ratio == 0.0:
        raise ValueError("alpha_ratio must be nonzero")

    eye = np.eye(8, dtype=complex)
    t4 = build_t_four_form(gamma_data)

    w0 = eye.copy()
    w4 = {
        key: (4.0 / (alpha_ratio * alpha_ratio)) * value
        for key, value in t4.items()
    }
    w8 = (16.0 / (alpha_ratio**4)) * eye

    y2: dict[PairKey, np.ndarray] = {}
    y6: dict[HexKey, np.ndarray] = {
        key: np.zeros((8, 8), dtype=complex) for key in combinations(range(8), 6)
    }

    for a, b in combinations(range(8), 2):
        gamma_ab = np.zeros((8, 8), dtype=complex)
        for i in range(8):
            for j in range(8):
                gamma_ab[i, j] = signed_gamma_ss(gamma_data, i, j)[a, b]
        y2[(a, b)] = (-1.0j / alpha_ratio) * gamma_ab

        comp = complement((a, b))
        sign = inversion_sign((a, b) + comp)
        y6[comp] += (-4.0j * sign / (alpha_ratio**3)) * gamma_ab

    max_w_symmetry_error = 0.0
    max_y_antisymmetry_error = 0.0
    max_t_trace_error = 0.0
    max_trace_y2_error = 0.0
    max_trace_y6_error = 0.0

    for value in w4.values():
        max_w_symmetry_error = max(
            max_w_symmetry_error, float(np.max(np.abs(value - value.T)))
        )
        max_t_trace_error = max(
            max_t_trace_error, float(abs(np.trace(value)))
        )
    max_w_symmetry_error = max(
        max_w_symmetry_error, float(np.max(np.abs(w0 - w0.T))), float(np.max(np.abs(w8 - w8.T)))
    )

    for value in y2.values():
        max_y_antisymmetry_error = max(
            max_y_antisymmetry_error, float(np.max(np.abs(value + value.T)))
        )
        max_trace_y2_error = max(max_trace_y2_error, float(abs(np.trace(value))))
    for value in y6.values():
        max_y_antisymmetry_error = max(
            max_y_antisymmetry_error, float(np.max(np.abs(value + value.T)))
        )
        max_trace_y6_error = max(max_trace_y6_error, float(abs(np.trace(value))))

    return VPrefactorData(
        alpha_ratio=alpha_ratio,
        w0=w0,
        w4=w4,
        w8=w8,
        y2=y2,
        y6=y6,
        t4=t4,
        max_w_symmetry_error=max_w_symmetry_error,
        max_y_antisymmetry_error=max_y_antisymmetry_error,
        max_t_trace_error=max_t_trace_error,
        max_trace_y2_error=max_trace_y2_error,
        max_trace_y6_error=max_trace_y6_error,
    )


def format_key(key: tuple[int, ...]) -> str:
    return "(" + ",".join(str(index + 1) for index in key) + ")"


def print_summary(alpha_ratio: float) -> None:
    data = build_v_prefactor(alpha_ratio)
    print("=" * 78)
    print("GS ZERO-MODE PREFACTOR: v_IJ(Y)")
    print("=" * 78)
    print(f"alpha ratio           : {data.alpha_ratio:.6g}")
    print(f"degree-2 monomials    : {len(data.y2)}")
    print(f"degree-4 monomials    : {len(data.w4)}")
    print(f"degree-6 monomials    : {len(data.y6)}")
    print("degree-8 monomials    : 1")
    print()
    print(f"max w symmetry error  : {data.max_w_symmetry_error:.3e}")
    print(f"max y antisym error   : {data.max_y_antisymmetry_error:.3e}")
    print(f"max tr(t4) error      : {data.max_t_trace_error:.3e}")
    print(f"max tr(y2) error      : {data.max_trace_y2_error:.3e}")
    print(f"max tr(y6) error      : {data.max_trace_y6_error:.3e}")
    print()
    print(
        "This is the imported continuum coefficient set in one fixed SO(8) "
        "basis. The remaining missing ingredient for the full graviton check is "
        "the external-state / Clifford-module dictionary."
    )


def top_terms(
    scalar_coeffs: dict[tuple[int, ...], complex], limit: int
) -> list[tuple[tuple[int, ...], complex]]:
    return sorted(
        scalar_coeffs.items(),
        key=lambda item: float(abs(item[1])),
        reverse=True,
    )[:limit]


def print_sample(alpha_ratio: float, ij_pair: tuple[int, int], limit: int) -> None:
    i, j = ij_pair
    if not (1 <= i <= 8 and 1 <= j <= 8):
        raise ValueError("I,J must lie in {1,...,8}")

    data = build_v_prefactor(alpha_ratio)
    i0 = i - 1
    j0 = j - 1

    print("=" * 78)
    print(f"SAMPLE v_{{{i}{j}}}(Y) COEFFICIENTS")
    print("=" * 78)
    print(f"w0 : {data.w0[i0, j0]}")
    print(f"w8 : {data.w8[i0, j0]}")
    print()

    y2_terms = {key: value[i0, j0] for key, value in data.y2.items() if abs(value[i0, j0]) > 1e-12}
    w4_terms = {key: value[i0, j0] for key, value in data.w4.items() if abs(value[i0, j0]) > 1e-12}
    y6_terms = {key: value[i0, j0] for key, value in data.y6.items() if abs(value[i0, j0]) > 1e-12}

    print(f"nonzero degree-2 terms: {len(y2_terms)}")
    for key, value in top_terms(y2_terms, limit):
        print(f"  Y^{format_key(key)} : {value}")
    print()

    print(f"nonzero degree-4 terms: {len(w4_terms)}")
    for key, value in top_terms(w4_terms, limit):
        print(f"  Y^{format_key(key)} : {value}")
    print()

    print(f"nonzero degree-6 terms: {len(y6_terms)}")
    for key, value in top_terms(y6_terms, limit):
        print(f"  Y^{format_key(key)} : {value}")


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
        help="the continuum alpha in the imported formulas (default: 1)",
    )
    parser.add_argument(
        "--sample-ij",
        type=parse_ij_pair,
        default=None,
        help="print the explicit coefficient support of v_{IJ}(Y) for one I,J pair",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="maximum number of sample monomials to print per degree",
    )
    args = parser.parse_args()

    print_summary(args.alpha_ratio)
    if args.sample_ij is not None:
        print()
        print_sample(args.alpha_ratio, args.sample_ij, args.limit)


if __name__ == "__main__":
    main()
