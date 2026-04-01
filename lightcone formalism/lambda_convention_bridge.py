#!/usr/bin/env python3
r"""
Continuum convention bridge for the cubic GS fermionic variable.

The local Dijkgraaf--Motl variable is defined directly at the interaction point
by the regulated local field

    Lambda_DM^a = sqrt(z/2) theta^a(z) + i sqrt(\bar z / 2) \tilde theta^a(\bar z),

and their octic polynomials are written with no explicit alpha-dependence in
the coefficients. In contrast, the Brink/Green/Schwarz / Pankiewicz-Stefanski
oscillator formulas are usually written in terms of a zero-mode variable
Lambda_GSB with explicit powers of `alpha` in the coefficients.

Dijkgraaf--Motl state explicitly (their eq. `vztah`) that

    Lambda_DM^a = sqrt(2 / alpha) Lambda_GSB^a.

This helper verifies that the coefficient tensors imported in
`gs_zero_mode_prefactor.py` obey exactly that convention bridge:

1. build the alpha-dependent PS/GSB coefficients,
2. rescale degree-d monomials by `(alpha/2)^(d/2)`,
3. obtain an alpha-independent DM-normalized polynomial.

This does not derive the finite-N lattice interaction-point fermion. What it
does settle is that the continuum DM-vs-PS difference is a clean convention
conversion, not an additional source of ambiguity in the local lattice rebuild.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import gs_zero_mode_prefactor as gp
import so8_gamma


PairKey = tuple[int, int]
QuadKey = tuple[int, int, int, int]
HexKey = tuple[int, int, int, int, int, int]


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


def degree_rescaling(alpha_ratio: float, degree: int) -> float:
    if degree % 2 != 0:
        raise ValueError("only even Grassmann degrees occur")
    return float((alpha_ratio / 2.0) ** (degree // 2))


@dataclass(frozen=True)
class DMNormalizedPrefactorData:
    alpha_ratio: float
    w0: np.ndarray
    w4: dict[QuadKey, np.ndarray]
    w8: np.ndarray
    y2: dict[PairKey, np.ndarray]
    y6: dict[HexKey, np.ndarray]


def build_dm_normalized_prefactor(alpha_ratio: float) -> DMNormalizedPrefactorData:
    ps = gp.build_v_prefactor(alpha_ratio)
    return DMNormalizedPrefactorData(
        alpha_ratio=float(alpha_ratio),
        w0=ps.w0.copy(),
        w4={
            key: degree_rescaling(alpha_ratio, 4) * value
            for key, value in ps.w4.items()
        },
        w8=degree_rescaling(alpha_ratio, 8) * ps.w8,
        y2={
            key: degree_rescaling(alpha_ratio, 2) * value
            for key, value in ps.y2.items()
        },
        y6={
            key: degree_rescaling(alpha_ratio, 6) * value
            for key, value in ps.y6.items()
        },
    )


def signed_gamma_ss(gamma_data: so8_gamma.SO8GammaData, i: int, j: int) -> np.ndarray:
    if i == j:
        return np.zeros((8, 8), dtype=complex)
    if i < j:
        return gamma_data.gamma_ss[(i + 1, j + 1)]
    return -gamma_data.gamma_ss[(j + 1, i + 1)]


def expected_dm_normalized_prefactor() -> DMNormalizedPrefactorData:
    gamma_data = so8_gamma.so8_gamma_data()
    base = gp.build_v_prefactor(2.0)

    y2: dict[PairKey, np.ndarray] = {}
    y6: dict[HexKey, np.ndarray] = {
        key: np.zeros((8, 8), dtype=complex) for key in base.y6.keys()
    }

    for a, b in base.y2.keys():
        gamma_ab = np.zeros((8, 8), dtype=complex)
        for i in range(8):
            for j in range(8):
                gamma_ab[i, j] = signed_gamma_ss(gamma_data, i, j)[a, b]
        y2[(a, b)] = (-0.5j) * gamma_ab

        complement = tuple(index for index in range(8) if index not in (a, b))
        sign = gp.inversion_sign((a, b) + complement)
        y6[complement] += (-0.5j * sign) * gamma_ab

    return DMNormalizedPrefactorData(
        alpha_ratio=0.0,
        w0=np.eye(8, dtype=complex),
        w4={key: value.copy() for key, value in base.t4.items()},
        w8=np.eye(8, dtype=complex),
        y2=y2,
        y6=y6,
    )


def compare_dm_normalized_to_expected(alpha_ratio: float) -> dict[str, Any]:
    observed = build_dm_normalized_prefactor(alpha_ratio)
    expected = expected_dm_normalized_prefactor()

    max_w0_error = float(np.max(np.abs(observed.w0 - expected.w0)))
    max_w8_error = float(np.max(np.abs(observed.w8 - expected.w8)))
    max_w4_error = max(
        float(np.max(np.abs(observed.w4[key] - expected.w4[key])))
        for key in observed.w4
    )
    max_y2_error = max(
        float(np.max(np.abs(observed.y2[key] - expected.y2[key])))
        for key in observed.y2
    )
    max_y6_error = max(
        float(np.max(np.abs(observed.y6[key] - expected.y6[key])))
        for key in observed.y6
    )

    return {
        "alpha_ratio": float(alpha_ratio),
        "max_w0_error": max_w0_error,
        "max_w4_error": max_w4_error,
        "max_w8_error": max_w8_error,
        "max_y2_error": max_y2_error,
        "max_y6_error": max_y6_error,
        "max_error": max(
            max_w0_error,
            max_w4_error,
            max_w8_error,
            max_y2_error,
            max_y6_error,
        ),
    }


def alpha_independence_scan(alpha_values: tuple[float, ...]) -> dict[str, Any]:
    datasets = [build_dm_normalized_prefactor(alpha) for alpha in alpha_values]
    anchor = datasets[0]

    max_w0_error = 0.0
    max_w4_error = 0.0
    max_w8_error = 0.0
    max_y2_error = 0.0
    max_y6_error = 0.0
    for dataset in datasets[1:]:
        max_w0_error = max(max_w0_error, float(np.max(np.abs(dataset.w0 - anchor.w0))))
        max_w8_error = max(max_w8_error, float(np.max(np.abs(dataset.w8 - anchor.w8))))
        max_w4_error = max(
            max_w4_error,
            max(
                float(np.max(np.abs(dataset.w4[key] - anchor.w4[key])))
                for key in anchor.w4
            ),
        )
        max_y2_error = max(
            max_y2_error,
            max(
                float(np.max(np.abs(dataset.y2[key] - anchor.y2[key])))
                for key in anchor.y2
            ),
        )
        max_y6_error = max(
            max_y6_error,
            max(
                float(np.max(np.abs(dataset.y6[key] - anchor.y6[key])))
                for key in anchor.y6
            ),
        )

    return {
        "alpha_values": [float(value) for value in alpha_values],
        "max_w0_error": max_w0_error,
        "max_w4_error": max_w4_error,
        "max_w8_error": max_w8_error,
        "max_y2_error": max_y2_error,
        "max_y6_error": max_y6_error,
        "max_error": max(
            max_w0_error,
            max_w4_error,
            max_w8_error,
            max_y2_error,
            max_y6_error,
        ),
    }


def print_report(alpha_ratio: float) -> None:
    comparison = compare_dm_normalized_to_expected(alpha_ratio)
    scan = alpha_independence_scan((0.4, 0.7, 1.0, 1.7))
    print("=" * 96)
    print("DM <-> PS / GSB LAMBDA CONVENTION BRIDGE")
    print("=" * 96)
    print(f"alpha_ratio = {alpha_ratio:.6f}")
    print(
        "expected-form max error: "
        f"{comparison['max_error']:.3e} "
        f"(w0 {comparison['max_w0_error']:.3e}, "
        f"w4 {comparison['max_w4_error']:.3e}, "
        f"w8 {comparison['max_w8_error']:.3e}, "
        f"y2 {comparison['max_y2_error']:.3e}, "
        f"y6 {comparison['max_y6_error']:.3e})"
    )
    print(
        "alpha-independence max error: "
        f"{scan['max_error']:.3e} "
        f"over alpha = {scan['alpha_values']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-ratio", type=float, default=0.7)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    comparison = compare_dm_normalized_to_expected(args.alpha_ratio)
    scan = alpha_independence_scan((0.4, 0.7, 1.0, 1.7))
    report = {"comparison": comparison, "alpha_independence": scan}
    print_report(args.alpha_ratio)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n")
        print(f"Wrote JSON report to {args.json_out}")


if __name__ == "__main__":
    main()
