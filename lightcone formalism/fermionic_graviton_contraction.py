#!/usr/bin/env python3
"""
Explicit tree-level fermionic zero-mode contraction for the superstring graviton vertex.

This module works in the local Lambda-superfield convention suggested by the
matrix-string / light-cone dictionary

    v^{ij}(Lambda) <-> 16 Sigma^j Sigma_tilde^i.

In this convention a closed-string vector-vector ground state with polarization
epsilon_{ij} is represented by the Grassmann polynomial

    Psi_epsilon(lambda) = (1/16) epsilon_{ij} v^{ji}(lambda).

The cubic kinematical measure contains the zero-mode delta function

    Delta^8(lambda_1 + lambda_2 + lambda_3),

so the tree-level fermionic factor reduces to a 16-Grassmann integral

    A_F(e1,e2,e3; T_bos)
      = int d^8 lambda_1 d^8 lambda_2
          Psi_1(lambda_1)
          Psi_2(lambda_2)
          Psi_3(lambda_1 + lambda_2)
          T_bos^{IJ} v_{IJ}(lambda lambda_2 - (1-lambda) lambda_1),

with lambda = alpha_1 / alpha_3 and

    T_bos^{IJ} = A_delta delta^{IJ} + B_qq qhat^I qhat^J.

This does not yet fix the overall continuum normalization of the full cubic
vertex, but it *does* complete the missing fermionic zero-mode contraction in a
definite local convention and makes concrete graviton / dilaton / B-field
channel checks possible.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gs_zero_mode_prefactor as gp
import superstring_prefactor_check as sp


DEFAULT_D_PERP = 8
FULL_TOP_16 = tuple(range(16))
_V_CACHE: dict[tuple[float, int, int, bool], dict[tuple[int, ...], complex]] = {}
_WAVEFUNCTION_CACHE: dict[
    tuple[float, tuple[complex, ...]], dict[tuple[int, ...], complex]
] = {}
_STATE_PRODUCT_CACHE: dict[
    tuple[float, tuple[complex, ...], tuple[complex, ...], tuple[complex, ...]],
    dict[tuple[int, ...], complex],
] = {}
_BASIS_PREF_CACHE: dict[
    tuple[float, bool], tuple[dict[tuple[int, ...], complex], dict[tuple[int, ...], complex]]
] = {}
_CHANNEL_RESPONSE_CACHE: dict[
    tuple[
        float,
        bool,
        float,
        tuple[complex, ...],
        tuple[complex, ...],
        tuple[complex, ...],
    ],
    tuple[complex, complex],
] = {}


BENCHMARK_CHANNELS = (
    ("perp23", "perp23", "parallel"),
    ("perp23", "perp24", "parallel"),
    ("parallel", "perp23", "perp23"),
    ("perp23", "perp23", "dilaton"),
    ("parallel", "parallel", "dilaton"),
    ("perp23", "perp23", "b23"),
    ("parallel", "parallel", "b23"),
)


def benchmark_response_closed_forms(lambda_ratio: float) -> dict[tuple[str, str, str], complex]:
    """
    Closed forms observed for the trace-dropped benchmark pure responses.

    These formulas are verified numerically on the default lambda grid and
    additional off-grid spot checks used in the regression tests.
    """
    lambda_ratio = float(lambda_ratio)
    diag = 4.0 * math.sqrt(14.0) * (1.0 - lambda_ratio) ** 2
    return {
        ("perp23", "perp23", "parallel"): complex(diag),
        ("perp23", "perp24", "parallel"): complex(0.5 * diag),
        ("parallel", "perp23", "perp23"): complex(diag / (lambda_ratio**2)),
        ("perp23", "perp23", "dilaton"): 0.0j,
        ("parallel", "parallel", "dilaton"): 0.0j,
        ("perp23", "perp23", "b23"): 0.0j,
        ("parallel", "parallel", "b23"): 0.0j,
    }


def benchmark_trace_dropped_amplitude_closed_forms(
    lambda_ratio: float,
    b_qq: complex,
) -> dict[tuple[str, str, str], complex]:
    """
    Closed forms for the trace-dropped assembled benchmark channels.

    For the benchmark channels used in the superstring scans, the trace-dropped
    fermionic contraction kills the A_delta contribution and leaves only the
    universal qq-response profile multiplied by the bosonic coefficient B_qq.
    """
    responses = benchmark_response_closed_forms(lambda_ratio)
    return {
        channel: complex(b_qq) * response
        for channel, response in responses.items()
    }


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


def merge_sign(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    inversions = 0
    for left_index in left:
        for right_index in right:
            if left_index > right_index:
                inversions += 1
    return -1 if inversions % 2 else 1


def multiply_sparse(
    left: dict[tuple[int, ...], complex],
    right: dict[tuple[int, ...], complex],
) -> dict[tuple[int, ...], complex]:
    out: dict[tuple[int, ...], complex] = {}
    for left_key, left_value in left.items():
        left_set = set(left_key)
        for right_key, right_value in right.items():
            if left_set.intersection(right_key):
                continue
            merged_key = tuple(sorted(left_key + right_key))
            out[merged_key] = out.get(merged_key, 0.0j) + (
                merge_sign(left_key, right_key) * left_value * right_value
            )
    return {key: value for key, value in out.items() if abs(value) > 1.0e-12}


def integrate_16(poly: dict[tuple[int, ...], complex]) -> complex:
    return complex(poly.get(FULL_TOP_16, 0.0j))


def _float_key(value: float) -> float:
    return round(float(value), 12)


def _epsilon_key(epsilon: np.ndarray) -> tuple[complex, ...]:
    epsilon = np.asarray(epsilon, dtype=complex)
    if epsilon.shape != (8, 8):
        raise ValueError("epsilon must be 8x8")
    return tuple(complex(value) for value in epsilon.ravel())


def _sparse_scale(
    poly: dict[tuple[int, ...], complex],
    scalar: complex,
) -> dict[tuple[int, ...], complex]:
    if abs(scalar) < 1.0e-15:
        return {}
    return {
        monomial: scalar * coefficient
        for monomial, coefficient in poly.items()
        if abs(scalar * coefficient) > 1.0e-12
    }


def _sparse_add(
    left: dict[tuple[int, ...], complex],
    right: dict[tuple[int, ...], complex],
) -> dict[tuple[int, ...], complex]:
    out = dict(left)
    for monomial, coefficient in right.items():
        out[monomial] = out.get(monomial, 0.0j) + coefficient
    return {key: value for key, value in out.items() if abs(value) > 1.0e-12}


def v_prefactor_polynomial(
    alpha_ratio: float,
    i: int,
    j: int,
    trace_dropped: bool = False,
) -> dict[tuple[int, ...], complex]:
    key = (float(alpha_ratio), int(i), int(j), bool(trace_dropped))
    if key in _V_CACHE:
        return _V_CACHE[key]

    prefactor = gp.build_v_prefactor(alpha_ratio)
    poly: dict[tuple[int, ...], complex] = {}
    if not trace_dropped:
        poly[()] = complex(prefactor.w0[i, j])
        poly[tuple(range(8))] = complex(prefactor.w8[i, j])
    for store in (prefactor.y2, prefactor.w4, prefactor.y6):
        for monomial, value in store.items():
            coefficient = complex(value[i, j])
            if abs(coefficient) > 1.0e-12:
                poly[monomial] = poly.get(monomial, 0.0j) + coefficient

    _V_CACHE[key] = {
        monomial: coefficient
        for monomial, coefficient in poly.items()
        if abs(coefficient) > 1.0e-12
    }
    return _V_CACHE[key]


def embed_leg(
    poly: dict[tuple[int, ...], complex],
    leg: int,
) -> dict[tuple[int, ...], complex]:
    offset = 8 * leg
    return {
        tuple(offset + index for index in monomial): coefficient
        for monomial, coefficient in poly.items()
    }


def substitute_two_leg(
    poly: dict[tuple[int, ...], complex],
    coeff_leg1: complex,
    coeff_leg2: complex,
) -> dict[tuple[int, ...], complex]:
    """
    Replace each local Lambda^a by coeff_leg1 * lambda_1^a + coeff_leg2 * lambda_2^a.
    """
    total: dict[tuple[int, ...], complex] = {}
    for monomial, coefficient in poly.items():
        terms: dict[tuple[int, ...], complex] = {(): coefficient}
        for index in monomial:
            factor = {
                (index,): coeff_leg1,
                (8 + index,): coeff_leg2,
            }
            terms = multiply_sparse(terms, factor)
        for key, value in terms.items():
            total[key] = total.get(key, 0.0j) + value
    return {key: value for key, value in total.items() if abs(value) > 1.0e-12}


def graviton_wavefunction(
    epsilon: np.ndarray,
    external_alpha_ratio: float = 1.0,
) -> dict[tuple[int, ...], complex]:
    """
    Build Psi_epsilon(lambda) = (1/16) epsilon_{ij} v^{ji}(lambda).
    """
    epsilon = np.asarray(epsilon, dtype=complex)
    key = (_float_key(external_alpha_ratio), _epsilon_key(epsilon))
    if key in _WAVEFUNCTION_CACHE:
        return _WAVEFUNCTION_CACHE[key]

    out: dict[tuple[int, ...], complex] = {}
    for i in range(8):
        for j in range(8):
            coefficient = epsilon[i, j] / 16.0
            if abs(coefficient) < 1.0e-15:
                continue
            base = v_prefactor_polynomial(external_alpha_ratio, j, i, False)
            for monomial, value in base.items():
                out[monomial] = out.get(monomial, 0.0j) + coefficient * value
    _WAVEFUNCTION_CACHE[key] = {
        monomial: coefficient
        for monomial, coefficient in out.items()
        if abs(coefficient) > 1.0e-12
    }
    return _WAVEFUNCTION_CACHE[key]


def polarization_tensors(d_perp: int = DEFAULT_D_PERP) -> dict[str, np.ndarray]:
    if d_perp != 8:
        raise ValueError("the current zero-mode implementation is fixed to SO(8)")

    e1 = np.zeros(d_perp, dtype=float)
    e2 = np.zeros(d_perp, dtype=float)
    e3 = np.zeros(d_perp, dtype=float)
    e4 = np.zeros(d_perp, dtype=float)
    e1[0] = 1.0
    e2[1] = 1.0
    e3[2] = 1.0
    e4[3] = 1.0
    identity = np.eye(d_perp, dtype=float)
    qhat_qhat = np.outer(e1, e1)

    return {
        "parallel": math.sqrt(d_perp / (d_perp - 1.0))
        * (qhat_qhat - identity / d_perp),
        "perp23": (np.outer(e2, e2) - np.outer(e3, e3)) / math.sqrt(2.0),
        "perp24": (np.outer(e2, e2) - np.outer(e4, e4)) / math.sqrt(2.0),
        "dilaton": identity / math.sqrt(d_perp),
        "b23": (np.outer(e2, e3) - np.outer(e3, e2)) / math.sqrt(2.0),
    }


def bosonic_tensor_from_prefactor_data(data: sp.PrefactorData) -> np.ndarray:
    tensor = data.a_delta_reduced * np.eye(8, dtype=complex)
    tensor[0, 0] += data.b_qq_reduced
    return tensor


def external_state_product(
    epsilon_1: np.ndarray,
    epsilon_2: np.ndarray,
    epsilon_3: np.ndarray,
    external_alpha_ratio: float = 1.0,
) -> dict[tuple[int, ...], complex]:
    key = (
        _float_key(external_alpha_ratio),
        _epsilon_key(epsilon_1),
        _epsilon_key(epsilon_2),
        _epsilon_key(epsilon_3),
    )
    if key in _STATE_PRODUCT_CACHE:
        return _STATE_PRODUCT_CACHE[key]

    state_1 = embed_leg(graviton_wavefunction(epsilon_1, external_alpha_ratio), 0)
    state_2 = embed_leg(graviton_wavefunction(epsilon_2, external_alpha_ratio), 1)
    state_3 = substitute_two_leg(
        graviton_wavefunction(epsilon_3, external_alpha_ratio),
        1.0,
        1.0,
    )
    _STATE_PRODUCT_CACHE[key] = multiply_sparse(
        state_1,
        multiply_sparse(
            state_2,
            state_3,
        ),
    )
    return _STATE_PRODUCT_CACHE[key]


def basis_prefactor_polynomials(
    lambda_ratio: float,
    trace_dropped: bool = True,
) -> tuple[dict[tuple[int, ...], complex], dict[tuple[int, ...], complex]]:
    key = (_float_key(lambda_ratio), bool(trace_dropped))
    if key in _BASIS_PREF_CACHE:
        return _BASIS_PREF_CACHE[key]

    alpha_1 = float(lambda_ratio)
    alpha_2 = 1.0 - alpha_1

    delta_piece: dict[tuple[int, ...], complex] = {}
    for i in range(8):
        base = substitute_two_leg(
            v_prefactor_polynomial(lambda_ratio, i, i, trace_dropped),
            -alpha_2,
            alpha_1,
        )
        delta_piece = _sparse_add(delta_piece, base)

    qq_piece = substitute_two_leg(
        v_prefactor_polynomial(lambda_ratio, 0, 0, trace_dropped),
        -alpha_2,
        alpha_1,
    )
    _BASIS_PREF_CACHE[key] = (delta_piece, qq_piece)
    return _BASIS_PREF_CACHE[key]


def fermionic_channel_responses(
    epsilon_1: np.ndarray,
    epsilon_2: np.ndarray,
    epsilon_3: np.ndarray,
    lambda_ratio: float,
    trace_dropped: bool = True,
    external_alpha_ratio: float = 1.0,
) -> tuple[complex, complex]:
    key = (
        _float_key(lambda_ratio),
        bool(trace_dropped),
        _float_key(external_alpha_ratio),
        _epsilon_key(epsilon_1),
        _epsilon_key(epsilon_2),
        _epsilon_key(epsilon_3),
    )
    if key in _CHANNEL_RESPONSE_CACHE:
        return _CHANNEL_RESPONSE_CACHE[key]

    external = external_state_product(
        epsilon_1,
        epsilon_2,
        epsilon_3,
        external_alpha_ratio=external_alpha_ratio,
    )
    delta_piece, qq_piece = basis_prefactor_polynomials(
        lambda_ratio,
        trace_dropped=trace_dropped,
    )
    delta_response = integrate_16(multiply_sparse(external, delta_piece))
    qq_response = integrate_16(multiply_sparse(external, qq_piece))
    _CHANNEL_RESPONSE_CACHE[key] = (delta_response, qq_response)
    return _CHANNEL_RESPONSE_CACHE[key]


def fermionic_channel_amplitude_from_ab(
    epsilon_1: np.ndarray,
    epsilon_2: np.ndarray,
    epsilon_3: np.ndarray,
    a_delta: complex,
    b_qq: complex,
    lambda_ratio: float,
    trace_dropped: bool = True,
    external_alpha_ratio: float = 1.0,
) -> complex:
    delta_response, qq_response = fermionic_channel_responses(
        epsilon_1,
        epsilon_2,
        epsilon_3,
        lambda_ratio,
        trace_dropped=trace_dropped,
        external_alpha_ratio=external_alpha_ratio,
    )
    return complex(a_delta) * delta_response + complex(b_qq) * qq_response


def fermionic_channel_amplitude(
    epsilon_1: np.ndarray,
    epsilon_2: np.ndarray,
    epsilon_3: np.ndarray,
    bosonic_tensor: np.ndarray,
    lambda_ratio: float,
    trace_dropped: bool = True,
    external_alpha_ratio: float = 1.0,
) -> complex:
    bosonic_tensor = np.asarray(bosonic_tensor, dtype=complex)
    if bosonic_tensor.shape != (8, 8):
        raise ValueError("bosonic_tensor must be 8x8")

    diagonal = np.diag(bosonic_tensor)
    off_diagonal = bosonic_tensor - np.diag(diagonal)
    if (
        np.max(np.abs(off_diagonal)) < 1.0e-14
        and np.max(np.abs(diagonal[1:] - diagonal[1])) < 1.0e-14
    ):
        a_delta = complex(diagonal[1])
        b_qq = complex(diagonal[0] - diagonal[1])
        return fermionic_channel_amplitude_from_ab(
            epsilon_1,
            epsilon_2,
            epsilon_3,
            a_delta,
            b_qq,
            lambda_ratio,
            trace_dropped=trace_dropped,
            external_alpha_ratio=external_alpha_ratio,
        )

    alpha_1 = float(lambda_ratio)
    alpha_2 = 1.0 - alpha_1
    external = external_state_product(
        epsilon_1,
        epsilon_2,
        epsilon_3,
        external_alpha_ratio=external_alpha_ratio,
    )
    prefactor_poly: dict[tuple[int, ...], complex] = {}
    for i in range(8):
        for j in range(8):
            coefficient = complex(bosonic_tensor[i, j])
            if abs(coefficient) < 1.0e-15:
                continue
            base = substitute_two_leg(
                v_prefactor_polynomial(lambda_ratio, i, j, trace_dropped),
                -alpha_2,
                alpha_1,
            )
            prefactor_poly = _sparse_add(prefactor_poly, _sparse_scale(base, coefficient))
    return integrate_16(multiply_sparse(external, prefactor_poly))


def channel_response_report(
    lambda_ratio: float,
    trace_dropped: bool = True,
    external_alpha_ratio: float = 1.0,
    channels: tuple[tuple[str, str, str], ...] = BENCHMARK_CHANNELS,
) -> dict[str, Any]:
    polarizations = polarization_tensors()
    rows = []
    for eps1_name, eps2_name, eps3_name in channels:
        delta_response, qq_response = fermionic_channel_responses(
            polarizations[eps1_name],
            polarizations[eps2_name],
            polarizations[eps3_name],
            lambda_ratio,
            trace_dropped=trace_dropped,
            external_alpha_ratio=external_alpha_ratio,
        )
        rows.append(
            {
                "channels": [eps1_name, eps2_name, eps3_name],
                "delta_response": delta_response,
                "qq_response": qq_response,
            }
        )

    return {
        "parameters": {
            "lambda_ratio": float(lambda_ratio),
            "trace_dropped": bool(trace_dropped),
            "external_alpha_ratio": float(external_alpha_ratio),
        },
        "rows": rows,
    }


def channel_report(
    n1: int,
    n2: int,
    alpha_prime: float = 1.0,
    left_variant: str = "second_order",
    right_variant: str = "second_order",
    trace_dropped: bool = True,
) -> dict[str, Any]:
    polarizations = polarization_tensors()
    prefactor = sp.prefactor_data(
        n1,
        n2,
        alpha_prime,
        left_variant=left_variant,
        right_variant=right_variant,
    )
    bosonic_tensor = bosonic_tensor_from_prefactor_data(prefactor)
    lambda_ratio = n1 / (n1 + n2)

    channels = [
        ("perp23", "perp23", "parallel"),
        ("perp23", "perp24", "parallel"),
        ("parallel", "parallel", "parallel"),
        ("parallel", "perp23", "perp23"),
        ("perp23", "perp23", "dilaton"),
        ("parallel", "parallel", "dilaton"),
        ("perp23", "perp23", "b23"),
        ("parallel", "parallel", "b23"),
    ]

    rows = []
    for left_name, middle_name, right_name in channels:
        value = fermionic_channel_amplitude(
            polarizations[left_name],
            polarizations[middle_name],
            polarizations[right_name],
            bosonic_tensor,
            lambda_ratio,
            trace_dropped=trace_dropped,
        )
        rows.append(
            {
                "channels": [left_name, middle_name, right_name],
                "amplitude": value,
            }
        )

    return {
        "parameters": {
            "n1": n1,
            "n2": n2,
            "alpha_prime": alpha_prime,
            "left_variant": left_variant,
            "right_variant": right_variant,
            "trace_dropped": trace_dropped,
            "lambda_ratio": lambda_ratio,
        },
        "bosonic_prefactor": {
            "A_delta": prefactor.a_delta_reduced,
            "B_qq": prefactor.b_qq_reduced,
        },
        "rows": rows,
    }


def print_report(report: dict[str, Any]) -> None:
    params = report["parameters"]
    bos = report["bosonic_prefactor"]
    print("=" * 108)
    print("FERMIONIC TREE-LEVEL GRAVITON CONTRACTION")
    print("=" * 108)
    print(
        f"n1={params['n1']} n2={params['n2']} lambda={params['lambda_ratio']:.6f} "
        f"trace_dropped={params['trace_dropped']}"
    )
    print(
        f"A_delta={bos['A_delta']:.9f}   B_qq={bos['B_qq']:.9f}   "
        f"left={params['left_variant']}   right={params['right_variant']}"
    )
    print()
    print(f"{'eps1':>10s} {'eps2':>10s} {'eps3':>10s} {'amplitude':>24s}")
    print("-" * 62)
    for row in report["rows"]:
        eps1, eps2, eps3 = row["channels"]
        amp = complex(row["amplitude"])
        print(
            f"{eps1:>10s} {eps2:>10s} {eps3:>10s} "
            f"{amp.real:12.9f}{amp.imag:+12.9f}i"
        )


def markdown_report(report: dict[str, Any]) -> str:
    params = report["parameters"]
    bos = report["bosonic_prefactor"]
    lines = [
        "# Fermionic Tree-Level Graviton Contraction",
        "",
        (
            f"- `n1 = {params['n1']}`, `n2 = {params['n2']}`, "
            f"`lambda = {params['lambda_ratio']:.6f}`, "
            f"`trace_dropped = {params['trace_dropped']}`"
        ),
        (
            f"- `A_delta = {bos['A_delta']:.9f}`, "
            f"`B_qq = {bos['B_qq']:.9f}`, "
            f"`left = {params['left_variant']}`, "
            f"`right = {params['right_variant']}`"
        ),
        "",
        "| eps1 | eps2 | eps3 | amplitude |",
        "|---|---|---|---:|",
    ]
    for row in report["rows"]:
        amplitude = row["amplitude"]
        if isinstance(amplitude, dict):
            amp = complex(amplitude["real"], amplitude["imag"])
        else:
            amp = complex(amplitude)
        lines.append(
            f"| {row['channels'][0]} | {row['channels'][1]} | {row['channels'][2]} | "
            f"{amp.real:.9f}{amp.imag:+.9f}i |"
        )
    lines.append("")
    return "\n".join(lines)


def print_response_report(report: dict[str, Any]) -> None:
    params = report["parameters"]
    print("=" * 108)
    print("FERMIONIC ZERO-MODE CHANNEL RESPONSES")
    print("=" * 108)
    print(
        f"lambda={params['lambda_ratio']:.6f} "
        f"trace_dropped={params['trace_dropped']} "
        f"external_alpha_ratio={params['external_alpha_ratio']:.6f}"
    )
    print()
    print(
        f"{'eps1':>10s} {'eps2':>10s} {'eps3':>10s} "
        f"{'R_delta':>22s} {'R_qq':>22s}"
    )
    print("-" * 92)
    for row in report["rows"]:
        eps1, eps2, eps3 = row["channels"]
        delta_response = complex(row["delta_response"])
        qq_response = complex(row["qq_response"])
        print(
            f"{eps1:>10s} {eps2:>10s} {eps3:>10s} "
            f"{delta_response.real:11.8f}{delta_response.imag:+11.8f}i "
            f"{qq_response.real:11.8f}{qq_response.imag:+11.8f}i"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n1", type=int, default=128)
    parser.add_argument("--n2", type=int, default=192)
    parser.add_argument("--lambda-ratio", type=float, default=None)
    parser.add_argument("--alpha-prime", type=float, default=1.0)
    parser.add_argument("--left-variant", type=str, default="second_order")
    parser.add_argument("--right-variant", type=str, default="second_order")
    parser.add_argument(
        "--no-trace-drop",
        action="store_true",
        help="keep the delta_{IJ} pieces of v_{IJ} instead of using the on-shell trace drop",
    )
    parser.add_argument(
        "--response-report",
        action="store_true",
        help="print the pure fermionic channel responses R_delta and R_qq instead of assembled amplitudes",
    )
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--markdown-out", type=str, default=None)
    args = parser.parse_args()

    if args.response_report:
        lambda_ratio = (
            float(args.lambda_ratio)
            if args.lambda_ratio is not None
            else args.n1 / (args.n1 + args.n2)
        )
        report = channel_response_report(
            lambda_ratio,
            trace_dropped=not args.no_trace_drop,
        )
        print_response_report(report)
    else:
        report = channel_report(
            args.n1,
            args.n2,
            alpha_prime=args.alpha_prime,
            left_variant=args.left_variant,
            right_variant=args.right_variant,
            trace_dropped=not args.no_trace_drop,
        )
        print_report(report)

    if args.json_out is not None:
        json_path = Path(args.json_out)
        json_path.write_text(
            json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n"
        )
        print(f"\nWrote JSON report to {json_path}")
    if args.markdown_out is not None:
        md_path = Path(args.markdown_out)
        md_path.write_text(markdown_report(json_safe(report)) + "\n")
        print(f"Wrote markdown report to {md_path}")


if __name__ == "__main__":
    main()
