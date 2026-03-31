#!/usr/bin/env python3
"""
Unreduced local fermionic channel polynomials for the superstring cubic vertex.

This module sits one step beyond the exact local prefactor expansion:

    Lambda_join = coeff_1 * lambda_1 + coeff_2 * lambda_2 + Xi_loc,

where:
- lambda_1, lambda_2 are the surviving reduced zero modes after the cubic
  delta-function reduction,
- Xi_loc is the explicit 8-component local nonzero-mode correction.

For a chosen external polarization channel, we integrate over lambda_1 and
lambda_2 exactly but keep Xi_loc symbolic. The result is an exact finite
Grassmann polynomial in Xi_loc. Its Xi-degree-0 term is the current reduced
ansatz response; the higher Xi-degree sectors are the local corrections that
still need to be integrated out against the fermionic nonzero-mode overlap.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import fermionic_graviton_contraction as fgc
import local_interaction_point_fermion as lif


FULL_TOP_16 = tuple(range(16))


def _sparse_add(
    left: dict[tuple[int, ...], complex],
    right: dict[tuple[int, ...], complex],
) -> dict[tuple[int, ...], complex]:
    out = dict(left)
    for monomial, coefficient in right.items():
        out[monomial] = out.get(monomial, 0.0j) + coefficient
    return {key: value for key, value in out.items() if abs(value) > 1.0e-12}


def substitute_two_leg_plus_xi(
    poly: dict[tuple[int, ...], complex],
    coeff_leg1: complex,
    coeff_leg2: complex,
) -> dict[tuple[int, ...], complex]:
    """
    Replace each Lambda^a by coeff_1 * lambda_1^a + coeff_2 * lambda_2^a + xi^a.

    Variable ordering:
    - lambda_1^a -> indices 0..7
    - lambda_2^a -> indices 8..15
    - xi^a       -> indices 16..23
    """
    total: dict[tuple[int, ...], complex] = {}
    for monomial, coefficient in poly.items():
        terms: dict[tuple[int, ...], complex] = {(): coefficient}
        for index in monomial:
            factor = {
                (index,): coeff_leg1,
                (8 + index,): coeff_leg2,
                (16 + index,): 1.0,
            }
            terms = fgc.multiply_sparse(terms, factor)
        total = _sparse_add(total, terms)
    return total


def integrate_lambda_16_keep_xi(
    poly: dict[tuple[int, ...], complex],
) -> dict[tuple[int, ...], complex]:
    """
    Integrate over lambda_1 and lambda_2 while keeping Xi_loc symbolic.

    Because indices 0..15 always sort before 16..23, the remaining Xi_loc
    monomial is read off without extra permutation signs.
    """
    out: dict[tuple[int, ...], complex] = {}
    for monomial, coefficient in poly.items():
        lambda_part = tuple(index for index in monomial if index < 16)
        if lambda_part != FULL_TOP_16:
            continue
        xi_part = tuple(index - 16 for index in monomial if index >= 16)
        out[xi_part] = out.get(xi_part, 0.0j) + coefficient
    return {key: value for key, value in out.items() if abs(value) > 1.0e-12}


def basis_prefactor_local_polynomials(
    lambda_ratio: float,
    trace_dropped: bool = True,
) -> tuple[dict[tuple[int, ...], complex], dict[tuple[int, ...], complex]]:
    coeff_leg1, coeff_leg2 = lif.reduced_lambda_zero_mode_substitution_weights(
        lambda_ratio
    )

    delta_piece: dict[tuple[int, ...], complex] = {}
    for i in range(8):
        base = substitute_two_leg_plus_xi(
            fgc.v_prefactor_polynomial(lambda_ratio, i, i, trace_dropped),
            coeff_leg1,
            coeff_leg2,
        )
        delta_piece = _sparse_add(delta_piece, base)

    qq_piece = substitute_two_leg_plus_xi(
        fgc.v_prefactor_polynomial(lambda_ratio, 0, 0, trace_dropped),
        coeff_leg1,
        coeff_leg2,
    )
    return delta_piece, qq_piece


def local_channel_response_polynomial(
    epsilon_1: np.ndarray,
    epsilon_2: np.ndarray,
    epsilon_3: np.ndarray,
    lambda_ratio: float,
    *,
    response_kind: str = "qq",
    trace_dropped: bool = True,
    external_alpha_ratio: float = 1.0,
) -> dict[tuple[int, ...], complex]:
    """
    Compute the exact Xi_loc polynomial for one channel response.

    response_kind:
    - "qq"    -> T_bos^{IJ} = qhat^I qhat^J
    - "delta" -> T_bos^{IJ} = delta^{IJ}
    """
    external = fgc.external_state_product(
        epsilon_1,
        epsilon_2,
        epsilon_3,
        external_alpha_ratio=external_alpha_ratio,
    )
    delta_piece, qq_piece = basis_prefactor_local_polynomials(
        lambda_ratio,
        trace_dropped=trace_dropped,
    )
    if response_kind == "qq":
        prefactor = qq_piece
    elif response_kind == "delta":
        prefactor = delta_piece
    else:
        raise ValueError("response_kind must be 'qq' or 'delta'")

    full = fgc.multiply_sparse(external, prefactor)
    return integrate_lambda_16_keep_xi(full)


def xi_degree_profile(
    poly: dict[tuple[int, ...], complex],
) -> dict[int, int]:
    profile: dict[int, int] = {}
    for monomial in poly:
        degree = len(monomial)
        profile[degree] = profile.get(degree, 0) + 1
    return dict(sorted(profile.items()))


def xi_zero_component(
    poly: dict[tuple[int, ...], complex],
) -> complex:
    return complex(poly.get((), 0.0j))


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
