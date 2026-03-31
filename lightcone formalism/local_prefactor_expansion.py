#!/usr/bin/env python3
"""
Exact mixed-variable expansion of the local superstring prefactor.

This helper does not yet integrate out the fermionic nonzero modes. Instead it
builds the local prefactor symbol in a mixed Grassmann basis

    Lambda_local^a = Lambda_lat^a + Xi_local^a,

where:

- `Lambda_lat^a` is the reduced overlap-constrained zero-mode coordinate,
- `Xi_local^a` is an abstract 8-component Grassmann correction standing for the
  explicit nonzero-mode part of the local interaction-point fermion.

The resulting expansion is exact at the level of the continuum SO(8) symbol
v_{IJ}. It cleanly separates:

1. the reduced ansatz piece (Xi-degree 0),
2. the genuinely local correction sectors (Xi-degree > 0),

so the remaining task is a separate Grassmann-moment reduction of the
Xi-dependent pieces against the exact overlap.
"""

from __future__ import annotations

from typing import Any

import fermionic_graviton_contraction as fgc
import numpy as np


def _sparse_add(
    left: dict[tuple[int, ...], complex],
    right: dict[tuple[int, ...], complex],
) -> dict[tuple[int, ...], complex]:
    out = dict(left)
    for monomial, coefficient in right.items():
        out[monomial] = out.get(monomial, 0.0j) + coefficient
    return {key: value for key, value in out.items() if abs(value) > 1.0e-12}


def substitute_lambda_plus_xi(
    poly: dict[tuple[int, ...], complex],
) -> dict[tuple[int, ...], complex]:
    """
    Replace each Lambda^a by lambda^a + xi^a.

    Variable ordering:
    - lambda^a uses indices 0..7,
    - xi^a uses indices 8..15.
    """
    total: dict[tuple[int, ...], complex] = {}
    for monomial, coefficient in poly.items():
        terms: dict[tuple[int, ...], complex] = {(): coefficient}
        for index in monomial:
            factor = {
                (index,): 1.0,
                (8 + index,): 1.0,
            }
            terms = fgc.multiply_sparse(terms, factor)
        total = _sparse_add(total, terms)
    return total


def split_by_xi_degree(
    poly: dict[tuple[int, ...], complex],
) -> dict[int, dict[tuple[int, ...], complex]]:
    """
    Split a mixed (lambda, xi) polynomial by xi-degree.
    """
    pieces: dict[int, dict[tuple[int, ...], complex]] = {}
    for monomial, coefficient in poly.items():
        xi_degree = sum(1 for index in monomial if index >= 8)
        piece = pieces.setdefault(xi_degree, {})
        piece[monomial] = piece.get(monomial, 0.0j) + coefficient
    return {
        degree: {
            monomial: coeff
            for monomial, coeff in piece.items()
            if abs(coeff) > 1.0e-12
        }
        for degree, piece in pieces.items()
    }


def recompose_split(
    pieces: dict[int, dict[tuple[int, ...], complex]],
) -> dict[tuple[int, ...], complex]:
    """
    Recombine the xi-degree decomposition.
    """
    total: dict[tuple[int, ...], complex] = {}
    for piece in pieces.values():
        total = _sparse_add(total, piece)
    return total


def set_xi_zero(
    poly: dict[tuple[int, ...], complex],
) -> dict[tuple[int, ...], complex]:
    """
    Project to the reduced ansatz by setting all xi variables to zero.
    """
    out: dict[tuple[int, ...], complex] = {}
    for monomial, coefficient in poly.items():
        if any(index >= 8 for index in monomial):
            continue
        out[monomial] = out.get(monomial, 0.0j) + coefficient
    return out


def prefactor_mixed_expansion(
    alpha_ratio: float,
    i: int,
    j: int,
    trace_dropped: bool = False,
) -> dict[int, dict[tuple[int, ...], complex]]:
    """
    Exact xi-degree decomposition of v_{IJ}(Lambda_lat + Xi_local).
    """
    base = fgc.v_prefactor_polynomial(alpha_ratio, i, j, trace_dropped)
    shifted = substitute_lambda_plus_xi(base)
    return split_by_xi_degree(shifted)


def graviton_wavefunction_mixed_expansion(
    epsilon: np.ndarray,
    external_alpha_ratio: float = 1.0,
) -> dict[int, dict[tuple[int, ...], complex]]:
    """
    Exact xi-degree decomposition of Psi_epsilon(lambda + xi).
    """
    base = fgc.graviton_wavefunction(epsilon, external_alpha_ratio)
    shifted = substitute_lambda_plus_xi(base)
    return split_by_xi_degree(shifted)


def xi_degree_profile(
    pieces: dict[int, dict[tuple[int, ...], complex]],
) -> dict[int, int]:
    """
    Return the number of nonzero monomials in each xi-degree sector.
    """
    return {degree: len(piece) for degree, piece in pieces.items() if piece}


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
