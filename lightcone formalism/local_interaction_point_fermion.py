#!/usr/bin/env python3
"""
Finite-N join-local fermionic data for the discrete superstring cubic vertex.

This module does not yet define the final local Green-Schwarz interaction-point
fermion. Instead it makes the exact lattice join data explicit and separates
them cleanly from the current reduced zero-mode ansatz.

In the same real zero-sum basis `S_r` used in the bosonic scripts, each site
fermion on leg `r` decomposes exactly as

    theta_n^(r) = theta_av^(r) + sum_m S_r[n,m] vartheta_m^(r),

where `theta_av^(r)` is the leg average and `vartheta_m^(r)` are the real
nonzero-mode coordinates. This identity is the correct starting point for a
local interaction-point rebuild: the site variables at the join are local UV
data, while the averaged variables are reduced IR data.

The current tree-level reduced-Lambda calculations continue to use the PS-like
substitution weights exposed here by `reduced_lambda_zero_mode_substitution_`
`weights`. Future local computations should instead start from the join-local
site data returned by `join_local_fermion_data`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import tachyon_check as tc


@dataclass(frozen=True)
class SiteFermionDecomposition:
    """Exact average-plus-oscillator decomposition of one site variable."""

    n_sites: int
    site_index: int
    average_weight: float
    oscillator_row: np.ndarray
    selector: np.ndarray
    reconstructed_selector: np.ndarray
    reconstruction_error: float


@dataclass(frozen=True)
class JoinLocalFermionData:
    """
    Exact local fermionic data attached to the unfolded cubic join.

    `theta_I_plus` is the site value at `n=0` on leg 1, and `theta_I_minus` is
    the site value at `n=0` on leg 2. Their one-sided arc differences are the
    natural finite-N local fermionic analogues of the bosonic join stencils.
    """

    n1: int
    n2: int
    n3: int
    theta_i_plus: SiteFermionDecomposition
    theta_i_minus: SiteFermionDecomposition
    nabla_plus_selector: np.ndarray
    nabla_minus_selector: np.ndarray
    nabla_plus_oscillator_row: np.ndarray
    nabla_minus_oscillator_row: np.ndarray
    reduced_lambda_average_weights: tuple[float, float]


def average_selector(n_sites: int) -> np.ndarray:
    """Uniform average row on one leg."""
    return np.full(n_sites, 1.0 / n_sites, dtype=float)


def site_selector(n_sites: int, site_index: int) -> np.ndarray:
    """Row vector extracting the site value at `site_index`."""
    selector = np.zeros(n_sites, dtype=float)
    selector[site_index] = 1.0
    return selector


def oscillator_projection_row(n_sites: int, selector: np.ndarray) -> np.ndarray:
    """
    Real nonzero-mode row associated with a site-space selector.

    If `theta = theta_av * 1 + S_r vartheta`, then `selector @ theta` has the
    nonzero-mode coefficient row `selector @ S_r`.
    """
    basis, _ = tc.real_zero_sum_basis(n_sites)
    return selector @ basis


def site_fermion_decomposition(
    n_sites: int,
    site_index: int,
) -> SiteFermionDecomposition:
    """Return the exact decomposition of one site fermion into average + modes."""
    basis, _ = tc.real_zero_sum_basis(n_sites)
    selector = site_selector(n_sites, site_index)
    average = average_selector(n_sites)
    oscillator_row = selector @ basis
    reconstructed = average + basis @ oscillator_row
    error = float(np.linalg.norm(reconstructed - selector, ord=np.inf))
    return SiteFermionDecomposition(
        n_sites=n_sites,
        site_index=site_index,
        average_weight=1.0 / n_sites,
        oscillator_row=oscillator_row,
        selector=selector,
        reconstructed_selector=reconstructed,
        reconstruction_error=error,
    )


def forward_arc_difference_selector(n_sites: int) -> np.ndarray:
    """One-sided forward difference at the join: theta_1 - theta_0."""
    selector = np.zeros(n_sites, dtype=float)
    selector[0] = -1.0
    selector[1] = 1.0
    return selector


def backward_arc_difference_selector(n_sites: int) -> np.ndarray:
    """One-sided backward difference at the join: theta_0 - theta_{N-1}."""
    selector = np.zeros(n_sites, dtype=float)
    selector[0] = 1.0
    selector[-1] = -1.0
    return selector


def reduced_lambda_average_weights(n1: int, n2: int) -> tuple[float, float]:
    """
    Weights of the current reduced overlap coordinate on the leg averages.

    Lambda_lat = sqrt(N1 N2 / N3) * (theta_av^(1) - theta_av^(2)).
    """
    n3 = n1 + n2
    scale = math.sqrt(n1 * n2 / n3)
    return (scale, -scale)


def reduced_lambda_zero_mode_substitution_weights(
    lambda_ratio: float,
) -> tuple[float, float]:
    """
    Current reduced-Lambda substitution used in the three-point ansatz.

    After the delta-function reduction lambda_3 = lambda_1 + lambda_2, the
    continuum-inspired PS combination is represented as

        Lambda -> -(1-lambda) lambda_1 + lambda lambda_2,

    with lambda = alpha_1 / alpha_3.
    """
    alpha_1 = float(lambda_ratio)
    alpha_2 = 1.0 - alpha_1
    return (-alpha_2, alpha_1)


def join_local_fermion_data(n1: int, n2: int) -> JoinLocalFermionData:
    """
    Exact finite-N local fermionic data at the unfolded cubic join.

    This helper only packages the raw local data and their decomposition in the
    real nonzero-mode basis. It does not yet choose a branch-point regulator or
    a preferred local lattice interaction-point fermion.
    """
    theta_i_plus = site_fermion_decomposition(n1, 0)
    theta_i_minus = site_fermion_decomposition(n2, 0)

    nabla_plus_selector = forward_arc_difference_selector(n1)
    nabla_minus_selector = backward_arc_difference_selector(n2)

    return JoinLocalFermionData(
        n1=n1,
        n2=n2,
        n3=n1 + n2,
        theta_i_plus=theta_i_plus,
        theta_i_minus=theta_i_minus,
        nabla_plus_selector=nabla_plus_selector,
        nabla_minus_selector=nabla_minus_selector,
        nabla_plus_oscillator_row=oscillator_projection_row(n1, nabla_plus_selector),
        nabla_minus_oscillator_row=oscillator_projection_row(n2, nabla_minus_selector),
        reduced_lambda_average_weights=reduced_lambda_average_weights(n1, n2),
    )


def local_join_summary(n1: int, n2: int) -> dict[str, object]:
    """Structured report for debugging and note-writing."""
    data = join_local_fermion_data(n1, n2)
    return {
        "n1": n1,
        "n2": n2,
        "n3": n1 + n2,
        "theta_i_plus_reconstruction_error": data.theta_i_plus.reconstruction_error,
        "theta_i_minus_reconstruction_error": data.theta_i_minus.reconstruction_error,
        "nabla_plus_average_sum": float(np.sum(data.nabla_plus_selector)),
        "nabla_minus_average_sum": float(np.sum(data.nabla_minus_selector)),
        "reduced_lambda_average_weights": list(data.reduced_lambda_average_weights),
        "theta_i_plus_oscillator_norm": float(
            np.linalg.norm(data.theta_i_plus.oscillator_row)
        ),
        "theta_i_minus_oscillator_norm": float(
            np.linalg.norm(data.theta_i_minus.oscillator_row)
        ),
        "nabla_plus_oscillator_norm": float(
            np.linalg.norm(data.nabla_plus_oscillator_row)
        ),
        "nabla_minus_oscillator_norm": float(
            np.linalg.norm(data.nabla_minus_oscillator_row)
        ),
    }
