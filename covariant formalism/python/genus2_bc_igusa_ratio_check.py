from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compact_partition as cp
import ell_to_tau as elt
import partition_function as pf
import riemann_surface_tools as rst
from genus2_one_point import _stored_graph_to_ribbon_graph


def _shared_genus2_large_l_coefficients():
    families = []
    for topology in (1, 2, 3):
        families.append(
            (
                _stored_graph_to_ribbon_graph(cp.get_stored_genus2_graph(topology)),
                (1, 1, 1, 1, 1, 1, 1, 1, 1),
            )
        )
    families.append(
        (
            _stored_graph_to_ribbon_graph(cp.get_stored_genus2_graph(1)),
            (1, 1, 1, 1, 1, 1, 1, 1, 2),
        )
    )
    return rst.fit_genus_universal_aprime_coefficients(
        families,
        scales=(200, 220, 260, 300),
    )


def _renormalized_det_factor(ribbon_graph, edge_lengths, *, gamma: float, alpha: float) -> float:
    A_prime = np.asarray(
        pf.traced_matter_matrix_f1(ribbon_graph, edge_lengths),
        dtype=np.float64,
    )
    A_prime = 0.5 * (A_prime + A_prime.T)
    sign, logdet = np.linalg.slogdet(A_prime)
    if sign <= 0:
        raise RuntimeError(f"det A' is not positive for edge lengths {edge_lengths}.")
    total_length = 2 * sum(edge_lengths)
    return rst.renormalized_aprime_factor_from_raw_log_values(
        [total_length],
        [-0.5 * float(logdet)],
        gamma=gamma,
        alpha=alpha,
    )


def _canonical_bbb_correlator(
    surface,
    b_points,
    *,
    anchor_b_points,
    anchor_c_point,
    divisor_points,
    z1,
    nmax: int = 12,
):
    Delta = rst.riemann_constant_vector(surface)
    zeta_b = np.sum(
        np.asarray([rst.abel_map(point, surface) for point in b_points], dtype=np.complex128),
        axis=0,
    )
    theta_arg = zeta_b - 3.0 * Delta
    theta_val = rst.riemann_theta(theta_arg, surface.Omega, nmax=nmax)

    prime_bb = np.complex128(1.0)
    for idx, zi in enumerate(b_points):
        for zj in b_points[idx + 1 :]:
            prime_bb *= rst.prime_form(zi, zj, surface, nmax=nmax)

    sigma_b = np.complex128(1.0)
    for zi in b_points:
        sigma_b *= rst.canonical_sigma_value(
            zi,
            surface,
            anchor_b_points=anchor_b_points,
            anchor_c_point=anchor_c_point,
            divisor_points=divisor_points,
            normalization_point=anchor_c_point,
            z1=z1,
            Delta=Delta,
            nmax=nmax,
        ) ** 3

    return np.complex128(theta_val * prime_bb * sigma_b / np.sqrt(z1))


def _direct_bbb_component_data(surface, b_points, *, z1, nmax: int = 12):
    Delta = rst.riemann_constant_vector(surface)
    sigma_vals = rst.genus2_sigma_values_from_lambda_one(
        b_points,
        surface,
        z1=z1,
        Delta=Delta,
        nmax=nmax,
    )
    sigma_prod = np.prod(np.asarray(sigma_vals, dtype=np.complex128))
    zeta_b = np.sum(
        np.asarray([rst.abel_map(point, surface) for point in b_points], dtype=np.complex128),
        axis=0,
    )
    theta_arg = zeta_b - 3.0 * Delta
    theta_val = rst.riemann_theta(theta_arg, surface.Omega, nmax=nmax)
    prime_bb = np.complex128(1.0)
    for idx, zi in enumerate(b_points):
        for zj in b_points[idx + 1 :]:
            prime_bb *= rst.prime_form(zi, zj, surface, nmax=nmax)
    corr = np.complex128(theta_val * prime_bb * sigma_prod**3 / np.sqrt(z1))
    return {
        "theta": theta_val,
        "prime_bb": prime_bb,
        "sigma_vals": sigma_vals,
        "sigma_prod": sigma_prod,
        "corr": corr,
    }


def _s_matrix_determinant(surface, b_points):
    rows = []
    for point in b_points:
        omega_1 = rst._evaluate_one_form(surface.normalized_forms[0], point)
        omega_2 = rst._evaluate_one_form(surface.normalized_forms[1], point)
        rows.append([omega_1**2, omega_2**2, omega_1 * omega_2])
    matrix = np.asarray(rows, dtype=np.complex128)
    return np.complex128(np.linalg.det(matrix))


def _surface_data_for_ratio_check(ribbon_graph, edge_lengths, *, gamma: float, alpha: float):
    renormalized_det_factor = _renormalized_det_factor(
        ribbon_graph,
        edge_lengths,
        gamma=gamma,
        alpha=alpha,
    )
    surface = rst.build_surface_from_ribbon_graph(ribbon_graph, edge_lengths)
    abs_z1_sq = rst.canonical_abs_z1_sq(
        surface,
        renormalized_det_factor=renormalized_det_factor,
    )
    z1 = rst.canonical_chiral_z1(abs_z1_sq)
    return surface, renormalized_det_factor, abs_z1_sq, z1


def main():
    shared = _shared_genus2_large_l_coefficients()

    ribbon_graph = _stored_graph_to_ribbon_graph(cp.get_stored_genus2_graph(1))
    edge_lengths_1 = [300] * 9
    edge_lengths_2 = [250, 250, 250, 250, 250, 250, 250, 250, 700]

    b_points = [0.08 + 0.12j, -0.14 + 0.16j, 0.19 - 0.09j]
    anchor_b_points = [0.12 + 0.08j, -0.11 + 0.17j]
    anchor_c_point = -0.19 + 0.13j
    divisor_points = [0.21 + 0.09j, -0.16 + 0.12j]

    data_1 = _surface_data_for_ratio_check(
        ribbon_graph,
        edge_lengths_1,
        gamma=shared.gamma,
        alpha=shared.alpha,
    )
    data_2 = _surface_data_for_ratio_check(
        ribbon_graph,
        edge_lengths_2,
        gamma=shared.gamma,
        alpha=shared.alpha,
    )

    surface_1, ren_1, abs_z1_sq_1, z1_1 = data_1
    surface_2, ren_2, abs_z1_sq_2, z1_2 = data_2

    pieces_1 = _direct_bbb_component_data(
        surface_1,
        b_points,
        z1=z1_1,
    )
    pieces_2 = _direct_bbb_component_data(
        surface_2,
        b_points,
        z1=z1_2,
    )
    corr_1 = pieces_1["corr"]
    corr_2 = pieces_2["corr"]
    old_corr_1 = _canonical_bbb_correlator(
        surface_1,
        b_points,
        anchor_b_points=anchor_b_points,
        anchor_c_point=anchor_c_point,
        divisor_points=divisor_points,
        z1=z1_1,
    )
    old_corr_2 = _canonical_bbb_correlator(
        surface_2,
        b_points,
        anchor_b_points=anchor_b_points,
        anchor_c_point=anchor_c_point,
        divisor_points=divisor_points,
        z1=z1_2,
    )

    sdet_1 = _s_matrix_determinant(surface_1, b_points)
    sdet_2 = _s_matrix_determinant(surface_2, b_points)
    chi10_1 = elt.igusa_chi10_genus2(surface_1.Omega, nmax=12, normalization="product")
    chi10_2 = elt.igusa_chi10_genus2(surface_2.Omega, nmax=12, normalization="product")

    lhs = (abs(corr_1) ** 2 / abs(corr_2) ** 2) * ((abs_z1_sq_1 / abs_z1_sq_2) ** 26)
    rhs = (abs(sdet_1) ** 2 / abs(sdet_2) ** 2) * (abs(chi10_2) ** 2 / abs(chi10_1) ** 2)

    print("Genus-2 bc / Igusa ratio check")
    print()
    print("Shared large-L coefficients")
    print(f"  gamma = {shared.gamma}")
    print(f"  alpha = {shared.alpha}")
    print()
    print("Modulus 1")
    print(f"  edge lengths = {edge_lengths_1}")
    print(f"  total L = {2 * sum(edge_lengths_1)}")
    print(f"  Omega =\n{surface_1.Omega}")
    print(f"  renormalized det factor = {ren_1}")
    print(f"  |Z1|^2 = {abs_z1_sq_1}")
    print(f"  direct <bbb> = {corr_1}")
    print(f"  old anchor-based <bbb> = {old_corr_1}")
    print(f"  |theta| = {abs(pieces_1['theta'])}")
    print(f"  |prime_bb| = {abs(pieces_1['prime_bb'])}")
    print(f"  sigma values = {pieces_1['sigma_vals']}")
    print(f"  |sigma_1 sigma_2 sigma_3| = {abs(pieces_1['sigma_prod'])}")
    print(f"  det S = {sdet_1}")
    print(f"  chi10 = {chi10_1}")
    print()
    print("Modulus 2")
    print(f"  edge lengths = {edge_lengths_2}")
    print(f"  total L = {2 * sum(edge_lengths_2)}")
    print(f"  Omega =\n{surface_2.Omega}")
    print(f"  renormalized det factor = {ren_2}")
    print(f"  |Z1|^2 = {abs_z1_sq_2}")
    print(f"  direct <bbb> = {corr_2}")
    print(f"  old anchor-based <bbb> = {old_corr_2}")
    print(f"  |theta| = {abs(pieces_2['theta'])}")
    print(f"  |prime_bb| = {abs(pieces_2['prime_bb'])}")
    print(f"  sigma values = {pieces_2['sigma_vals']}")
    print(f"  |sigma_1 sigma_2 sigma_3| = {abs(pieces_2['sigma_prod'])}")
    print(f"  det S = {sdet_2}")
    print(f"  chi10 = {chi10_2}")
    print()
    print("Ratio comparison")
    print(f"  LHS = {lhs}")
    print(f"  RHS = {rhs}")
    print(f"  LHS / RHS = {lhs / rhs}")
    print(f"  relative difference = {abs(lhs - rhs) / abs(rhs)}")


if __name__ == "__main__":
    main()
