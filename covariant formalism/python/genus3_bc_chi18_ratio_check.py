from __future__ import annotations

import argparse
import ast
import os
import sys
from itertools import combinations_with_replacement

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import genus3_t_duality as g3
import partition_function as pf
import riemann_surface_tools as rst


GENUS3_DEFAULT_TOPOLOGIES_FOR_FIT = (1, 541, 1000, 1726)
GENUS3_DEFAULT_SCALES = (200, 220, 260, 300)
GENUS3_DEFAULT_EDGE_LENGTHS_1 = (300,) * 15
GENUS3_DEFAULT_EDGE_LENGTHS_2 = (
    250,
    250,
    250,
    250,
    250,
    250,
    250,
    250,
    250,
    250,
    250,
    250,
    250,
    250,
    700,
)
GENUS3_DEFAULT_B_POINTS = (
    0.08 + 0.12j,
    -0.14 + 0.16j,
    0.19 - 0.09j,
    0.21 + 0.09j,
    -0.16 + 0.12j,
    0.05 - 0.18j,
)
GENUS3_DEFAULT_ANCHOR_B_POINTS = (
    0.12 + 0.08j,
    -0.11 + 0.17j,
    0.18 - 0.07j,
)
GENUS3_DEFAULT_ANCHOR_C_POINT = -0.19 + 0.13j
GENUS3_DEFAULT_DIVISOR_POINTS = (
    0.17 + 0.05j,
    -0.12 + 0.11j,
    0.11 - 0.09j,
)
GENUS3_DEFAULT_NMAX = 5


def _parse_complex_sequence(text: str, *, expected_len: int | None = None) -> tuple[np.complex128, ...]:
    values = ast.literal_eval(text)
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Expected a Python list/tuple literal, got {text!r}.")
    out = tuple(np.complex128(complex(value)) for value in values)
    if expected_len is not None and len(out) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(out)}.")
    return out


def _parse_int_sequence(text: str, *, expected_len: int | None = None) -> tuple[int, ...]:
    values = ast.literal_eval(text)
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Expected a Python list/tuple literal, got {text!r}.")
    out = tuple(int(value) for value in values)
    if expected_len is not None and len(out) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(out)}.")
    return out


def _ensure_disjoint_points(label_a: str, points_a, label_b: str, points_b, *, tol: float = 1e-12) -> None:
    for a in points_a:
        za = np.complex128(a)
        for b in points_b:
            zb = np.complex128(b)
            if abs(za - zb) < tol:
                raise ValueError(
                    f"{label_a} and {label_b} must be disjoint; found overlapping point {za!r}."
                )


def _stored_graph_to_ribbon_graph(graph_data):
    """Reconstruct a rotation system from the stored one-face boundary data."""
    edges_labeled = tuple(graph_data["edges"])
    boundary = tuple(graph_data["boundary"])
    edges = [(a, b) for _, a, b in edges_labeled]
    verts = sorted({v for _, a, b in edges_labeled for v in (a, b)})

    succ = {v: {} for v in verts}
    for i, (_, to_v, e_label) in enumerate(boundary):
        next_from, _, next_e = boundary[(i + 1) % len(boundary)]
        if next_from != to_v:
            raise ValueError(
                f"Boundary is not contiguous at segment {i + 1}: "
                f"{boundary[i]} followed by {boundary[(i + 1) % len(boundary)]}"
            )
        succ[to_v][e_label - 1] = next_e - 1

    rotation = {}
    for v in verts:
        incident = [idx for idx, (a, b) in enumerate(edges) if a == v or b == v]
        start = incident[0]
        order = [start]
        cur = start
        while True:
            nxt = succ[v][cur]
            if nxt == start:
                break
            order.append(nxt)
            cur = nxt
        rotation[v] = order

    return edges, verts, rotation


def _shared_genus3_large_l_coefficients():
    families = []
    for topology in GENUS3_DEFAULT_TOPOLOGIES_FOR_FIT:
        families.append(
            (
                _stored_graph_to_ribbon_graph(g3.get_stored_genus3_graph(topology)),
                (1,) * 15,
            )
        )
    families.append(
        (
            _stored_graph_to_ribbon_graph(g3.get_stored_genus3_graph(1)),
            (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2),
        )
    )
    return rst.fit_genus_universal_aprime_coefficients(
        families,
        scales=GENUS3_DEFAULT_SCALES,
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


def _surface_data_for_ratio_check(ribbon_graph, edge_lengths, *, gamma: float, alpha: float):
    renormalized_det_factor = _renormalized_det_factor(
        ribbon_graph,
        edge_lengths,
        gamma=gamma,
        alpha=alpha,
    )
    surface = rst.build_surface_from_ribbon_graph(ribbon_graph, edge_lengths)
    abs_zchiral_sq = rst.canonical_abs_zchiral_sq(
        surface,
        renormalized_det_factor=renormalized_det_factor,
    )
    abs_z1_sq = rst.canonical_abs_z1_sq(
        surface,
        renormalized_det_factor=renormalized_det_factor,
    )
    z1 = rst.canonical_chiral_z1(abs_z1_sq)
    return surface, renormalized_det_factor, abs_zchiral_sq, abs_z1_sq, z1


def _chi18_genus3(Omega, *, nmax: int) -> np.complex128:
    Omega = np.asarray(Omega, dtype=np.complex128)
    if Omega.shape != (3, 3):
        raise ValueError(f"chi_18 requires a genus-3 period matrix, got shape {Omega.shape}.")
    even_chars = rst.theta_characteristics(3, parity="even")
    if len(even_chars) != 36:
        raise RuntimeError(f"Expected 36 even genus-3 characteristics, got {len(even_chars)}.")

    zero = np.zeros(3, dtype=np.complex128)
    product = np.complex128(1.0)
    for characteristic in even_chars:
        product *= rst.riemann_theta(
            zero,
            Omega,
            characteristic=characteristic,
            nmax=nmax,
        )
    return np.complex128(product)


def _quadratic_monomial_matrix(surface, points) -> np.ndarray:
    if surface.genus != 3:
        raise ValueError(f"Expected genus 3 surface, got genus {surface.genus}.")
    points = tuple(np.complex128(point) for point in points)
    if len(points) != 6:
        raise ValueError(f"Need exactly 6 b-points for genus 3, got {len(points)}.")

    monomial_pairs = tuple(combinations_with_replacement(range(3), 2))
    rows = []
    for point in points:
        omegas = [
            rst._evaluate_one_form(surface.normalized_forms[idx], point)
            for idx in range(3)
        ]
        rows.append([omegas[i] * omegas[j] for i, j in monomial_pairs])
    return np.asarray(rows, dtype=np.complex128)


def _s_matrix_determinant_genus3(surface, b_points) -> np.complex128:
    matrix = _quadratic_monomial_matrix(surface, b_points)
    return np.complex128(np.linalg.det(matrix))


def _canonical_pure_b_correlator(
    surface,
    b_points,
    *,
    anchor_b_points,
    anchor_c_point,
    divisor_points,
    z1,
    nmax: int,
):
    raw = rst.bc_correlator(
        b_points,
        [],
        surface,
        lambda_weight=2.0,
        divisor_points=divisor_points,
        normalization_point=anchor_c_point,
        z1=z1,
        nmax=nmax,
    )
    sigma_scale = rst.sigma_scale_from_z1(
        anchor_b_points,
        anchor_c_point,
        surface,
        divisor_points=divisor_points,
        normalization_point=anchor_c_point,
        z1=z1,
        nmax=nmax,
    )
    weight = 2.0 * 2.0 - 1.0
    sigma_power = int(round(weight * len(b_points)))
    return np.complex128(raw * sigma_scale**sigma_power), np.complex128(sigma_scale)


def run_ratio_check(
    *,
    topology: int,
    edge_lengths_1,
    edge_lengths_2,
    b_points,
    anchor_b_points,
    anchor_c_point,
    divisor_points,
    nmax: int,
    verbose: bool = True,
):
    _ensure_disjoint_points("divisor_points", divisor_points, "b_points", b_points)
    _ensure_disjoint_points("divisor_points", divisor_points, "anchor_b_points", anchor_b_points)
    _ensure_disjoint_points("divisor_points", divisor_points, "anchor_c_point", (anchor_c_point,))

    if verbose:
        print("Fitting shared large-L genus-3 coefficients...", flush=True)
    ribbon_graph = _stored_graph_to_ribbon_graph(g3.get_stored_genus3_graph(int(topology)))
    shared = _shared_genus3_large_l_coefficients()

    if verbose:
        print("Building modulus 1 surface and determinant normalization...", flush=True)
    data_1 = _surface_data_for_ratio_check(
        ribbon_graph,
        edge_lengths_1,
        gamma=shared.gamma,
        alpha=shared.alpha,
    )
    if verbose:
        print("Building modulus 2 surface and determinant normalization...", flush=True)
    data_2 = _surface_data_for_ratio_check(
        ribbon_graph,
        edge_lengths_2,
        gamma=shared.gamma,
        alpha=shared.alpha,
    )

    surface_1, ren_1, abs_zchiral_sq_1, abs_z1_sq_1, z1_1 = data_1
    surface_2, ren_2, abs_zchiral_sq_2, abs_z1_sq_2, z1_2 = data_2

    if verbose:
        print("Evaluating canonically normalized genus-3 six-b correlators...", flush=True)
    corr_1, sigma_scale_1 = _canonical_pure_b_correlator(
        surface_1,
        b_points,
        anchor_b_points=anchor_b_points,
        anchor_c_point=anchor_c_point,
        divisor_points=divisor_points,
        z1=z1_1,
        nmax=nmax,
    )
    corr_2, sigma_scale_2 = _canonical_pure_b_correlator(
        surface_2,
        b_points,
        anchor_b_points=anchor_b_points,
        anchor_c_point=anchor_c_point,
        divisor_points=divisor_points,
        z1=z1_2,
        nmax=nmax,
    )

    if verbose:
        print("Evaluating det(S) and chi_18 on both moduli...", flush=True)
    sdet_1 = _s_matrix_determinant_genus3(surface_1, b_points)
    sdet_2 = _s_matrix_determinant_genus3(surface_2, b_points)
    chi18_1 = _chi18_genus3(surface_1.Omega, nmax=nmax)
    chi18_2 = _chi18_genus3(surface_2.Omega, nmax=nmax)

    lhs = (abs(corr_1) ** 2 / abs(corr_2) ** 2) * ((abs_zchiral_sq_1 / abs_zchiral_sq_2) ** 26)
    rhs = (abs(sdet_1) ** 2 / abs(sdet_2) ** 2) * (abs(chi18_2) / abs(chi18_1))

    return {
        "shared": shared,
        "surface_1": surface_1,
        "surface_2": surface_2,
        "ren_1": ren_1,
        "ren_2": ren_2,
        "abs_zchiral_sq_1": abs_zchiral_sq_1,
        "abs_zchiral_sq_2": abs_zchiral_sq_2,
        "abs_z1_sq_1": abs_z1_sq_1,
        "abs_z1_sq_2": abs_z1_sq_2,
        "z1_1": z1_1,
        "z1_2": z1_2,
        "corr_1": corr_1,
        "corr_2": corr_2,
        "sigma_scale_1": sigma_scale_1,
        "sigma_scale_2": sigma_scale_2,
        "sdet_1": sdet_1,
        "sdet_2": sdet_2,
        "chi18_1": chi18_1,
        "chi18_2": chi18_2,
        "lhs": float(lhs),
        "rhs": float(rhs),
        "relative_error": float(abs(lhs - rhs) / max(abs(lhs), abs(rhs))),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check the genus-3 squared bc / chi_18 relation at two moduli."
    )
    parser.add_argument(
        "--topology",
        type=int,
        default=1,
        help="Stored genus-3 topology index. Default: 1",
    )
    parser.add_argument(
        "--edge-lengths-1",
        default=repr(list(GENUS3_DEFAULT_EDGE_LENGTHS_1)),
        help="Python list/tuple literal with 15 edge lengths for modulus 1.",
    )
    parser.add_argument(
        "--edge-lengths-2",
        default=repr(list(GENUS3_DEFAULT_EDGE_LENGTHS_2)),
        help="Python list/tuple literal with 15 edge lengths for modulus 2.",
    )
    parser.add_argument(
        "--b-points",
        default=repr(list(GENUS3_DEFAULT_B_POINTS)),
        help="Python list/tuple literal with 6 complex b-insertion points.",
    )
    parser.add_argument(
        "--anchor-b-points",
        default=repr(list(GENUS3_DEFAULT_ANCHOR_B_POINTS)),
        help="Python list/tuple literal with 3 anchor b-points for sigma normalization.",
    )
    parser.add_argument(
        "--anchor-c-point",
        default=repr(GENUS3_DEFAULT_ANCHOR_C_POINT),
        help="Complex anchor c-point used in the lambda=1 sigma normalization.",
    )
    parser.add_argument(
        "--divisor-points",
        default=repr(list(GENUS3_DEFAULT_DIVISOR_POINTS)),
        help="Python list/tuple literal with 3 divisor points for sigma ratios.",
    )
    parser.add_argument(
        "--nmax",
        type=int,
        default=GENUS3_DEFAULT_NMAX,
        help="Theta truncation radius used in bc and chi_18 evaluations. Default: 5",
    )
    args = parser.parse_args()

    edge_lengths_1 = _parse_int_sequence(args.edge_lengths_1, expected_len=15)
    edge_lengths_2 = _parse_int_sequence(args.edge_lengths_2, expected_len=15)
    b_points = _parse_complex_sequence(args.b_points, expected_len=6)
    anchor_b_points = _parse_complex_sequence(args.anchor_b_points, expected_len=3)
    anchor_c_point = np.complex128(complex(ast.literal_eval(args.anchor_c_point)))
    divisor_points = _parse_complex_sequence(args.divisor_points, expected_len=3)

    result = run_ratio_check(
        topology=args.topology,
        edge_lengths_1=edge_lengths_1,
        edge_lengths_2=edge_lengths_2,
        b_points=b_points,
        anchor_b_points=anchor_b_points,
        anchor_c_point=anchor_c_point,
        divisor_points=divisor_points,
        nmax=args.nmax,
        verbose=True,
    )

    print("Genus-3 bc / chi_18 squared-ratio check")
    print()
    print("Shared large-L coefficients")
    print(f"  gamma = {result['shared'].gamma}")
    print(f"  alpha = {result['shared'].alpha}")
    print()
    print("Modulus 1")
    print(f"  edge lengths = {list(edge_lengths_1)}")
    print(f"  total L = {2 * sum(edge_lengths_1)}")
    print(f"  Omega =\n{result['surface_1'].Omega}")
    print(f"  renormalized det factor = {result['ren_1']}")
    print(f"  |Z_chiral|^2 = {result['abs_zchiral_sq_1']}")
    print(f"  |Z1|^2 = {result['abs_z1_sq_1']}")
    print(f"  sigma scale C = {result['sigma_scale_1']}")
    print(f"  canonical <b^6> = {result['corr_1']}")
    print(f"  det S = {result['sdet_1']}")
    print(f"  chi18 = {result['chi18_1']}")
    print()
    print("Modulus 2")
    print(f"  edge lengths = {list(edge_lengths_2)}")
    print(f"  total L = {2 * sum(edge_lengths_2)}")
    print(f"  Omega =\n{result['surface_2'].Omega}")
    print(f"  renormalized det factor = {result['ren_2']}")
    print(f"  |Z_chiral|^2 = {result['abs_zchiral_sq_2']}")
    print(f"  |Z1|^2 = {result['abs_z1_sq_2']}")
    print(f"  sigma scale C = {result['sigma_scale_2']}")
    print(f"  canonical <b^6> = {result['corr_2']}")
    print(f"  det S = {result['sdet_2']}")
    print(f"  chi18 = {result['chi18_2']}")
    print()
    print("Ratio comparison")
    print("  LHS uses |<b^6>|^2 * |Z_chiral|^52")
    print(f"  lhs = {result['lhs']}")
    print("  RHS uses |det S|^2 / |chi18|")
    print(f"  rhs = {result['rhs']}")
    print(f"  relative error = {result['relative_error']}")


if __name__ == "__main__":
    main()
