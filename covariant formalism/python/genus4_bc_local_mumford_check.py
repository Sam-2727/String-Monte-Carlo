from __future__ import annotations

import argparse
import ast
import os
import sys
from itertools import combinations_with_replacement

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import genus4_single_topology_t_duality as g4
import riemann_surface_tools as rst


GENUS4_DEFAULT_EDGE_LENGTHS = (24,) * 21
GENUS4_DEFAULT_B_POINTS_1 = (
    0.08 + 0.12j,
    -0.14 + 0.16j,
    0.19 - 0.09j,
    0.21 + 0.09j,
    -0.16 + 0.12j,
    0.05 - 0.18j,
    -0.07 - 0.11j,
    0.13 + 0.04j,
    -0.09 + 0.07j,
)
GENUS4_DEFAULT_B_POINTS_2 = (
    -0.03 + 0.14j,
    0.11 + 0.16j,
    -0.18 + 0.05j,
    0.17 - 0.12j,
    -0.11 - 0.15j,
    0.06 + 0.19j,
    0.14 - 0.03j,
    -0.05 - 0.10j,
    0.02 + 0.07j,
)
GENUS4_DEFAULT_DIVISOR_POINTS = (
    0.17 + 0.05j,
    -0.12 + 0.11j,
    0.11 - 0.09j,
    -0.15 - 0.06j,
)
GENUS4_DEFAULT_ALT_DIVISOR_POINTS = (
    0.15 + 0.02j,
    -0.10 + 0.14j,
    0.09 - 0.12j,
    -0.13 - 0.03j,
)
GENUS4_DEFAULT_ANCHOR_POINT = -0.19 + 0.13j
GENUS4_DEFAULT_SAMPLE_POINTS = (
    0.02 + 0.03j,
    -0.04 + 0.11j,
    0.09 - 0.05j,
    0.16 + 0.02j,
    -0.12 + 0.06j,
    0.06 - 0.14j,
    -0.09 - 0.08j,
    0.18 - 0.11j,
    -0.17 + 0.03j,
    0.11 + 0.15j,
    -0.02 - 0.17j,
    0.14 + 0.09j,
)
GENUS4_DEFAULT_FORM_IDX = 1
GENUS4_DEFAULT_ZERO_RADIUS = 0.99
GENUS4_DEFAULT_NMAX = 4


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


def _genus4_ribbon_graph(topology: int):
    graph_data, topology_count = g4.current_genus4_graph_data(int(topology))
    return _stored_graph_to_ribbon_graph(graph_data), topology_count


def _canonical_delta_with_fallback(
    surface,
    *,
    preferred_form_idx: int,
    zero_radius: float,
    nmax: int,
):
    attempted: list[tuple[int, str]] = []
    candidate_indices = [int(preferred_form_idx)] + [
        idx for idx in range(surface.genus) if idx != int(preferred_form_idx)
    ]
    for form_idx in candidate_indices:
        try:
            Delta = rst.riemann_constant_vector_canonical(
                surface,
                form_idx=form_idx,
                zero_radius=zero_radius,
                nmax=nmax,
            )
            return np.asarray(Delta, dtype=np.complex128), int(form_idx), tuple(attempted)
        except Exception as exc:  # pragma: no cover - diagnostic path
            attempted.append((int(form_idx), str(exc)))
    details = "\n".join(f"  form_idx={idx}: {message}" for idx, message in attempted)
    raise RuntimeError("Could not construct a canonical genus-4 Delta.\n" + details)


def _quadratic_monomial_pairs_genus4():
    return tuple(combinations_with_replacement(range(4), 2))


def _quadratic_monomial_labels_genus4():
    return (
        "w1^2",
        "w1 w2",
        "w1 w3",
        "w1 w4",
        "w2^2",
        "w2 w3",
        "w2 w4",
        "w3^2",
        "w3 w4",
        "w4^2",
    )


def _quadratic_monomial_matrix(surface, points) -> np.ndarray:
    if surface.genus != 4:
        raise ValueError(f"Expected genus 4 surface, got genus {surface.genus}.")
    points = tuple(np.complex128(point) for point in points)
    monomial_pairs = _quadratic_monomial_pairs_genus4()
    rows = []
    for point in points:
        omegas = [
            rst._evaluate_one_form(surface.normalized_forms[idx], point)
            for idx in range(4)
        ]
        rows.append([omegas[i] * omegas[j] for i, j in monomial_pairs])
    return np.asarray(rows, dtype=np.complex128)


def _quadric_coefficients_from_sample_points(surface, sample_points) -> dict:
    matrix = _quadratic_monomial_matrix(surface, sample_points)
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    coeffs = np.asarray(vh[-1, :], dtype=np.complex128)
    residual = matrix @ coeffs
    relative_residual = float(
        np.linalg.norm(residual) / max(np.linalg.norm(matrix) * np.linalg.norm(coeffs), 1e-30)
    )
    return {
        "matrix": matrix,
        "coeffs": coeffs,
        "smallest_singular_value": float(singular_values[-1]),
        "relative_residual": relative_residual,
    }


def _local_chart_ratio(surface, b_points_1, b_points_2, coeffs, *, coeff_tol: float = 1e-10) -> dict:
    matrix_1 = _quadratic_monomial_matrix(surface, b_points_1)
    matrix_2 = _quadratic_monomial_matrix(surface, b_points_2)
    coeffs = np.asarray(coeffs, dtype=np.complex128)

    ratios = []
    for omitted_idx, coeff in enumerate(coeffs):
        if abs(coeff) < coeff_tol:
            continue
        cols = [col for col in range(10) if col != omitted_idx]
        det_1 = np.complex128(np.linalg.det(matrix_1[:, cols]))
        det_2 = np.complex128(np.linalg.det(matrix_2[:, cols]))
        if abs(det_2) == 0.0:
            continue
        ratios.append(
            {
                "omitted_idx": int(omitted_idx),
                "coeff": coeff,
                "det_1": det_1,
                "det_2": det_2,
                "ratio_mod_sq": float(abs(det_1 / det_2) ** 2),
            }
        )

    if not ratios:
        raise RuntimeError("No usable genus-4 quadric chart had nonzero coefficient and determinant.")

    chosen = max(ratios, key=lambda item: abs(item["coeff"]))
    values = [item["ratio_mod_sq"] for item in ratios]
    spread = float(max(values) / min(values) - 1.0) if min(values) > 0.0 else float("inf")
    return {
        "chosen": chosen,
        "all": tuple(ratios),
        "chart_spread": spread,
    }


def run_local_check(
    *,
    topology: int,
    edge_lengths,
    b_points_1,
    b_points_2,
    divisor_points,
    anchor_point,
    sample_points,
    preferred_form_idx: int,
    zero_radius: float,
    nmax: int,
    alt_divisor_points=None,
    verbose: bool = True,
):
    _ensure_disjoint_points("divisor_points", divisor_points, "b_points_1", b_points_1)
    _ensure_disjoint_points("divisor_points", divisor_points, "b_points_2", b_points_2)
    _ensure_disjoint_points("divisor_points", divisor_points, "anchor_point", (anchor_point,))
    if alt_divisor_points is not None:
        _ensure_disjoint_points("alt_divisor_points", alt_divisor_points, "b_points_1", b_points_1)
        _ensure_disjoint_points("alt_divisor_points", alt_divisor_points, "b_points_2", b_points_2)
        _ensure_disjoint_points("alt_divisor_points", alt_divisor_points, "anchor_point", (anchor_point,))

    if verbose:
        print("Generating genus-4 topology with Fast_Ribbon_Generator data...", flush=True)
    ribbon_graph, topology_count = _genus4_ribbon_graph(int(topology))

    if verbose:
        print("Building genus-4 Riemann surface...", flush=True)
    surface = rst.build_surface_from_ribbon_graph(ribbon_graph, edge_lengths)

    if verbose:
        print("Constructing canonical Delta with genus-4 fallback over one-form choices...", flush=True)
    Delta, used_form_idx, delta_failures = _canonical_delta_with_fallback(
        surface,
        preferred_form_idx=preferred_form_idx,
        zero_radius=zero_radius,
        nmax=nmax,
    )

    if verbose:
        print("Evaluating two nine-b ghost correlators on the same surface...", flush=True)
    corr_1 = rst.bc_correlator(
        b_points_1,
        [],
        surface,
        lambda_weight=2.0,
        divisor_points=divisor_points,
        normalization_point=anchor_point,
        Delta=Delta,
        nmax=nmax,
    )
    corr_2 = rst.bc_correlator(
        b_points_2,
        [],
        surface,
        lambda_weight=2.0,
        divisor_points=divisor_points,
        normalization_point=anchor_point,
        Delta=Delta,
        nmax=nmax,
    )
    ghost_ratio = float(abs(corr_1) ** 2 / abs(corr_2) ** 2)

    alt_ghost_ratio = None
    alt_divisor_relative_shift = None
    if alt_divisor_points is not None:
        if verbose:
            print("Repeating the ghost ratio with an alternate divisor as a stability diagnostic...", flush=True)
        alt_corr_1 = rst.bc_correlator(
            b_points_1,
            [],
            surface,
            lambda_weight=2.0,
            divisor_points=alt_divisor_points,
            normalization_point=anchor_point,
            Delta=Delta,
            nmax=nmax,
        )
        alt_corr_2 = rst.bc_correlator(
            b_points_2,
            [],
            surface,
            lambda_weight=2.0,
            divisor_points=alt_divisor_points,
            normalization_point=anchor_point,
            Delta=Delta,
            nmax=nmax,
        )
        alt_ghost_ratio = float(abs(alt_corr_1) ** 2 / abs(alt_corr_2) ** 2)
        alt_divisor_relative_shift = float(
            abs(alt_ghost_ratio - ghost_ratio) / max(abs(alt_ghost_ratio), abs(ghost_ratio))
        )

    if verbose:
        print("Extracting the genus-4 quadric relation and local Mumford chart ratios...", flush=True)
    quadric = _quadric_coefficients_from_sample_points(surface, sample_points)
    local_chart = _local_chart_ratio(
        surface,
        b_points_1,
        b_points_2,
        quadric["coeffs"],
    )

    local_ratio = float(local_chart["chosen"]["ratio_mod_sq"])
    relative_error = float(abs(ghost_ratio - local_ratio) / max(abs(ghost_ratio), abs(local_ratio)))

    return {
        "topology_count": int(topology_count),
        "surface": surface,
        "Delta": Delta,
        "used_form_idx": int(used_form_idx),
        "delta_failures": delta_failures,
        "corr_1": corr_1,
        "corr_2": corr_2,
        "ghost_ratio": ghost_ratio,
        "alt_ghost_ratio": alt_ghost_ratio,
        "alt_divisor_relative_shift": alt_divisor_relative_shift,
        "quadric": quadric,
        "local_chart": local_chart,
        "local_ratio": local_ratio,
        "relative_error": relative_error,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check the genus-4 local Mumford-chart ratio against a same-surface "
            "nine-b ghost-correlator ratio."
        )
    )
    parser.add_argument(
        "--topology",
        type=int,
        default=1,
        help="Topology index in the current genus-4 Fast_Ribbon_Generator output. Default: 1",
    )
    parser.add_argument(
        "--edge-lengths",
        default=repr(list(GENUS4_DEFAULT_EDGE_LENGTHS)),
        help="Python list/tuple literal with 21 genus-4 edge lengths.",
    )
    parser.add_argument(
        "--b-points-1",
        default=repr(list(GENUS4_DEFAULT_B_POINTS_1)),
        help="Python list/tuple literal with 9 complex b-insertion points for configuration 1.",
    )
    parser.add_argument(
        "--b-points-2",
        default=repr(list(GENUS4_DEFAULT_B_POINTS_2)),
        help="Python list/tuple literal with 9 complex b-insertion points for configuration 2.",
    )
    parser.add_argument(
        "--divisor-points",
        default=repr(list(GENUS4_DEFAULT_DIVISOR_POINTS)),
        help="Python list/tuple literal with 4 divisor points used for the main ghost ratio.",
    )
    parser.add_argument(
        "--alt-divisor-points",
        default=repr(list(GENUS4_DEFAULT_ALT_DIVISOR_POINTS)),
        help=(
            "Optional Python list/tuple literal with 4 alternate divisor points used only "
            "for a stability diagnostic. Pass 'None' to disable."
        ),
    )
    parser.add_argument(
        "--anchor-point",
        default=repr(GENUS4_DEFAULT_ANCHOR_POINT),
        help="Complex normalization point supplied to sigma_value inside bc_correlator.",
    )
    parser.add_argument(
        "--sample-points",
        default=repr(list(GENUS4_DEFAULT_SAMPLE_POINTS)),
        help="Python list/tuple literal with sample points used to extract the genus-4 quadric relation.",
    )
    parser.add_argument(
        "--form-idx",
        type=int,
        default=GENUS4_DEFAULT_FORM_IDX,
        help="Preferred one-form index for the canonical-Delta construction. Default: 1",
    )
    parser.add_argument(
        "--zero-radius",
        type=float,
        default=GENUS4_DEFAULT_ZERO_RADIUS,
        help="Root-radius cutoff passed to riemann_constant_vector_canonical. Default: 0.99",
    )
    parser.add_argument(
        "--nmax",
        type=int,
        default=GENUS4_DEFAULT_NMAX,
        help="Theta truncation radius used in the ghost correlators and Delta filter. Default: 4",
    )
    args = parser.parse_args()

    edge_lengths = _parse_int_sequence(args.edge_lengths, expected_len=21)
    b_points_1 = _parse_complex_sequence(args.b_points_1, expected_len=9)
    b_points_2 = _parse_complex_sequence(args.b_points_2, expected_len=9)
    divisor_points = _parse_complex_sequence(args.divisor_points, expected_len=4)
    anchor_point = np.complex128(complex(ast.literal_eval(args.anchor_point)))
    sample_points = _parse_complex_sequence(args.sample_points)
    alt_divisor_points = None
    if args.alt_divisor_points.strip() != "None":
        alt_divisor_points = _parse_complex_sequence(args.alt_divisor_points, expected_len=4)

    result = run_local_check(
        topology=args.topology,
        edge_lengths=edge_lengths,
        b_points_1=b_points_1,
        b_points_2=b_points_2,
        divisor_points=divisor_points,
        anchor_point=anchor_point,
        sample_points=sample_points,
        preferred_form_idx=args.form_idx,
        zero_radius=args.zero_radius,
        nmax=args.nmax,
        alt_divisor_points=alt_divisor_points,
        verbose=True,
    )

    labels = _quadratic_monomial_labels_genus4()
    chosen = result["local_chart"]["chosen"]

    print("Genus-4 local Mumford / ghost same-surface ratio check")
    print()
    print("Surface data")
    print(f"  generator topology count = {result['topology_count']}")
    print(f"  topology index = {args.topology}")
    print(f"  edge lengths = {list(edge_lengths)}")
    print(f"  Omega =\n{result['surface'].Omega}")
    print(f"  Delta form index used = {result['used_form_idx']}")
    if result["delta_failures"]:
        print("  earlier Delta attempts:")
        for idx, message in result["delta_failures"]:
            print(f"    form_idx={idx}: {message}")
    print()
    print("Ghost correlators")
    print(f"  <b^9> configuration 1 = {result['corr_1']}")
    print(f"  <b^9> configuration 2 = {result['corr_2']}")
    print(f"  |<b^9>_1|^2 / |<b^9>_2|^2 = {result['ghost_ratio']}")
    if result["alt_ghost_ratio"] is not None:
        print(f"  alternate-divisor ghost ratio = {result['alt_ghost_ratio']}")
        print(f"  alternate-divisor relative shift = {result['alt_divisor_relative_shift']}")
    print()
    print("Genus-4 quadric data")
    print(f"  smallest singular value = {result['quadric']['smallest_singular_value']}")
    print(f"  relative quadric residual = {result['quadric']['relative_residual']}")
    print(f"  chosen omission chart = {chosen['omitted_idx']} ({labels[chosen['omitted_idx']]})")
    print(f"  chosen chart coefficient = {chosen['coeff']}")
    print(f"  local chart ratio |M_1|^2 / |M_2|^2 = {result['local_ratio']}")
    print(f"  omission-chart spread = {result['local_chart']['chart_spread']}")
    print()
    print("Comparison")
    print("  Expected same-surface relation:")
    print("    |<b^9>_1|^2 / |<b^9>_2|^2 ?= |M_1|^2 / |M_2|^2")
    print(f"  lhs = {result['ghost_ratio']}")
    print(f"  rhs = {result['local_ratio']}")
    print(f"  relative error = {result['relative_error']}")


if __name__ == "__main__":
    main()
