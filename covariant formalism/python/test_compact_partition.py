import unittest
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compact_partition as cp
import ell_to_tau as elt
import genus3_t_duality as g3


def _incidence_matrix(oriented_edges):
    vertices = sorted({v for edge in oriented_edges for v in edge})
    vidx = {v: i for i, v in enumerate(vertices)}
    mat = np.zeros((len(vertices), len(oriented_edges)), dtype=int)
    for edge_idx, (src, dst) in enumerate(oriented_edges):
        mat[vidx[src], edge_idx] = 1
        mat[vidx[dst], edge_idx] = -1
    return mat


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


def _analytic_compact_lattice_sum(Omega: np.ndarray, R: float, N: int) -> float:
    """
    Truncated analytic compact-boson lattice sum for a period matrix Omega.

    This is the radius-dependent classical sum
      sum_{m,n in Z^g} exp(-pi R^2 (m+Omega n)^T (Im Omega)^(-1) (m+bar(Omega) n))
    with each component of m and n truncated to [-N, N].
    """
    Omega = np.asarray(Omega, dtype=np.complex128)
    if Omega.ndim == 0:
        Omega = Omega.reshape(1, 1)

    g = Omega.shape[0]
    side = np.arange(-N, N + 1, dtype=float)
    grids = np.meshgrid(*([side] * g), indexing="ij")
    pts = np.stack(grids, axis=-1).reshape(-1, g)
    inv_im = np.linalg.inv(Omega.imag)

    total = 0.0
    for n in pts:
        z = pts + n @ Omega.T
        quad = np.einsum("ni,ij,nj->n", z.conj(), inv_im, z, optimize=True).real
        total += float(np.sum(np.exp(-np.pi * (R ** 2) * quad)))
    return total


def _check_higher_genus_ratio(graph_data, edge_lengths, R1, R2, *,
                              theta_cutoff, analytic_cutoff):
    """
    Compare the compact-partition theta ratio against the analytic lattice-sum ratio.

    The analytic expression in the TeX notes already isolates the radius-dependent
    classical lattice sum, so the comparison is between the reduced theta sums
    themselves rather than the full graph partition function with its explicit
    lattice zero-mode factor.
    """
    ribbon_graph = _stored_graph_to_ribbon_graph(graph_data)
    forms = elt.make_cyl_eqn_improved_higher_genus(ribbon_graph, edge_lengths)
    Omega = elt.period_matrix(forms=forms, ribbon_graph=ribbon_graph, ell_list=edge_lengths)

    geom = cp.compact_boson_graph_geometry(edge_lengths, graph_data)
    ratio_lattice = (
        cp.theta_sum_reduced(geom["T_reduced"], R1, N=theta_cutoff)
        / cp.theta_sum_reduced(geom["T_reduced"], R2, N=theta_cutoff)
    )
    ratio_analytic = (
        _analytic_compact_lattice_sum(Omega, R1, analytic_cutoff)
        / _analytic_compact_lattice_sum(Omega, R2, analytic_cutoff)
    )
    return ratio_lattice, ratio_analytic


class TestCompactPartitionHigherGenus(unittest.TestCase):
    def test_graph_path_reproduces_torus_special_case(self):
        L, l1, l2, R = 24, 3, 4, 1.3
        l3 = L // 2 - l1 - l2

        Mat = cp.direct_mat_n_fast(L)
        Aprime = cp.direct_red_traced_mat(L, l1, l2, Mat)
        W = cp.mat_w(L, l1, l2, Mat)
        T1 = cp.mat_t_first_part(L, l1, l2, Mat)
        T2 = cp.mat_t_second_part(L, l1, l2, W, Aprime)
        Tp = cp.mat_t_prime(cp.symm(T1 - T2))
        z_torus = cp.partition_function_z(Aprime, R, Tp, N=6)

        z_graph = cp.compute_graph_compact_partition(
            [l1, l2, l3],
            R,
            cp.GENUS1_F1_GRAPH_DATA,
            N=6,
        )["Z"]

        self.assertAlmostEqual(z_graph, z_torus, places=12)

    def test_genus2_topologies_have_four_holonomies(self):
        edge_lengths = [1, 2, 2, 1, 2, 2, 1, 2, 2]
        values = []

        for topology in range(1, 5):
            graph_data = cp.get_stored_genus2_graph(topology)
            geom = cp.compute_graph_compact_partition(edge_lengths, 1.2, graph_data, N=3)
            incidence = _incidence_matrix(cp._gluing_data(edge_lengths, graph_data)["oriented_edges"])

            self.assertEqual(geom["A_prime"].shape, (14, 14))
            self.assertEqual(geom["T_edge"].shape, (9, 9))
            self.assertEqual(geom["cycle_basis"].shape, (9, 4))
            self.assertEqual(geom["T_reduced"].shape, (4, 4))
            self.assertEqual(np.linalg.matrix_rank(geom["cycle_basis"]), 4)
            np.testing.assert_array_equal(incidence @ geom["cycle_basis"], 0)
            self.assertTrue(np.isfinite(geom["Z"]))
            self.assertGreater(geom["Z"], 0.0)
            values.append(geom["Z"])

        self.assertGreater(max(values) - min(values), 1e-6)

    def test_genus2_theta_ratio_matches_analytic_period_matrix_sum(self):
        ratio_lattice, ratio_analytic = _check_higher_genus_ratio(
            cp.get_stored_genus2_graph(1),
            [40, 42, 44, 46, 48, 50, 52, 54, 56],
            1.2,
            1.6,
            theta_cutoff=4,
            analytic_cutoff=3,
        )
        rel = abs(ratio_lattice - ratio_analytic) / abs(ratio_analytic)
        self.assertLess(rel, 1.0e-3)

    def test_genus3_theta_ratio_matches_analytic_period_matrix_sum(self):
        ratio_lattice, ratio_analytic = _check_higher_genus_ratio(
            g3.get_stored_genus3_graph(1),
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
            1.2,
            1.6,
            theta_cutoff=3,
            analytic_cutoff=2,
        )
        rel = abs(ratio_lattice - ratio_analytic) / abs(ratio_analytic)
        self.assertLess(rel, 1.0e-3)


if __name__ == "__main__":
    unittest.main()
