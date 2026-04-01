import unittest
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compact_partition as cp


def _incidence_matrix(oriented_edges):
    vertices = sorted({v for edge in oriented_edges for v in edge})
    vidx = {v: i for i, v in enumerate(vertices)}
    mat = np.zeros((len(vertices), len(oriented_edges)), dtype=int)
    for edge_idx, (src, dst) in enumerate(oriented_edges):
        mat[vidx[src], edge_idx] = 1
        mat[vidx[dst], edge_idx] = -1
    return mat


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


if __name__ == "__main__":
    unittest.main()
