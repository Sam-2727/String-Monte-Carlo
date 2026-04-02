import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compact_partition as cp
import genus2_one_point as g2
import torus_one_point as t1


class TestGenus2OnePoint(unittest.TestCase):
    def test_random_genus1_comparison_helper(self):
        rows = g2.compare_against_torus_random_samples(
            n_samples=2,
            low=3,
            high=6,
            seed=20260401,
            R=1.2,
            winding_cutoff=4,
        )

        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertLess(row.abs_error, 1e-12)
            self.assertLess(row.rel_error, 1e-12)

    def test_graph_formula_reproduces_torus_special_case(self):
        L, l1, l2, R = 24, 3, 4, 1.3
        holonomy_cutoff = 6
        l3 = L // 2 - l1 - l2

        graph_value = g2.discretized_graph_disk_one_point(
            [l1, l2, l3],
            R,
            cp.GENUS1_F1_GRAPH_DATA,
            holonomy_cutoff=holonomy_cutoff,
        )
        torus_value = t1.discretized_disk_one_point(
            L,
            l1,
            l2,
            R,
            n_winding=holonomy_cutoff,
        )

        self.assertAlmostEqual(graph_value.real, torus_value.real, places=12)
        self.assertAlmostEqual(graph_value.imag, torus_value.imag, places=12)

    def test_genus2_one_point_returns_period_data(self):
        result = g2.compute_genus2_one_point(
            edge_lengths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
            R=1.1,
            topology=1,
            holonomy_cutoff=3,
            include_period_data=True,
        )

        self.assertEqual(result.holonomy_dim, 4)
        self.assertEqual(result.T_reduced.shape, (4, 4))
        self.assertEqual(result.Xi11_reduced.shape, (4, 4))
        self.assertTrue(np.isfinite(result.theta_sum))
        self.assertGreater(result.theta_sum, 0.0)
        self.assertIsNotNone(result.period_data)
        self.assertEqual(result.period_data.Omega.shape, (2, 2))
        eigvals = np.linalg.eigvalsh(np.imag(result.period_data.Omega))
        self.assertTrue(np.all(eigvals > 0.0))


if __name__ == "__main__":
    unittest.main()
