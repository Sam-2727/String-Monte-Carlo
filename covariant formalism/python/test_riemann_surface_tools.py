import os
import sys
import unittest

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import riemann_surface_tools as rst


def _cycle_period(surface, cycle_terms, form_idx: int = 0) -> np.complex128:
    total = np.complex128(0.0)
    F = surface.antiderivatives[form_idx]
    for edge_idx, coeff in cycle_terms:
        z0, z1 = surface.edge_midpoints[int(edge_idx)]
        total += np.complex128(coeff) * (F(z1) - F(z0))
    return np.complex128(total)


class TestRiemannSurfaceTools(unittest.TestCase):
    def test_genus1_riemann_constant_matches_half_period_formula(self):
        surface = rst.build_surface_data(L=20, l1=3, l2=4)
        Delta = rst.riemann_constant_vector(surface)
        expected = 0.5 * (1.0 - surface.tau)

        self.assertEqual(Delta.shape, (1,))
        self.assertAlmostEqual(Delta[0].real, expected.real, places=10)
        self.assertAlmostEqual(Delta[0].imag, expected.imag, places=10)

    def test_genus1_abel_map_reproduces_a_and_b_periods(self):
        surface = rst.build_surface_data(L=20, l1=3, l2=4)
        self.assertEqual(surface.genus, 1)

        alpha = surface.basis_pairs[0]["alpha"]
        beta = surface.basis_pairs[0]["beta"]
        alpha_period = _cycle_period(surface, alpha)
        beta_period = _cycle_period(surface, beta)

        self.assertAlmostEqual(alpha_period.real, 1.0, places=9)
        self.assertAlmostEqual(alpha_period.imag, 0.0, places=9)
        self.assertAlmostEqual(beta_period.real, surface.tau.real, places=9)
        self.assertAlmostEqual(beta_period.imag, surface.tau.imag, places=9)

    def test_genus1_theta_with_characteristics_matches_jacobi(self):
        tau = np.array([[0.37 + 0.91j]], dtype=np.complex128)
        y = np.array([0.23 + 0.07j], dtype=np.complex128)
        q = mp.exp(mp.pi * 1j * mp.mpc(complex(tau[0, 0])))
        z = mp.pi * mp.mpc(complex(y[0]))

        cases = [
            (((0,), (0,)), complex(mp.jtheta(3, z, q))),
            (((1,), (0,)), complex(mp.jtheta(2, z, q))),
            (((0,), (1,)), complex(mp.jtheta(4, z, q))),
            (((1,), (1,)), -complex(mp.jtheta(1, z, q))),
        ]

        for characteristic, expected in cases:
            got = rst.riemann_theta(y, tau, characteristic=characteristic, nmax=8)
            self.assertAlmostEqual(got.real, expected.real, places=12)
            self.assertAlmostEqual(got.imag, expected.imag, places=12)

    def test_prime_form_has_correct_local_limit_genus1(self):
        surface = rst.build_surface_data(L=20, l1=3, l2=4)
        z = np.complex128(0.21 + 0.17j)
        eps = np.complex128(1.0e-6 * (1.0 + 0.4j))
        w = z + eps

        E = rst.prime_form(z, w, surface, characteristic=((1,), (1,)), nmax=8)
        ratio = E / (z - w)

        self.assertAlmostEqual(ratio.real, 1.0, places=6)
        self.assertAlmostEqual(ratio.imag, 0.0, places=6)

    def test_genus1_sigma_ratio_is_divisor_independent(self):
        surface = rst.build_surface_data(L=20, l1=3, l2=4)
        z = np.complex128(0.31 - 0.12j)
        w0 = np.complex128(-0.17 + 0.14j)

        sigma_a = rst.sigma_value(
            z,
            surface,
            divisor_points=[0.23 + 0.11j],
            normalization_point=w0,
            nmax=8,
        )
        sigma_b = rst.sigma_value(
            z,
            surface,
            divisor_points=[-0.09 + 0.27j],
            normalization_point=w0,
            nmax=8,
        )

        self.assertAlmostEqual(sigma_a.real, sigma_b.real, places=9)
        self.assertAlmostEqual(sigma_a.imag, sigma_b.imag, places=9)

    def test_genus1_sigma_normalization_is_imposed(self):
        surface = rst.build_surface_data(L=20, l1=3, l2=4)
        w0 = np.complex128(-0.17 + 0.14j)
        sigma = rst.sigma_value(
            w0,
            surface,
            divisor_points=[0.23 + 0.11j],
            normalization_point=w0,
            nmax=8,
        )
        self.assertAlmostEqual(sigma.real, 1.0, places=12)
        self.assertAlmostEqual(sigma.imag, 0.0, places=12)

    def test_genus2_theta_constant_matches_existing_helper(self):
        Omega = np.array(
            [
                [0.9j, 0.11 + 0.07j],
                [0.11 + 0.07j, 1.2j],
            ],
            dtype=np.complex128,
        )
        char = ((1, 0), (0, 1))
        got = rst.riemann_theta(
            np.zeros(2, dtype=np.complex128),
            Omega,
            characteristic=char,
            nmax=8,
        )

        import ell_to_tau as elt

        expected = elt.riemann_theta_constant_genus2(Omega, char, nmax=8)
        self.assertAlmostEqual(got.real, expected.real, places=12)
        self.assertAlmostEqual(got.imag, expected.imag, places=12)


if __name__ == "__main__":
    unittest.main()
