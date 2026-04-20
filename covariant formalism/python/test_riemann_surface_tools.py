import os
import sys
import unittest

import mpmath as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compact_partition as cp
import ell_to_tau as elt
from genus2_one_point import _stored_graph_to_ribbon_graph
import riemann_surface_tools as rst


def _cycle_period(surface, cycle_terms, form_idx: int = 0) -> np.complex128:
    total = np.complex128(0.0)
    F = surface.antiderivatives[form_idx]
    for edge_idx, coeff in cycle_terms:
        z0, z1 = surface.edge_midpoints[int(edge_idx)]
        total += np.complex128(coeff) * (F(z1) - F(z0))
    return np.complex128(total)


def _cycle_vector(cycle_terms, n_edges: int) -> np.ndarray:
    vec = np.zeros(n_edges, dtype=int)
    for edge_idx, coeff in cycle_terms:
        vec[int(edge_idx)] += int(coeff)
    return vec


def _identified_edge_jump_errors(surface, chord_intersection_matrix: np.ndarray) -> list[float]:
    n_edges = chord_intersection_matrix.shape[0]
    alpha_vecs = [
        _cycle_vector(pair["alpha"], n_edges)
        for pair in surface.basis_pairs
    ]
    beta_vecs = [
        _cycle_vector(pair["beta"], n_edges)
        for pair in surface.basis_pairs
    ]
    Omega = np.asarray(surface.Omega, dtype=np.complex128)

    errors = []
    for edge_idx, (z0, z1) in sorted(surface.edge_midpoints.items()):
        raw_edge = np.zeros(n_edges, dtype=int)
        raw_edge[int(edge_idx)] = 1

        m = np.asarray(
            [raw_edge @ chord_intersection_matrix @ beta for beta in beta_vecs],
            dtype=np.int64,
        )
        n = np.asarray(
            [-(raw_edge @ chord_intersection_matrix @ alpha) for alpha in alpha_vecs],
            dtype=np.int64,
        )

        predicted = m.astype(np.complex128) + Omega @ n.astype(np.complex128)
        actual = rst.abel_difference(z1, z0, surface)
        errors.append(float(np.max(np.abs(actual - predicted))))
    return errors


class TestRiemannSurfaceTools(unittest.TestCase):
    def test_large_l_fit_recovers_synthetic_coefficients(self):
        total_lengths = [1200, 1600, 2000, 2600]
        c0 = 1.25
        gamma = 0.031
        alpha = -1.375
        log_values = [
            c0 + gamma * L + alpha * np.log(L)
            for L in total_lengths
        ]

        fit = rst._fit_large_l_behavior(total_lengths, log_values)

        self.assertAlmostEqual(fit.c, c0, places=12)
        self.assertAlmostEqual(fit.gamma, gamma, places=12)
        self.assertAlmostEqual(fit.alpha, alpha, places=12)
        self.assertAlmostEqual(fit.finite_part, np.exp(c0), places=11)

    def test_universal_large_l_fit_recovers_shared_coefficients(self):
        gamma = 0.027
        alpha = -1.125
        c_values = [1.2, -0.4, 0.75]
        family_samples = []
        length_sets = [
            [1200, 1600, 2200],
            [1400, 1800, 2600, 3200],
            [1000, 1500, 2100],
        ]

        for c0, total_lengths in zip(c_values, length_sets):
            log_values = [
                c0 + gamma * L + alpha * np.log(L)
                for L in total_lengths
            ]
            family_samples.append((total_lengths, log_values))

        fit = rst.fit_universal_large_l_coefficients(family_samples)

        self.assertAlmostEqual(fit.gamma, gamma, places=12)
        self.assertAlmostEqual(fit.alpha, alpha, places=12)
        for got, expected in zip(fit.family_constants, c_values):
            self.assertAlmostEqual(got, expected, places=12)

    def test_renormalized_factor_from_fixed_coefficients_recovers_finite_part(self):
        total_lengths = [1200, 1600, 2000, 2600]
        c0 = 0.85
        gamma = 0.019
        alpha = -1.5
        log_values = [
            c0 + gamma * L + alpha * np.log(L)
            for L in total_lengths
        ]
        determinants = [np.exp(-2.0 * value) for value in log_values]

        finite_from_logs = rst.renormalized_aprime_factor_from_raw_log_values(
            total_lengths,
            log_values,
            gamma=gamma,
            alpha=alpha,
        )
        finite_from_det = rst.renormalized_aprime_factor_from_raw_det(
            total_lengths,
            determinants,
            gamma=gamma,
            alpha=alpha,
        )

        self.assertAlmostEqual(finite_from_logs, np.exp(c0), places=12)
        self.assertAlmostEqual(finite_from_det, np.exp(c0), places=12)

    def test_fixed_large_l_coefficients_allow_single_sample(self):
        L = 1800
        c0 = 0.42
        gamma = 0.013
        alpha = -1.25
        log_value = c0 + gamma * L + alpha * np.log(L)

        finite = rst.renormalized_aprime_factor_from_raw_log_values(
            [L],
            [log_value],
            gamma=gamma,
            alpha=alpha,
        )

        self.assertAlmostEqual(finite, np.exp(c0), places=12)

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

    def test_genus1_identified_boundary_points_differ_by_period_lattice(self):
        surface = rst.build_surface_data(L=3600, l1=500, l2=600)
        ribbon_graph = (
            [(1, 2), (1, 2), (1, 2)],
            [1, 2],
            {1: [0, 1, 2], 2: [0, 1, 2]},
        )
        chord_data = elt.edge_chord_intersection_matrix(ribbon_graph)
        errors = _identified_edge_jump_errors(
            surface,
            np.asarray(chord_data["intersection_matrix"], dtype=int),
        )

        self.assertLess(max(errors), 1e-6)

    def test_genus2_identified_boundary_points_differ_by_period_lattice(self):
        graph_data = cp.get_stored_genus2_graph(1)
        ribbon_graph = _stored_graph_to_ribbon_graph(graph_data)
        edge_lengths = [100] * 9
        forms = elt.make_cyl_eqn_improved_higher_genus(ribbon_graph, edge_lengths)
        surface = rst.build_surface_data(
            forms=forms,
            ribbon_graph=ribbon_graph,
            ell_list=edge_lengths,
        )
        chord_data = elt.edge_chord_intersection_matrix(ribbon_graph)
        errors = _identified_edge_jump_errors(
            surface,
            np.asarray(chord_data["intersection_matrix"], dtype=int),
        )

        self.assertLess(max(errors), 1e-5)

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

    def test_genus1_abs_z1_sq_pipeline_is_self_consistent(self):
        ribbon_graph = (
            [(1, 2), (1, 2), (1, 2)],
            [1, 2],
            {1: [0, 1, 2], 2: [0, 1, 2]},
        )
        result = rst.estimate_abs_z1_sq(
            ribbon_graph,
            (1, 1, 1),
            scales=(200, 240, 300),
            b_points=[0.21 + 0.17j],
            c_point=-0.17 + 0.14j,
            divisor_points=[0.23 + 0.11j],
            normalization_point=-0.17 + 0.14j,
            nmax=8,
        )

        direct = rst.abs_z1_sq_from_lambda_one(
            [0.21 + 0.17j],
            -0.17 + 0.14j,
            result.surface,
            divisor_points=[0.23 + 0.11j],
            normalization_point=-0.17 + 0.14j,
            nmax=8,
        )
        reconstructed = rst.abs_z1_sq_from_renormalized_det(
            result.surface,
            normalization_factor=result.normalization_factor,
            renormalized_det_factor=result.renormalized_det_factor,
        )

        self.assertAlmostEqual(result.abs_z1_sq, direct, places=10)
        self.assertAlmostEqual(result.abs_z1_sq, reconstructed, places=12)
        self.assertGreater(result.normalization_factor, 0.0)
        self.assertGreater(result.renormalized_det_factor, 0.0)

    def test_canonical_abs_z1_sq_uses_n1_equal_one_convention(self):
        ribbon_graph = (
            [(1, 2), (1, 2), (1, 2)],
            [1, 2],
            {1: [0, 1, 2], 2: [0, 1, 2]},
        )
        result = rst.estimate_canonical_abs_z1_sq(
            ribbon_graph,
            (1, 1, 1),
            scales=(200, 240, 300),
        )
        expected = rst.abs_z1_sq_from_renormalized_det(
            result.surface,
            renormalized_det_factor=result.renormalized_det_factor,
        )

        self.assertEqual(result.normalization_factor, 1.0)
        self.assertAlmostEqual(result.abs_z1_sq, expected, places=12)

    def test_sigma_scale_from_z1_requires_higher_genus(self):
        surface = rst.build_surface_data(L=20, l1=3, l2=4)
        with self.assertRaises(ValueError):
            rst.sigma_scale_from_z1(
                [0.21 + 0.17j],
                -0.17 + 0.14j,
                surface,
                divisor_points=[0.23 + 0.11j],
                normalization_point=0.0 + 0.0j,
                z1=1.7,
                nmax=8,
            )

    def test_genus2_sigma_scale_from_z1_matches_lambda_one_equation(self):
        graph_data = cp.get_stored_genus2_graph(1)
        ribbon_graph = _stored_graph_to_ribbon_graph(graph_data)
        edge_lengths = [100] * 9
        forms = elt.make_cyl_eqn_improved_higher_genus(ribbon_graph, edge_lengths)
        surface = rst.build_surface_data(
            forms=forms,
            ribbon_graph=ribbon_graph,
            ell_list=edge_lengths,
        )

        z1 = np.complex128(1.7)
        anchor_b_points = [0.12 + 0.08j, -0.07 + 0.18j]
        anchor_c_point = 0.04 - 0.19j
        divisor_points = [0.23 + 0.11j, -0.17 + 0.14j]

        scale = rst.sigma_scale_from_z1(
            anchor_b_points,
            anchor_c_point,
            surface,
            divisor_points=divisor_points,
            normalization_point=0.0 + 0.0j,
            z1=z1,
            nmax=8,
        )
        a_tilde = rst.lambda_one_geometric_z1_factor(
            anchor_b_points,
            anchor_c_point,
            surface,
            divisor_points=divisor_points,
            normalization_point=0.0 + 0.0j,
            nmax=8,
        )
        target = z1 * np.sqrt(z1)

        self.assertAlmostEqual((scale * a_tilde).real, target.real, places=9)
        self.assertAlmostEqual((scale * a_tilde).imag, target.imag, places=9)

    def test_genus2_direct_sigma_values_satisfy_lambda_one_equations(self):
        graph_data = cp.get_stored_genus2_graph(1)
        ribbon_graph = _stored_graph_to_ribbon_graph(graph_data)
        edge_lengths = [100] * 9
        forms = elt.make_cyl_eqn_improved_higher_genus(ribbon_graph, edge_lengths)
        surface = rst.build_surface_data(
            forms=forms,
            ribbon_graph=ribbon_graph,
            ell_list=edge_lengths,
        )

        z1 = np.complex128(1.7)
        points = [0.12 + 0.08j, -0.07 + 0.18j, 0.04 - 0.19j]
        Delta = rst.riemann_constant_vector(surface)
        s1, s2, s3 = rst.genus2_sigma_values_from_lambda_one(
            points,
            surface,
            z1=z1,
            Delta=Delta,
            nmax=8,
        )
        h12 = rst.genus2_lambda_one_sigma_kernel(
            points[0],
            points[1],
            points[2],
            surface,
            z1=z1,
            Delta=Delta,
            nmax=8,
        )
        h13 = rst.genus2_lambda_one_sigma_kernel(
            points[0],
            points[2],
            points[1],
            surface,
            z1=z1,
            Delta=Delta,
            nmax=8,
        )
        h23 = rst.genus2_lambda_one_sigma_kernel(
            points[1],
            points[2],
            points[0],
            surface,
            z1=z1,
            Delta=Delta,
            nmax=8,
        )

        self.assertLess(abs(s3 - s1 * s2 * h12), 1e-9)
        self.assertLess(abs(s2 - s1 * s3 * h13), 1e-9)
        self.assertLess(abs(s1 - s2 * s3 * h23), 1e-9)

    def test_genus2_direct_bbb_correlator_is_anchor_free(self):
        graph_data = cp.get_stored_genus2_graph(1)
        ribbon_graph = _stored_graph_to_ribbon_graph(graph_data)
        edge_lengths = [100] * 9
        forms = elt.make_cyl_eqn_improved_higher_genus(ribbon_graph, edge_lengths)
        surface = rst.build_surface_data(
            forms=forms,
            ribbon_graph=ribbon_graph,
            ell_list=edge_lengths,
        )

        z1 = np.complex128(1.7)
        b_points = [0.12 + 0.08j, -0.07 + 0.18j, 0.04 - 0.19j]
        direct = rst.genus2_bbb_correlator_from_lambda_one(
            b_points,
            surface,
            z1=z1,
            nmax=8,
        )

        anchor_b_points = [0.21 + 0.17j, -0.11 + 0.14j]
        anchor_c_point = 0.03 - 0.12j
        divisor_points = [0.23 + 0.11j, -0.17 + 0.14j]
        old_style = np.complex128(1.0)
        Delta = rst.riemann_constant_vector(surface)
        theta_arg = np.sum(
            np.asarray([rst.abel_map(point, surface) for point in b_points], dtype=np.complex128),
            axis=0,
        ) - 3.0 * Delta
        theta_val = rst.riemann_theta(theta_arg, surface.Omega, nmax=8)
        for idx, zi in enumerate(b_points):
            for zj in b_points[idx + 1 :]:
                old_style *= rst.prime_form(zi, zj, surface, nmax=8)
        sigma_prod = np.complex128(1.0)
        for zi in b_points:
            sigma_prod *= rst.canonical_sigma_value(
                zi,
                surface,
                anchor_b_points=anchor_b_points,
                anchor_c_point=anchor_c_point,
                divisor_points=divisor_points,
                normalization_point=anchor_c_point,
                z1=z1,
                Delta=Delta,
                nmax=8,
            ) ** 3
        old_style = np.complex128(theta_val * old_style * sigma_prod / np.sqrt(z1))

        rel_diff = abs(abs(direct) - abs(old_style)) / abs(direct)
        self.assertGreater(rel_diff, 1e-3)

    def test_bc_correlator_selection_rule_is_enforced(self):
        surface = rst.build_surface_data(L=20, l1=3, l2=4)
        with self.assertRaises(ValueError):
            rst.bc_correlator_geometric_factor(
                [0.21 + 0.17j],
                [],
                surface,
                lambda_weight=2.0,
                divisor_points=[0.23 + 0.11j],
                normalization_point=0.0 + 0.0j,
                nmax=8,
            )

    def test_genus1_lambda_one_correlator_matches_special_z1_helper(self):
        surface = rst.build_surface_data(L=20, l1=3, l2=4)
        z = np.complex128(0.21 + 0.17j)
        w = np.complex128(-0.17 + 0.14j)
        divisor = [0.23 + 0.11j]

        geometric = rst.bc_correlator_geometric_factor(
            [z],
            [w],
            surface,
            lambda_weight=1.0,
            divisor_points=divisor,
            normalization_point=w,
            nmax=8,
        )
        omega_z = rst._evaluate_one_form(surface.normalized_forms[0], z)
        special = rst.lambda_one_geometric_z1_factor(
            [z],
            w,
            surface,
            divisor_points=divisor,
            normalization_point=w,
            nmax=8,
        )

        self.assertAlmostEqual((geometric / omega_z).real, special.real, places=10)
        self.assertAlmostEqual((geometric / omega_z).imag, special.imag, places=10)

    def test_canonical_riemann_constant_matches_genus1_half_period_modulo_lattice(self):
        surface = rst.build_surface_data(L=20, l1=3, l2=4)
        Delta_can = rst.riemann_constant_vector_canonical(surface, nmax=8)
        Delta_old = rst.riemann_constant_vector(surface)
        diff = Delta_can[0] - Delta_old[0]
        tau = surface.tau
        a_re = diff.real - round(diff.real)
        a_im = diff.imag
        m_n_n = a_im / tau.imag
        n = round(m_n_n)
        m = round(diff.real - n * tau.real)
        residual = diff - (m + n * tau)
        self.assertLess(abs(residual), 1e-8)

    def test_canonical_riemann_constant_satisfies_vanishing_genus2_symmetric(self):
        graph_data = cp.get_stored_genus2_graph(1)
        ribbon_graph = _stored_graph_to_ribbon_graph(graph_data)
        edge_lengths = [100] * 9
        forms = elt.make_cyl_eqn_improved_higher_genus(ribbon_graph, edge_lengths)
        surface = rst.build_surface_data(
            forms=forms,
            ribbon_graph=ribbon_graph,
            ell_list=edge_lengths,
        )
        Delta = rst.riemann_constant_vector_canonical(surface, nmax=6)
        test_points = [
            0.08 + 0.12j,
            -0.14 + 0.16j,
            0.19 - 0.09j,
            0.21 + 0.09j,
            -0.16 + 0.12j,
        ]
        values = [
            abs(rst.riemann_theta(
                rst.abel_map(p, surface) - Delta,
                surface.Omega,
                nmax=6,
            ))
            for p in test_points
        ]
        self.assertLess(max(values), 1e-4)

    def test_canonical_riemann_constant_satisfies_vanishing_genus2_asymmetric(self):
        graph_data = cp.get_stored_genus2_graph(1)
        ribbon_graph = _stored_graph_to_ribbon_graph(graph_data)
        edge_lengths = [210, 230, 250, 270, 290, 310, 330, 350, 370]
        forms = elt.make_cyl_eqn_improved_higher_genus(ribbon_graph, edge_lengths)
        surface = rst.build_surface_data(
            forms=forms,
            ribbon_graph=ribbon_graph,
            ell_list=edge_lengths,
        )
        Delta = rst.riemann_constant_vector_canonical(surface, nmax=6)
        test_points = [
            0.08 + 0.12j,
            -0.14 + 0.16j,
            0.19 - 0.09j,
            0.21 + 0.09j,
            -0.16 + 0.12j,
        ]
        values = [
            abs(rst.riemann_theta(
                rst.abel_map(p, surface) - Delta,
                surface.Omega,
                nmax=6,
            ))
            for p in test_points
        ]
        self.assertLess(max(values), 1e-4)

    def test_canonical_riemann_constant_makes_sigma_ratio_divisor_independent_genus2(self):
        graph_data = cp.get_stored_genus2_graph(1)
        ribbon_graph = _stored_graph_to_ribbon_graph(graph_data)
        edge_lengths = [210, 230, 250, 270, 290, 310, 330, 350, 370]
        forms = elt.make_cyl_eqn_improved_higher_genus(ribbon_graph, edge_lengths)
        surface = rst.build_surface_data(
            forms=forms,
            ribbon_graph=ribbon_graph,
            ell_list=edge_lengths,
        )
        Delta = rst.riemann_constant_vector_canonical(surface, nmax=6)
        z = 0.08 + 0.12j
        w = -0.19 + 0.13j
        divisors = [
            [0.21 + 0.09j, -0.16 + 0.12j],
            [0.17 + 0.05j, -0.12 + 0.11j],
            [0.22 + 0.08j, -0.09 + 0.16j],
        ]
        ratios = [
            rst.sigma_ratio(z, w, surface, divisor_points=d, Delta=Delta, nmax=6)
            for d in divisors
        ]
        mags = [abs(r) for r in ratios]
        spread = max(mags) / min(mags)
        self.assertLess(spread - 1.0, 1e-4)


if __name__ == "__main__":
    unittest.main()
