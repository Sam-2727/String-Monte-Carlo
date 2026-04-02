import unittest
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compact_partition as cp
import ell_to_tau as elt
import genus3_t_duality as g3

THETA_CUTOFF = 4
GENUS2_RANDOM_SEED = 20260402
GENUS3_RANDOM_SEED = 20260403


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


def _full_ratio_from_sums(sum_r1: float, sum_r2: float, R1: float, R2: float) -> float:
    """Equation-(9.11) style compact-boson ratio, including the explicit zero-mode factor."""
    return (R1 * sum_r1) / (R2 * sum_r2)


def _random_topology_case(*, seed: int, topology_count: int, edge_count: int) -> dict:
    """Reproducible random test geometry with edge lengths in [50, 300]."""
    rng = np.random.default_rng(seed)
    topology = int(rng.integers(1, topology_count + 1))
    edge_lengths = [int(x) for x in rng.integers(50, 301, size=edge_count)]
    return {
        "seed": seed,
        "topology": topology,
        "edge_lengths": edge_lengths,
    }


def _genus2_random_case() -> dict:
    return _random_topology_case(
        seed=GENUS2_RANDOM_SEED,
        topology_count=len(cp.GENUS2_F1_GRAPH_DATA),
        edge_count=9,
    )


def _genus3_random_case() -> dict:
    return _random_topology_case(
        seed=GENUS3_RANDOM_SEED,
        topology_count=g3.GENUS3_GRAPH_COUNT,
        edge_count=15,
    )


def _check_higher_genus_ratio(graph_data, edge_lengths, R1, R2, *,
                              theta_cutoff, analytic_cutoff):
    """
    Compare the full compact-boson ratio in equation (9.11).

    Both sides include the explicit compact-boson zero-mode factor R, so this
    compares
      (R1 * Theta_graph(R1)) / (R2 * Theta_graph(R2))
    against
      (R1 * LatticeSum_analytic(R1)) / (R2 * LatticeSum_analytic(R2)).
    """
    data = _higher_genus_ratio_data(
        graph_data,
        edge_lengths,
        R1,
        R2,
        theta_cutoff=theta_cutoff,
        analytic_cutoff=analytic_cutoff,
    )
    return data["graph_ratio"], data["analytic_ratio"]


def _higher_genus_ratio_data(graph_data, edge_lengths, R1, R2, *,
                             theta_cutoff, analytic_cutoff):
    """Return the full graph-vs-analytic ratio comparison data for one geometry."""
    ribbon_graph = _stored_graph_to_ribbon_graph(graph_data)
    forms = elt.make_cyl_eqn_improved_higher_genus(ribbon_graph, edge_lengths)
    pdata = elt.period_matrix(
        forms=forms,
        ribbon_graph=ribbon_graph,
        ell_list=edge_lengths,
        return_data=True,
    )
    Omega = np.asarray(pdata["Omega"], dtype=np.complex128)

    geom = cp.compact_boson_graph_geometry(edge_lengths, graph_data)
    lattice_sum_r1 = cp.theta_sum_reduced(geom["T_reduced"], R1, N=theta_cutoff)
    lattice_sum_r2 = cp.theta_sum_reduced(geom["T_reduced"], R2, N=theta_cutoff)
    analytic_sum_r1 = _analytic_compact_lattice_sum(Omega, R1, analytic_cutoff)
    analytic_sum_r2 = _analytic_compact_lattice_sum(Omega, R2, analytic_cutoff)

    ratio_lattice = _full_ratio_from_sums(
        lattice_sum_r1,
        lattice_sum_r2,
        R1,
        R2,
    )
    ratio_analytic = _full_ratio_from_sums(
        analytic_sum_r1,
        analytic_sum_r2,
        R1,
        R2,
    )
    resid = ratio_lattice - ratio_analytic
    rel = abs(resid) / abs(ratio_analytic)
    return {
        "Omega": Omega,
        "graph_ratio": ratio_lattice,
        "analytic_ratio": ratio_analytic,
        "residual": resid,
        "rel_error": rel,
    }


def _print_ratio_report(label, graph_data, edge_lengths, R1, R2, *,
                        theta_cutoff, analytic_cutoff, topology=None, seed=None):
    """Print a readable analytic-vs-graph comparison for one higher-genus geometry."""
    ratio_lattice, ratio_analytic = _check_higher_genus_ratio(
        graph_data,
        edge_lengths,
        R1,
        R2,
        theta_cutoff=theta_cutoff,
        analytic_cutoff=analytic_cutoff,
    )
    resid = ratio_lattice - ratio_analytic
    rel = abs(resid) / abs(ratio_analytic)

    print(label)
    if topology is not None:
        print(f"  topology         = {topology}")
    if seed is not None:
        print(f"  rng seed         = {seed}")
    print(f"  edge_lengths     = {edge_lengths}")
    print(f"  R1, R2           = {R1}, {R2}")
    print(f"  theta_cutoff     = {theta_cutoff}")
    print(f"  analytic_cutoff  = {analytic_cutoff}")
    print(f"  graph ratio      = {ratio_lattice:.15g}")
    print(f"  analytic ratio   = {ratio_analytic:.15g}")
    print(f"  residual         = {resid:.6e}")
    print(f"  relative error   = {rel:.6e}")
    print()


def print_reference_reports():
    """Diagnostic output for reproducible random higher-genus checks."""
    genus2_case = _genus2_random_case()
    genus3_case = _genus3_random_case()

    print("Higher-genus compact-boson ratio checks")
    print("Comparing (R1 * Theta_graph(R1)) / (R2 * Theta_graph(R2))")
    print("against   (R1 * LatticeSum_analytic(R1)) / (R2 * LatticeSum_analytic(R2))")
    print("Each topology and edge-length set below is chosen reproducibly from a fixed RNG seed.")
    print()

    _print_ratio_report(
        "Genus 2 random sample",
        cp.get_stored_genus2_graph(genus2_case["topology"]),
        genus2_case["edge_lengths"],
        1.2,
        1.6,
        theta_cutoff=THETA_CUTOFF,
        analytic_cutoff=3,
        topology=genus2_case["topology"],
        seed=genus2_case["seed"],
    )

    _print_ratio_report(
        "Genus 3 random sample",
        g3.get_stored_genus3_graph(genus3_case["topology"]),
        genus3_case["edge_lengths"],
        1.2,
        1.6,
        theta_cutoff=THETA_CUTOFF,
        analytic_cutoff=3,
        topology=genus3_case["topology"],
        seed=genus3_case["seed"],
    )


class TestCompactPartitionHigherGenus(unittest.TestCase):
    @unittest.skip("Focused on the genus-2 all-topology theta-ratio check.")
    def test_graph_path_reproduces_torus_special_case(self):
        L, l1, l2, R = 24, 3, 4, 1.3
        l3 = L // 2 - l1 - l2

        Mat = cp.direct_mat_n_fast(L)
        Aprime = cp.direct_red_traced_mat(L, l1, l2, Mat)
        W = cp.mat_w(L, l1, l2, Mat)
        T1 = cp.mat_t_first_part(L, l1, l2, Mat)
        T2 = cp.mat_t_second_part(L, l1, l2, W, Aprime)
        Tp = cp.mat_t_prime(cp.symm(T1 - T2))
        z_torus = cp.partition_function_z(Aprime, R, Tp, N=THETA_CUTOFF)

        z_graph = cp.compute_graph_compact_partition(
            [l1, l2, l3],
            R,
            cp.GENUS1_F1_GRAPH_DATA,
            N=THETA_CUTOFF,
        )["Z"]

        self.assertAlmostEqual(z_graph, z_torus, places=12)

    @unittest.skip("Focused on the genus-2 all-topology theta-ratio check.")
    def test_genus2_topologies_have_four_holonomies(self):
        edge_lengths = [1, 2, 2, 1, 2, 2, 1, 2, 2]
        values = []

        for topology in range(1, len(cp.GENUS2_F1_GRAPH_DATA) + 1):
            graph_data = cp.get_stored_genus2_graph(topology)
            geom = cp.compute_graph_compact_partition(edge_lengths, 1.2, graph_data, N=THETA_CUTOFF)
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
        edge_lengths = _genus2_random_case()["edge_lengths"]
        for topology in range(1, len(cp.GENUS2_F1_GRAPH_DATA) + 1):
            ratio_lattice, ratio_analytic = _check_higher_genus_ratio(
                cp.get_stored_genus2_graph(topology),
                edge_lengths,
                1.2,
                1.6,
                theta_cutoff=THETA_CUTOFF,
                analytic_cutoff=3,
            )
            rel = abs(ratio_lattice - ratio_analytic) / abs(ratio_analytic)
            with self.subTest(topology=topology):
                self.assertLess(rel, 1.0e-3)

    @unittest.skip("Focused on the genus-2 all-topology theta-ratio check.")
    def test_genus3_theta_ratio_matches_analytic_period_matrix_sum(self):
        case = _genus3_random_case()
        ratio_lattice, ratio_analytic = _check_higher_genus_ratio(
            g3.get_stored_genus3_graph(case["topology"]),
            case["edge_lengths"],
            1.2,
            1.6,
            theta_cutoff=THETA_CUTOFF,
            analytic_cutoff=3,
        )
        rel = abs(ratio_lattice - ratio_analytic) / abs(ratio_analytic)
        self.assertLess(rel, 1.0e-3)


def print_genus2_theta_ratio_report_all_topologies():
    """Direct-run report for the genus-2 theta-ratio check across all stored topologies."""
    edge_lengths = _genus2_random_case()["edge_lengths"]
    R1, R2 = 1.2, 1.6
    failures = []

    for topology in range(1, len(cp.GENUS2_F1_GRAPH_DATA) + 1):
        data = _higher_genus_ratio_data(
            cp.get_stored_genus2_graph(topology),
            edge_lengths,
            R1,
            R2,
            theta_cutoff=THETA_CUTOFF,
            analytic_cutoff=3,
        )

        print(f"topology = {topology}")
        print(f"edge_lengths = {edge_lengths}")
        print("Omega =")
        print(np.array2string(data["Omega"], precision=10, suppress_small=False))
        print(f"R1 = {R1}")
        print(f"R2 = {R2}")
        print(f"graph ratio = {data['graph_ratio']:.15g}")
        print(f"analytic ratio = {data['analytic_ratio']:.15g}")
        print(f"residual = {data['residual']:.6e}")
        print(f"relerror = {data['rel_error']:.6e}")
        print()

        if data["rel_error"] >= 1.0e-3:
            failures.append((topology, data["rel_error"]))

    if failures:
        raise AssertionError(f"Genus-2 theta-ratio failures: {failures}")


if __name__ == "__main__":
    print_genus2_theta_ratio_report_all_topologies()
