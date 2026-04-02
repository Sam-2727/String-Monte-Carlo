"""
genus2_one_point.py

Numerical one-point function of the compact-boson operator
    :partial X partial X:
on a genus-2, one-puncture ribbon-graph surface in the discretized disc frame.

This is the higher-genus analogue of torus_one_point.py. The core formula is
the same Gaussian insertion computation, but the two torus windings are now
replaced by the four independent holonomies obtained from the graph cycle
basis in compact_partition.py.

For a fixed genus-2 topology and edge-length list, the script computes

    < :partial X partial X:(0) >_disc
      = (pi/2) Tr(C_11 A'^(-1))
        + 4 pi^2 R^2
          [sum_n (n^T Xi_11 n) exp(-4 pi R^2 n^T T' n)]
          / [sum_n exp(-4 pi R^2 n^T T' n)],

where n in Z^4 runs over the independent holonomy lattice. The corresponding
stress tensor is

    <T(0)>_disc = -(1/alpha') < :partial X partial X:(0) >_disc.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

import compact_partition as cp
import ell_to_tau as elt


DEFAULT_ALPHA_PRIME = 1.0
DEFAULT_EDGE_LENGTHS = (2, 2, 2, 2, 2, 2, 2, 2, 2)


@dataclass
class GraphGeometry:
    L: int
    Aprime: np.ndarray
    P: np.ndarray
    Q_reduced: np.ndarray
    U_reduced: np.ndarray
    T_reduced: np.ndarray
    cycle_basis: np.ndarray


@dataclass
class GraphDiskComponents:
    oscillator_disk: complex
    classical_disk: complex
    compact_trace_11: complex
    Xi11_reduced: np.ndarray
    theta_sum: float


@dataclass
class PeriodDiagnostics:
    Omega: np.ndarray
    omega0: tuple[complex, complex]
    quadratic_basis0: tuple[complex, complex, complex]
    basis_pairs: list


@dataclass
class Genus2OnePointResult:
    topology: int
    edge_lengths: tuple[int, ...]
    R: float
    alpha_prime: float
    holonomy_dim: int
    holonomy_cutoff: int
    disk_one_point: complex
    disk_stress_tensor: complex
    oscillator_disk: complex
    classical_disk: complex
    compact_trace_11: complex
    Xi11_reduced: np.ndarray
    T_reduced: np.ndarray
    theta_sum: float
    period_data: PeriodDiagnostics | None


@dataclass
class Genus1ComparisonRow:
    sample_index: int
    edge_lengths: tuple[int, int, int]
    L: int
    R: float
    winding_cutoff: int
    graph_value: complex
    torus_value: complex
    abs_error: float
    rel_error: float


def _full_one_form(f, z: complex) -> complex:
    """Return the full one-form coefficient f(z), not the split (singular, poly)."""
    value = f(z)
    if isinstance(value, tuple):
        singular, poly = value
        return complex(singular * poly)
    return complex(value)


def _format_complex(z: complex) -> str:
    return f"{z.real:.12g} {z.imag:+.12g}i"


def _format_matrix(mat: np.ndarray) -> str:
    rows = []
    for row in np.asarray(mat):
        rows.append("[" + ", ".join(_format_complex(complex(val)) for val in row) + "]")
    return "[\n  " + ",\n  ".join(rows) + "\n]"


def _stored_graph_to_ribbon_graph(graph_data: dict):
    """
    Reconstruct a ribbon-graph rotation system from stored one-face graph data.

    This matches the helper used in test_compact_partition.py, kept local here
    so the script can be run standalone.
    """
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


def _mode_quadratic_matrix(L: int, m: int, n: int) -> np.ndarray:
    """Full-boundary quadratic form for the holomorphic insertion X_m X_n."""
    idx = np.arange(1, L + 1, dtype=np.float64)
    phase_m = np.exp(-2j * np.pi * m * idx / L)
    phase_n = np.exp(-2j * np.pi * n * idx / L)
    return np.outer(phase_m, phase_n) / (L * L)


def _graph_lattice_setup(edge_lengths, graph_data: dict) -> GraphGeometry:
    """
    Build the higher-genus insertion data in the same reduced basis as A'.

    The compact_partition reduction is the higher-genus analogue of the torus
    zero-sum slice:

        X(K[0]) = -sum_{j>0} X(K[j]).

    The reconstruction matrix P must therefore use the same basis, rather than
    a simple gauge choice X(K[0]) = 0.
    """
    gluing = cp._gluing_data(edge_lengths, graph_data)
    K = gluing["K"]
    prime = gluing["prime"]
    edge_K_cols = gluing["edge_K_cols"]

    L = gluing["L"]
    Mat = cp.direct_mat_n_fast(L)
    base_geom = cp.compact_boson_graph_geometry(edge_lengths, graph_data, Mat=Mat)

    Aprime = np.asarray(base_geom["A_prime"], dtype=np.complex128)
    W = np.asarray(base_geom["W"], dtype=np.complex128)
    cycle_basis = np.asarray(base_geom["cycle_basis"], dtype=np.complex128)
    T_reduced = np.asarray(base_geom["T_reduced"], dtype=np.complex128)

    n_red = K.size - 1
    n_edges = len(edge_K_cols)

    U_edge = np.zeros((n_red, n_edges), dtype=np.complex128)
    edge_of_k = np.full(K.size, -1, dtype=np.int64)
    for edge_idx, cols in enumerate(edge_K_cols):
        U_edge[:, edge_idx] = W[:, cols].sum(axis=1)
        edge_of_k[cols] = edge_idx

    P = np.zeros((L, n_red), dtype=np.complex128)
    Q_edge = np.zeros((L, n_edges), dtype=np.complex128)
    eye = np.eye(n_red, dtype=np.complex128)
    base_row = -np.ones(n_red, dtype=np.complex128)
    for idx, point in enumerate(K):
        row = base_row if idx == 0 else eye[idx - 1]
        P[point, :] = row
        P[prime[point], :] = row
        Q_edge[prime[point], edge_of_k[idx]] = 1.0

    return GraphGeometry(
        L=L,
        Aprime=Aprime,
        P=P,
        Q_reduced=Q_edge @ cycle_basis,
        U_reduced=U_edge @ cycle_basis,
        T_reduced=T_reduced,
        cycle_basis=cycle_basis,
    )


def _integer_lattice_points(dim: int, cutoff: int) -> np.ndarray:
    """Return all lattice points n in Z^dim with each coordinate in [-cutoff, cutoff]."""
    side = np.arange(-cutoff, cutoff + 1, dtype=np.float64)
    grids = np.meshgrid(*([side] * dim), indexing="ij")
    return np.stack(grids, axis=-1).reshape(-1, dim)


def _quadratic_form_values(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Evaluate n^T matrix n on a list of row vectors n."""
    return np.einsum("ni,ij,nj->n", points, matrix, points, optimize=True)


def discretized_graph_disk_components(edge_lengths,
                                      R: float,
                                      graph_data: dict,
                                      holonomy_cutoff: int = 5,
                                      mode_m: int = 1,
                                      mode_n: int = 1) -> GraphDiskComponents:
    """
    Compute the graph-based oscillator/classical split of the disk one-point function.

    This is the direct higher-genus generalization of
    torus_one_point._discretized_disk_components.
    """
    geom = _graph_lattice_setup(edge_lengths, graph_data)
    Nmn = _mode_quadratic_matrix(geom.L, mode_m, mode_n)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Cmn = geom.P.T @ Nmn @ geom.P
        Ymn = geom.P.T @ Nmn @ geom.Q_reduced
        Emn = geom.Q_reduced.T @ Nmn @ geom.Q_reduced

        AC = np.linalg.solve(geom.Aprime, Cmn)
        AU = np.linalg.solve(geom.Aprime, geom.U_reduced)
        Xi = Emn - 2.0 * (Ymn.T @ AU) + (geom.U_reduced.T @ AC @ AU)
        Xi = 0.5 * (Xi + Xi.T)

    holonomy_dim = geom.T_reduced.shape[0]
    points = _integer_lattice_points(holonomy_dim, holonomy_cutoff)
    quad_t = _quadratic_form_values(geom.T_reduced, points)
    quad_xi = _quadratic_form_values(Xi, points)
    weights = np.exp(-4.0 * np.pi * R * R * quad_t.real)
    theta_sum = float(np.sum(weights))

    oscillator_disk = 0.5 * np.pi * np.trace(AC)
    classical_disk = 4.0 * np.pi * np.pi * R * R * np.sum(quad_xi * weights) / theta_sum

    return GraphDiskComponents(
        oscillator_disk=complex(oscillator_disk),
        classical_disk=complex(classical_disk),
        compact_trace_11=complex(np.trace(AC)),
        Xi11_reduced=np.asarray(Xi, dtype=np.complex128),
        theta_sum=theta_sum,
    )


def discretized_graph_disk_one_point(edge_lengths,
                                     R: float,
                                     graph_data: dict,
                                     holonomy_cutoff: int = 5,
                                     mode_m: int = 1,
                                     mode_n: int = 1) -> complex:
    """Return the normalized disk one-point function of :partial X partial X:."""
    comp = discretized_graph_disk_components(
        edge_lengths,
        R,
        graph_data,
        holonomy_cutoff=holonomy_cutoff,
        mode_m=mode_m,
        mode_n=mode_n,
    )
    return complex(comp.oscillator_disk + comp.classical_disk)


def _period_diagnostics(edge_lengths, graph_data: dict) -> PeriodDiagnostics:
    """
    Compute genus-2 period-matrix data and A-normalized one-form values at z=0.

    The values omega_I(0) are not used in the lattice one-point computation
    itself, but they are useful diagnostics for relating the disc insertion to
    the canonical genus-2 holomorphic basis.
    """
    ribbon_graph = _stored_graph_to_ribbon_graph(graph_data)
    forms = elt.make_cyl_eqn_improved_higher_genus(ribbon_graph, edge_lengths)
    if len(forms) != 2:
        raise ValueError(f"Expected two holomorphic one-forms for genus 2, got {len(forms)}.")

    norm_data = elt.normalize_holomorphic_forms(
        forms,
        ribbon_graph=ribbon_graph,
        ell_list=edge_lengths,
        return_data=True,
    )
    Omega = np.asarray(norm_data["Omega"], dtype=np.complex128)
    if Omega.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 genus-2 period matrix, got shape {Omega.shape}.")

    normalized_forms = norm_data["normalized_forms"]
    omega0 = tuple(_full_one_form(f, 0.0) for f in normalized_forms)
    quadratic_basis0 = (
        omega0[0] * omega0[0],
        omega0[0] * omega0[1],
        omega0[1] * omega0[1],
    )
    return PeriodDiagnostics(
        Omega=Omega,
        omega0=(complex(omega0[0]), complex(omega0[1])),
        quadratic_basis0=tuple(complex(val) for val in quadratic_basis0),
        basis_pairs=list(norm_data["basis_pairs"]),
    )


def compute_genus2_one_point(edge_lengths=DEFAULT_EDGE_LENGTHS,
                             R: float = 1.0,
                             topology: int = 1,
                             holonomy_cutoff: int = 5,
                             alpha_prime: float = DEFAULT_ALPHA_PRIME,
                             include_period_data: bool = True) -> Genus2OnePointResult:
    """
    Compute the genus-2 compact-boson one-point function for a stored topology.

    Parameters
    ----------
    edge_lengths
        Positive integer edge lengths for the 9 graph edges.
    R
        Compactification radius.
    topology
        Which stored genus-2 one-face ribbon-graph topology to use, 1..4.
    holonomy_cutoff
        Truncation N for the four-dimensional holonomy sum n_i in [-N, N].
    alpha_prime
        Worldsheet normalization. The script computes :partial X partial X:
        directly and reports the stress tensor as -1/alpha' times that value.
    include_period_data
        Whether to also compute Omega and the A-normalized one-form values at
        the puncture.
    """
    graph_data = cp.get_stored_genus2_graph(topology)
    edge_lengths = tuple(int(x) for x in edge_lengths)
    geom = _graph_lattice_setup(edge_lengths, graph_data)
    if geom.T_reduced.shape != (4, 4):
        raise ValueError(
            "Expected a genus-2 holonomy matrix of shape (4, 4), "
            f"got {geom.T_reduced.shape}."
        )

    comp = discretized_graph_disk_components(
        edge_lengths,
        R,
        graph_data,
        holonomy_cutoff=holonomy_cutoff,
        mode_m=1,
        mode_n=1,
    )
    disk_one_point = complex(comp.oscillator_disk + comp.classical_disk)
    disk_stress_tensor = complex(-disk_one_point / alpha_prime)
    period_data = _period_diagnostics(edge_lengths, graph_data) if include_period_data else None

    return Genus2OnePointResult(
        topology=topology,
        edge_lengths=edge_lengths,
        R=R,
        alpha_prime=alpha_prime,
        holonomy_dim=int(geom.T_reduced.shape[0]),
        holonomy_cutoff=holonomy_cutoff,
        disk_one_point=disk_one_point,
        disk_stress_tensor=disk_stress_tensor,
        oscillator_disk=comp.oscillator_disk,
        classical_disk=comp.classical_disk,
        compact_trace_11=comp.compact_trace_11,
        Xi11_reduced=comp.Xi11_reduced,
        T_reduced=geom.T_reduced,
        theta_sum=comp.theta_sum,
        period_data=period_data,
    )


def compare_against_torus_random_samples(*,
                                         n_samples: int = 10,
                                         low: int = 50,
                                         high: int = 300,
                                         seed: int = 20260401,
                                         R: float = 1.0,
                                         winding_cutoff: int = 5) -> list[Genus1ComparisonRow]:
    """
    Compare the graph-based genus-1 special case against torus_one_point.py.

    A random triple (l1, l2, l3) is sampled with each entry in [low, high].
    The torus code is then evaluated at

        L = 2 (l1 + l2 + l3),

    so that its third segment length is exactly the sampled l3.
    """
    import torus_one_point as t1

    rng = np.random.default_rng(seed)
    rows = []
    for sample_index in range(1, n_samples + 1):
        edge_lengths = tuple(
            int(x) for x in rng.integers(low, high + 1, size=3)
        )
        l1, l2, l3 = edge_lengths
        L = 2 * (l1 + l2 + l3)

        graph_value = discretized_graph_disk_one_point(
            edge_lengths,
            R,
            cp.GENUS1_F1_GRAPH_DATA,
            holonomy_cutoff=winding_cutoff,
        )
        torus_value = t1.discretized_disk_one_point(
            L,
            l1,
            l2,
            R,
            n_winding=winding_cutoff,
        )
        abs_error = abs(graph_value - torus_value)
        rel_error = abs_error / max(abs(torus_value), 1.0e-14)
        rows.append(
            Genus1ComparisonRow(
                sample_index=sample_index,
                edge_lengths=edge_lengths,
                L=L,
                R=R,
                winding_cutoff=winding_cutoff,
                graph_value=complex(graph_value),
                torus_value=complex(torus_value),
                abs_error=float(abs_error),
                rel_error=float(rel_error),
            )
        )
    return rows


def _print_result(result: Genus2OnePointResult, *, decompose: bool = False):
    print(f"topology                = {result.topology}")
    print(f"edge_lengths            = {list(result.edge_lengths)}")
    if result.period_data is not None:
        print("Omega                   =")
        print(_format_matrix(result.period_data.Omega))
    print(f"<dXdX(0)>               = {_format_complex(result.disk_one_point)}")

    if decompose:
        print(f"R                       = {result.R}")
        print(f"alpha'                  = {result.alpha_prime}")
        print(f"holonomy dimension      = {result.holonomy_dim}")
        print(f"holonomy cutoff         = {result.holonomy_cutoff}")
        print(f"<T(0)>_disc             = {_format_complex(result.disk_stress_tensor)}")
        print(f"oscillator piece        = {_format_complex(result.oscillator_disk)}")
        print(f"classical piece         = {_format_complex(result.classical_disk)}")
        print(f"Tr(C11 A'^-1)           = {_format_complex(result.compact_trace_11)}")
        print(f"theta sum               = {result.theta_sum:.12g}")
        print("T_reduced               =")
        print(_format_matrix(result.T_reduced))
        print("Xi11_reduced            =")
        print(_format_matrix(result.Xi11_reduced))
        if result.period_data is not None:
            print(f"omega_1(0)              = {_format_complex(result.period_data.omega0[0])}")
            print(f"omega_2(0)              = {_format_complex(result.period_data.omega0[1])}")
            print(f"omega_1(0)^2            = {_format_complex(result.period_data.quadratic_basis0[0])}")
            print(f"omega_1(0)omega_2(0)    = {_format_complex(result.period_data.quadratic_basis0[1])}")
            print(f"omega_2(0)^2            = {_format_complex(result.period_data.quadratic_basis0[2])}")


def _print_genus1_comparison(rows: list[Genus1ComparisonRow], *, seed: int):
    print("Genus-1 comparison against torus_one_point.py")
    print(f"seed                    = {seed}")
    if rows:
        print(f"R                       = {rows[0].R}")
        print(f"winding cutoff          = {rows[0].winding_cutoff}")
    for row in rows:
        print(
            f"sample {row.sample_index:2d}: "
            f"(l1, l2, l3) = {row.edge_lengths}, "
            f"L = {row.L}, "
            f"abs err = {row.abs_error:.6e}, "
            f"rel err = {row.rel_error:.6e}"
        )
    if rows:
        max_abs = max(row.abs_error for row in rows)
        max_rel = max(row.rel_error for row in rows)
        print(f"max abs err             = {max_abs:.6e}")
        print(f"max rel err             = {max_rel:.6e}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute the genus-2 compact-boson one-point function < :partial X partial X: >."
    )
    parser.add_argument(
        "--topology",
        type=int,
        default=1,
        help="Stored genus-2 one-face ribbon-graph topology, in 1..4.",
    )
    parser.add_argument(
        "--edge-lengths",
        type=int,
        nargs=9,
        default=list(DEFAULT_EDGE_LENGTHS),
        help="Nine positive integer edge lengths.",
    )
    parser.add_argument(
        "--R",
        type=float,
        default=1.0,
        help="Compactification radius.",
    )
    parser.add_argument(
        "--alpha-prime",
        type=float,
        default=DEFAULT_ALPHA_PRIME,
        help="Worldsheet normalization alpha'.",
    )
    parser.add_argument(
        "--nhol",
        type=int,
        default=5,
        help="Holonomy cutoff N with n_i in [-N, N] for the four-dimensional sum.",
    )
    parser.add_argument(
        "--decompose",
        action="store_true",
        help="Also print the oscillator/classical split and the reduced quadratic forms.",
    )
    parser.add_argument(
        "--skip-period-data",
        action="store_true",
        help="Skip the period-matrix and normalized one-form diagnostics.",
    )
    parser.add_argument(
        "--compare-genus1-random",
        action="store_true",
        help="Compare the genus-1 special case against torus_one_point.py on random triples.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of random genus-1 triples used by --compare-genus1-random.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260401,
        help="Random seed used by --compare-genus1-random.",
    )
    args = parser.parse_args()

    if args.compare_genus1_random:
        rows = compare_against_torus_random_samples(
            n_samples=args.samples,
            low=50,
            high=300,
            seed=args.seed,
            R=args.R,
            winding_cutoff=args.nhol,
        )
        _print_genus1_comparison(rows, seed=args.seed)
        return

    result = compute_genus2_one_point(
        edge_lengths=args.edge_lengths,
        R=args.R,
        topology=args.topology,
        holonomy_cutoff=args.nhol,
        alpha_prime=args.alpha_prime,
        include_period_data=not args.skip_period_data,
    )
    _print_result(result, decompose=args.decompose)


if __name__ == "__main__":
    main()
