from __future__ import annotations

import numpy as np
import partition_function as pf
import sympy as sp
from typing import Callable, Iterable, Sequence, Tuple
from itertools import combinations, product

import mpmath as mp
from scipy.integrate import quad


def dedekind_eta(tau: complex) -> complex:
    tau = mp.mpc(tau)
    if mp.im(tau) <= 0:
        raise ValueError("Need Im(tau) > 0.")
    q = mp.e**(2*mp.pi*1j*tau)
    eta = mp.e**(mp.pi*1j*tau/12) * mp.qp(q)  # <-- key change
    return complex(eta)

def Z(tau: complex, prec: int = 50) -> float:
    tau = complex(tau)
    eta = dedekind_eta(tau)
    return (tau.imag ** (-13.0)) * (abs(eta) ** (-48.0))

def theta3_eta_sqrt(L: int, l1: int, l2: int, n: int = 3):
    """
    Compute sqrt(theta_n(0, tau) / eta(tau)) where tau = P2/P1
    from the periods of the Strebel differential with parameters (L, l1, l2).

    n selects the Jacobi theta function (1-4).
    Uses mpmath.jtheta and the existing dedekind_eta helper.
    """
    P1, P2, P3 = periods_improved(L, l1, l2)
    tau = P2 / P1

    tau_mp = mp.mpc(tau)
    nome = mp.exp(mp.pi * 1j * tau_mp)
    theta_n = mp.jtheta(n, 0, nome)
    eta = mp.mpc(dedekind_eta(tau))

    return abs(theta_n / eta)


def make_cyl_eqn(L: int, l1: int, l2: int, *,
                      dtype=np.complex128,
                      chop_tol: float = 1e-12,
                      use_cholesky: bool = True):
    """
    Optimized factory for the CylEqn polynomial, mirroring Mathematica CylEqnFastOptimized.

    Builds the overdetermined matrix B (m x m, with m=L/2),
    forms gram = B^H B (m x m Hermitian),
    adds +1 to gram[0,0] to impose A1=1 in normal-equation form,
    solves gram * coeffs = e1,
    returns callable f(z) = Σ coeffs[n] z^n  (n=0..m-1) with Chop-like cleanup.

    Notes:
      - This uses normal equations, which are fast but less numerically stable than lstsq.
      - If you see sensitivity, set use_cholesky=False and switch to np.linalg.solve,
        or use the slower/safer np.linalg.lstsq approach.
    """
    if L % 2 != 0:
        raise ValueError("L must be even.")
    m = L // 2
    if l1 + l2 > m:
        raise ValueError("Need l1 + l2 <= L/2.")

    # nList = 1..m
    n = np.arange(1, m + 1, dtype=np.float64)

    # anglesL = piecewise mapped angles (length m)
    # Mathematica:
    # 2π*(m + l1 + 1 - k)/L, k=1..l1
    # 2π*(m + 2l1 + l2 + 1 - k)/L, k=l1+1..l1+l2
    # 2π*(L + l1 + l2 + 1 - k)/L, k=l1+l2+1..m
    k = np.arange(1, m + 1, dtype=np.float64)
    mapped = np.empty_like(k)

    if l1 > 0:
        mapped[:l1] = (m + l1 + 1) - k[:l1]
    if l2 > 0:
        mapped[l1:l1 + l2] = (m + 2 * l1 + l2 + 1) - k[l1:l1 + l2]
    if l1 + l2 < m:
        mapped[l1 + l2:] = (L + l1 + l2 + 1) - k[l1 + l2:]

    twopi_over_L = 2.0 * np.pi / L
    anglesL = twopi_over_L * mapped          # length m
    anglesR = twopi_over_L * k               # length m

    # B = exp(i outer(anglesL, n)) + exp(i outer(anglesR, n))
    # (your Mathematica uses Take[anglesR, Length[anglesL]]; lengths are both m here)
    B = np.exp(1j * anglesL[:, None] * n[None, :]) + np.exp(1j * anglesR[:, None] * n[None, :])
    B = B.astype(dtype, copy=False)

    # gram = B^H B
    gram = (B.conj().T @ B).astype(dtype, copy=False)

    # gram[[1,1]] += 1  (Mathematica indexing) => gram[0,0] += 1
    gram[0, 0] += 1.0

    # rhs = e1
    rhs = np.zeros(m, dtype=dtype)
    rhs[0] = 1.0

    # Solve gram * coeffs = rhs
    if use_cholesky:
        # Cholesky is fastest if gram is HPD (it should be in this construction).
        Lc = np.linalg.cholesky(gram)               # gram = Lc Lc^H
        y = np.linalg.solve(Lc, rhs)
        coeffs = np.linalg.solve(Lc.conj().T, y)
    else:
        coeffs = np.linalg.solve(gram, rhs)

    # Chop coeffs once
    coeffs = coeffs.copy()
    coeffs.real[np.abs(coeffs.real) < chop_tol] = 0.0
    coeffs.imag[np.abs(coeffs.imag) < chop_tol] = 0.0
    
    #FIGURE THIS OUT LATER -- COULD BE WRONG
    coeffs[-1]=0.0

    # Fast polynomial evaluation (Horner): Σ_{n=0..m-1} coeffs[n] z^n
    def f(z):
        z = np.complex128(z)
        acc = np.complex128(0.0)
        # Horner for ascending powers: reverse coefficients
        for c in coeffs[::-1]:
            acc = acc * z + c
        # Chop output
        re = 0.0 if abs(acc.real) < chop_tol else acc.real
        im = 0.0 if abs(acc.imag) < chop_tol else acc.imag
        return np.complex128(re + 1j * im)

    # Optional metadata
    f.L = L
    f.m = m
    f.l1 = l1
    f.l2 = l2
    f.coeffs = coeffs
    return f

def make_cyl_eqn_improved(L: int, l1: int, l2: int, *,
                           dtype=np.complex128,
                           chop_tol: float = 1e-12):
    """
    Python translation of Mathematica CylEqnImproved.

    Improved version of make_cyl_eqn that factors out the cube-root
    singularities analytically via an 'improvement' pre-factor, and
    uses a half-step angle shift (-π/L).

    Returns a callable f(z) -> (singular, polynomial) where the full
    cylinder equation is singular * polynomial.
    """
    if L % 2 != 0:
        raise ValueError("L must be even.")
    m = L // 2
    l3 = m - l1 - l2
    if l3 < 0:
        raise ValueError("Need l1 + l2 <= L/2.")

    twopi_over_L = 2.0 * np.pi / L
    half_step = np.pi / L

    # Special case: all three lengths equal
    if l1 == l2 == l3:
        w1 = np.exp(2j * (-twopi_over_L * l1))
        w2 = np.exp(2j * (-twopi_over_L * (l1 + l2)))
        def f_equal(z):
            s = (1 - z**2)**(-1/3) * (1 - z**2 * w1)**(-1/3) * (1 - z**2 * w2)**(-1/3)
            return (s, 1.0)
        f_equal.L, f_equal.l1, f_equal.l2, f_equal.m = L, l1, l2, m
        f_equal.coeffs = np.array([1.0])
        return f_equal

    # Parity shift: +1 if all odd, -1 if all even, 0 otherwise
    shift = ((l1 % 2) * (l2 % 2) * (l3 % 2)
             - ((l1 + 1) % 2) * ((l2 + 1) % 2) * ((l3 + 1) % 2))
    n_coeffs = m - shift
    n_list = np.arange(1, n_coeffs + 1, dtype=np.float64)

    # Angles with half-step shift
    k = np.arange(1, m + 1, dtype=np.float64)
    mapped = np.empty_like(k)
    if l1 > 0:
        mapped[:l1] = (m + l1 + 1) - k[:l1]
    if l2 > 0:
        mapped[l1:l1 + l2] = (m + 2 * l1 + l2 + 1) - k[l1:l1 + l2]
    if l1 + l2 < m:
        mapped[l1 + l2:] = (L + l1 + l2 + 1) - k[l1 + l2:]

    angles_L = twopi_over_L * mapped - half_step
    angles_R = twopi_over_L * k - half_step

    # Improvement (singular) factor
    phase1 = twopi_over_L * l1
    phase2 = twopi_over_L * (l1 + l2)

    if (l3 == 0) or (l2==0) or (l1==0):
        def improvement(angles):
            return ((1 - np.exp(2j * angles))**(-1/2) *
                    (1 - np.exp(2j * (angles - phase1)))**(-1/2))
    else:
        def improvement(angles):
            return ((1 - np.exp(2j * angles))**(-1/3) *
                    (1 - np.exp(2j * (angles - phase1)))**(-1/3) *
                    (1 - np.exp(2j * (angles - phase2)))**(-1/3))

    imp_L = improvement(angles_L)
    imp_R = improvement(angles_R[:len(angles_L)])

    # B matrix: diag(imp) @ exp(i * outer(angles, nList))
    exp_L = np.exp(1j * np.outer(angles_L, n_list))
    exp_R = np.exp(1j * np.outer(angles_R[:len(angles_L)], n_list))
    B = (imp_L[:, None] * exp_L + imp_R[:, None] * exp_R).astype(dtype)

    # Gram matrix (plain transpose, matching Mathematica Transpose[B].B)
    gram = (B.T @ B).astype(dtype)
    gram[0, 0] += 1.0

    # Solve gram * coeffs = e1
    rhs = np.zeros(n_coeffs, dtype=dtype)
    rhs[0] = 1.0
    coeffs = np.linalg.solve(gram, rhs)

    # Chop small values
    coeffs = coeffs.copy()
    coeffs.real[np.abs(coeffs.real) < chop_tol] = 0.0
    coeffs.imag[np.abs(coeffs.imag) < chop_tol] = 0.0

    # Precompute singular factor phases
    w1 = np.exp(2j * (-phase1))
    w2 = np.exp(2j * (-phase2))

    def f(z):
        z = np.complex128(z)
        if l3 == 0:
            singular = ((1 - z**2)**(-1/2) *
                        (1 - z**2 * w1)**(-1/2))
        else:
            singular = ((1 - z**2)**(-1/3) *
                        (1 - z**2 * w1)**(-1/3) *
                        (1 - z**2 * w2)**(-1/3))
        # Horner evaluation of polynomial
        acc = np.complex128(0.0)
        for c in coeffs[::-1]:
            acc = acc * z + c
        re = 0.0 if abs(acc.real) < chop_tol else acc.real
        im = 0.0 if abs(acc.imag) < chop_tol else acc.imag
        poly = np.complex128(re + 1j * im)
        return (singular, poly)

    f.L = L
    f.m = m
    f.l1 = l1
    f.l2 = l2
    f.coeffs = coeffs
    return f

def make_cyl_eqn_general_genus(ribbon_graph, ell_list, *,
                                dtype=np.complex128,
                                chop_tol: float = 1e-12):
    """
    Construct a basis of holomorphic one-forms for a general-genus ribbon graph.

    Generalizes make_cyl_eqn: the sewing constraints are read from the
    ribbon graph boundary sequence. Each edge appears twice on the disc
    boundary; the one-form must match (with orientation reversal) across
    each sewing. For genus g, the constraint matrix has a g-dimensional
    null space, giving g independent holomorphic one-forms.

    Parameters
    ----------
    ribbon_graph : tuple (edges, vertices, rotation)
        A single ribbon graph from generate_ribbon_graphs. Must have F=1.
    ell_list : list of int
        Edge lengths [l_1, ..., l_E].  L = 2 * sum(ell_list).

    Returns
    -------
    list of callable
        g functions f_i(z) = sum_{n=0}^{m-1} coeffs[n] * z^n.
        Each has .coeffs, .L, .m, .genus attributes.
    """
    from ribbon_graph_generator import (
        _trace_boundary, _get_all_face_boundaries
    )

    edges_graph, verts, rotation = ribbon_graph
    E = len(edges_graph)

    if len(ell_list) != E:
        raise ValueError(f"Need {E} edge lengths, got {len(ell_list)}")

    m = sum(ell_list)
    L = 2 * m

    # Verify F = 1
    all_faces = _get_all_face_boundaries(edges_graph, verts, rotation)
    n_faces = len(all_faces)
    if n_faces != 1:
        raise ValueError(f"Requires F=1 ribbon graph, got F={n_faces}")

    genus = (2 - len(verts) + E - n_faces) // 2

    # Trace the single face boundary: list of (from_v, to_v, edge_idx)
    boundary = _trace_boundary(edges_graph, verts, rotation)

    # Cumulative start position (in discrete units) for each half-edge
    starts = []
    pos = 0
    for _, _, eidx in boundary:
        starts.append(pos)
        pos += ell_list[eidx]
    assert pos == L, f"Boundary length {pos} != L={L}"

    # Two boundary positions for each edge
    edge_occ = {}
    for i, (_, _, eidx) in enumerate(boundary):
        edge_occ.setdefault(eidx, []).append(i)

    # ------------------------------------------------------------------
    # Build constraint matrix B  (m rows x m columns)
    #
    # Basis functions: z^n for n = 1..m, where z = exp(2*pi*i*pos/L).
    # For edge e (length l_e) with boundary occurrences at i1, i2:
    #   point k in first occurrence pairs with point (l_e+1-k) in second
    #   (orientation reversal).
    # Constraint: sum_n a_n (z_R^n + z_L^n) = 0.
    # ------------------------------------------------------------------
    n_basis = np.arange(1, m + 1, dtype=np.float64)
    twopi_over_L = 2.0 * np.pi / L

    rows = []
    for eidx in range(E):
        le = ell_list[eidx]
        i1, i2 = edge_occ[eidx]
        s1, s2 = starts[i1], starts[i2]

        for k in range(1, le + 1):
            theta_R = twopi_over_L * (s1 + k)
            theta_L = twopi_over_L * (s2 + le + 1 - k)
            row = np.exp(1j * n_basis * theta_R) + np.exp(1j * n_basis * theta_L)
            rows.append(row)

    B = np.array(rows, dtype=dtype)

    # ------------------------------------------------------------------
    # Null space via SVD  (dimension should equal genus)
    # ------------------------------------------------------------------
    U, S, Vh = np.linalg.svd(B)
    tol = max(B.shape) * S[0] * np.finfo(float).eps * 100
    null_dim = int(np.sum(S < tol))

    if null_dim != genus:
        print(f"Warning: null space dim = {null_dim}, expected genus = {genus}")
        print(f"Smallest singular values: {S[max(0,len(S)-5):]}")

    # Last null_dim rows of Vh span the null space
    null_vectors = Vh[-null_dim:] if null_dim > 0 else np.empty((0, m), dtype=dtype)

    # ------------------------------------------------------------------
    # Build one-form functions (same Horner convention as make_cyl_eqn)
    # ------------------------------------------------------------------
    forms = []
    for i in range(null_dim):
        coeffs = null_vectors[i].copy()
        coeffs.real[np.abs(coeffs.real) < chop_tol] = 0.0
        coeffs.imag[np.abs(coeffs.imag) < chop_tol] = 0.0

        def _make_f(c, _L=L, _m=m, _g=genus, _tol=chop_tol):
            def f(z):
                z = np.complex128(z)
                acc = np.complex128(0.0)
                for coeff in c[::-1]:
                    acc = acc * z + coeff
                re = 0.0 if abs(acc.real) < _tol else acc.real
                im = 0.0 if abs(acc.imag) < _tol else acc.imag
                return np.complex128(re + 1j * im)
            f.coeffs = c
            f.L = _L
            f.m = _m
            f.genus = _g
            return f

        forms.append(_make_f(coeffs))

    return forms


def make_cyl_eqn_improved_higher_genus(ribbon_graph, ell_list, *,
                                       dtype=np.complex128,
                                       chop_tol: float = 1e-12,
                                       n_forms: int | None = None,
                                       zero_col_tol: float | None = None):
    """
    Improved higher-genus analogue of make_cyl_eqn_improved.

    For an F=1 ribbon graph, trace the unique disc boundary, factor out the
    expected cubic prevertex singularities, and solve for the most-null
    regular parts of the sewn one-form constraints.

    The returned functions have the same calling convention as
    make_cyl_eqn_improved:

      f(z) -> (singular_factor, regular_polynomial)

    and the full pulled-back differential is represented by

      (singular_factor * regular_polynomial) dz.
    """
    from ribbon_graph_generator import (
        _trace_boundary, _get_all_face_boundaries,
    )

    edges_graph, verts, rotation = ribbon_graph
    E = len(edges_graph)

    if len(ell_list) != E:
        raise ValueError(f"Need {E} edge lengths, got {len(ell_list)}")
    if any(le <= 0 for le in ell_list):
        raise ValueError("make_cyl_eqn_improved_higher_genus assumes all edge lengths are positive.")

    m = int(sum(ell_list))
    L = 2 * m

    all_faces = _get_all_face_boundaries(edges_graph, verts, rotation)
    n_faces = len(all_faces)
    if n_faces != 1:
        raise ValueError(f"Requires F=1 ribbon graph, got F={n_faces}")

    genus = (2 - len(verts) + E - n_faces) // 2
    if n_forms is None:
        n_forms = genus
    if n_forms <= 0:
        return []

    boundary = _trace_boundary(edges_graph, verts, rotation)

    starts = []
    pos = 0
    for _, _, eidx in boundary:
        starts.append(pos)
        pos += ell_list[eidx]
    if pos != L:
        raise ValueError(f"Boundary length {pos} != L={L}")

    edge_occ: dict[int, list[int]] = {}
    for i, (_, _, eidx) in enumerate(boundary):
        edge_occ.setdefault(eidx, []).append(i)

    bad_edges = [eidx for eidx, occ in edge_occ.items() if len(occ) != 2]
    if bad_edges:
        raise ValueError(f"Each edge must occur exactly twice on the F=1 boundary, bad edges={bad_edges}")

    twopi_over_L = 2.0 * np.pi / L
    half_step = np.pi / L

    singular_phases = twopi_over_L * np.asarray(starts, dtype=np.float64)
    singular_points = np.exp(1j * singular_phases).astype(dtype, copy=False)

    sample_angles_R = []
    sample_angles_L = []
    sample_edges = []
    for eidx in range(E):
        le = ell_list[eidx]
        i1, i2 = edge_occ[eidx]
        s1, s2 = starts[i1], starts[i2]

        k = np.arange(1, le + 1, dtype=np.float64)
        sample_angles_R.append(twopi_over_L * (s1 + k) - half_step)
        sample_angles_L.append(twopi_over_L * (s2 + le + 1 - k) - half_step)
        sample_edges.extend([eidx] * le)

    angles_R = np.concatenate(sample_angles_R) if sample_angles_R else np.empty(0, dtype=np.float64)
    angles_L = np.concatenate(sample_angles_L) if sample_angles_L else np.empty(0, dtype=np.float64)

    def boundary_improvement(angles: np.ndarray) -> np.ndarray:
        phase_diff = np.exp(1j * (angles[:, None] - singular_phases[None, :]))
        vals = np.prod((1.0 - phase_diff) ** (-1 / 3), axis=1)
        return np.asarray(vals, dtype=dtype)

    imp_R = boundary_improvement(angles_R)
    imp_L = boundary_improvement(angles_L)

    n_basis_full = np.arange(1, m + 1, dtype=np.float64)
    exp_R = np.exp(1j * np.outer(angles_R, n_basis_full))
    exp_L = np.exp(1j * np.outer(angles_L, n_basis_full))
    B_full = (imp_R[:, None] * exp_R + imp_L[:, None] * exp_L).astype(dtype, copy=False)

    col_norms = np.linalg.norm(B_full, axis=0)
    if zero_col_tol is None:
        scale = float(np.max(col_norms)) if col_norms.size else 1.0
        zero_col_tol = max(B_full.shape) * np.finfo(float).eps * max(scale, 1.0) * 100.0
    active_mask = col_norms > zero_col_tol
    # Genus-1 special case: keep the constant regular part available even if
    # its seam column vanishes exactly. In the equal-length theta-graph case
    # this zero column is the physical solution p(z)=1, not a spurious
    # redundant mode.
    if genus == 1 and active_mask.size:
        active_mask[0] = True
    if not np.any(active_mask):
        raise ValueError("All basis columns were removed as numerically zero.")

    B = B_full[:, active_mask]

    U, S, Vh = np.linalg.svd(B, full_matrices=False)
    if Vh.shape[0] < n_forms:
        raise ValueError(f"Need at least {n_forms} singular vectors, only have {Vh.shape[0]}")

    # Use the g smallest right singular vectors only to choose a stable set of
    # coefficient pivots. The actual forms are then obtained from a Hermitian
    # block solve, which is the higher-genus analogue of the genus-1 improved
    # solver.
    null_hint = np.asarray(Vh[-n_forms:].conj().T, dtype=dtype)

    def choose_pivots(subspace: np.ndarray) -> np.ndarray:
        n_active, n_basis = subspace.shape
        if n_basis == 1:
            if np.abs(subspace[0, 0]) > chop_tol:
                return np.array([0], dtype=int)
            return np.array([int(np.argmax(np.abs(subspace[:, 0])))], dtype=int)

        chosen: list[int] = []
        remaining = list(range(n_active))
        for _ in range(n_basis):
            best_idx = None
            best_score = -1.0
            for idx in remaining:
                trial = chosen + [idx]
                block = subspace[trial, :]
                score = float(np.linalg.svd(block, compute_uv=False)[-1])
                if score > best_score + 1e-15:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                raise ValueError("Failed to choose pivot coefficients for higher-genus solver.")
            chosen.append(best_idx)
            remaining.remove(best_idx)
        return np.asarray(chosen, dtype=int)

    pivot_local = choose_pivots(null_hint)

    selector = np.zeros((n_forms, B.shape[1]), dtype=dtype)
    for row, col in enumerate(pivot_local):
        selector[row, col] = 1.0

    gram = (B.conj().T @ B).astype(dtype, copy=False)
    rhs = selector.conj().T
    coeffs_active_matrix = np.linalg.solve(gram + selector.conj().T @ selector, rhs)

    # Rotate the solved columns so that the chosen pivot coefficients are
    # exactly the identity matrix.
    pivot_values = selector @ coeffs_active_matrix
    pivot_svals = np.linalg.svd(pivot_values, compute_uv=False)
    if pivot_svals[-1] <= chop_tol:
        raise ValueError("Higher-genus normalization pivots are singular.")
    coeffs_active_matrix = coeffs_active_matrix @ np.linalg.inv(pivot_values)

    forms = []
    for i in range(n_forms):
        coeffs_active = np.asarray(coeffs_active_matrix[:, i], dtype=dtype).copy()
        coeffs_full = np.zeros(m, dtype=dtype)
        coeffs_full[active_mask] = coeffs_active

        coeffs_full.real[np.abs(coeffs_full.real) < chop_tol] = 0.0
        coeffs_full.imag[np.abs(coeffs_full.imag) < chop_tol] = 0.0

        def _make_f(c, _points=singular_points, _L=L, _m=m, _g=genus,
                    _boundary=boundary, _starts=tuple(starts),
                    _active=active_mask.copy(), _singvals=S.copy(),
                    _sample_edges=tuple(sample_edges), _pivots=tuple(int(x) for x in pivot_local),
                    _tol=chop_tol):
            def f(z):
                z = np.complex128(z)
                singular = np.prod((1.0 - z / _points) ** (-1 / 3))

                acc = np.complex128(0.0)
                for coeff in c[::-1]:
                    acc = acc * z + coeff

                re = 0.0 if abs(acc.real) < _tol else acc.real
                im = 0.0 if abs(acc.imag) < _tol else acc.imag
                poly = np.complex128(re + 1j * im)

                s_re = 0.0 if abs(singular.real) < _tol else singular.real
                s_im = 0.0 if abs(singular.imag) < _tol else singular.imag
                singular_out = np.complex128(s_re + 1j * s_im)
                return (singular_out, poly)

            f.coeffs = c
            f.L = _L
            f.m = _m
            f.genus = _g
            f.boundary = _boundary
            f.starts = _starts
            f.active_mask = _active
            f.singular_points = _points
            f.singular_values = _singvals
            f.sample_edges = _sample_edges
            f.pivot_indices = _pivots
            f.solver = "hermitian_block"
            return f

        forms.append(_make_f(coeffs_full))

    return forms


def _boundary_edge_chord_data(ribbon_graph):
    """Return boundary-word and chord-endpoint data for an F=1 ribbon graph."""
    from ribbon_graph_generator import (
        _trace_boundary, _get_all_face_boundaries,
    )

    edges_graph, verts, rotation = ribbon_graph
    all_faces = _get_all_face_boundaries(edges_graph, verts, rotation)
    if len(all_faces) != 1:
        raise ValueError(f"Requires F=1 ribbon graph, got F={len(all_faces)}")

    boundary = _trace_boundary(edges_graph, verts, rotation)
    edge_word = [step[2] for step in boundary]

    edge_positions: dict[int, list[int]] = {}
    for pos, edge_idx in enumerate(edge_word):
        edge_positions.setdefault(edge_idx, []).append(pos)

    bad_edges = [edge_idx for edge_idx, occ in edge_positions.items() if len(occ) != 2]
    if bad_edges:
        raise ValueError(
            "Each edge must occur exactly twice on the traced F=1 boundary, "
            f"bad edges={bad_edges}"
        )

    return boundary, tuple(edge_word), {
        edge_idx: tuple(occ) for edge_idx, occ in edge_positions.items()
    }


def _edge_chord_data_from_boundary(boundary, *, genus: int):
    """Build boundary-word and chord-intersection data from a traced F=1 boundary."""
    edge_word = tuple(step[2] for step in boundary)

    edge_positions: dict[int, list[int]] = {}
    for pos, edge_idx in enumerate(edge_word):
        edge_positions.setdefault(edge_idx, []).append(pos)

    bad_edges = [edge_idx for edge_idx, occ in edge_positions.items() if len(occ) != 2]
    if bad_edges:
        raise ValueError(
            "Each edge must occur exactly twice on the traced F=1 boundary, "
            f"bad edges={bad_edges}"
        )

    sorted_positions = {
        edge_idx: tuple(edge_positions[edge_idx])
        for edge_idx in sorted(edge_positions)
    }

    n_edges = len(sorted_positions)
    J = np.zeros((n_edges, n_edges), dtype=int)
    for edge_a in range(n_edges):
        if edge_a not in sorted_positions:
            raise ValueError(f"Edge {edge_a} did not appear on the traced boundary.")
        for edge_b in range(edge_a + 1, n_edges):
            if edge_b not in sorted_positions:
                raise ValueError(f"Edge {edge_b} did not appear on the traced boundary.")
            sign = _signed_boundary_chord_intersection(
                sorted_positions[edge_a], sorted_positions[edge_b]
            )
            J[edge_a, edge_b] = sign
            J[edge_b, edge_a] = -sign

    rank = int(sp.Matrix(J).rank())
    if rank != 2 * genus:
        raise ValueError(
            "Boundary-chord intersection matrix has unexpected rank: "
            f"rank={rank}, expected 2g={2 * genus}"
        )

    return {
        "boundary": tuple(boundary),
        "boundary_word": edge_word,
        "edge_positions": sorted_positions,
        "intersection_matrix": J,
        "genus": genus,
    }


def _signed_boundary_chord_intersection(pos_a, pos_b) -> int:
    """Signed crossing of two oriented boundary chords."""
    a1, a2 = sorted((int(pos_a[0]), int(pos_a[1])))
    b1, b2 = sorted((int(pos_b[0]), int(pos_b[1])))
    if a1 < b1 < a2 < b2:
        return 1
    if b1 < a1 < b2 < a2:
        return -1
    return 0


def edge_chord_intersection_matrix(ribbon_graph):
    """
    Build the signed intersection matrix of edge-chords in the disc frame.

    For an F=1 ribbon graph, each edge appears twice on the traced boundary.
    Joining those two appearances by a chord gives a candidate homology cycle
    in the sewn surface. Two such chords intersect iff their endpoints
    interleave along the boundary word.

    Returns
    -------
    dict
        {
            "boundary": traced boundary half-edges,
            "boundary_word": tuple of edge indices,
            "edge_positions": {edge_idx: (p, q)},
            "intersection_matrix": antisymmetric integer ndarray,
            "genus": genus inferred from Euler characteristic,
        }
    """
    edges_graph, verts, _ = ribbon_graph
    boundary, _, _ = _boundary_edge_chord_data(ribbon_graph)
    n_faces = 1
    genus = (2 - len(verts) + len(edges_graph) - n_faces) // 2
    return _edge_chord_data_from_boundary(boundary, genus=genus)


def _find_edge_homology_basis_from_chord_data(chord_data):
    """Internal backtracking search for a raw-edge symplectic basis."""
    J = chord_data["intersection_matrix"]
    genus = int(chord_data["genus"])
    n_edges = J.shape[0]

    used: set[int] = set()
    basis_pairs: list[dict[str, list[tuple[int, int]]]] = []

    def current_alpha_edges():
        return [pair["alpha"][0][0] for pair in basis_pairs]

    def current_beta_edges():
        return [pair["beta"][0][0] for pair in basis_pairs]

    def alpha_candidates():
        candidates = []
        for edge_idx in range(n_edges):
            if edge_idx in used:
                continue
            if any(J[edge_idx, alpha_idx] != 0 for alpha_idx in current_alpha_edges()):
                continue
            if any(J[edge_idx, beta_idx] != 0 for beta_idx in current_beta_edges()):
                continue

            partner_count = 0
            for beta_idx in range(n_edges):
                if beta_idx == edge_idx or beta_idx in used:
                    continue
                if abs(J[edge_idx, beta_idx]) != 1:
                    continue
                if any(J[beta_idx, alpha_idx] != 0 for alpha_idx in current_alpha_edges()):
                    continue
                if any(J[beta_idx, beta_idx_prev] != 0 for beta_idx_prev in current_beta_edges()):
                    continue
                partner_count += 1

            if partner_count > 0:
                candidates.append((partner_count, edge_idx))

        candidates.sort()
        return [edge_idx for _, edge_idx in candidates]

    def beta_candidates(alpha_idx: int):
        candidates = []
        for beta_idx in range(n_edges):
            if beta_idx == alpha_idx or beta_idx in used:
                continue
            if abs(J[alpha_idx, beta_idx]) != 1:
                continue
            if any(J[beta_idx, alpha_idx_prev] != 0 for alpha_idx_prev in current_alpha_edges()):
                continue
            if any(J[beta_idx, beta_idx_prev] != 0 for beta_idx_prev in current_beta_edges()):
                continue
            candidates.append(beta_idx)
        candidates.sort()
        return candidates

    def backtrack() -> bool:
        if len(basis_pairs) == genus:
            return True

        for alpha_idx in alpha_candidates():
            beta_list = beta_candidates(alpha_idx)
            for beta_idx in beta_list:
                beta_coeff = int(J[alpha_idx, beta_idx])
                basis_pairs.append({
                    "alpha": [(alpha_idx, 1)],
                    "beta": [(beta_idx, beta_coeff)],
                })
                used.add(alpha_idx)
                used.add(beta_idx)

                if backtrack():
                    return True

                used.remove(beta_idx)
                used.remove(alpha_idx)
                basis_pairs.pop()

        return False

    if not backtrack():
        raise ValueError(
            "No raw-edge symplectic basis was found from the boundary chords. "
            "This graph may require integer combinations of edge-cycles."
        )

    out = dict(chord_data)
    out["basis_pairs"] = basis_pairs
    return out


def find_edge_homology_basis(ribbon_graph):
    """
    Find paired alpha/beta cycles directly from edge-chords in the disc frame.

    The algorithm uses only the F=1 boundary word. Each edge gives an oriented
    chord joining its two boundary occurrences. It then searches for raw edges
    whose chord intersection matrix is already in symplectic form.

    Returns
    -------
    dict
        {
            "genus": g,
            "boundary_word": (...),
            "edge_positions": {edge_idx: (p, q)},
            "intersection_matrix": J,
            "basis_pairs": [
                {"alpha": [(edge_idx, coeff)], "beta": [(edge_idx, coeff)]},
                ...
            ],
        }

    Notes
    -----
    The coefficient is +/- 1 and records whether the raw chord orientation
    (first boundary occurrence to second) is used as-is or reversed. The ith
    dictionary pairs alpha_i with its dual beta_i.
    """
    chord_data = edge_chord_intersection_matrix(ribbon_graph)
    return _find_edge_homology_basis_from_chord_data(chord_data)


def _cycle_terms_to_edge_vector(cycle_terms, n_edges: int, *, cycle_name: str):
    """Convert a cycle term list [(edge_idx, coeff), ...] into an integer edge vector."""
    if cycle_terms is None:
        raise ValueError(f"{cycle_name} is missing.")
    if not isinstance(cycle_terms, (list, tuple)):
        raise ValueError(f"{cycle_name} must be a list/tuple of (edge_idx, coeff) pairs.")
    if len(cycle_terms) == 0:
        raise ValueError(f"{cycle_name} must contain at least one edge term.")

    vec = np.zeros(n_edges, dtype=int)
    for term_idx, term in enumerate(cycle_terms):
        if not isinstance(term, (list, tuple)) or len(term) != 2:
            raise ValueError(
                f"{cycle_name} term {term_idx} must be a pair (edge_idx, coeff), got {term!r}."
            )
        edge_idx, coeff = term
        try:
            edge_idx_int = int(edge_idx)
        except Exception as exc:
            raise ValueError(
                f"{cycle_name} term {term_idx} has non-integer edge index {edge_idx!r}."
            ) from exc
        if edge_idx_int != edge_idx:
            raise ValueError(
                f"{cycle_name} term {term_idx} has non-integer edge index {edge_idx!r}."
            )
        if not (0 <= edge_idx_int < n_edges):
            raise ValueError(
                f"{cycle_name} term {term_idx} uses edge {edge_idx_int}, "
                f"but valid edge indices are 0..{n_edges - 1}."
            )

        try:
            coeff_int = int(coeff)
        except Exception as exc:
            raise ValueError(
                f"{cycle_name} term {term_idx} has non-integer coefficient {coeff!r}."
            ) from exc
        if coeff_int != coeff:
            raise ValueError(
                f"{cycle_name} term {term_idx} has non-integer coefficient {coeff!r}."
            )
        if coeff_int == 0:
            continue

        vec[edge_idx_int] += coeff_int

    if not np.any(vec):
        raise ValueError(f"{cycle_name} reduces to the zero cycle.")
    return vec


def validate_edge_homology_basis(chord_data, basis_pairs, *, expected_genus: int | None = None):
    """
    Validate a custom alpha/beta basis against the chord intersection matrix.

    Parameters
    ----------
    chord_data : dict
        Output of edge_chord_intersection_matrix(...) or compatible data.
    basis_pairs : sequence
        A list of dictionaries
        {"alpha": [(edge_idx, coeff), ...], "beta": [(edge_idx, coeff), ...]}.
    expected_genus : int, optional
        If provided, require len(basis_pairs) == expected_genus.

    Returns
    -------
    list[dict]
        Canonicalized basis pairs with integer coefficients.
    """
    J = np.asarray(chord_data["intersection_matrix"], dtype=int)
    n_edges = J.shape[0]
    genus = int(chord_data["genus"])

    if not isinstance(basis_pairs, (list, tuple)):
        raise ValueError("Custom cycles must be a list/tuple of alpha/beta pair dictionaries.")
    if len(basis_pairs) == 0:
        raise ValueError("Custom cycles must contain at least one alpha/beta pair.")

    if expected_genus is not None and len(basis_pairs) != expected_genus:
        raise ValueError(
            f"Expected {expected_genus} alpha/beta pairs, got {len(basis_pairs)}."
        )
    if len(basis_pairs) > genus:
        raise ValueError(
            f"Received {len(basis_pairs)} alpha/beta pairs, but the graph genus is only {genus}."
        )

    alpha_vectors = []
    beta_vectors = []
    canonical_pairs = []
    for pair_idx, pair in enumerate(basis_pairs, start=1):
        if not isinstance(pair, dict):
            raise ValueError(f"Basis pair {pair_idx} must be a dict with 'alpha' and 'beta' entries.")
        if "alpha" not in pair or "beta" not in pair:
            raise ValueError(f"Basis pair {pair_idx} must contain both 'alpha' and 'beta'.")

        alpha_vec = _cycle_terms_to_edge_vector(
            pair["alpha"], n_edges, cycle_name=f"alpha_{pair_idx}"
        )
        beta_vec = _cycle_terms_to_edge_vector(
            pair["beta"], n_edges, cycle_name=f"beta_{pair_idx}"
        )

        alpha_vectors.append(alpha_vec)
        beta_vectors.append(beta_vec)
        canonical_pairs.append({
            "alpha": [(edge_idx, int(coeff)) for edge_idx, coeff in enumerate(alpha_vec) if coeff != 0],
            "beta": [(edge_idx, int(coeff)) for edge_idx, coeff in enumerate(beta_vec) if coeff != 0],
        })

    A = np.stack(alpha_vectors, axis=0)
    B = np.stack(beta_vectors, axis=0)
    aa = A @ J @ A.T
    bb = B @ J @ B.T
    ab = A @ J @ B.T
    target = np.eye(len(basis_pairs), dtype=int)

    if not np.array_equal(aa, np.zeros_like(aa)):
        raise ValueError(f"Custom alpha cycles are not mutually isotropic: alpha·alpha = {aa}.")
    if not np.array_equal(bb, np.zeros_like(bb)):
        raise ValueError(f"Custom beta cycles are not mutually isotropic: beta·beta = {bb}.")
    if not np.array_equal(ab, target):
        raise ValueError(
            "Custom cycles are not symplectic: expected alpha_i·beta_j = delta_ij, "
            f"got {ab}."
        )

    return canonical_pairs


def _build_starts_from_boundary(boundary, ell_list):
    """Cumulative boundary-length starts for each boundary half-edge."""
    starts = []
    pos = 0
    for _, _, edge_idx in boundary:
        starts.append(pos)
        pos += int(ell_list[edge_idx])
    L = 2 * int(sum(ell_list))
    if pos != L:
        raise ValueError(f"Boundary length {pos} != L={L}")
    return tuple(starts)


def _edge_midpoint_pairs(boundary, starts, ell_list):
    """Return midpoint representatives for each edge-cycle in the disc frame."""
    edge_occ: dict[int, list[int]] = {}
    for i, (_, _, edge_idx) in enumerate(boundary):
        edge_occ.setdefault(edge_idx, []).append(i)

    bad_edges = [edge_idx for edge_idx, occ in edge_occ.items() if len(occ) != 2]
    if bad_edges:
        raise ValueError(
            "Each edge must occur exactly twice on the traced F=1 boundary, "
            f"bad edges={bad_edges}"
        )

    L = 2 * int(sum(ell_list))
    twopi_over_L = 2.0 * np.pi / L

    pairs = {}
    for edge_idx, occ in edge_occ.items():
        i1, i2 = occ
        le = float(ell_list[edge_idx])
        theta_1 = twopi_over_L * (float(starts[i1]) + 0.5 * le)
        theta_2 = twopi_over_L * (float(starts[i2]) + 0.5 * le)
        pairs[edge_idx] = (
            np.complex128(np.exp(1j * theta_1)),
            np.complex128(np.exp(1j * theta_2)),
        )
    return pairs


def _evaluate_pulled_back_one_form(f, z):
    """Evaluate a one-form callable, supporting both raw and improved conventions."""
    val = f(z)
    if isinstance(val, tuple) and len(val) == 2:
        return np.complex128(val[0]) * np.complex128(val[1])
    return np.complex128(val)


def _make_radial_antiderivative(f, *, quad_limit: int = 200):
    """Return F(p) = integral from 0 to p along the straight radial segment."""
    try:
        test_val = f(np.complex128(0.0))
    except Exception:
        test_val = None

    if not (isinstance(test_val, tuple) and len(test_val) == 2) and hasattr(f, "coeffs"):
        return make_antiderivative_from_f(f)

    cache: dict[tuple[float, float], np.complex128] = {}

    def F(p):
        p = np.complex128(p)
        key = (float(np.real(p)), float(np.imag(p)))
        if key in cache:
            return cache[key]

        def integrand_re(t):
            return (_evaluate_pulled_back_one_form(f, t * p) * p).real

        def integrand_im(t):
            return (_evaluate_pulled_back_one_form(f, t * p) * p).imag

        re_part, _ = quad(integrand_re, 0.0, 1.0, limit=quad_limit)
        im_part, _ = quad(integrand_im, 0.0, 1.0, limit=quad_limit)
        out = np.complex128(re_part + 1j * im_part)
        cache[key] = out
        return out

    return F


def _infer_period_data_from_forms(forms):
    """Infer boundary/length data from higher-genus form metadata when available."""
    if not forms:
        raise ValueError("Need at least one holomorphic one-form.")

    ref = forms[0]
    boundary = getattr(ref, "boundary", None)
    starts = getattr(ref, "starts", None)
    sample_edges = getattr(ref, "sample_edges", None)
    genus = getattr(ref, "genus", None)
    if boundary is None or starts is None or sample_edges is None or genus is None:
        raise ValueError(
            "When ribbon_graph/ell_list are omitted, the forms must carry "
            "boundary, starts, sample_edges, and genus metadata."
        )

    from collections import Counter

    counts = Counter(int(edge_idx) for edge_idx in sample_edges)
    max_edge = max(step[2] for step in boundary)
    ell_list = [int(counts.get(edge_idx, 0)) for edge_idx in range(max_edge + 1)]
    if any(le <= 0 for le in ell_list):
        raise ValueError("Could not infer strictly positive edge lengths from form metadata.")

    chord_data = _edge_chord_data_from_boundary(tuple(boundary), genus=int(genus))
    return tuple(boundary), tuple(starts), ell_list, chord_data


def _period_matrix_from_forms(forms, *, boundary, starts, ell_list, basis_pairs):
    """Compute A/B periods and Omega once the disc-frame data are fixed."""
    edge_pairs = _edge_midpoint_pairs(boundary, starts, ell_list)
    antiderivatives = [_make_radial_antiderivative(f) for f in forms]

    g = len(forms)
    if len(basis_pairs) != g:
        raise ValueError(
            f"Need as many homology pairs as one-forms. Got {len(basis_pairs)} pairs for {g} forms."
        )

    def cycle_period(F, cycle_terms):
        total = np.complex128(0.0)
        for edge_idx, coeff in cycle_terms:
            z_start, z_end = edge_pairs[int(edge_idx)]
            total += np.complex128(coeff) * (F(z_end) - F(z_start))
        return np.complex128(total)

    A = np.zeros((g, g), dtype=np.complex128)
    B = np.zeros((g, g), dtype=np.complex128)
    for i, F in enumerate(antiderivatives):
        for j, pair in enumerate(basis_pairs):
            A[i, j] = cycle_period(F, pair["alpha"])
            B[i, j] = cycle_period(F, pair["beta"])

    Omega = np.linalg.solve(A, B)
    return A, B, Omega, edge_pairs


def period_matrix(*, forms=None, ribbon_graph=None, ell_list=None,
                  L: int | None = None, l1: int | None = None, l2: int | None = None,
                  basis_pairs=None, custom_cycles=None, return_data: bool = False):
    """
    Compute the period matrix from disc-frame edge cycles.

    Two entry points are supported:

    1. Higher genus / general case:
         period_matrix(forms=[omega_1, ..., omega_g], ribbon_graph=rg, ell_list=[...])
       If ribbon_graph/ell_list are omitted, the first form must carry the
       higher-genus metadata attached by imimproved_higher_genus.

    2. Genus-1 convenience case:
         period_matrix(L=L, l1=l1, l2=l2)
       which uses make_cyl_eqn_improved on the theta graph.

    Returns
    -------
    np.ndarray or np.complex128
        The normalized period matrix Omega. For genus 1 this is returned as a
        scalar tau. If return_data=True, return a dictionary containing Omega,
        A/B period matrices, the chosen basis pairs, and the edge midpoint data.
    """
    if basis_pairs is not None and custom_cycles is not None:
        raise ValueError("Pass either basis_pairs=... or custom_cycles=..., not both.")

    if forms is None:
        if L is None or l1 is None or l2 is None:
            raise ValueError("Provide either forms=... or the genus-1 data (L, l1, l2).")
        if L % 2 != 0:
            raise ValueError("L must be even.")

        m = L // 2
        l3 = m - l1 - l2
        if l3 < 0:
            raise ValueError("Need l1 + l2 <= L/2.")
        ribbon_graph = (
            [(1, 2), (1, 2), (1, 2)],
            [1, 2],
            {1: [0, 1, 2], 2: [0, 1, 2]},
        )
        ell_list = [int(l1), int(l2), int(l3)]
        forms = [make_cyl_eqn_improved(int(L), int(l1), int(l2))]
        chord_data = find_edge_homology_basis(ribbon_graph)
        boundary = chord_data["boundary"]
        starts = _build_starts_from_boundary(boundary, ell_list)
    else:
        if callable(forms):
            forms = [forms]
        else:
            forms = list(forms)
        if not forms:
            raise ValueError("Need at least one holomorphic one-form.")

        if ribbon_graph is not None:
            if ell_list is None:
                raise ValueError("If ribbon_graph is provided, ell_list is also required.")
            boundary, _, _ = _boundary_edge_chord_data(ribbon_graph)
            starts = _build_starts_from_boundary(boundary, ell_list)
            genus = len(forms)
            chord_data = edge_chord_intersection_matrix(ribbon_graph)
            if int(chord_data["genus"]) != genus:
                raise ValueError(
                    f"Ribbon-graph genus {chord_data['genus']} does not match number of forms {genus}."
                )
        else:
            if ell_list is not None:
                raise ValueError("If ell_list is provided, ribbon_graph must also be provided.")
            boundary, starts, ell_list, chord_data = _infer_period_data_from_forms(forms)

    requested_basis = custom_cycles if custom_cycles is not None else basis_pairs
    if requested_basis is None:
        basis_pairs = _find_edge_homology_basis_from_chord_data(chord_data)["basis_pairs"]
    else:
        if isinstance(requested_basis, dict) and "basis_pairs" in requested_basis:
            requested_basis = requested_basis["basis_pairs"]
        basis_pairs = validate_edge_homology_basis(
            chord_data,
            requested_basis,
            expected_genus=len(forms),
        )

    A, B, Omega, edge_pairs = _period_matrix_from_forms(
        forms,
        boundary=boundary,
        starts=starts,
        ell_list=ell_list,
        basis_pairs=basis_pairs,
    )

    result = {
        "Omega": Omega,
        "A_periods": A,
        "B_periods": B,
        "basis_pairs": basis_pairs,
        "edge_midpoints": edge_pairs,
    }
    if Omega.shape == (1, 1):
        result["tau"] = np.complex128(Omega[0, 0])

    if return_data:
        return result
    if Omega.shape == (1, 1):
        return np.complex128(Omega[0, 0])
    return Omega


def _theta_graph_ribbon():
    """Canonical theta-graph ribbon data used throughout the genus-1 helpers."""
    return (
        [(1, 2), (1, 2), (1, 2)],
        [1, 2],
        {1: [0, 1, 2], 2: [0, 1, 2]},
    )


def _theta_graph_lengths(L: int, l1: int, l2: int) -> tuple[int, int, int]:
    """Return (m, l1, l2, l3) consistency data for the genus-1 theta graph."""
    if L % 2 != 0:
        raise ValueError("L must be even.")
    m = L // 2
    l3 = m - l1 - l2
    if l3 < 0:
        raise ValueError("Need l1 + l2 <= L/2.")
    return m, int(l1), int(l2), int(l3)


def _theta_graph_prevertex_data(L: int, l1: int, l2: int):
    """
    Singular-point data for make_cyl_eqn_improved on the theta graph.

    This is the exact singular-point convention underlying calculate_b:
    the generic case has 6 cubic prevertices, while the degenerate
    l_i = 0 case has 4 square-root singularities.
    """
    m, l1, l2, l3 = _theta_graph_lengths(L, l1, l2)

    if l1 == 0 or l2 == 0 or l3 == 0:
        pts = np.array(
            [
                1.0,
                np.exp(2j * np.pi * l1 / L),
                -1.0,
                -np.exp(2j * np.pi * l1 / L),
            ],
            dtype=np.complex128,
        )
        return {
            "singular_points": pts,
            "regularization_power": 1 / 2,
            "boundary": None,
            "starts": None,
            "groups": None,
            "ell_list": [int(l1), int(l2), int(l3)],
        }

    ribbon_graph = _theta_graph_ribbon()
    ell_list = [int(l1), int(l2), int(l3)]
    boundary, _, _ = _boundary_edge_chord_data(ribbon_graph)
    starts = tuple(_build_starts_from_boundary(boundary, ell_list))
    singular_points = np.exp(2j * np.pi * np.asarray(starts, dtype=np.float64) / L).astype(
        np.complex128, copy=False
    )
    groups = _surface_vertex_groups_from_boundary(boundary)
    return {
        "singular_points": singular_points,
        "regularization_power": 1 / 3,
        "boundary": boundary,
        "starts": starts,
        "groups": groups,
        "ell_list": ell_list,
    }


def _ensure_form_list(forms):
    """Normalize a single callable or iterable of callables into a list."""
    if callable(forms):
        return [forms]
    return list(forms)


def _surface_vertex_groups_from_boundary(boundary):
    """Group singular boundary start-points by their surface vertex label."""
    groups = {}
    for idx, (frm, _, _) in enumerate(boundary):
        groups.setdefault(int(frm), []).append(int(idx))
    return {int(vid): tuple(int(i) for i in idxs) for vid, idxs in groups.items()}


def _make_prevertex_singular_form(
    coeffs,
    singular_points,
    *,
    regularization_power: float,
    chop_tol: float = 1e-12,
    metadata: dict | None = None,
):
    """
    Build a callable of the usual improved-form shape
        f(z) -> (singular_factor, regular_polynomial)
    from explicit singular points and polynomial coefficients.
    """
    coeffs = np.asarray(coeffs, dtype=np.complex128).copy()
    points = np.asarray(singular_points, dtype=np.complex128).copy()
    meta = {} if metadata is None else dict(metadata)

    def f(z):
        z = np.complex128(z)
        singular = np.prod((1.0 - z / points) ** (-regularization_power))
        acc = np.complex128(0.0)
        for c in coeffs[::-1]:
            acc = acc * z + c
        re = 0.0 if abs(acc.real) < chop_tol else acc.real
        im = 0.0 if abs(acc.imag) < chop_tol else acc.imag
        poly = np.complex128(re + 1j * im)

        s_re = 0.0 if abs(singular.real) < chop_tol else singular.real
        s_im = 0.0 if abs(singular.imag) < chop_tol else singular.imag
        return np.complex128(s_re + 1j * s_im), poly

    f.coeffs = coeffs
    f.singular_points = points
    f.regularization_power = float(regularization_power)
    for key, value in meta.items():
        setattr(f, key, value)
    return f


def _infer_prevertex_geometry_from_forms(forms, *, ribbon_graph=None, ell_list=None):
    """
    Infer singular-point / boundary metadata for improved forms.

    Supported inputs:
    - higher-genus forms from make_cyl_eqn_improved_higher_genus
    - genus-1 forms from make_cyl_eqn_improved via their (L,l1,l2) metadata
    """
    forms = _ensure_form_list(forms)
    if not forms:
        raise ValueError("Need at least one form.")
    ref = forms[0]

    if ribbon_graph is not None:
        if ell_list is None:
            raise ValueError("If ribbon_graph is provided, ell_list is also required.")
        boundary, _, chord_data = _boundary_edge_chord_data(ribbon_graph)
        starts = tuple(_build_starts_from_boundary(boundary, ell_list))
        singular_points = np.exp(
            2j * np.pi * np.asarray(starts, dtype=np.float64) / (2 * sum(ell_list))
        ).astype(np.complex128, copy=False)
        return {
            "boundary": boundary,
            "starts": starts,
            "groups": _surface_vertex_groups_from_boundary(boundary),
            "singular_points": singular_points,
            "regularization_power": float(getattr(ref, "regularization_power", 1 / 3)),
            "ell_list": list(int(x) for x in ell_list),
        }

    singular_points = getattr(ref, "singular_points", None)
    boundary = getattr(ref, "boundary", None)
    starts = getattr(ref, "starts", None)
    if singular_points is not None:
        if boundary is not None:
            groups = _surface_vertex_groups_from_boundary(boundary)
        else:
            groups = None
        return {
            "boundary": boundary,
            "starts": tuple(starts) if starts is not None else None,
            "groups": groups,
            "singular_points": np.asarray(singular_points, dtype=np.complex128),
            "regularization_power": float(getattr(ref, "regularization_power", 1 / 3)),
            "ell_list": None,
        }

    L = getattr(ref, "L", None)
    l1 = getattr(ref, "l1", None)
    l2 = getattr(ref, "l2", None)
    if L is None or l1 is None or l2 is None:
        raise ValueError(
            "Could not infer singular-point data from the supplied forms. "
            "Pass ribbon_graph=... and ell_list=..., or use forms carrying "
            "singular_points / (L,l1,l2) metadata."
        )
    return _theta_graph_prevertex_data(int(L), int(l1), int(l2))


def normalize_holomorphic_forms(
    forms,
    *,
    ribbon_graph=None,
    ell_list=None,
    basis_pairs=None,
    custom_cycles=None,
    return_data: bool = False,
):
    """
    A-normalize a basis of holomorphic one-forms.

    If the input forms carry polynomial/singular-point metadata, the returned
    normalized forms are rebuilt as the same improved-form type so they can be
    used by the local-nu extraction code. For a single genus-1 form, this is
    simply division by its A-period.
    """
    forms = _ensure_form_list(forms)
    pdata = period_matrix(
        forms=forms,
        ribbon_graph=ribbon_graph,
        ell_list=ell_list,
        basis_pairs=basis_pairs,
        custom_cycles=custom_cycles,
        return_data=True,
    )
    A = np.asarray(pdata["A_periods"], dtype=np.complex128)
    transform = np.linalg.inv(A)
    geom = _infer_prevertex_geometry_from_forms(forms, ribbon_graph=ribbon_graph, ell_list=ell_list)

    normalized_forms = []
    for i in range(transform.shape[0]):
        weights = transform[i, :]
        coeffs = None
        if all(getattr(f, "coeffs", None) is not None for f in forms):
            coeffs = np.zeros_like(np.asarray(forms[0].coeffs, dtype=np.complex128))
            for weight, f in zip(weights, forms):
                coeffs = coeffs + np.complex128(weight) * np.asarray(f.coeffs, dtype=np.complex128)

        metadata = {
            "normalized_by_A": True,
            "normalization_weights": np.asarray(weights, dtype=np.complex128),
            "boundary": geom["boundary"],
            "starts": geom["starts"],
            "genus": len(forms),
        }
        if geom["ell_list"] is not None:
            metadata["ell_list"] = tuple(int(x) for x in geom["ell_list"])

        if coeffs is not None:
            normalized_forms.append(
                _make_prevertex_singular_form(
                    coeffs,
                    geom["singular_points"],
                    regularization_power=geom["regularization_power"],
                    metadata=metadata,
                )
            )
            continue

        # Fallback path: preserve numerical action, but without coefficient metadata.
        def _make_linear_form(weights_row):
            def f(z):
                total = np.complex128(0.0)
                for w, base in zip(weights_row, forms):
                    total += np.complex128(w) * _evaluate_pulled_back_one_form(base, z)
                return total

            for key, value in metadata.items():
                setattr(f, key, value)
            return f

        normalized_forms.append(_make_linear_form(weights.copy()))

    if return_data:
        out = dict(pdata)
        out["normalized_forms"] = normalized_forms
        out["A_normalization_matrix"] = transform
        return out
    return normalized_forms


def _raw_local_nu_from_coeffs(coeffs, singular_points, regularization_power):
    """Exact regularized local coefficients from the singular-factor ansatz."""
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    points = np.asarray(singular_points, dtype=np.complex128)
    n_pts = points.size
    out = np.zeros(n_pts, dtype=np.complex128)

    for a, z in enumerate(points):
        prefactor = np.complex128(1.0)
        for b, w in enumerate(points):
            if b == a:
                continue
            prefactor *= (1.0 - z / w) ** (-regularization_power)
        out[a] = prefactor * _poly_eval(coeffs, z)
    return out


def calculate_nu(
    *,
    forms=None,
    ribbon_graph=None,
    ell_list=None,
    L: int | None = None,
    l1: int | None = None,
    l2: int | None = None,
    normalize_A: bool = False,
    basis_pairs=None,
    custom_cycles=None,
    return_data: bool = False,
):
    """
    Compute the regularized local coefficients nu.

    Genus-1 theta-graph usage:
        calculate_nu(L=L, l1=l1, l2=l2)
    returns the exact same list as calculate_b.

    Higher-genus usage:
        calculate_nu(forms=[omega_1,...,omega_g], ribbon_graph=rg, ell_list=[...])
    returns a matrix of shape (g, n_prevertices), where entry (I,a) is
        nu_{I,a} = lim_{z->z_a} (1-z/z_a)^p \\hat f_I(z)
    with p = 1/3 for the current cubic higher-genus construction.
    """
    if forms is None:
        if L is None or l1 is None or l2 is None:
            raise ValueError("Provide either forms=... or the genus-1 data (L, l1, l2).")
        base_form = make_cyl_eqn_improved(int(L), int(l1), int(l2))
        forms = [base_form]
        geom = _theta_graph_prevertex_data(int(L), int(l1), int(l2))
        if normalize_A:
            norm_data = normalize_holomorphic_forms(
                forms,
                ribbon_graph=_theta_graph_ribbon() if geom["boundary"] is not None else None,
                ell_list=geom["ell_list"] if geom["boundary"] is not None else None,
                basis_pairs=basis_pairs,
                custom_cycles=custom_cycles,
                return_data=True,
            )
            forms = norm_data["normalized_forms"]
            geom = _infer_prevertex_geometry_from_forms(forms)
        matrix = np.asarray(
            [
                _raw_local_nu_from_coeffs(
                    np.asarray(forms[0].coeffs, dtype=np.complex128),
                    geom["singular_points"],
                    geom["regularization_power"],
                )
            ],
            dtype=np.complex128,
        )
        if return_data:
            return {
                "nu_matrix": matrix,
                "singular_points": geom["singular_points"],
                "regularization_power": geom["regularization_power"],
                "groups": geom["groups"],
                "forms": forms,
            }
        return list(matrix[0])

    forms = _ensure_form_list(forms)
    if normalize_A:
        norm_data = normalize_holomorphic_forms(
            forms,
            ribbon_graph=ribbon_graph,
            ell_list=ell_list,
            basis_pairs=basis_pairs,
            custom_cycles=custom_cycles,
            return_data=True,
        )
        forms = norm_data["normalized_forms"]
    geom = _infer_prevertex_geometry_from_forms(forms, ribbon_graph=ribbon_graph, ell_list=ell_list)

    matrix = []
    for f in forms:
        coeffs = getattr(f, "coeffs", None)
        if coeffs is None:
            raise ValueError("calculate_nu currently requires forms carrying .coeffs metadata.")
        matrix.append(
            _raw_local_nu_from_coeffs(coeffs, geom["singular_points"], geom["regularization_power"])
        )
    matrix = np.asarray(matrix, dtype=np.complex128)

    if return_data:
        return {
            "nu_matrix": matrix,
            "singular_points": geom["singular_points"],
            "regularization_power": geom["regularization_power"],
            "groups": geom["groups"],
            "forms": forms,
        }
    return matrix


def _align_group_by_overlap(columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Phase-align a set of nu-vectors by maximizing overlap with the first column.

    This is a provisional higher-genus averaging rule. It is exact in the
    trivial g=1 case up to the historical genus-1 phase convention, and for
    g>1 it provides a stable numerical alignment when the three prevertex
    vectors at a cubic vertex are already nearly proportional.
    """
    cols = np.asarray(columns, dtype=np.complex128)
    if cols.ndim != 2:
        raise ValueError("Expected a 2D array of shape (g, n_group).")
    n_group = cols.shape[1]
    aligned = np.zeros_like(cols)
    phases = np.ones(n_group, dtype=np.complex128)
    aligned[:, 0] = cols[:, 0]

    ref = cols[:, 0]
    ref_norm = np.linalg.norm(ref)
    for j in range(1, n_group):
        vec = cols[:, j]
        if ref_norm <= 1e-15 or np.linalg.norm(vec) <= 1e-15:
            phase = np.complex128(1.0)
        else:
            overlap = np.vdot(ref, vec)
            phase = np.complex128(1.0) if abs(overlap) <= 1e-15 else np.exp(-1j * np.angle(overlap))
        phases[j] = phase
        aligned[:, j] = phase * vec
    return aligned, phases


def nu_group_proportionality_errors(nu_matrix, groups):
    """
    Measure how close the raw nu-vectors in each vertex-group are to
    being scalar multiples of one another.

    Exact holomorphic one-forms on the same surface vertex should make these
    errors small. This is therefore a useful diagnostic when testing the
    current higher-genus ansatz.
    """
    nu_matrix = np.asarray(nu_matrix, dtype=np.complex128)
    out = {}
    for key, idxs in groups.items():
        cols = nu_matrix[:, list(idxs)]
        ref = cols[:, 0]
        ref_norm_sq = float(np.vdot(ref, ref).real)
        errs = []
        for j in range(1, cols.shape[1]):
            vec = cols[:, j]
            if ref_norm_sq <= 1e-30:
                lam = np.complex128(0.0)
            else:
                lam = np.vdot(ref, vec) / ref_norm_sq
            resid = vec - lam * ref
            denom = max(np.linalg.norm(vec), np.linalg.norm(ref), 1e-30)
            errs.append(float(np.linalg.norm(resid) / denom))
        out[int(key)] = 0.0 if not errs else max(errs)
    return out


def average_nu(
    *,
    nus=None,
    forms=None,
    ribbon_graph=None,
    ell_list=None,
    L: int | None = None,
    l1: int | None = None,
    l2: int | None = None,
    normalize_A: bool = False,
    basis_pairs=None,
    custom_cycles=None,
    method: str = "auto",
    return_data: bool = False,
):
    """
    Average local nu-data.

    Historical genus-1 theta-graph convention:
        average_nu(L=..., l1=..., l2=...)
    reproduces average_b(calculate_b(...)) exactly.

    Higher genus:
        average_nu(forms=[...], ribbon_graph=..., ell_list=..., method='overlap')
    groups the raw nu-vectors by surface vertex and phase-aligns each triple
    by overlap before averaging.
    """
    if nus is None:
        if forms is None:
            if L is None or l1 is None or l2 is None:
                raise ValueError("Provide nus=..., forms=..., or the genus-1 data (L, l1, l2).")
            raw = calculate_nu(L=L, l1=l1, l2=l2, normalize_A=normalize_A, return_data=True)
            raw_matrix = np.asarray(raw["nu_matrix"], dtype=np.complex128)
            if method == "auto":
                method = "genus1_exact"
            if method == "genus1_exact":
                avg = np.asarray(
                    average_b(int(L), int(l1), int(l2), list(raw_matrix[0])),
                    dtype=np.complex128,
                )
                if return_data:
                    return {
                        "averaged_nu": avg,
                        "raw_nu": raw_matrix,
                        "method": "genus1_exact",
                    }
                return list(avg)
            nus = raw
        else:
            nus = calculate_nu(
                forms=forms,
                ribbon_graph=ribbon_graph,
                ell_list=ell_list,
                normalize_A=normalize_A,
                basis_pairs=basis_pairs,
                custom_cycles=custom_cycles,
                return_data=True,
            )

    if isinstance(nus, dict):
        nu_matrix = np.asarray(nus["nu_matrix"], dtype=np.complex128)
        groups = nus.get("groups")
    else:
        nu_matrix = np.asarray(nus, dtype=np.complex128)
        groups = None

    if method == "auto":
        method = "overlap"
    if method != "overlap":
        raise ValueError(f"Unsupported higher-genus averaging method {method!r}.")
    if groups is None:
        raise ValueError("Higher-genus averaging needs explicit surface-vertex groups.")

    averaged_cols = []
    phase_data = {}
    for key in sorted(groups):
        idxs = tuple(int(i) for i in groups[key])
        cols = nu_matrix[:, list(idxs)]
        aligned, phases = _align_group_by_overlap(cols)
        averaged_cols.append(np.mean(aligned, axis=1))
        phase_data[int(key)] = phases
    averaged = np.column_stack(averaged_cols) if averaged_cols else np.empty((nu_matrix.shape[0], 0))

    if return_data:
        return {
            "averaged_nu": averaged,
            "raw_nu": nu_matrix,
            "groups": groups,
            "group_errors": nu_group_proportionality_errors(nu_matrix, groups),
            "phases": phase_data,
            "method": method,
        }
    return averaged


def quadratic_differential_nu_vectors(averaged_nu):
    """
    Build the genus-2 local quadratic-differential vectors
        (nu_1^2, nu_1 nu_2, nu_2^2)
    from averaged one-form coefficients.
    """
    averaged_nu = np.asarray(averaged_nu, dtype=np.complex128)
    if averaged_nu.shape[0] != 2:
        raise ValueError(
            f"quadratic_differential_nu_vectors requires a genus-2 nu matrix. Got shape {averaged_nu.shape}."
        )
    nu1 = averaged_nu[0]
    nu2 = averaged_nu[1]
    return np.vstack([nu1**2, nu1 * nu2, nu2**2])


def triple_determinants_from_nu(averaged_nu):
    """
    Compute all genus-2 triple determinants
        Delta_{abc} = det(q_a, q_b, q_c)
    from averaged local nu-vectors.
    """
    qvecs = quadratic_differential_nu_vectors(averaged_nu)
    n_vertices = qvecs.shape[1]
    out = {}
    for triple in combinations(range(n_vertices), 3):
        block = qvecs[:, list(triple)]
        out[tuple(int(i) for i in triple)] = np.linalg.det(block)
    return out


def genus2_nu_factor_from_triples(
    triple_dets,
    *,
    mode: str = "mean_abs_det2",
    triple=None,
):
    """
    Build a genus-2 local nu-factor from the triple determinants Delta_{abc}.

    Parameters
    ----------
    triple_dets
        Dictionary mapping triples (a,b,c) to Delta_{abc}.
    mode
        How to choose the local factor. Supported values are:
        - 'mean_abs_det2' : arithmetic mean of |Delta|^2 over all triples
        - 'max_abs_det'   : use the relabeling-invariant triple with largest |Delta|^2
        - 'fixed_triple'  : use the explicitly supplied triple=(a,b,c)
    triple
        Explicit triple used when mode='fixed_triple'.

    Returns
    -------
    dict
        A dictionary with keys:
        - 'nu_factor'
        - 'selected_triple'
        - 'selected_value'
        - 'mode'
    """
    if not triple_dets:
        raise ValueError("Need at least one triple determinant to build a genus-2 nu-factor.")

    abs_sq = {
        tuple(int(i) for i in key): float(abs(val) ** 2)
        for key, val in triple_dets.items()
    }

    if mode == "mean_abs_det2":
        vals = np.asarray(list(abs_sq.values()), dtype=np.float64)
        return {
            "nu_factor": float(np.mean(vals)),
            "selected_triple": None,
            "selected_value": None,
            "mode": mode,
        }

    if mode == "fixed_triple":
        if triple is None:
            raise ValueError("mode='fixed_triple' requires triple=(a,b,c).")
        key = tuple(int(i) for i in triple)
        if key not in abs_sq:
            raise ValueError(
                f"Requested triple {key} is not available. "
                f"Available triples are {sorted(abs_sq)}."
            )
        return {
            "nu_factor": abs_sq[key],
            "selected_triple": key,
            "selected_value": triple_dets[key],
            "mode": mode,
        }

    if mode == "max_abs_det":
        key = max(sorted(abs_sq), key=lambda k: abs_sq[k])
        return {
            "nu_factor": abs_sq[key],
            "selected_triple": key,
            "selected_value": triple_dets[key],
            "mode": mode,
        }

    raise ValueError(f"Unsupported genus-2 nu-factor mode {mode!r}.")


def symmetric_nu_triple_factor(averaged_nu):
    """
    Symmetric genus-2 candidate replacing the genus-1 |nu|^4 factor:
    the average of |Delta_{abc}|^2 over all triples of cubic vertices.
    """
    triple_dets = triple_determinants_from_nu(averaged_nu)
    if not triple_dets:
        raise ValueError("Need at least three averaged cubic vertices to build triple determinants.")
    vals = np.asarray([abs(v) ** 2 for v in triple_dets.values()], dtype=np.float64)
    return float(np.mean(vals))


def genus2_even_characteristics():
    """Return the 10 even genus-2 theta characteristics in binary notation."""
    chars = []
    for a in product((0, 1), repeat=2):
        for b in product((0, 1), repeat=2):
            parity = (a[0] * b[0] + a[1] * b[1]) % 2
            if parity == 0:
                chars.append((tuple(int(x) for x in a), tuple(int(x) for x in b)))
    return tuple(chars)


def _theta_truncation_genus2(Omega, tol: float):
    """
    Choose a square lattice truncation for genus-2 theta constants from the
    smallest eigenvalue of Im(Omega).
    """
    Omega = np.asarray(Omega, dtype=np.complex128)
    im_omega = np.asarray(np.imag(Omega), dtype=np.float64)
    evals = np.linalg.eigvalsh(im_omega)
    lam_min = float(np.min(evals))
    if lam_min <= 0:
        raise ValueError("Need Im(Omega) positive definite to evaluate theta constants.")
    tol = max(float(tol), 1e-16)
    return max(4, int(np.ceil(np.sqrt(-np.log(tol) / (np.pi * lam_min)))) + 2)


def riemann_theta_constant_genus2(
    Omega,
    characteristic,
    *,
    nmax: int | None = None,
    tol: float = 1e-12,
):
    """
    Compute the genus-2 theta constant theta[delta](0|Omega) by direct lattice summation.

    The characteristic is passed in binary notation:
        characteristic = ((a1,a2), (b1,b2)),  ai,bi in {0,1}
    corresponding to epsilon = a/2 and delta = b/2.
    """
    Omega = np.asarray(Omega, dtype=np.complex128)
    if Omega.shape != (2, 2):
        raise ValueError("riemann_theta_constant_genus2 requires a 2x2 period matrix.")
    if nmax is None:
        nmax = _theta_truncation_genus2(Omega, tol)

    a_bits, b_bits = characteristic
    eps = 0.5 * np.asarray(a_bits, dtype=np.float64)
    delt = 0.5 * np.asarray(b_bits, dtype=np.float64)

    rng = np.arange(-nmax, nmax + 1, dtype=np.float64)
    n1, n2 = np.meshgrid(rng, rng, indexing="ij")
    v1 = n1 + eps[0]
    v2 = n2 + eps[1]

    quad_form = (
        Omega[0, 0] * v1 * v1
        + 2.0 * Omega[0, 1] * v1 * v2
        + Omega[1, 1] * v2 * v2
    )
    linear_part = 2.0 * (v1 * delt[0] + v2 * delt[1])
    terms = np.exp(1j * np.pi * (quad_form + linear_part))
    return np.complex128(np.sum(terms))


def genus2_theta_constants(Omega, *, nmax: int | None = None, tol: float = 1e-12):
    """Return the 10 even genus-2 theta constants as a dictionary."""
    return {
        char: riemann_theta_constant_genus2(Omega, char, nmax=nmax, tol=tol)
        for char in genus2_even_characteristics()
    }


def igusa_chi10_genus2(
    Omega,
    *,
    nmax: int | None = None,
    tol: float = 1e-12,
    normalization: str = "product",
):
    """
    Compute a genus-2 Igusa-cusp-form candidate from even theta constants.

    normalization='product' returns the explicit even-theta product
        prod_even theta[delta](0|Omega)^2
    which differs from the standard chi_10 by a fixed convention-dependent
    constant. This avoids hiding that overall-normalization ambiguity.

    For convenience, two common literature normalizations are also exposed:
        'igusa_2^-12' :  -2^{-12} * product
        'igusa_2^-14' :  -2^{-14} * product
    """
    thetas = genus2_theta_constants(Omega, nmax=nmax, tol=tol)
    product_form = np.complex128(1.0)
    for val in thetas.values():
        product_form *= np.complex128(val) ** 2

    if normalization == "product":
        return product_form
    if normalization == "igusa_2^-12":
        return np.complex128(-(2.0 ** -12) * product_form)
    if normalization == "igusa_2^-14":
        return np.complex128(-(2.0 ** -14) * product_form)
    raise ValueError(f"Unsupported chi_10 normalization {normalization!r}.")


def genus2_matter_bc_candidate(
    forms,
    *,
    ribbon_graph,
    ell_list,
    basis_pairs=None,
    custom_cycles=None,
    averaging_method: str = "overlap",
    chi10_normalization: str = "product",
    theta_nmax: int | None = None,
    theta_tol: float = 1e-12,
    nu_factor_mode: str = "mean_abs_det2",
    nu_factor_triple=None,
):
    """
    Package the current genus-2 matter+bc candidate comparison ingredients.

    Returns a dictionary containing:
    - A-normalized forms and period matrix data
    - raw prevertex nu-data
    - averaged surface-vertex nu-data
    - quadratic-differential triple determinants
    - chi_10(Omega) from even theta constants
    - the modular factor (det Im Omega)^(-13) |chi_10|^{-2}
    - the candidate local nu-factor based on averaged triple determinants

    By default, the current code uses the original symmetric genus-2 guess
    \mathcal N_\nu = mean_{a<b<c} |Delta_{abc}|^2.
    """
    forms = _ensure_form_list(forms)
    if len(forms) != 2:
        raise ValueError("genus2_matter_bc_candidate currently expects exactly two one-forms.")

    norm_data = normalize_holomorphic_forms(
        forms,
        ribbon_graph=ribbon_graph,
        ell_list=ell_list,
        basis_pairs=basis_pairs,
        custom_cycles=custom_cycles,
        return_data=True,
    )
    norm_forms = norm_data["normalized_forms"]
    Omega = np.asarray(norm_data["Omega"], dtype=np.complex128)

    raw = calculate_nu(forms=norm_forms, ribbon_graph=ribbon_graph, ell_list=ell_list, return_data=True)
    avg = average_nu(nus=raw, method=averaging_method, return_data=True)
    averaged_nu = np.asarray(avg["averaged_nu"], dtype=np.complex128)
    triple_dets = triple_determinants_from_nu(averaged_nu)
    nu_factor_data = genus2_nu_factor_from_triples(
        triple_dets,
        mode=nu_factor_mode,
        triple=nu_factor_triple,
    )
    nu_factor = float(nu_factor_data["nu_factor"])
    im_omega = np.asarray(np.imag(Omega), dtype=np.float64)
    im_evals = np.linalg.eigvalsh(im_omega)
    if float(np.min(im_evals)) <= 0.0:
        raise ValueError(
            "Current period matrix is not in the genus-2 Siegel upper half plane; "
            f"Im(Omega) eigenvalues are {im_evals}."
        )
    chi10 = igusa_chi10_genus2(
        Omega,
        nmax=theta_nmax,
        tol=theta_tol,
        normalization=chi10_normalization,
    )
    det_im_omega = float(np.linalg.det(im_omega))
    modular_factor = (det_im_omega ** (-13.0)) * (abs(chi10) ** (-2.0))

    out = dict(norm_data)
    out.update(
        {
            "raw_nu": raw["nu_matrix"],
            "nu_groups": raw["groups"],
            "group_errors": avg["group_errors"],
            "averaged_nu": averaged_nu,
            "quadratic_nu_vectors": quadratic_differential_nu_vectors(averaged_nu),
            "triple_determinants": triple_dets,
            "nu_factor": nu_factor,
            "nu_factor_mode": nu_factor_data["mode"],
            "selected_triple": nu_factor_data["selected_triple"],
            "selected_triple_value": nu_factor_data["selected_value"],
            "chi10": chi10,
            "det_im_omega": det_im_omega,
            "modular_factor": modular_factor,
            "candidate_factor": modular_factor * nu_factor,
        }
    )
    return out


def periods_improved(L: int, l1: int, l2: int, f=None):
    """
    Python translation of Mathematica PeriodsImproved.

    Computes three period integrals by numerically integrating
    f(z) (from make_cyl_eqn_improved) along straight-line paths
    from 0 to each of three PreCorner points on the unit circle.

    Uses scipy.integrate.quad since the integrand has (1-z²)^{-1/3}
    singularities and no closed-form antiderivative.

    Returns: (P1, P2, P3) complex128.
    """
    if L % 2 != 0:
        raise ValueError("L must be even.")
    m = L // 2
    l3 = m - l1 - l2
    if l3 < 0:
        raise ValueError("Need l1 + l2 <= L/2.")
    if f is None:
        f = make_cyl_eqn_improved(L, l1, l2)

    twopi_over_L = 2.0 * np.pi / L
    theta1 = twopi_over_L * l1
    theta2 = twopi_over_L * l2

    pre_corners = [
        np.complex128(1.0),                       # singularity of (1-z²)^{-1/3}
        np.exp(1j * (theta1 + theta2)),            # singularity of 2nd factor
        np.exp(1j * (np.pi + theta1)),             # singularity of 3rd factor
    ]

    def integrate_to(p):
        """∫₀ᵖ f(z) dz  via parameterization z(t)=t·p, t∈[0,1]."""
        def integrand_re(t):
            singular, poly = f(t * p)
            return (singular * poly * p).real
        def integrand_im(t):
            singular, poly = f(t * p)
            return (singular * poly * p).imag
        re_part, _ = quad(integrand_re, 0, 1, limit=200)
        im_part, _ = quad(integrand_im, 0, 1, limit=200)
        return np.complex128(re_part + 1j * im_part)

    corners = [integrate_to(p) for p in pre_corners]

    P1 = corners[0] - corners[2]
    P2 = corners[1] - corners[2]
    P3 = corners[0] - corners[1]
    return (P1, P2, P3)

def periods(L: int,
                         l1: int,
                         l2: int,
                         m1: int | None = None,
                         m2: int | None = None,
                         m3: int | None = None):
    """
    PeriodsFastGivenF with signature (L, l1, l2, f, m1=None, m2=None, m3=None).

    Defaults if m1/m2/m3 not provided:
        m1 = Round(l1/2),  m2 = Round(l2/2),  m3 = Round(l3/2)
    where Round matches Mathematica (ties-to-even).

    f must have attribute f.coeffs of length m=L/2 for:
        f(z) = Σ_{n=1..m} c_n z^(n-1)

    Returns: (P1, P2, P3) complex128.
    """
    if L % 2 != 0:
        raise ValueError("L must be even.")
    m = L // 2
    l3 = m - l1 - l2
    if l3 < 0:
        raise ValueError("Need l1 + l2 <= L/2 so that l3 >= 0.")

    # Mathematica Round[x] uses ties-to-even; Python round() does too.
    def round_mathematica(x: float) -> int:
        return int(round(x))

    if m1 is None:
        m1 = round_mathematica(l1 / 2)
    if m2 is None:
        m2 = round_mathematica(l2 / 2)
    if m3 is None:
        m3 = round_mathematica(l3 / 2)
    f = make_cyl_eqn(L,l1,l2)
    coeffs = getattr(f, "coeffs", None)
    if coeffs is None:
        raise ValueError("Input function f must have attribute `.coeffs` (from make_cyl_eqn_fast).")
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    if coeffs.shape[0] != m:
        raise ValueError(f"Coefficient length mismatch: expected {m}, got {coeffs.shape[0]}.")

    n = np.arange(1, m + 1, dtype=np.float64)
    anti = coeffs / n  # for F(w) = Σ (c_n/n) w^n = w * Σ anti[j] w^j

    def F(w: np.complex128) -> np.complex128:
        w = np.complex128(w)
        acc = np.complex128(0.0)
        for a in anti[::-1]:
            acc = acc * w + a
        return w * acc

    k1 = int(m1)
    k2 = int(l1 + m2)
    k3 = int(l1 + l2 + m3)

    def exp_angle(s: float) -> np.complex128:
        return np.complex128(np.exp(2j * np.pi * (s / L)))

    w1L = exp_angle(m + l1 + 1 - k1)
    w1R = exp_angle(k1)

    w2L = exp_angle(m + 2 * l1 + l2 + 1 - k2)
    w2R = exp_angle(k2)

    w3L = exp_angle(L + l1 + l2 + 1 - k3)
    w3R = exp_angle(k3)

    return (np.complex128(F(w1L) - F(w1R)),
            np.complex128(F(w2L) - F(w2R)),
            np.complex128(F(w3L) - F(w3R)))

def periods_given_f(L: int,
                         l1: int,
                         l2: int,
                         f,
                         m1: int | None = None,
                         m2: int | None = None,
                         m3: int | None = None):
    """
    PeriodsFastGivenF with signature (L, l1, l2, f, m1=None, m2=None, m3=None).

    Defaults if m1/m2/m3 not provided:
        m1 = Round(l1/2),  m2 = Round(l2/2),  m3 = Round(l3/2)
    where Round matches Mathematica (ties-to-even).

    f must have attribute f.coeffs of length m=L/2 for:
        f(z) = Σ_{n=1..m} c_n z^(n-1)

    Returns: (P1, P2, P3) complex128.
    """
    if L % 2 != 0:
        raise ValueError("L must be even.")
    m = L // 2
    l3 = m - l1 - l2
    if l3 < 0:
        raise ValueError("Need l1 + l2 <= L/2 so that l3 >= 0.")

    # Mathematica Round[x] uses ties-to-even; Python round() does too.
    def round_mathematica(x: float) -> int:
        return int(round(x))

    if m1 is None:
        m1 = round_mathematica(l1 / 2)
    if m2 is None:
        m2 = round_mathematica(l2 / 2)
    if m3 is None:
        m3 = round_mathematica(l3 / 2)

    coeffs = getattr(f, "coeffs", None)
    if coeffs is None:
        raise ValueError("Input function f must have attribute `.coeffs` (from make_cyl_eqn_fast).")
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    if coeffs.shape[0] != m:
        raise ValueError(f"Coefficient length mismatch: expected {m}, got {coeffs.shape[0]}.")

    n = np.arange(1, m + 1, dtype=np.float64)
    anti = coeffs / n  # for F(w) = Σ (c_n/n) w^n = w * Σ anti[j] w^j

    def F(w: np.complex128) -> np.complex128:
        w = np.complex128(w)
        acc = np.complex128(0.0)
        for a in anti[::-1]:
            acc = acc * w + a
        return w * acc

    k1 = int(m1)
    k2 = int(l1 + m2)
    k3 = int(l1 + l2 + m3)

    def exp_angle(s: float) -> np.complex128:
        return np.complex128(np.exp(2j * np.pi * (s / L)))

    w1L = exp_angle(m + l1 + 1 - k1)
    w1R = exp_angle(k1)

    w2L = exp_angle(m + 2 * l1 + l2 + 1 - k2)
    w2R = exp_angle(k2)

    w3L = exp_angle(L + l1 + l2 + 1 - k3)
    w3R = exp_angle(k3)

    return (np.complex128(F(w1L) - F(w1R)),
            np.complex128(F(w2L) - F(w2R)),
            np.complex128(F(w3L) - F(w3R)))

def make_antiderivative_from_f(f):
    """
    Return a callable F(w) = ∫ f(w) dw with integration constant = 0,
    where f(z) = Σ_{n=1..m} c_n z^(n-1) and f.coeffs stores c_n.
    """
    coeffs = getattr(f, "coeffs", None)
    if coeffs is None:
        raise ValueError("f must have attribute `.coeffs` (from make_cyl_eqn_fast).")
    coeffs = np.asarray(coeffs, dtype=np.complex128)

    m = coeffs.shape[0]
    n = np.arange(1, m + 1, dtype=np.float64)
    anti = coeffs / n  # anti[n-1] = c_n / n

    def F(w):
        w = np.complex128(w)
        # F(w) = Σ_{n=1..m} (c_n/n) w^n = w * Σ_{j=0..m-1} anti[j] w^j
        acc = np.complex128(0.0)
        for a in anti[::-1]:
            acc = acc * w + a
        return w * acc

    # attach for debugging
    F.anti_coeffs = anti
    return F

def compute_period_derivative(
    L: int,
    l1: int,
    l2: int,
    m1: int,
    m2: int,
    m3: int,
    eps: int = 1,
) -> np.complex128:
    """
    Integer-safe translation of Mathematica computePeriodDerivativePrime.

    periods_fast is a callable:
        periods_fast(l1, l2, l3, m1, m2, m3) -> (P1, P2, P3)
    with P1,P2,P3 complex.

    eps is an integer (usually 1), so l1±eps etc remain integers.

    Returns:
        np.complex128 value of
          1/(4 epsN^2) (tauL * conj(tauR) - conj(tauL) * tauR)
        where epsN = eps / (L/2) and L = 2(l1+l2+l3).
    """
    if eps <= 0:
        raise ValueError("eps should be a positive integer.")

    # L is preserved by the +/- eps updates
   
    m = L // 2  # = l1+l2+l3
    l3 = m-l1-l2
    if 2 * m != L:
        raise ValueError("L must be even (it is by construction).")

    # Compute perturbed periods
    periodsLp = periods(L,l1+eps, l2, m1, m2, m3)
    periodsLm = periods(L,l1 - eps, l2, m1, m2, m3)
    periodsRp = periods(L, l1, l2 + eps, m1, m2, m3)
    periodsRm = periods(L, l1, l2 - eps, m1, m2, m3)

    # tau = P2/P1 (Mathematica [[2]]/[[1]])
    tauLp = periodsLp[1] / periodsLp[0]
    tauLm = periodsLm[1] / periodsLm[0]
    tauRp = periodsRp[1] / periodsRp[0]
    tauRm = periodsRm[1] / periodsRm[0]

    tauL = tauLp - tauLm
    tauR = tauRp - tauRm

    # epsN = eps/(L/2) = eps/m
    epsN = eps / float(m)

    out = (1.0 / (4.0 * epsN * epsN)) * (tauL * np.conjugate(tauR) - np.conjugate(tauL) * tauR)
    return np.complex128(out)

def pole_intercept(f, k0: int, L: int,
                   alpha_min: int = -40,
                   alpha_max: int = 0,
                   fit_low: int = -4,
                   fit_high: int = -1,
                   step: int = 2) -> float:
    """
    Python analogue of the Mathematica:

      PoleIntercept[f_, k0_, L_] := Module[...; lm["BestFitParameters"][[1]]]

    It builds samples
        w(alpha) = exp(i*pi*(2*k0/L + 2*alpha/L + 1/L))
    and fits
        log|f(w)| = a + b * log|w - w0|
    on alpha in [fit_low, fit_high], returning the intercept a.

    Parameters
    ----------
    f : callable
        The CylEqn-style function; takes complex w and returns complex f(w).
    k0 : int
    L : int
    alpha_min, alpha_max, fit_low, fit_high : int
        Match the Mathematica defaults.
    step : int
        Step for alpha table (Mathematica used 2).

    Returns
    -------
    float
        Intercept a from least-squares fit y = a + b x.
    """
    # Base point w0 = exp(i*pi*(2*k0+1)/L)
    w0 = np.exp(1j * np.pi * (2.0 * k0 + 1.0) / L)

    xs = []
    ys = []

    for alpha in range(alpha_min, alpha_max + 1, step):
        w = np.exp(1j * np.pi * (2.0 * k0 / L + 2.0 * alpha / L + 1.0 / L))
        result = f(w)
        if isinstance(result, tuple):
            singular, poly = result
            fw = singular * poly
        else:
            fw = result

        if fit_low <= alpha <= fit_high:
            dx = abs(w - w0)
            dy = abs(fw)

            # avoid log(0)
            if dx > 0 and dy > 0:
                xs.append(np.log(dx))
                ys.append(np.log(dy))

    if len(xs) < 2:
        raise ValueError("Not enough points in fit window to perform linear regression.")

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)

    # Fit y = a + b x via least squares
    A = np.vstack([np.ones_like(x), x]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    return float(a)

def pole_intercept_average(f, L:int,l1: int, l2: int,
                           alpha_min: int = -40,
                           alpha_max: int = 0,
                           fit_low: int = -4,
                           fit_high: int = -1,
                           step: int = 2) -> float:
    """
    Python analogue of Mathematica PoleInterceptAverage[f,l1,l2,l3].

    For L = 2*(l1+l2+l3) and
      ells = {l1, l1+l2+l3, 2*l1+2*l2+l3},
    it computes three PoleIntercept-style log-log fits around the three
    corresponding boundary points and returns the mean intercept.

    Fit model (same as before):
      log|f(w)| = a + b * log|w - w0|
    using alpha in [fit_low, fit_high], returning mean(a) over i=1..3.

    Parameters mirror the Mathematica defaults.
    """
    m = L // 2  # = l1+l2+l3
    l3 = m-l1-l2
    if 2 * m != L:
        raise ValueError("L must be even (it is by construction).")
    ells = [l1, l1 + l2 + l3, 2 * l1 + 2 * l2 + l3]

    intercepts = []

    for ell in ells:
        # base point for this ell
        w0 = np.exp(1j * np.pi * (2.0 * ell + 1.0) / L)

        xs = []
        ys = []

        for alpha in range(alpha_min, alpha_max + 1, step):
            w = np.exp(1j * np.pi * (2.0 * ell / L + 2.0 * alpha / L + 1.0 / L))
            result = f(w)
            if isinstance(result, tuple):
                singular, poly = result
                fw = singular * poly
            else:
                fw = result

            if fit_low <= alpha <= fit_high:
                dx = abs(w - w0)
                dy = abs(fw)
                if dx > 0 and dy > 0:
                    xs.append(np.log(dx))
                    ys.append(np.log(dy))

        if len(xs) < 2:
            raise ValueError(f"Not enough fit points for ell={ell}.")
        x = np.asarray(xs, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)

        # least squares fit y = a + b x
        A = np.vstack([np.ones_like(x), x]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        intercepts.append(float(a))

    return float(np.mean(intercepts))

def boundary_intercepts(L: int, l1: int, l2: int,
                        m1: int, m2: int, m3: int,
                        *,
                        make_cyl_eqn,
                        pole_intercept_average):
    """
    Python analogue of Mathematica:

      BoundaryIntercepts[l1,l2,l3,m1,m2,m3] := {int1, p2/p1, p3/p1, (p2/p1 - p3/p1), p1}

    Inputs
    ------
    l1,l2,l3 : ints
    m1,m2,m3 : ints (these are the "m1,m2,m3" in Mathematica)
    make_cyl_eqn : function
        Should build the CylEqnFastOptimized-like function for this geometry.
        Expected signature: make_cyl_eqn(L:int, l1:int, l2:int) -> object f
        where f is callable f(z) and has attribute f.coeffs of length m=L/2
        for f(z)=Σ_{n=1..m} c_n z^(n-1) with A[_]->0 already enforced.
    pole_intercept_average : function
        The Python PoleInterceptAverage(f,l1,l2,l3) you implemented.

    Returns
    -------
    (int1, tau2, tau3, tau2_minus_tau3, period1) as complex/float.
    """
    m = L // 2  # = l1+l2+l3
    l3 = m-l1-l2
    if 2 * m != L:
        raise ValueError("L must be even (it is by construction).")

    # Map indices exactly like Mathematica
    k1 = int(m1)
    k2 = int(l1 + m2)
    k3 = int(l1 + l2 + m3)

    # Build cyl (A[_]->0 already) as a callable with coeffs
    f = make_cyl_eqn(L=L, l1=l1, l2=l2)

    coeffs = getattr(f, "coeffs", None)
    if coeffs is None or len(coeffs) != m:
        raise ValueError("make_cyl_eqn must return callable with .coeffs length L/2.")

    coeffs = np.asarray(coeffs, dtype=np.complex128)

    # Antiderivative coefficients for F(z)=∫f(z)dz = Σ (c_n/n) z^n  (n=1..m)
    n = np.arange(1, m + 1, dtype=np.float64)
    anti = coeffs / n  # length m

    def F(w: np.complex128) -> np.complex128:
        """
        Evaluate F(w)=Σ_{n=1..m} (c_n/n) w^n using Horner:
          Σ anti[n-1] w^n = w * Σ anti[n-1] w^(n-1)
        """
        w = np.complex128(w)
        acc = np.complex128(0.0)
        for a in anti[::-1]:
            acc = acc * w + a
        return w * acc

    def exp_angle(s: float) -> np.complex128:
        return np.complex128(np.exp(2j * np.pi * (s / L)))

    # Period endpoints exactly as in Mathematica:
    # period1 = F[exp(2πi*(L/2 + l1 - k1 + 1)/L)] - F[exp(2πi*k1/L)]
    w1L = exp_angle((L/2) + l1 - k1 + 1)
    w1R = exp_angle(k1)
    period1 = F(w1L) - F(w1R)

    # period2 = F[exp(2πi*(L/2 + 2l1 + l2 - k2 + 1)/L)] - F[exp(2πi*k2/L)]
    w2L = exp_angle((L/2) + 2*l1 + l2 - k2 + 1)
    w2R = exp_angle(k2)
    period2 = F(w2L) - F(w2R)

    # period3 = F[exp(2πi*(L + l1 + l2 - k3 + 1)/L)] - F[exp(2πi*k3/L)]
    w3L = exp_angle(L + l1 + l2 - k3 + 1)
    w3R = exp_angle(k3)
    period3 = F(w3L) - F(w3R)

    # int1 = PoleInterceptAverage[f, l1, l2, l3]
    int1 = pole_intercept_average(f, l1, l2, l3)

    # Chop analogue: zero tiny numerical noise
    def chop(z, tol=1e-12):
        z = complex(z)
        re = 0.0 if abs(z.real) < tol else z.real
        im = 0.0 if abs(z.imag) < tol else z.imag
        return re + 1j * im

    tau2 = chop(period2 / period1)
    tau3 = chop(period3 / period1)
    diff = chop(tau2 - tau3)

    return (float(int1), tau2, tau3, diff, chop(period1))

def compute_rhs(L: int,
                l1: int,
                l2: int,
                m1: int | None = None,
                m2: int | None = None,
                m3: int | None = None):
    """
    computeRHS with signature (L, l1, l2, m1=None, m2=None, m3=None),
    calling helper functions by their known global names.

    Uses:
      make_cyl_eqn
      periods_given_f
      pole_intercept_average
      compute_period_derivative_prime_int (and constructs periods_fast internally)
      Z_of_tau
    """
    if L % 2 != 0:
        raise ValueError("L must be even.")
    half = L // 2
    l3 = half - l1 - l2
    if l3 < 0:
        raise ValueError("Need l1 + l2 <= L/2 so that l3 >= 0.")

    # Mathematica Round ties-to-even; Python round() does too.
    def round_mathematica(x: float) -> int:
        return int(round(x))

    if m1 is None:
        m1 = round_mathematica(l1 / 2)
    if m2 is None:
        m2 = round_mathematica(l2 / 2)
    if m3 is None:
        m3 = round_mathematica(l3 / 2)

    # Build f
    f = make_cyl_eqn_improved(L=L, l1=l1, l2=l2)  # must be callable and have f.coeffs

    # Periods (PeriodsFastGivenF analogue)
    P1, P2, P3 = periods_improved(L=L, l1=l1, l2=l2, f=f)
    tau = P2 / P1

    # fZero = Abs[(f/P1) at z=0]
    coeffs = getattr(f, "coeffs", None)
    if coeffs is None:
        raise ValueError("make_cyl_eqn must return an object with attribute `.coeffs`.")
    f_at_0 = np.complex128(coeffs[0])  # f(0)
    fZero = np.abs(f_at_0 / P1)
    fZero_inv_sq = fZero ** (-2)

    # normalized f2(c) = f(c)/P1
    def f2(c):
        result = f(c)
        if isinstance(result, tuple):
            singular, poly = result
            return np.complex128(singular * poly / P1)
        else:
            return np.complex128(result / P1)

    jacobian = compute_period_derivative(
        L=L, l1=l1, l2=l2,
        m1=m1, m2=m2, m3=m3,
        eps=1,
    )
    Ztau = Z(tau)
    eta_tau = dedekind_eta(tau)
    im_tau = mp.im(mp.mpc(tau))
    b= pole_intercept_average(f,L,l1,l2)

    return (fZero_inv_sq, eta_tau,im_tau, b, jacobian, tau,P1)

def _poly_eval(coeffs: np.ndarray, z: complex) -> np.complex128:
    """Horner eval for p(z)=Σ coeffs[k] z^k (ascending powers)."""
    z = np.complex128(z)
    acc = np.complex128(0.0)
    for c in coeffs[::-1]:
        acc = acc * z + c
    return acc

def integrate_f2_times_z_between(
    f: Callable[[complex], complex],
    a: complex,
    b: complex,
) -> np.complex128:
    """
    Compute ∫_{a}^{b} z * f(z)^2 dz using polynomial coefficients f.coeffs.

    Requires: f.coeffs are ascending-power coeffs for f(z)=Σ c_k z^k.
    """
    coeffs = getattr(f, "coeffs", None)
    if coeffs is None:
        raise ValueError("f must have attribute `.coeffs` (ascending power coefficients).")
    c = np.asarray(coeffs, dtype=np.complex128)
    m = c.size
    if m == 0:
        return np.complex128(0.0)

    # d = coeffs of f(z)^2  (ascending powers), length 2m-1
    d = np.convolve(c, c)

    # Antiderivative of z*f(z)^2:
    # z*f^2 = Σ_{n=0..2m-2} d[n] z^(n+1)
    # Integral = Σ d[n]/(n+2) z^(n+2)
    n = np.arange(d.size, dtype=np.float64)  # n = 0..2m-2
    anti = np.zeros(d.size + 2, dtype=np.complex128)  # powers 0..(2m)
    anti[2:] = d / (n + 2.0)  # coefficient for z^(n+2)

    return _poly_eval(anti, b) - _poly_eval(anti, a)

def integrate_f2_times_z_over_pairs(
    f: Callable[[complex], complex],
    pairs: Sequence[Tuple[complex, complex]],
) -> np.ndarray:
    """Vector convenience: apply integrate_f2_times_z_between to many (a,b) pairs."""
    out = [integrate_f2_times_z_between(f, a, b) for a, b in pairs]
    return np.asarray(out, dtype=np.complex128)


def average_b(L,l1: int, l2: int, bs: list) -> list:
    """
    Python translation of Mathematica Averageb.

    Averages the b values using phase rotations that account for
    the cyclic symmetry of the three edges.

    Parameters
    ----------
    l1, l2, l3 : int
        Edge lengths.
    bs : list
        List of (at least 3) complex b values from calculate_b.

    Returns
    -------
    list of 2 complex values (if any l_i == 0) or 3 complex values (general case).
    """
    if L % 2 != 0:
        raise ValueError("L must be even.")
    l3=int(L/2)-l2-l1

    if l1 == 0 or l2 == 0 or l3 == 0:
        b1 = np.mean([bs[0], bs[1] * np.exp(-1j * np.pi / 2) * np.exp(2j * np.pi * l1 / L)])
        b2 = np.mean([bs[1], bs[0] * np.exp(1j * np.pi / 2) * np.exp(-2j * np.pi * l1 / L)])
        return [b1, b2]

    b1 = np.mean([bs[0],
                   bs[1] * np.exp(-1j * np.pi / 3) * np.exp(2j * np.pi * l1 / L),
                   bs[2] * np.exp(1j * np.pi / 3) * np.exp(-2j * np.pi * l3 / L)])
    b2 = np.mean([bs[1],
                   bs[2] * np.exp(-1j * np.pi / 3) * np.exp(2j * np.pi * l2 / L),
                   bs[0] * np.exp(1j * np.pi / 3) * np.exp(-2j * np.pi * l1 / L)])
    b3 = np.mean([bs[2],
                   bs[0] * np.exp(-1j * np.pi / 3) * np.exp(2j * np.pi * l3 / L),
                   bs[1] * np.exp(1j * np.pi / 3) * np.exp(-2j * np.pi * l2 / L)])
    return [b1, b2, b3]


def calculate_b(L: int, l1: int, l2: int, f=None) -> list:
    """
    Python translation of Mathematica Calculateb.

    Evaluates the improved cylinder equation polynomial at special boundary
    points with appropriate prefactors.

    For l3=0 (or l1=0 or l2=0): 4 points with (-1/2) exponent prefactors.
    Otherwise: 6 points with (-1/3) exponent prefactors.

    Parameters
    ----------
    L, l1, l2 : int
    f : callable, optional
        If provided, should be the callable from make_cyl_eqn_improved.
        If None, computes it internally.

    Returns
    -------
    list of 4 complex values (if any l_i == 0) or 6 complex values (general case).
    """
    if f is None:
        f = make_cyl_eqn_improved(L, l1, l2)

    m = L // 2
    l3 = m - l1 - l2
    theta1 = 2 * np.pi * l1 / L
    theta2 = 2 * np.pi * l2 / L

    if l1 == 0 or l2 == 0 or l3 == 0:
        # 4 special points on unit circle
        zs = np.array([
            1.0,
            np.exp(1j * theta1),
            -1.0,
            -np.exp(1j * theta1)
        ], dtype=np.complex128)

        bs = []
        for i in range(4):
            z_val = zs[i]
            if i < 2:
                # (1 + z/zs[i]) * (1 - z²/zs[j]²)  where j is the other index in {0,1}
                j = 1 - i
                factor = ((1 + z_val / zs[i]) *
                          (1 - z_val**2 / zs[j]**2))
            else:
                # (1 - z/zs[j]) * (1 - z²/zs[k]²)  where j=i-2, k is the other in {0,1}
                j = i - 2
                k = 1 - j
                factor = ((1 - z_val / zs[j]) *
                          (1 - z_val**2 / zs[k]**2))

            prefactor = factor ** (-1/2)

            result = f(z_val)
            if isinstance(result, tuple):
                singular, poly = result
                val = poly
            else:
                val = result

            bs.append(prefactor * val)

        return bs

    # General case: 6 special points on unit circle
    zs = np.array([
        1.0,
        np.exp(1j * theta1),
        np.exp(1j * (theta1 + theta2)),
        -1.0,
        -np.exp(1j * theta1),
        -np.exp(1j * (theta1 + theta2))
    ], dtype=np.complex128)

    bs = []
    for i in range(6):
        z_val = zs[i]

        if i < 3:
            # (1 + z/zs[i]) * (1 - z²/zs[j]²) * (1 - z²/zs[k]²)
            j = (i + 1) % 3
            k = (i + 2) % 3
            factor = ((1 + z_val / zs[i]) *
                      (1 - z_val**2 / zs[j]**2) *
                      (1 - z_val**2 / zs[k]**2))
        else:
            # (1 - z/zs[j]) * (1 - z²/zs[k1]²) * (1 - z²/zs[k2]²)
            j = i - 3
            k1 = (j + 1) % 3
            k2 = (j + 2) % 3
            factor = ((1 - z_val / zs[j]) *
                      (1 - z_val**2 / zs[k1]**2) *
                      (1 - z_val**2 / zs[k2]**2))

        prefactor = factor ** (-1/3)

        result = f(z_val)
        if isinstance(result, tuple):
            singular, poly = result
            val = poly
        else:
            val = result

        bs.append(prefactor * val)

    return bs
