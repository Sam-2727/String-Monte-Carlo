"""
compact_partition.py

Python translation of compact_partition.nb.
Computes the torus partition function Z(l_i; R) of a compact boson
via the lattice wavefunctional method, and compares with the analytic
Zcompact(R, tau).

Zero-mode handling: the zero mode sum_k X(k) = 0 is imposed, so the
Gaussian integral runs over the (L/2-1)-dimensional subspace orthogonal
to the constant mode.  A' has shape (L/2-1, L/2-1).  This matches
MatAp in compact_partition.nb (returns tempTracedMat[1;;-2, 1;;-2]).

Periods are computed via CylEqnImproved / PeriodsImproved from periods.nb
(implemented in ell_to_tau.py as make_cyl_eqn_improved / periods_improved).
"""

import sys
import os
import math
from collections import deque
from typing import Dict, Sequence
import numpy as np
import mpmath as mp

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import ell_to_tau as elt          # make_cyl_eqn_improved, periods_improved


# ── Stored ribbon-graph data (generator output, kept local for reuse) ─────────

GENUS1_F1_GRAPH_DATA = {
    "edges": (
        (1, 1, 2),
        (2, 1, 2),
        (3, 1, 2),
    ),
    "boundary": (
        (1, 2, 1),
        (2, 1, 2),
        (1, 2, 3),
        (2, 1, 1),
        (1, 2, 2),
        (2, 1, 3),
    ),
    "sewing_pairs": (
        (1, 1, 4),
        (2, 2, 5),
        (3, 3, 6),
    ),
}

GENUS2_F1_GRAPH_DATA = (
    {
        "edges": (
            (1, 1, 4),
            (2, 1, 5),
            (3, 1, 6),
            (4, 2, 4),
            (5, 2, 5),
            (6, 2, 6),
            (7, 3, 4),
            (8, 3, 5),
            (9, 3, 6),
        ),
        "boundary": (
            (1, 4, 1),
            (4, 2, 4),
            (2, 5, 5),
            (5, 3, 8),
            (3, 6, 9),
            (6, 2, 6),
            (2, 4, 4),
            (4, 3, 7),
            (3, 5, 8),
            (5, 1, 2),
            (1, 6, 3),
            (6, 3, 9),
            (3, 4, 7),
            (4, 1, 1),
            (1, 5, 2),
            (5, 2, 5),
            (2, 6, 6),
            (6, 1, 3),
        ),
        "sewing_pairs": (
            (1, 1, 14),
            (2, 10, 15),
            (3, 11, 18),
            (4, 2, 7),
            (5, 3, 16),
            (6, 6, 17),
            (7, 8, 13),
            (8, 4, 9),
            (9, 5, 12),
        ),
    },
    {
        "edges": (
            (1, 1, 2),
            (2, 2, 3),
            (3, 3, 1),
            (4, 4, 5),
            (5, 5, 6),
            (6, 6, 4),
            (7, 1, 4),
            (8, 2, 5),
            (9, 3, 6),
        ),
        "boundary": (
            (1, 2, 1),
            (2, 3, 2),
            (3, 1, 3),
            (1, 4, 7),
            (4, 5, 4),
            (5, 6, 5),
            (6, 3, 9),
            (3, 2, 2),
            (2, 5, 8),
            (5, 4, 4),
            (4, 6, 6),
            (6, 5, 5),
            (5, 2, 8),
            (2, 1, 1),
            (1, 3, 3),
            (3, 6, 9),
            (6, 4, 6),
            (4, 1, 7),
        ),
        "sewing_pairs": (
            (1, 1, 14),
            (2, 2, 8),
            (3, 3, 15),
            (4, 5, 10),
            (5, 6, 12),
            (6, 11, 17),
            (7, 4, 18),
            (8, 9, 13),
            (9, 7, 16),
        ),
    },
    {
        "edges": (
            (1, 1, 2),
            (2, 2, 3),
            (3, 3, 1),
            (4, 4, 5),
            (5, 5, 6),
            (6, 6, 4),
            (7, 1, 4),
            (8, 2, 5),
            (9, 3, 6),
        ),
        "boundary": (
            (1, 2, 1),
            (2, 3, 2),
            (3, 1, 3),
            (1, 4, 7),
            (4, 5, 4),
            (5, 2, 8),
            (2, 1, 1),
            (1, 3, 3),
            (3, 6, 9),
            (6, 5, 5),
            (5, 4, 4),
            (4, 6, 6),
            (6, 3, 9),
            (3, 2, 2),
            (2, 5, 8),
            (5, 6, 5),
            (6, 4, 6),
            (4, 1, 7),
        ),
        "sewing_pairs": (
            (1, 1, 7),
            (2, 2, 14),
            (3, 3, 8),
            (4, 5, 11),
            (5, 10, 16),
            (6, 12, 17),
            (7, 4, 18),
            (8, 6, 15),
            (9, 9, 13),
        ),
    },
    {
        "edges": (
            (1, 1, 2),
            (2, 2, 3),
            (3, 3, 1),
            (4, 4, 5),
            (5, 5, 6),
            (6, 6, 4),
            (7, 1, 4),
            (8, 2, 5),
            (9, 3, 6),
        ),
        "boundary": (
            (1, 2, 1),
            (2, 3, 2),
            (3, 1, 3),
            (1, 4, 7),
            (4, 6, 6),
            (6, 5, 5),
            (5, 2, 8),
            (2, 1, 1),
            (1, 3, 3),
            (3, 6, 9),
            (6, 4, 6),
            (4, 5, 4),
            (5, 6, 5),
            (6, 3, 9),
            (3, 2, 2),
            (2, 5, 8),
            (5, 4, 4),
            (4, 1, 7),
        ),
        "sewing_pairs": (
            (1, 1, 8),
            (2, 2, 15),
            (3, 3, 9),
            (4, 12, 17),
            (5, 6, 13),
            (6, 5, 11),
            (7, 4, 18),
            (8, 7, 16),
            (9, 10, 14),
        ),
    },
)


# ── M matrix (DirectMatN) ──────────────────────────────────────────────────────

def direct_mat_n_fast(L: int) -> np.ndarray:
    """Vectorized circulant matrix construction (no Python loop)."""
    pi = math.pi
    c0 = math.cos(pi / L)
    s0 = math.sin(pi / L)

    d = np.arange(L, dtype=np.float64)
    kernel = -s0 / (L * (c0 - np.cos(2 * pi * d / L)))

    # Circulant: Mat[i, j] = kernel[(j - i) % L]
    idx = (np.arange(L)[:, None] - np.arange(L)[None, :]) % L
    return 0.5 * kernel[idx]


# ── Analytic partition functions ───────────────────────────────────────────────

def z_boson(tau: complex) -> float:
    """Zboson[tau] = 1 / (sqrt(Im tau) * |eta(tau)|^2)"""
    tau2 = complex(tau).imag
    eta = complex(mp.eta(mp.mpc(tau)))
    return 1.0 / (np.sqrt(tau2) * abs(eta) ** 2)


def z_compact(R: float, tau: complex, N: int = 20) -> float:
    """
    Zcompact[R, tau] = Zboson[tau] * R
                     * sum_{m,n=-N..N} exp(-pi R^2 / tau2 * |m*tau - n|^2)
    """
    tau = complex(tau)
    tau2 = tau.imag
    m_vals, n_vals = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1),
                                  indexing='ij')
    m_vals = m_vals.ravel().astype(float)
    n_vals = n_vals.ravel().astype(float)
    arg = m_vals * tau - n_vals
    exponent = -np.pi * R ** 2 / tau2 * (arg.real ** 2 + arg.imag ** 2)
    return z_boson(tau) * R * float(np.sum(np.exp(exponent)))


def z_fermion(tau: complex) -> float:
    """
    Zfermion[tau] = 1/2 * sum_{a=2..4} |theta_a(0|tau) / eta(tau)|^2
    where the nome is exp(i pi tau).
    """
    tau_mp = mp.mpc(tau)
    nome = mp.exp(mp.pi * 1j * tau_mp)
    eta = mp.eta(tau_mp)
    return float(0.5 * sum(
        abs(complex(mp.jtheta(a, 0, nome)) / complex(eta)) ** 2
        for a in range(2, 5)
    ))


# ── A' matrix (MatAp) ─────────────────────────────────────────────────────────

def direct_red_traced_mat(L: int, l1: int, l2: int, Mat: np.ndarray) -> np.ndarray:
    """
    Vectorized reproduction of Mathematica DirectRedTracedMat[L_, l1_, l2_, Mat_].

    Inputs:
      - L: even integer
      - l1, l2: integers
      - Mat: numpy array of shape (L, L)

    Output:
      - numpy array of shape (L/2 - 1, L/2 - 1)
    """
    assert L % 2 == 0, "L must be even"
    assert Mat.shape == (L, L), "Mat must be LxL"
    half = L // 2

    tempTracedMat = Mat[:half, :half].copy()

    # Build index arrays for all segment indices and their maps (0-indexed)
    if l1 + l2 == half:
        segments = [(1, l1), (l1 + 1, l1 + l2 - 1), (l1 + l2 + 1, half)]
        seg_flag = 2
    else:
        segments = [(1, l1), (l1 + 1, l1 + l2), (l1 + l2 + 1, half - 1)]
        seg_flag = 3

    # Map offsets (1-indexed): map_i(x) = offset_i - x
    # map1(x) = half + l1 + 1 - x
    # map2(x) = half + 2*l1 + l2 + 1 - x
    # map3(x) = mod1(L + l1 + l2 + 1 - x, L)
    map_offsets = [half + l1 + 1, half + 2 * l1 + l2 + 1, L + l1 + l2 + 1]

    # Build K (all segment indices, 0-indexed) and mapped_K (0-indexed)
    K_list = []
    mapped_K_list = []
    for i in range(seg_flag):
        a, b = segments[i]
        if a > b:
            continue
        seg_k = np.arange(a, b + 1)  # 1-indexed
        mapped_k = map_offsets[i] - seg_k  # 1-indexed
        if i == 2:  # map3 uses mod1
            mapped_k = ((mapped_k - 1) % L) + 1
        K_list.append(seg_k - 1)        # 0-indexed
        mapped_K_list.append(mapped_k - 1)  # 0-indexed

    K = np.concatenate(K_list)
    mapped_K = np.concatenate(mapped_K_list)
    # L_idx and mapped_L are the same arrays (same segments for rows and columns)
    L_idx = K.copy()
    mapped_L = mapped_K.copy()

    n0 = half - 1  # 0-indexed n (the L/2-th index)

    # mapped_n: the map applied to n (= half, 1-indexed)
    mapped_n_1idx = map_offsets[seg_flag - 1] - half  # 1-indexed
    if seg_flag - 1 == 2:
        mapped_n_1idx = ((mapped_n_1idx - 1) % L) + 1
    mn0 = mapped_n_1idx - 1  # 0-indexed

    # ── Row updates: tempTracedMat[K, n0] += ... ──
    tempTracedMat[K, n0] += (
        Mat[mapped_K, n0]
        + Mat[mapped_K, mn0]
        + Mat[K, mn0]
    )

    # ── Column updates: tempTracedMat[n0, L_idx] += ... ──
    tempTracedMat[n0, L_idx] += (
        Mat[mn0, L_idx]
        + Mat[mn0, mapped_L]
        + Mat[n0, mapped_L]
    )

    # ── Corner update ──
    tempTracedMat[n0, n0] += (
        Mat[mn0, n0] + Mat[mn0, mn0] + Mat[n0, mn0]
    )

    # ── Main block: vectorized over all (k, l) pairs ──
    # The n-th row/col values are now frozen (n0 is outside segment ranges)
    row_n = tempTracedMat[K, n0]       # shape (len(K),)
    col_n = tempTracedMat[n0, L_idx]   # shape (len(L_idx),)
    corner = tempTracedMat[n0, n0]     # scalar

    tempTracedMat[np.ix_(K, L_idx)] += (
        Mat[np.ix_(mapped_K, L_idx)]
        + Mat[np.ix_(mapped_K, mapped_L)]
        + Mat[np.ix_(K, mapped_L)]
        + (-row_n[:, None] - col_n[None, :] + corner)
    )

    return tempTracedMat[:half - 1, :half - 1]


# ── Segment structure ──────────────────────────────────────────────────────────

def _segment_maps(L: int, l1: int, l2: int):
    """
    Returns (segments, map_0idx) where:
      segments   = [(1,l1), (l1+1,l1+l2), (l1+l2+1,L//2)]  1-indexed inclusive
      map_0idx   = 0-indexed array of length L//2:
                   map_0idx[k-1] = (primed partner of k) - 1  for k=1..L/2.

    The three segment maps are (1-indexed):
      map1(k) = L/2 + l1 + 1 - k          for k in seg 1
      map2(k) = L/2 + 2*l1 + l2 + 1 - k   for k in seg 2
      map3(k) = Mod[L + l1 + l2 + 1 - k, L, 1]  for k in seg 3
    """
    half = L // 2
    l3 = half - l1 - l2
    segments = [(1, l1), (l1 + 1, l1 + l2), (l1 + l2 + 1, half)]

    k = np.arange(1, half + 1, dtype=int)
    m = np.empty(half, dtype=int)
    if l1 > 0:
        m[:l1]      = half + l1 + 1 - k[:l1]
    if l2 > 0:
        m[l1:l1+l2] = half + 2*l1 + l2 + 1 - k[l1:l1+l2]
    if l3 > 0:
        raw          = L + l1 + l2 + 1 - k[l1+l2:]
        m[l1+l2:]   = ((raw - 1) % L) + 1   # Mod[., L, 1]

    return segments, m - 1   # 0-indexed


# ── Matrix W ──────────────────────────────────────────────────────────────────

def mat_w(L: int, l1: int, l2: int, Mat: np.ndarray) -> np.ndarray:
    """
    Python translation of MatW[L_, l1_, l2_, Mat_].

    Mat : (L, L) float array  (DirectMatN / 2)
    Returns W of shape (L/2-1, L/2):

      W[k, l] = Mat[k, map(l)] + Mat[map(k), map(l)]
              - Mat[L/2-1, map(l)] - Mat[map3(L/2), map(l)]

    where all indices are 0-based and map(.) is the primed-partner index.

    Corresponds to Mathematica return value tempW[[1;;-2, 1;;-1]],
    i.e. rows 1..L/2-1 (all except the last) and all L/2 columns.
    """
    half = L // 2
    l3   = half - l1 - l2
    _, m = _segment_maps(L, l1, l2)   # m[k-1] = 0-indexed primed partner of k

    col = m                            # mapped column index for l = 0..half-1

    # map3 applied to L/2 (1-indexed half → 0-indexed):
    #   Mod[L + l1 + l2 + 1 - half, L, 1] - 1 = (L - l3) % L
    map3_half = (L - l3) % L

    k0 = np.arange(half)

    # half-1 is the 0-indexed position of 1-indexed L/2.
    # It appears in two roles below:
    #   Mat[half-1, col]    → tempMat[[L/2, map_j(l)]] in Mathematica (row L/2)
    #   return [:half-1, :] → [[1;;-2, 1;;-1]] in Mathematica (all L/2 cols,
    #                          first L/2-1 rows; last row dropped because it
    #                          encodes the zero-mode constraint sum_k X(k) = 0)
    tempW = (Mat[k0[:, None],  col[None, :]]    # Mat[k,         map(l)]
           + Mat[m[:, None],   col[None, :]]    # Mat[map(k),    map(l)]
           - Mat[half - 1,     col][None, :]    # Mat[L/2,       map(l)]  (1-indexed)
           - Mat[map3_half,    col][None, :])   # Mat[map3(L/2), map(l)]

    # Mathematica returns tempW[[1;;-2, 1;;-1]]: all L/2 columns, first L/2-1 rows.
    return tempW[:half - 1, :]


# ── Matrix T, first part ───────────────────────────────────────────────────────

def mat_t_first_part(L: int, l1: int, l2: int, Mat: np.ndarray) -> np.ndarray:
    """
    Python translation of MatTFirstPart.

    T1[i, j] = sum_{m in seg_i, n in seg_j} Mat[map(m), map(n)]

    Returns (3, 3) float array.
    """
    segments, m = _segment_maps(L, l1, l2)
    T1 = np.zeros((3, 3))
    for i, (si_s, si_e) in enumerate(segments):
        rows = m[si_s - 1 : si_e]          # mapped row indices for seg i
        for j, (sj_s, sj_e) in enumerate(segments):
            cols = m[sj_s - 1 : sj_e]      # mapped col indices for seg j
            T1[i, j] = Mat[np.ix_(rows, cols)].sum()
    return T1


# ── Matrix T, second part ─────────────────────────────────────────────────────

def mat_t_second_part(L: int, l1: int, l2: int,
                      W: np.ndarray, Aprime: np.ndarray) -> np.ndarray:
    """
    Python translation of MatTSecondPart.

    u[i]   = sum of W columns in segment i,  shape (L/2-1,)
    w[i]   = A'^{-1} @ u[i]
    T2[i,j] = u[i] . w[j]

    W      : (L/2-1, L/2) array
    Aprime : (L/2-1, L/2-1) array
    Returns (3, 3) float array.
    """
    segments, _ = _segment_maps(L, l1, l2)
    n_red = W.shape[0]   # L/2 - 1

    u = np.zeros((3, n_red))
    for i, (si_s, si_e) in enumerate(segments):
        u[i] = W[:, si_s - 1 : si_e].sum(axis=1)

    # Solve A' @ w[i] = u[i] for all three right-hand sides simultaneously.
    # np.linalg.solve(A, B) with B=(n,3) solves A @ X = B, returning X=(n,3).
    w = np.linalg.solve(Aprime, u.T).T   # shape (3, n_red)

    return u @ w.T   # (3, n_red) @ (n_red, 3) → (3, 3)


# ── Symmetrise / T' / partition function ──────────────────────────────────────

def symm(T: np.ndarray) -> np.ndarray:
    """Symmetrize: (T + T^T) / 2"""
    return 0.5 * (T + T.T)


def mat_t_prime(T: np.ndarray) -> np.ndarray:
    """
    Python translation of MatTprime.

    sgn = [-1, 1]  (Mathematica 1-indexed as {-1, 1})
    T'[i, j] = T[i,j] + sgn[i]*T[2,j] + sgn[j]*T[i,2] + sgn[i]*sgn[j]*T[2,2]
    for i, j in {0, 1}  (0-indexed; Mathematica index 3 → Python index 2).

    Explicit values (0-indexed):
      T'[0,0] = T[0,0] - T[2,0] - T[0,2] + T[2,2]
      T'[0,1] = T[0,1] - T[2,1] + T[0,2] - T[2,2]
      T'[1,0] = T[1,0] + T[2,0] - T[1,2] - T[2,2]
      T'[1,1] = T[1,1] + T[2,1] + T[1,2] + T[2,2]
    """
    sgn = np.array([-1.0, 1.0])
    Tp = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            Tp[i, j] = (T[i, j]
                        + sgn[i] * T[2, j]
                        + sgn[j] * T[i, 2]
                        + sgn[i] * sgn[j] * T[2, 2])
    return Tp


def t_duality(R: float, Tp: np.ndarray, N: int = 20) -> float:
    """
    Python translation of Tduality[R_, T_] from compact_partition.nb.

    Tduality[R, T] =
        R   * sum_{s1,s2} exp(-4π R²   * [s1,s2] T [s1,s2]^T)
      - 1/R * sum_{s1,s2} exp(-4π/R²   * [s1,s2] T [s1,s2]^T)

    For a T-dual theory this should vanish.
    """
    sv = np.arange(-N, N + 1, dtype=float)
    s1, s2 = np.meshgrid(sv, sv)
    s1, s2 = s1.ravel(), s2.ravel()

    quad = (s1 ** 2 * Tp[0, 0]
            + s1 * s2 * Tp[0, 1]
            + s2 * s1 * Tp[1, 0]
            + s2 ** 2 * Tp[1, 1])

    term1 = R       * float(np.sum(np.exp(-4.0 * np.pi * R ** 2       * quad)))
    term2 = (1.0/R) * float(np.sum(np.exp(-4.0 * np.pi * (1.0/R**2)  * quad)))
    return term1 - term2


def partition_function_z(Aprime: np.ndarray, R: float, Tp: np.ndarray,
                          N: int = 20) -> float:
    """
    Python translation of PartitionFunctionZ.

    Z = R * det(A')^{-1/2}
          * sum_{s1,s2=-N..N} exp(-2 pi R^2 [s1,s2] Tp [s1,s2]^T)

    Numerically, the Tp produced by the lattice construction matches the
    torus zero-mode metric G(tau) with 2*Tp ~= G(tau), not 4*Tp ~= G(tau).
    Using the 2*pi normalization restores the expected R <-> 1/R T-duality
    of the theta sum.
    """
    _, logdet = np.linalg.slogdet(Aprime)
    det_factor = np.exp(-0.5 * logdet)

    sv = np.arange(-N, N + 1, dtype=float)
    s1, s2 = np.meshgrid(sv, sv)
    s1, s2 = s1.ravel(), s2.ravel()

    exponent = -4.0 * np.pi * R ** 2 * (
        s1 ** 2 * Tp[0, 0]
        + s1 * s2 * Tp[0, 1]
        + s2 * s1 * Tp[1, 0]
        + s2 ** 2 * Tp[1, 1]
    )
    theta_sum = np.sum(np.exp(exponent))
    return float(R * det_factor * theta_sum)


# ── Graph-based higher-genus compact boson ────────────────────────────────────

def get_stored_genus2_graph(topology: int) -> dict:
    """Return one of the four stored genus-2, one-face ribbon graphs."""
    if not 1 <= topology <= len(GENUS2_F1_GRAPH_DATA):
        raise ValueError(
            f"topology must be in 1..{len(GENUS2_F1_GRAPH_DATA)}, got {topology}"
        )
    return GENUS2_F1_GRAPH_DATA[topology - 1]


def _validate_edge_lengths(edge_lengths: Sequence[int], n_edges: int) -> np.ndarray:
    """Return positive integer edge lengths as a numpy array."""
    if len(edge_lengths) != n_edges:
        raise ValueError(f"Expected {n_edges} edge lengths, got {len(edge_lengths)}")
    arr = np.asarray(edge_lengths, dtype=int)
    if np.any(arr <= 0):
        raise ValueError("All edge lengths must be positive integers")
    return arr


def _gluing_data(edge_lengths: Sequence[int], graph_data: dict) -> dict:
    """
    Build the global point/segment bookkeeping from stored F=1 ribbon data.

    The stored boundary and sewing data is sufficient to reconstruct:
      - the global prime map k -> k'
      - the independent-point set K (all points on first segments)
      - the oriented edge list used for the shift-lattice cycle basis
    """
    boundary = tuple(graph_data["boundary"])
    sewing_pairs = tuple(graph_data["sewing_pairs"])
    n_edges = len(graph_data["edges"])
    n_segments = len(boundary)
    edge_lengths = _validate_edge_lengths(edge_lengths, n_edges)

    segment_to_edge = np.zeros(n_segments + 1, dtype=int)
    first_seg = np.zeros(n_edges + 1, dtype=int)
    second_seg = np.zeros(n_edges + 1, dtype=int)

    for edge, s1, s2 in sewing_pairs:
        lo, hi = sorted((s1, s2))
        segment_to_edge[lo] = edge
        segment_to_edge[hi] = edge
        first_seg[edge] = lo
        second_seg[edge] = hi

    if np.any(segment_to_edge[1:] == 0):
        raise ValueError("Stored ribbon data has incomplete segment-to-edge coverage")

    segment_lengths = np.zeros(n_segments + 1, dtype=int)
    for seg in range(1, n_segments + 1):
        segment_lengths[seg] = edge_lengths[segment_to_edge[seg] - 1]

    segment_start = np.zeros(n_segments + 1, dtype=int)
    segment_end = np.zeros(n_segments + 1, dtype=int)
    cursor = 1
    for seg in range(1, n_segments + 1):
        segment_start[seg] = cursor
        cursor += segment_lengths[seg]
        segment_end[seg] = cursor - 1
    L = cursor - 1

    prime = np.full(L, -1, dtype=int)
    edge_points = []
    oriented_edges = []
    for edge in range(1, n_edges + 1):
        seg1 = first_seg[edge]
        seg2 = second_seg[edge]
        ell = edge_lengths[edge - 1]
        local = np.arange(ell, dtype=int)
        pts1 = (segment_start[seg1] - 1) + local
        pts2 = (segment_end[seg2] - 1) - local
        prime[pts1] = pts2
        prime[pts2] = pts1
        edge_points.append(pts1)

        frm, to, edge_label = boundary[seg1 - 1]
        if edge_label != edge:
            raise ValueError(
                f"Boundary/edge mismatch on edge {edge}: saw boundary edge {edge_label}"
            )
        oriented_edges.append((frm, to))

    if np.any(prime < 0):
        raise ValueError("Prime map construction failed: some boundary points were unpaired")

    K = np.sort(np.concatenate(edge_points))
    if K[0] != 0:
        raise ValueError("Expected point 1 to belong to K after sorting")

    K_lookup = np.full(L, -1, dtype=int)
    K_lookup[K] = np.arange(K.size)
    edge_K_cols = [K_lookup[pts] for pts in edge_points]

    return {
        "L": L,
        "K": K,
        "prime": prime,
        "edge_points": edge_points,
        "edge_K_cols": edge_K_cols,
        "oriented_edges": oriented_edges,
    }


def _tree_path(tree_adj: Dict[int, list], start: int, end: int):
    """Return the unique tree path start -> end as (u, v, edge_idx) steps."""
    prev = {start: (None, None)}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == end:
            break
        for nxt, edge_idx in tree_adj[node]:
            if nxt in prev:
                continue
            prev[nxt] = (node, edge_idx)
            queue.append(nxt)

    if end not in prev:
        raise ValueError(f"No tree path between vertices {start} and {end}")

    steps = []
    cur = end
    while cur != start:
        prv, edge_idx = prev[cur]
        steps.append((prv, cur, edge_idx))
        cur = prv
    steps.reverse()
    return steps


def _fundamental_cycle_basis(oriented_edges) -> np.ndarray:
    """
    Integer cycle basis for the shift lattice.

    Each column is a fundamental cycle vector in Z^E, so s = B n automatically
    solves the vertex constraints and n runs over the independent holonomies.
    """
    n_edges = len(oriented_edges)
    vertices = sorted({v for edge in oriented_edges for v in edge})
    parent = {v: v for v in vertices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            return True
        return False

    tree_edges = []
    non_tree_edges = []
    for edge_idx, (u, v) in enumerate(oriented_edges):
        if union(u, v):
            tree_edges.append(edge_idx)
        else:
            non_tree_edges.append(edge_idx)

    tree_adj = {v: [] for v in vertices}
    for edge_idx in tree_edges:
        u, v = oriented_edges[edge_idx]
        tree_adj[u].append((v, edge_idx))
        tree_adj[v].append((u, edge_idx))

    cycles = []
    for edge_idx in non_tree_edges:
        u, v = oriented_edges[edge_idx]
        cycle = np.zeros(n_edges, dtype=int)
        cycle[edge_idx] = 1
        for a, b, tree_edge_idx in _tree_path(tree_adj, v, u):
            src, dst = oriented_edges[tree_edge_idx]
            cycle[tree_edge_idx] = 1 if (a, b) == (src, dst) else -1
        cycles.append(cycle)

    if not cycles:
        return np.zeros((n_edges, 0), dtype=int)
    return np.column_stack(cycles)


def compact_boson_graph_geometry(edge_lengths: Sequence[int], graph_data: dict,
                                 Mat: np.ndarray | None = None) -> dict:
    """
    Build A', W, the full edge-space T, and its reduced holonomy form.

    This is the higher-genus generalization of the torus-specific MatAp / MatW /
    MatT construction, with the gluing encoded by stored ribbon-graph data.
    """
    gluing = _gluing_data(edge_lengths, graph_data)
    L = gluing["L"]
    K = gluing["K"]
    prime = gluing["prime"]
    edge_points = gluing["edge_points"]
    edge_K_cols = gluing["edge_K_cols"]
    oriented_edges = gluing["oriented_edges"]

    if Mat is None:
        Mat = direct_mat_n_fast(L)
    elif Mat.shape != (L, L):
        raise ValueError(f"Mat must have shape {(L, L)}, got {Mat.shape}")

    Kp = prime[K]
    A = (
        Mat[np.ix_(K, K)]
        + Mat[np.ix_(Kp, K)]
        + Mat[np.ix_(K, Kp)]
        + Mat[np.ix_(Kp, Kp)]
    )
    Aprime = A[1:, 1:] - A[1:, [0]] - A[[0], 1:] + A[0, 0]
    Aprime = 0.5 * (Aprime + Aprime.T)

    Kred = K[1:]
    Kprimed = prime[K]
    one = K[0]
    one_prime = prime[one]
    W = (
        Mat[np.ix_(Kred, Kprimed)]
        + Mat[np.ix_(prime[Kred], Kprimed)]
        - Mat[one, Kprimed][None, :]
        - Mat[one_prime, Kprimed][None, :]
    )

    n_edges = len(edge_points)
    T1 = np.zeros((n_edges, n_edges), dtype=float)
    primed_edge_points = [prime[pts] for pts in edge_points]
    for a in range(n_edges):
        rows = primed_edge_points[a]
        for b in range(n_edges):
            cols = primed_edge_points[b]
            T1[a, b] = Mat[np.ix_(rows, cols)].sum()

    U = np.zeros((K.size - 1, n_edges), dtype=float)
    for edge_idx, cols in enumerate(edge_K_cols):
        U[:, edge_idx] = W[:, cols].sum(axis=1)

    solved = np.linalg.solve(Aprime, U)
    T2 = U.T @ solved
    T = 0.5 * ((T1 - T2) + (T1 - T2).T)

    cycle_basis = _fundamental_cycle_basis(oriented_edges)
    T_reduced = cycle_basis.T @ T @ cycle_basis
    T_reduced = 0.5 * (T_reduced + T_reduced.T)

    return {
        "L": L,
        "A_prime": Aprime,
        "W": W,
        "T_edge": T,
        "cycle_basis": cycle_basis,
        "T_reduced": T_reduced,
    }


def theta_sum_reduced(T_reduced: np.ndarray, R: float, N: int = 6) -> float:
    """Finite Siegel-theta truncation for the reduced holonomy lattice."""
    dim = T_reduced.shape[0]
    side = np.arange(-N, N + 1, dtype=float)
    grids = np.meshgrid(*([side] * dim), indexing='ij')
    points = np.stack(grids, axis=-1).reshape(-1, dim)
    quad = np.einsum("ni,ij,nj->n", points, T_reduced, points, optimize=True)
    return float(np.sum(np.exp(-4.0 * np.pi * R ** 2 * quad)))


def t_duality_residual_reduced(T_reduced: np.ndarray, R: float, N: int = 6) -> float:
    """
    T-duality residual for the reduced higher-genus theta form.

    This checks the physical duality relation implied by
      Z_R(Sigma) = Z_{1/R}(Sigma)
    for the full compact-boson partition function when the same
    normalization is used on both sides:

      R * Theta(4 i R^2 T') - R^(-1) * Theta(4 i R^(-2) T').
    """
    inv_R = 1.0 / R
    return (
        R * theta_sum_reduced(T_reduced, R, N=N)
        - inv_R * theta_sum_reduced(T_reduced, inv_R, N=N)
    )


def partition_function_graph(Aprime: np.ndarray, R: float, T_reduced: np.ndarray,
                             N: int = 6) -> float:
    """
    Graph-based compact boson partition function.

    This follows the same normalization convention as partition_function_z:
    the moduli-independent 2*pi factor from the zero mode is omitted so the
    higher-genus code stays consistent with the existing torus implementation.
    """
    _, logdet = np.linalg.slogdet(Aprime)
    det_factor = np.exp(-0.5 * logdet)
    return float(R * det_factor * theta_sum_reduced(T_reduced, R, N=N))


def compute_graph_compact_partition(edge_lengths: Sequence[int], R: float,
                                    graph_data: dict, N: int = 6) -> dict:
    """Compute the lattice compact-boson partition function for stored graph data."""
    geom = compact_boson_graph_geometry(edge_lengths, graph_data)
    geom["Z"] = partition_function_graph(geom["A_prime"], R, geom["T_reduced"], N=N)
    return geom


def compute_genus2_partition(edge_lengths: Sequence[int], R: float,
                             topology: int = 1, N: int = 6) -> dict:
    """Convenience wrapper for the stored genus-2 one-face ribbon graphs."""
    graph_data = get_stored_genus2_graph(topology)
    result = compute_graph_compact_partition(edge_lengths, R, graph_data, N=N)
    result["topology"] = topology
    return result


# ── Core computation ───────────────────────────────────────────────────────────

def compute_z(L: int, l1: int, l2: int, R: float):
    """
    Compute the lattice partition function and the analytic Zcompact
    for given L, l1, l2, R.

    Returns (Z_lattice, Z_analytic, tau).
    """
    l3 = L // 2 - l1 - l2
    assert l3 > 0, f"Need l3 > 0, got l3={l3}"

    # M matrix (L x L), with the factor of 1/2 already included
    Mat = direct_mat_n_fast(L)

    # A' = MatAp  (L/2-1 x L/2-1)
    Aprime = direct_red_traced_mat(L, l1, l2, Mat)

    # W  (L/2-1 x L/2)
    W = mat_w(L, l1, l2, Mat)

    # T = Symm(T1 - T2)
    T1 = mat_t_first_part(L, l1, l2, Mat)
    T2 = mat_t_second_part(L, l1, l2, W, Aprime)
    T  = symm(T1 - T2)

    # T' (2x2)
    Tp = mat_t_prime(T)

    # Lattice partition function
    Z_lat = partition_function_z(Aprime, R, Tp)

    # tau via CylEqnImproved + PeriodsImproved
    f       = elt.make_cyl_eqn_improved(L, l1, l2)
    P1, P2, _ = elt.periods_improved(L, l1, l2, f)
    tau     = P2 / P1

    # Analytic partition function
    Z_ana = z_compact(R, tau)

    return Z_lat, Z_ana, tau


# ── Reproduce compact_partition.nb benchmarks ──────────────────────────────────

if __name__ == "__main__":
    # --- Bench: L=502, l1=l2=101, R=2 ---
    L, l1, l2, R = 1000, 150, 250, 1.0
    l3 = L // 2 - l1 - l2
    print(f"Bench: L={L}, l1={l1}, l2={l2}, l3={l3}, R={R}")
    Z_bench, Z_bench_ana, tau_bench = compute_z(L, l1, l2, R)
    print(f"  tau         = {tau_bench}")
    print(f"  Z_lattice   = {Z_bench:.10g}")
    print(f"  Z_analytic  = {Z_bench_ana:.10g}")
    print(f"  ratio       = {Z_bench / Z_bench_ana:.10g}")

    print(f"\nCheckTable (L={L}, l1={l1}, l2={l2}, r from 1.0 to 2.0):")
    print(f"{'r':>5}  {'Re(tau)':>12}  {'Im(tau)':>12}"
          f"  {'Z/Z_bench':>14}  {'Zana/Zana_bench':>16}.")
    for r in np.arange(1.0, 2.0, 0.05):
        Z_k, Z_k_ana, tau_k = compute_z(L, l1, l2, r)
        print(f"{r:>5.2f}  {tau_k.real:>12.8f}  {tau_k.imag:>12.8f}"
              f"  {Z_k / Z_bench:>14.8f}  {Z_k_ana / Z_bench_ana:>16.8f}")

    # --- T-duality test: L=500, l1=25, l2=150 ---
    print("\n" + "=" * 60)
    L_td, l1_td, l2_td = 1000, 25, 150
    l3_td = L_td // 2 - l1_td - l2_td
    print(f"T-duality check: L={L_td}, l1={l1_td}, l2={l2_td}, l3={l3_td}")
    print(f"  (should vanish: Tduality[R, Tp] = R Theta(R) - R^(-1) Theta(1/R) ≈ 0)")
    print(f"{'R':>5}  {'Tduality(R, Tp)':>20}")

    # Tp depends only on geometry, not R — compute once.
    Mat_td   = direct_mat_n_fast(L_td)
    Ap_td    = direct_red_traced_mat(L_td, l1_td, l2_td, Mat_td)
    W_td     = mat_w(L_td, l1_td, l2_td, Mat_td)
    T1_td    = mat_t_first_part(L_td, l1_td, l2_td, Mat_td)
    T2_td    = mat_t_second_part(L_td, l1_td, l2_td, W_td, Ap_td)
    Tp_td    = mat_t_prime(symm(T1_td - T2_td))

    for r in np.arange(1.1, 2.01, 0.1):
        val = t_duality(r, Tp_td)
        print(f"{r:>5.1f}  {val:>20.6e}")

    # --- Genus-2 R^2-theta check at L=3000 ---
    print("\n" + "=" * 60)
    edge_lengths_g2 = [120, 130, 140, 150, 160, 170, 180, 200, 250]
    total_L_g2 = 2 * sum(edge_lengths_g2)
    r_values_g2 = [1.2, 1.5, 1.8]
    print("Genus-2 R^2-theta check:")
    print(f"  edge lengths = {edge_lengths_g2}")
    print(f"  total L      = {total_L_g2}")
    print("  check: R^2 * Theta(4 i R^2 T') - R^(-2) * Theta(4 i R^(-2) T')")

    theta_cutoff = 5
    print(f"\n  theta cutoff N = {theta_cutoff}")
    for topology in range(1, 5):
        graph_data = get_stored_genus2_graph(topology)
        geom = compact_boson_graph_geometry(edge_lengths_g2, graph_data)
        print(f"  topology {topology}")
        for r in r_values_g2:
            inv_r = 1.0 / r
            theta_r = theta_sum_reduced(geom["T_reduced"], r, N=theta_cutoff)
            theta_inv = theta_sum_reduced(
                geom["T_reduced"], inv_r, N=theta_cutoff
            )
            lhs = (r ** 2) * theta_r
            rhs = (inv_r ** 2) * theta_inv
            resid = lhs - rhs
            rel = abs(resid) / max(abs(lhs), abs(rhs))
            print(
                f"    R={r:>3.1f}  lhs={lhs:>14.10f}  rhs={rhs:>14.10f}"
                f"  resid={resid:>12.6e}  rel={rel:>10.3e}"
            )
