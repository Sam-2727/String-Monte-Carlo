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
import numpy as np
import mpmath as mp

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import ell_to_tau as elt          # make_cyl_eqn_improved, periods_improved


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
    return kernel[idx]


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

    exponent = -2.0 * np.pi * R ** 2 * (
        s1 ** 2 * Tp[0, 0]
        + s1 * s2 * Tp[0, 1]
        + s2 * s1 * Tp[1, 0]
        + s2 ** 2 * Tp[1, 1]
    )
    theta_sum = np.sum(np.exp(exponent))
    return float(R * det_factor * theta_sum)


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
