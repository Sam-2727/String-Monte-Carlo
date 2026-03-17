import numpy as np
import math
from typing import Iterable,Callable, List, Tuple
import mpmath as mp
mp.mp.dps = 50


def mod1(x: int, L: int) -> int:
    """
    Mathematica Mod[x, L, 1] for integer x, returning in {1,...,L}.
    """
    return ((x - 1) % L) + 1


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

def logdet_cholesky(M: np.ndarray, symmetrize: bool = False) -> float:
    """
    Safer log(det(M)) using Cholesky. Raises if not positive definite.
    """
    M = np.asarray(M)
    if symmetrize:
        M = 0.5 * (M + M.T)

    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError as e:
        # Not PD (or numerically not PD)
        raise np.linalg.LinAlgError("Cholesky failed: matrix not positive definite.") from e

    return 2.0 * np.sum(np.log(np.diag(L)))

def bmat(L: int, l1: int, l2: int, dtype=np.complex128) -> np.ndarray:
    """
    Python/Numpy version of Mathematica:

    Bmat[L_, l1_, l2_] := Join[
      k1=1..l1:   exp(2π i m k1/L) - exp(2π i m/L (L/2 + l1 + 1 - k1))
      k2=l1+1..l1+l2: exp(2π i m k2/L) - exp(2π i m/L (L/2 + 2 l1 + l2 + 1 - k2))
      k3=l1+l2+1..L/2: exp(2π i m k3/L) - exp(2π i m/L (L + l1 + l2 + 1 - k3))
    ]

    Returns a (L/2) x (L/2) complex matrix with rows ordered by k=1..L/2
    and columns by m=1..L/2.
    """
    assert L % 2 == 0, "L must be even"
    half = L // 2

    m = np.arange(1, half + 1, dtype=np.float64)  # m=1..L/2
    twopi_i_over_L = 2j * np.pi / L

    rows = []

    # Block 1: k1 = 1..l1
    for k in range(1, l1 + 1):
        a = np.exp(twopi_i_over_L * (m * k))
        b = np.exp(twopi_i_over_L * (m * (half + l1 + 1 - k)))
        rows.append(a - b)

    # Block 2: k2 = l1+1 .. l1+l2
    for k in range(l1 + 1, l1 + l2 + 1):
        a = np.exp(twopi_i_over_L * (m * k))
        b = np.exp(twopi_i_over_L * (m * (half + 2 * l1 + l2 + 1 - k)))
        rows.append(a - b)

    # Block 3: k3 = l1+l2+1 .. L/2
    for k in range(l1 + l2 + 1, half + 1):
        a = np.exp(twopi_i_over_L * (m * k))
        b = np.exp(twopi_i_over_L * (m * (L + l1 + l2 + 1 - k)))
        rows.append(a - b)

    B = np.vstack(rows).astype(dtype, copy=False)
    assert B.shape == (half, half), f"Expected {(half, half)}, got {B.shape}"
    return B

def bmat_reduced(L: int, l1: int, l2: int, dtype=np.complex128):
    """
    Compute B with the m=1 column removed and its left null vector.

    B1 = B[:, 1:]  has shape (L/2) x (L/2-1).
    The left null vector v satisfies B1^T v = 0, i.e.
      sum_k B1[k,m] v[k] = 0  for all m = 2,...,L/2.
    In the continuum limit v_k -> 1/f(z_k).

    Returns
    -------
    B1 : ndarray, shape (L/2, L/2-1)
        The B matrix with the first mode column removed.
    v : ndarray, shape (L/2,)
        The left null vector (unit-normalized).
    """
    B = bmat(L, l1, l2, dtype=dtype)
    B1 = B[:, 1:]  # remove m=1 column
    # Need null(B1^T), not null(B1^H). For complex matrices these differ.
    # SVD of B1^T: the right null vector gives null(B1^T).
    _, _, Vh = np.linalg.svd(B1.T, full_matrices=True)
    v = Vh[-1, :].conj()  # null vector of B1^T
    return B1, v


_LOG64 = float(np.log(64.0))
_LOG64_MP = mp.log(mp.mpf(64))

def bdet_mp(L: int, l1: int, l2: int) -> mp.mpf:
    return mp.e ** bdet_log(L, l1, l2)

def prime_det_mp(L: int, l1: int, l2: int) -> mp.mpf:
    return mp.e ** prime_det_log(L, l1, l2)

def bdet_log(L: int, l1: int, l2: int) -> mp.mpf:
    """
    Returns log(Bdet) where
      Bdet = |det(B)|^2 / 64^L

    Computed as:
      log Bdet = 2*log|det(B)| - L*log(64)
    using np.linalg.slogdet for stability.
    """
    if L % 2 != 0:
        raise ValueError("L must be even")
    half = L // 2

    m = np.arange(1, half + 1, dtype=np.float64)
    k = np.arange(1, half + 1, dtype=np.float64)

    f = np.empty_like(k)
    if l1 > 0:
        f[:l1] = half + l1 + 1 - k[:l1]
    if l2 > 0:
        f[l1:l1 + l2] = half + 2 * l1 + l2 + 1 - k[l1:l1 + l2]
    if l1 + l2 < half:
        f[l1 + l2:] = L + l1 + l2 + 1 - k[l1 + l2:]

    twopi_i_over_L = 2j * np.pi / L
    phase1 = twopi_i_over_L * (k[:, None] * m[None, :])
    phase2 = twopi_i_over_L * (f[:, None] * m[None, :])
    B = np.exp(phase1) - np.exp(phase2)

    _, logabsdet = np.linalg.slogdet(B)  # log |det(B)| as float64
    # lift to mp.mpf for high-precision log arithmetic
    return mp.mpf(2.0 * logabsdet) - mp.mpf(L) * _LOG64_MP


def ddet_log(L: int, l1: int, l2: int,
             signs: tuple[bool, bool, bool] = (False, False, False)) -> mp.mpf:
    """
    Returns log(Ddet) where
      Ddet = |det(D)|^2 / 64^L

    D is the fermionic analogue of B, with a half-integer shift
    (m - 1/2) on the mode index:
      D[m,n] = exp(2πi(m-1/2)n/L) + s_j * i * exp(2πi(m-1/2)mapped(n)/L)

    signs = (s1, s2, s3): one bool per segment (l1, l2, l3).
      False -> -i  (default),  True -> +i.

    Computed as:
      log Ddet = 2*log|det(D)| - L*log(64)
    using np.linalg.slogdet for stability.
    """
    if L % 2 != 0:
        raise ValueError("L must be even")
    half = L // 2

    m = np.arange(1, half + 1, dtype=np.float64)   # row: mode index
    n = np.arange(1, half + 1, dtype=np.float64)   # col: position index

    # same piecewise map as B, applied to column index n
    mapped_n = np.empty_like(n)
    if l1 > 0:
        mapped_n[:l1] = half + l1 + 1 - n[:l1]
    if l2 > 0:
        mapped_n[l1:l1 + l2] = half + 2 * l1 + l2 + 1 - n[l1:l1 + l2]
    if l1 + l2 < half:
        mapped_n[l1 + l2:] = L + l1 + l2 + 1 - n[l1 + l2:]

    # Per-column sign vector: +1j if True, -1j if False
    sign_vec = np.empty(half, dtype=np.complex128)
    if l1 > 0:
        sign_vec[:l1] = 1j if signs[0] else -1j
    if l2 > 0:
        sign_vec[l1:l1 + l2] = 1j if signs[1] else -1j
    if l1 + l2 < half:
        sign_vec[l1 + l2:] = 1j if signs[2] else -1j

    twopi_i_over_L = 2j * np.pi / L
    row_half = m - 0.5   # half-integer shift: (m - 1/2)

    phase1 = twopi_i_over_L * (row_half[:, None] * n[None, :])
    phase2 = twopi_i_over_L * (row_half[:, None] * mapped_n[None, :])
    D = np.exp(phase1) + sign_vec[None, :] * np.exp(phase2)

    _, logabsdet = np.linalg.slogdet(D)
    return mp.mpf(2.0 * logabsdet) - mp.mpf(L) * _LOG64_MP



def prime_det_log(L: int, l1: int, l2: int) -> mp.mpf:
    """
    Returns log(primeDet) where
      primeDet = exp(logdet(A')) / (L/2)

    Fused implementation that exploits circulant structure of Mat to avoid
    building the full L×L matrix. Only the kernel (length L) and the
    half×half reduced matrix are allocated.
    """
    if L % 2 != 0:
        raise ValueError("L must be even")
    half = L // 2
    pi = math.pi

    # ── Kernel of the circulant matrix: Mat[i,j] = kernel[(j-i) % L] ──
    c0 = math.cos(pi / L)
    s0 = math.sin(pi / L)
    d = np.arange(L, dtype=np.float64)
    kernel = -s0 / (L * (c0 - np.cos(2 * pi * d / L)))

    # ── Initial tempTracedMat = Mat[:half, :half] via circulant ──
    idx = np.arange(half)
    tempTracedMat = kernel[(idx[None, :] - idx[:, None]) % L]

    # ── Build segment index arrays (0-indexed) and their maps ──
    if l1 + l2 == half:
        segments = [(1, l1), (l1 + 1, l1 + l2 - 1), (l1 + l2 + 1, half)]
        seg_flag = 2
    else:
        segments = [(1, l1), (l1 + 1, l1 + l2), (l1 + l2 + 1, half - 1)]
        seg_flag = 3

    map_offsets = [half + l1 + 1, half + 2 * l1 + l2 + 1, L + l1 + l2 + 1]

    K_list = []
    mapped_K_list = []
    for i in range(seg_flag):
        a, b = segments[i]
        if a > b:
            continue
        seg_k = np.arange(a, b + 1)
        mapped_k = map_offsets[i] - seg_k
        if i == 2:
            mapped_k = ((mapped_k - 1) % L) + 1
        K_list.append(seg_k - 1)
        mapped_K_list.append(mapped_k - 1)

    K = np.concatenate(K_list)
    mapped_K = np.concatenate(mapped_K_list)
    L_idx = K
    mapped_L = mapped_K

    n0 = half - 1

    mapped_n_1idx = map_offsets[seg_flag - 1] - half
    if seg_flag - 1 == 2:
        mapped_n_1idx = ((mapped_n_1idx - 1) % L) + 1
    mn0 = mapped_n_1idx - 1

    # ── Row updates: tempTracedMat[K, n0] += Mat[mapped_K, n0] + ... ──
    tempTracedMat[K, n0] += (
        kernel[(n0 - mapped_K) % L]
        + kernel[(mn0 - mapped_K) % L]
        + kernel[(mn0 - K) % L]
    )

    # ── Column updates: tempTracedMat[n0, L_idx] += ... ──
    tempTracedMat[n0, L_idx] += (
        kernel[(L_idx - mn0) % L]
        + kernel[(mapped_L - mn0) % L]
        + kernel[(mapped_L - n0) % L]
    )

    # ── Corner update ──
    tempTracedMat[n0, n0] += (
        kernel[(n0 - mn0) % L] + kernel[0] + kernel[(mn0 - n0) % L]
    )

    # ── Main block: vectorized, all from kernel ──
    row_n = tempTracedMat[K, n0]
    col_n = tempTracedMat[n0, L_idx]
    corner = tempTracedMat[n0, n0]

    tempTracedMat[np.ix_(K, L_idx)] += (
        kernel[(L_idx[None, :] - mapped_K[:, None]) % L]
        + kernel[(mapped_L[None, :] - mapped_K[:, None]) % L]
        + kernel[(mapped_L[None, :] - K[:, None]) % L]
        + (-row_n[:, None] - col_n[None, :] + corner)
    )

    Aprime = tempTracedMat[:half - 1, :half - 1]
    logdet = logdet_cholesky(Aprime)

    return mp.mpf(logdet) - mp.log(mp.mpf(half))


def combined_det2_log(L: int, l1: int, l2: int) -> tuple[mp.mpf, mp.mpf]:
    """
    Log-space analogue of combined_det2.
    Returns (log_bdet, log_prime_det).
    """
    return (bdet_log(L, l1, l2), -13*prime_det_log(L, l1, l2))