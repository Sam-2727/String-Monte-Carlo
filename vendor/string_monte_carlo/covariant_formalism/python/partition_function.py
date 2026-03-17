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
    pi = math.pi
    c0 = math.cos(pi / L)
    s0 = math.sin(pi / L)

    kernel = np.array(
        [-s0 / (L * (c0 - math.cos(2 * pi * d / L))) for d in range(L)],
        dtype=np.float64
    )

    Mat = np.empty((L, L), dtype=np.float64)
    for i in range(L):
        Mat[i, :] = np.roll(kernel, i)  # <-- NOTE: +i, not -i
    return Mat


def direct_mat_n(L: int, dtype=np.float64) -> np.ndarray:
    """
    Python version of DirectMatN[L, NPrec], following your double Do loop.
    This is O(L^2) and matches the indexing logic with Mod[...,L,1].
    """
    pi = math.pi
    sin = math.sin
    cos = math.cos

    Mat = np.zeros((L, L), dtype=dtype)

    s0 = sin(pi / L)
    c0 = cos(pi / L)

    # Mathematica: Do[ ..., {k,1,L}, {n,1,L}]
    for k in range(1, L + 1):
        denom = (c0 - cos((2 * k * pi) / L))
        term = 0.5 * (-(s0 / denom)) / L  # matches your N[1/2 * (...) / L]
        for n in range(1, L + 1):
            i = mod1(n, L) - 1
            j = mod1(n + k, L) - 1
            Mat[i, j] += term
            Mat[j, i] += term
    return Mat


def _segments_inclusive(r: Tuple[int, int]) -> List[int]:
    """
    Inclusive integer segment [a,b], empty if a>b.
    """
    a, b = r
    if a > b:
        return []
    return list(range(a, b + 1))

import numpy as np

def mod1(x: int, L: int) -> int:
    """Mathematica Mod[x, L, 1] for integers: returns in {1,...,L}."""
    return ((x - 1) % L) + 1


def direct_red_traced_mat(L: int, l1: int, l2: int, Mat: np.ndarray) -> np.ndarray:
    """
    Python reproduction of Mathematica:

      DirectRedTracedMat[L_, l1_, l2_, Mat_] := ...

    Inputs:
      - L: even integer
      - l1, l2: integers
      - Mat: numpy array of shape (L, L)

    Output:
      - numpy array of shape (L/2 - 1, L/2 - 1)
        equal to tempTracedMat[[1;;-2,1;;-2]] in Mathematica.

    Notes:
      - Internally uses 1-indexed k,l,i,j logic to match Mathematica exactly.
      - Uses inclusive segment ranges like Mathematica {k, a, b}.
    """
    assert L % 2 == 0, "L must be even"
    assert Mat.shape == (L, L), "Mat must be LxL"
    half = L // 2

    tempMat = Mat
    tempTracedMat = tempMat[:half, :half].copy()  # Mat[[1;;L/2,1;;L/2]]

    # segmentMaps (each returns a 1-indexed site in {1,...,L})
    def map1(x: int) -> int:
        return half + l1 + 1 - x

    def map2(x: int) -> int:
        return half + 2 * l1 + l2 + 1 - x

    def map3(x: int) -> int:
        return mod1(L + l1 + l2 + 1 - x, L)

    segmentMaps = [map1, map2, map3]

    # segments and segmentFlag logic (as in Mathematica)
    if l1 + l2 == half:
        # segments = {{1,l1},{l1+1,l1+l2-1},{l1+l2+1,L/2}}
        segments = [(1, l1), (l1 + 1, l1 + l2 - 1), (l1 + l2 + 1, half)]
        segmentFlag = 2
    else:
        # segments = {{1,l1},{l1+1,l1+l2},{l1+l2+1,L/2-1}}
        segments = [(1, l1), (l1 + 1, l1 + l2), (l1 + l2 + 1, half - 1)]
        segmentFlag = 3

    def rng(a: int, b: int):
        """Inclusive range [a,b] (Mathematica style). Empty if a>b."""
        return range(a, b + 1) if a <= b else range(0, 0)

    def idx0(x1: int) -> int:
        """Convert 1-indexed -> 0-indexed."""
        return x1 - 1

    n = half  # the special index L/2 (1-indexed)

    # segmentMaps[[segmentFlag]][L/2] in Mathematica
    mapped_n = segmentMaps[segmentFlag - 1](n)  # still 1-indexed

    # ---- The L/2-th row updates: tempTracedMat[[k, L/2]] += (...) ----
    for i in range(1, segmentFlag + 1):
        a, b = segments[i - 1]
        for k in rng(a, b):
            tempTracedMat[idx0(k), idx0(n)] += (
                tempMat[idx0(segmentMaps[i - 1](k)), idx0(n)]
                + tempMat[idx0(segmentMaps[i - 1](k)), idx0(mapped_n)]
                + tempMat[idx0(k), idx0(mapped_n)]
            )

    # ---- The L/2-th column updates: tempTracedMat[[L/2, l]] += (...) ----
    for j in range(1, segmentFlag + 1):
        a, b = segments[j - 1]
        for l in rng(a, b):
            tempTracedMat[idx0(n), idx0(l)] += (
                tempMat[idx0(mapped_n), idx0(l)]
                + tempMat[idx0(mapped_n), idx0(segmentMaps[j - 1](l))]
                + tempMat[idx0(n), idx0(segmentMaps[j - 1](l))]
            )

    # ---- The (L/2, L/2) corner update ----
    tempTracedMat[idx0(n), idx0(n)] += (
        tempMat[idx0(mapped_n), idx0(n)]
        + tempMat[idx0(mapped_n), idx0(mapped_n)]
        + tempMat[idx0(n), idx0(mapped_n)]
    )

    # ---- Main block updates (k,l in segments, excluding L/2 as in Mathematica) ----
    for i in range(1, segmentFlag + 1):
        ai, bi = segments[i - 1]
        for j in range(1, segmentFlag + 1):
            aj, bj = segments[j - 1]
            for k in rng(ai, bi):
                for l in rng(aj, bj):
                    tempTracedMat[idx0(k), idx0(l)] += (
                        tempMat[idx0(segmentMaps[i - 1](k)), idx0(l)]
                        + tempMat[idx0(segmentMaps[i - 1](k)), idx0(segmentMaps[j - 1](l))]
                        + tempMat[idx0(k), idx0(segmentMaps[j - 1](l))]
                        + (-tempTracedMat[idx0(k), idx0(n)]
                           - tempTracedMat[idx0(n), idx0(l)]
                           + tempTracedMat[idx0(n), idx0(n)])
                    )

    # Return tempTracedMat[[1;;-2,1;;-2]]
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

    So:
      log primeDet = logdet(A') - log(L/2)
    """
    if L % 2 != 0:
        raise ValueError("L must be even")
    half = L // 2

    Mat = direct_mat_n_fast(L)
    Aprime = direct_red_traced_mat(L, l1, l2, Mat)
    logdet = logdet_cholesky(Aprime)  # float64 log det

    return mp.mpf(logdet) - mp.log(mp.mpf(half))


def combined_det2_log(L: int, l1: int, l2: int) -> tuple[mp.mpf, mp.mpf]:
    """
    Log-space analogue of combined_det2.
    Returns (log_bdet, log_prime_det).
    """
    return (bdet_log(L, l1, l2), -13*prime_det_log(L, l1, l2))