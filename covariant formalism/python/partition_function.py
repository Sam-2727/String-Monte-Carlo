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


def _theta_graph_ribbon():
    """Canonical genus-1 theta graph used by the legacy determinant routines."""
    return (
        [(1, 2), (1, 2), (1, 2)],
        [1, 2],
        {1: [0, 1, 2], 2: [0, 1, 2]},
    )


def _f1_boundary_site_pairs(ribbon_graph, ell_list):
    """
    Enumerate the reduced boundary sites and their sewn partners for an F=1 ribbon graph.

    The reduced variables are chosen from the first occurrence of each edge along the
    traced disc boundary. If an edge of length ell_e appears on boundary segments
    p_first < p_second, then local site t=1,...,ell_e on the first copy is paired with
    site ell_e+1-t on the second copy.
    """
    from ribbon_graph_generator import _trace_boundary, _get_all_face_boundaries

    edges_graph, verts, rotation = ribbon_graph
    n_edges = len(edges_graph)
    if len(ell_list) != n_edges:
        raise ValueError(
            f"Need one length per edge: got {len(ell_list)} lengths for {n_edges} edges."
        )

    ell_arr = np.asarray(ell_list, dtype=int)
    if np.any(ell_arr <= 0):
        raise ValueError("All edge lengths must be positive integers for the determinant lattice.")

    all_faces = _get_all_face_boundaries(edges_graph, verts, rotation)
    if len(all_faces) != 1:
        raise ValueError(f"Requires an F=1 ribbon graph, got F={len(all_faces)}.")

    boundary = tuple(_trace_boundary(edges_graph, verts, rotation))
    n_segments = len(boundary)
    edge_word = [step[2] for step in boundary]

    edge_positions: dict[int, list[int]] = {}
    for seg_idx, edge_idx in enumerate(edge_word):
        edge_positions.setdefault(edge_idx, []).append(seg_idx)

    bad_edges = [edge_idx for edge_idx, occ in edge_positions.items() if len(occ) != 2]
    if bad_edges:
        raise ValueError(
            "Each edge must occur exactly twice on the traced boundary; "
            f"bad edges={bad_edges}"
        )

    starts = np.zeros(n_segments + 1, dtype=int)
    for seg_idx, edge_idx in enumerate(edge_word):
        starts[seg_idx + 1] = starts[seg_idx] + int(ell_arr[edge_idx])

    L = int(starts[-1])
    if L % 2 != 0:
        raise ValueError(f"Total boundary lattice length must be even, got L={L}.")

    first_occurrence = {edge_idx: min(pos) for edge_idx, pos in edge_positions.items()}

    rep_sites: list[int] = []
    partner_sites: list[int] = []
    rep_meta: list[tuple[int, int, int, int]] = []
    for seg_idx, edge_idx in enumerate(edge_word):
        if seg_idx != first_occurrence[edge_idx]:
            continue
        occ1, occ2 = edge_positions[edge_idx]
        edge_len = int(ell_arr[edge_idx])
        start1 = int(starts[occ1])
        start2 = int(starts[occ2])
        for local_idx in range(edge_len):
            rep_site = start1 + local_idx
            partner_site = start2 + (edge_len - 1 - local_idx)
            rep_sites.append(rep_site)
            partner_sites.append(partner_site)
            rep_meta.append((int(edge_idx), int(local_idx + 1), int(occ1), int(occ2)))

    rep_sites_arr = np.asarray(rep_sites, dtype=int)
    partner_sites_arr = np.asarray(partner_sites, dtype=int)
    if rep_sites_arr.size * 2 != L:
        raise ValueError(
            "Unexpected reduced-site count: "
            f"{rep_sites_arr.size} reduced sites but total boundary length L={L}."
        )

    return {
        "boundary": boundary,
        "edge_word": tuple(edge_word),
        "edge_positions": {edge_idx: tuple(pos) for edge_idx, pos in edge_positions.items()},
        "starts": starts,
        "L": L,
        "n_reduced": rep_sites_arr.size,
        "rep_sites": rep_sites_arr,
        "partner_sites": partner_sites_arr,
        "rep_site_metadata": tuple(rep_meta),
    }


def traced_bmat_f1(ribbon_graph, ell_list, dtype=np.complex128) -> np.ndarray:
    """
    Generic F=1 bc-ghost matrix from the sewn boundary lattice.

    Rows are reduced boundary sites, columns are positive Fourier modes.
    The sign convention matches the legacy genus-1 B matrix:
        exp(2π i m r / L) - exp(2π i m r_partner / L).

    This is the square Psi_c-style matrix with modes m=1,...,N_red.
    """
    data = _f1_boundary_site_pairs(ribbon_graph, ell_list)
    L = data["L"]
    n_red = data["n_reduced"]
    modes = np.arange(1, n_red + 1, dtype=np.float64)
    rep = data["rep_sites"].astype(np.float64) + 1.0
    partner = data["partner_sites"].astype(np.float64) + 1.0
    twopi_i_over_L = 2j * np.pi / L
    rows = []
    for rep_pos, partner_pos in zip(rep, partner):
        phase_rep = np.exp(twopi_i_over_L * (modes * float(rep_pos)))
        phase_partner = np.exp(twopi_i_over_L * (modes * float(partner_pos)))
        rows.append(phase_rep - phase_partner)
    return np.vstack(rows).astype(dtype, copy=False)


def traced_bmat_psi1_rect_f1(ribbon_graph, ell_list, dtype=np.complex128) -> np.ndarray:
    """
    Rectangular F=1 ghost matrix for the identity wavefunctional Psi_1.

    This removes the m=1 column from the square Psi_c-style matrix, leaving the
    modes m=2,...,N_red. With the row/column convention used here, the output
    has shape (N_red, N_red-1), where rows are reduced boundary sites and columns
    are the surviving Fourier modes.
    """
    B = traced_bmat_f1(ribbon_graph, ell_list, dtype=dtype)
    if B.shape[1] < 2:
        raise ValueError("Need at least two reduced Fourier modes to build the Psi_1 matrix.")
    return B[:, 1:]


def _left_null_vector_rectangular(Brect: np.ndarray) -> np.ndarray:
    """
    Unit-norm left null vector v of a full-column-rank rectangular matrix Brect.

    Here "left null" means Brect^T v = 0, which is the null direction in the
    reduced c-variable space when Brect is viewed as the constrained Psi_1 matrix
    with rows indexed by variables and columns indexed by mode factors.
    """
    _, _, Vh = np.linalg.svd(Brect.T, full_matrices=True)
    v = Vh[-1, :].conj()
    norm = np.linalg.norm(v)
    if norm == 0.0:
        raise np.linalg.LinAlgError("Rectangular Psi_1 matrix has a zero left-null vector.")
    return v / norm


def traced_bmat_psi1_quotient_f1(ribbon_graph, ell_list, dtype=np.complex128, pivot: int | None = None):
    """
    Explicit square quotient representative for the sewn Psi_1 ghost matrix.

    Returns
    -------
    B_tilde : ndarray, shape (N_red-1, N_red-1)
        Square matrix obtained by projecting the reduced c-variable space onto an
        orthonormal complement of the missing Psi_c direction.
    null_vec : ndarray, shape (N_red,)
        Unit-norm left null vector of the rectangular Psi_1 matrix.

    pivot : int or None
        Reduced-site index to eliminate. If None, use the entry of the left null
        vector with largest magnitude for numerical stability.

    Notes
    -----
    The quotient is defined by solving the bilinear null constraint
        v^T c = 0
    for one reduced variable c_p, where v is a left null vector of the rectangular
    Psi_1 matrix. This yields a concrete square matrix B_tilde whose determinant is
    the numerical ghost factor in the chosen elimination convention.
    """
    Brect = traced_bmat_psi1_rect_f1(ribbon_graph, ell_list, dtype=dtype)
    null_vec = _left_null_vector_rectangular(Brect)
    if pivot is None:
        pivot = int(np.argmax(np.abs(null_vec)))
    pivot = int(pivot)
    if pivot < 0 or pivot >= null_vec.size:
        raise IndexError(f"Pivot {pivot} is out of range for null vector of size {null_vec.size}.")
    if abs(null_vec[pivot]) == 0.0:
        raise np.linalg.LinAlgError("Chosen pivot lies in a zero component of the left null vector.")

    keep = [idx for idx in range(null_vec.size) if idx != pivot]
    B_tilde = Brect[keep, :] - (null_vec[keep, None] / null_vec[pivot]) * Brect[pivot : pivot + 1, :]
    return (
        np.asarray(B_tilde, dtype=dtype),
        np.asarray(null_vec, dtype=dtype),
        pivot,
    )


def _constant_mode_elimination_matrix(n_reduced: int) -> np.ndarray:
    """Matrix C implementing x_last = -sum_{i<last} x_i on the reduced lattice."""
    if n_reduced < 2:
        raise ValueError("Need at least two reduced sites to eliminate the constant mode.")
    C = np.zeros((n_reduced, n_reduced - 1), dtype=np.float64)
    C[: n_reduced - 1, :] = np.eye(n_reduced - 1, dtype=np.float64)
    C[n_reduced - 1, :] = -1.0
    return C


def traced_matter_full_matrix_f1(ribbon_graph, ell_list, *, kernel: np.ndarray | None = None) -> np.ndarray:
    """
    Generic sewn matter matrix before removing the constant zero mode.

    If x_red labels the independent boundary values and x_full = M x_red implements the
    sewing constraints x(partner)=x(rep), then this returns M^T A_full M without removing
    the constant zero mode.
    """
    data = _f1_boundary_site_pairs(ribbon_graph, ell_list)
    L = data["L"]
    rep = data["rep_sites"]
    partner = data["partner_sites"]

    if kernel is None:
        pi = math.pi
        c0 = math.cos(pi / L)
        s0 = math.sin(pi / L)
        d = np.arange(L, dtype=np.float64)
        kernel = -s0 / (L * (c0 - np.cos(2 * pi * d / L)))
    else:
        kernel = np.asarray(kernel, dtype=np.float64)
        if kernel.shape != (L,):
            raise ValueError(f"Kernel must have shape ({L},), got {kernel.shape}.")

    rr = kernel[(rep[:, None] - rep[None, :]) % L]
    rp = kernel[(rep[:, None] - partner[None, :]) % L]
    pr = kernel[(partner[:, None] - rep[None, :]) % L]
    pp = kernel[(partner[:, None] - partner[None, :]) % L]
    return rr + rp + pr + pp


def traced_matter_matrix_f1(ribbon_graph, ell_list, *, kernel: np.ndarray | None = None) -> np.ndarray:
    """
    Generic A' matrix for an F=1 ribbon graph, matching direct_red_traced_mat in genus 1.

    Starting from the sewn matrix on the L/2 independent boundary values, we project onto
    the codimension-one subspace orthogonal to the constant mode using the same linear
    elimination convention as the legacy genus-1 code.
    """
    A_sewn = traced_matter_full_matrix_f1(ribbon_graph, ell_list, kernel=kernel)
    C = _constant_mode_elimination_matrix(A_sewn.shape[0])
    return C.T @ A_sewn @ C


def traced_bdet_log_f1(ribbon_graph, ell_list) -> mp.mpf:
    """
    Generic log(Bdet) for an F=1 ribbon graph, in the same normalization as bdet_log.

    Bdet = |det(B)|^2 / 64^L
    """
    data = _f1_boundary_site_pairs(ribbon_graph, ell_list)
    B = traced_bmat_f1(ribbon_graph, ell_list)
    _, logabsdet = np.linalg.slogdet(B)
    return mp.mpf(2.0 * logabsdet) - mp.mpf(data["L"]) * _LOG64_MP


def traced_psi1_logabsdet_f1(
    ribbon_graph,
    ell_list,
    *,
    normalize_by_lattice: bool = False,
    pivot: int | None = None,
) -> mp.mpf:
    """
    Log of |det(B_tilde)| for the sewn identity-wavefunctional Psi_1 ghost matrix.

    The square quotient representative B_tilde is obtained by solving the bilinear
    null constraint v^T c = 0 for a stable pivot variable, where v is a left null
    vector of the rectangular Psi_1 matrix with the m=1 column removed.

    The Berezin measure on the quotient contributes an extra Jacobian |v_p| from
    the eliminated null direction, so the pivot-invariant quantity is

        |v_p| * |det(B_tilde,p)|.

    Parameters
    ----------
    normalize_by_lattice:
        If True, subtract L log 8 so that the returned quantity is the natural
        square-root analogue of the old Bdet normalization |det(B)|^2 / 64^L.
        The default False returns the raw log |det(B_tilde)|.
    pivot:
        Optional reduced-site index used for the elimination. If None, choose the
        stable pivot with largest |v_p|. The final value should be pivot-independent
        up to numerical precision.
    """
    data = _f1_boundary_site_pairs(ribbon_graph, ell_list)
    B_tilde, null_vec, used_pivot = traced_bmat_psi1_quotient_f1(
        ribbon_graph,
        ell_list,
        pivot=pivot,
    )
    _, logabsdet = np.linalg.slogdet(B_tilde)
    out = mp.mpf(logabsdet + np.log(abs(null_vec[used_pivot])))
    if normalize_by_lattice:
        out -= mp.mpf(data["L"]) * (_LOG64_MP / 2)
    return out


def traced_prime_det_log_f1(ribbon_graph, ell_list) -> mp.mpf:
    """
    Generic log(primeDet) for an F=1 ribbon graph, matching prime_det_log in genus 1.

    primeDet = det(A_minor) / n_red, where A_minor removes one representative of the
    constant zero mode from the reduced sewn matter matrix.
    """
    data = _f1_boundary_site_pairs(ribbon_graph, ell_list)
    A_prime = traced_matter_matrix_f1(ribbon_graph, ell_list)
    A_prime = 0.5 * (A_prime + A_prime.T)
    logdet = logdet_cholesky(A_prime)
    n_red = data["n_reduced"]
    return mp.mpf(logdet) - mp.log(mp.mpf(n_red))


def traced_combined_det_log_f1(ribbon_graph, ell_list, *, matter_power: int = -13) -> tuple[mp.mpf, mp.mpf]:
    """
    Generic F=1 analogue of combined_det2_log for an arbitrary one-face ribbon graph.

    Returns
    -------
    (log_bdet, matter_power * log_prime_det)
    """
    log_bdet = traced_bdet_log_f1(ribbon_graph, ell_list)
    log_prime = traced_prime_det_log_f1(ribbon_graph, ell_list)
    return log_bdet, mp.mpf(matter_power) * log_prime


def traced_matter_bc_log_f1(ribbon_graph, ell_list, *, matter_power: int = -13) -> mp.mpf:
    """
    Log of the higher-genus numerical analogue of bdet * prime_det^{matter_power}.

    For the current matter+bc application, the default is matter_power = -13, so this
    returns
        log(Bdet) - 13 log(primeDet).
    """
    log_bdet, weighted_log_prime = traced_combined_det_log_f1(
        ribbon_graph, ell_list, matter_power=matter_power
    )
    return log_bdet + weighted_log_prime


def traced_numeric_amplitude_log_psi1_f1(
    ribbon_graph,
    ell_list,
    *,
    matter_power: int = -13,
    normalize_ghost: bool = False,
    pivot: int | None = None,
) -> mp.mpf:
    """
    Log of the higher-genus numerical amplitude built from Psi_1 and A'.

    By default this returns
        log |det(B_tilde)| - 13 log(det A').

    If normalize_ghost=True, the ghost factor is replaced by its lattice-normalized
    version log(|det(B_tilde)| / 8^L).
    """
    logabsdet = traced_psi1_logabsdet_f1(
        ribbon_graph,
        ell_list,
        normalize_by_lattice=normalize_ghost,
        pivot=pivot,
    )
    log_prime = traced_prime_det_log_f1(ribbon_graph, ell_list)
    return logabsdet + mp.mpf(matter_power) * log_prime



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


def logdet_hermitian_cholesky(M: np.ndarray, symmetrize: bool = True) -> float:
    """
    Stable log(det(M)) for Hermitian positive-definite real or complex matrices.
    """
    M = np.asarray(M)
    if symmetrize:
        M = 0.5 * (M + M.conj().T)

    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            "Cholesky failed: Hermitian matrix is not positive definite."
        ) from e

    diag = np.real(np.diag(L))
    if np.any(diag <= 0.0):
        raise np.linalg.LinAlgError(
            "Cholesky produced a non-positive diagonal entry for a supposed positive matrix."
        )
    return 2.0 * np.sum(np.log(diag))

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
