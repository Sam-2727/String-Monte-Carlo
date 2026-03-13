#!/usr/bin/env python3
r"""
Diagnostic script for the bosonic cubic sewn oscillator quadratic form.

Important terminology:

This file historically used the phrase "Neumann coefficients" for the matrices
obtained after imposing the cubic overlap and eliminating the outgoing leg.
That language is too loose. The primary object computed here is the reduced
constrained quadratic form

    G_T = diag(M_1, M_2) + U^\dagger M_3 U,

or its DFT-basis blocks. Those blocks are physically useful for the Gaussian
vacuum amplitude, Schur complement, and determinant analysis, but they are not
the same thing as the standard continuum three-string squeeze matrix extracted
from the full three-leg oscillator vertex.

The purpose of this script is therefore:
1. build the exact discrete overlap,
2. form the reduced sewn quadratic form after eliminating leg 3,
3. inspect its raw and oscillator-normalized low-mode behavior,
4. separate that diagnostic from the distinct task of extracting the full
   three-leg squeeze matrix for a true continuum Neumann comparison.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tachyon_check as tc


def continuum_neumann_N11(m: int, n: int, alpha1: float, alpha2: float) -> float:
    """
    Placeholder for a true continuum Neumann comparison.

    The actual standard continuum coefficients should be compared against the
    full three-leg squeeze matrix extracted from the uneliminated overlap
    system, not against the reduced sewn quadratic form analyzed here.
    """
    # Placeholder — we check convergence instead of using the closed form
    pass


def discrete_neumann_matrix(n1: int, n2: int, alpha_prime: float = 1.0):
    """
    Extract the reduced sewn quadratic form from the constrained cubic overlap.

    Returns the (n1-1+n2-1+n3-1) x (n1-1+n2-1+n3-1) constrained
    quadratic-form data in the real nonzero-mode basis, organized by leg.

    After imposing q^(3) = U1 q^(1) + U2 q^(2) + xi*y and eliminating
    q^(3), the reduced quadratic form on the incoming modes Q = (q^(1), q^(2))
    is G_T = diag(M1, M2) + U^T M3 U, which is exactly the matrix already
    computed in tachyon_check. The blocks returned below are therefore the
    reduced constrained form, not the full three-leg squeeze matrix.
    """
    n3 = n1 + n2
    m1 = tc.mode_metric(n1, alpha_prime)
    m2 = tc.mode_metric(n2, alpha_prime)
    m3 = tc.mode_metric(n3, alpha_prime)
    u1, u2, xi = tc.overlap_data(n1, n2)

    # The constrained quadratic form after eliminating q^(3)
    # Q^T G_T Q + 2 y B_T^T Q + C_T y^2
    # where Q = (q^(1), q^(2))
    g_t = np.block([
        [m1 + u1.T @ m3 @ u1,  u1.T @ m3 @ u2],
        [u2.T @ m3 @ u1,        m2 + u2.T @ m3 @ u2],
    ])

    # Also expose the reduced blocks separately. These are convenient diagnostics
    # for the sewn Gaussian, but they should not be confused with the standard
    # full-vertex Neumann/squeeze data of lightcone string field theory.

    n11 = m1 + u1.T @ m3 @ u1
    n12 = u1.T @ m3 @ u2
    n22 = m2 + u2.T @ m3 @ u2

    # For the 13, 23 coefficients (before elimination):
    # N^{13}_{mn} = -M3 U1  (with appropriate signs from the overlap direction)
    # These are the cross-leg terms between incoming and outgoing
    n13 = -(m3 @ u1)  # (n3-1) x (n1-1)
    n23 = -(m3 @ u2)  # (n3-1) x (n2-1)
    n33 = m3           # (n3-1) x (n3-1)

    return {
        'G_T': g_t,
        'N11': n11, 'N12': n12, 'N22': n22,
        'N13': n13, 'N23': n23, 'N33': n33,
        'n1': n1, 'n2': n2, 'n3': n3,
    }


def check_neumann_convergence(m_idx: int, n_idx: int,
                               block: str = 'N11',
                               alpha_ratio: float = 2.0 / 3.0):
    """
    Check convergence of one reduced-block entry as the lattice is refined at a
    fixed circumference ratio.
    """
    results = []
    for scale in [4, 8, 16, 32, 64, 128]:
        n1 = int(round(scale * 2))
        n2 = int(round(scale * 3))
        data = discrete_neumann_matrix(n1, n2)
        mat = data[block]

        if m_idx < mat.shape[0] and n_idx < mat.shape[1]:
            val = mat[m_idx, n_idx]
            results.append((n1, n2, val))

    return results


def print_neumann_diagnostics():
    """Print convergence diagnostics for low-lying reduced-block entries."""
    print("=" * 78)
    print("REDUCED QUADRATIC-FORM ENTRY DIAGNOSTICS")
    print("=" * 78)
    print("Fixed ratio alpha1:alpha2 = 2:3 (so N1:N2 = 2:3)")
    print()

    # Check a few representative entries
    for block, label in [('N11', 'N^{11}'), ('N12', 'N^{12}'),
                          ('N22', 'N^{22}'), ('N13', 'N^{13}')]:
        print(f"--- Reduced block {label} ---")
        # Check the (0,0), (0,1), (1,0) entries (in zero-indexed real basis)
        for mi, ni in [(0, 0), (0, 1), (1, 1), (2, 2)]:
            results = check_neumann_convergence(mi, ni, block=block)
            if not results:
                continue
            print(f"  ({mi},{ni}): ", end="")
            for n1, n2, val in results:
                print(f" N1={n1:4d}:{val:12.8f}", end="")
            print()

            # Check convergence rate of the raw real-basis entry
            if len(results) >= 3:
                vals = [v for _, _, v in results]
                # Richardson extrapolation on last 3 values
                if abs(vals[-1] - vals[-2]) > 1e-15:
                    ratio = (vals[-2] - vals[-3]) / (vals[-1] - vals[-2])
                    if ratio > 1:
                        print(f"           convergence ratio: {ratio:.3f} "
                              f"(~O(1/N^{math.log(ratio)/math.log(2):.1f}))")
                else:
                    print(f"           already converged to machine precision")

        print()


def print_neumann_matrix_structure(n1: int, n2: int):
    """Print the structure of the reduced sewn quadratic-form blocks."""
    data = discrete_neumann_matrix(n1, n2)
    print("=" * 78)
    print(f"REDUCED QUADRATIC-FORM STRUCTURE FOR (N1,N2)=({n1},{n2})")
    print("=" * 78)

    # Print the diagonal dominance and off-diagonal decay
    for block_name in ['N11', 'N12', 'N22']:
        mat = data[block_name]
        diag = np.diag(mat) if mat.shape[0] == mat.shape[1] else None
        offdiag_max = np.max(np.abs(mat - np.diag(np.diag(mat)))) if diag is not None else np.max(np.abs(mat))

        print(f"\n  {block_name} ({mat.shape[0]}x{mat.shape[1]}):")
        if diag is not None:
            print(f"    diagonal range: [{np.min(diag):.6f}, {np.max(diag):.6f}]")
        print(f"    max |off-diagonal|: {offdiag_max:.6e}")

        # Print first few entries
        nshow = min(5, mat.shape[0], mat.shape[1])
        print(f"    top-left {nshow}x{nshow} block:")
        for i in range(nshow):
            row_str = "    "
            for j in range(nshow):
                row_str += f" {mat[i,j]:10.6f}"
            print(row_str)

    # Check: does N11 - M1 decay for large mode numbers?
    m1 = tc.mode_metric(n1, 1.0)
    overlap_contrib = data['N11'] - m1
    print(f"\n  Overlap contribution N11 - M1:")
    print(f"    max |N11 - M1|: {np.max(np.abs(overlap_contrib)):.6e}")
    print(f"    Frobenius norm: {np.linalg.norm(overlap_contrib, 'fro'):.6e}")

    # Verify: G_T should be the reduced matrix after eliminating q^(3)
    gt_check = np.block([
        [data['N11'], data['N12']],
        [data['N12'].T, data['N22']],
    ])
    gt_err = np.max(np.abs(gt_check - data['G_T']))
    print(f"\n  Consistency check |G_T - [N11,N12;N12^T,N22]|: {gt_err:.2e}")
    print()


def print_continuum_comparison(n1: int, n2: int):
    """
    Inspect the raw DFT-basis reduced blocks against the free-string metric.

    This is not yet a comparison with the standard continuum Neumann/squeeze
    matrix. It only shows how the reduced sewn quadratic form differs from the
    bare free-cylinder metric before any oscillator normalization is removed.
    """
    data = discrete_neumann_matrix(n1, n2)
    m1 = tc.mode_metric(n1, 1.0)

    print("=" * 78)
    print(f"RAW DFT-BASIS REDUCED-BLOCK COMPARISON FOR (N1,N2)=({n1},{n2})")
    print("=" * 78)

    n11_diag = np.diag(data['N11'])
    m1_diag = np.diag(m1)
    overlap_diag = n11_diag - m1_diag

    print(f"\n  Diagonal of N11 vs M1 (first 8 modes):")
    print(f"  {'mode':>5s} {'N11_diag':>12s} {'M1_diag':>12s} {'overlap':>12s} {'ratio':>12s}")
    nshow = min(8, len(n11_diag))
    for i in range(nshow):
        ratio = overlap_diag[i] / m1_diag[i] if abs(m1_diag[i]) > 1e-15 else float('nan')
        print(f"  {i:5d} {n11_diag[i]:12.8f} {m1_diag[i]:12.8f} "
              f"{overlap_diag[i]:12.8f} {ratio:12.8f}")

    # Check off-diagonal decay in N12
    n12 = data['N12']
    print(f"\n  Off-diagonal decay in N12 (first row):")
    nshow = min(8, n12.shape[1])
    for j in range(nshow):
        print(f"    N12[0,{j}] = {n12[0,j]:12.8e}")

    print()


def dft_overlap_data(n1: int, n2: int):
    """
    Compute the overlap matrices U_1, U_2, xi directly in the DFT basis.

    U_r[m, ell] = (1/sqrt(N3*Nr)) sum_{n} exp(-2pi i m n/N3) exp(2pi i ell n/Nr)

    Returns complex matrices for the nonzero modes m=1..N3-1, ell=1..Nr-1.
    """
    n3 = n1 + n2

    # Vectorized computation using outer products
    # U_1[m, ell] for m=1..N3-1, ell=1..N1-1
    m_idx = np.arange(1, n3)[:, None]     # (N3-1, 1)
    n_arr1 = np.arange(n1)[None, :]       # (1, N1)
    ell_idx1 = np.arange(1, n1)[None, :]  # (1, N1-1)

    # phase[m, n] = exp(-2pi i m n/N3)
    phase3_1 = np.exp(-2j * np.pi * m_idx * n_arr1 / n3)  # (N3-1, N1)
    # phase1[n, ell] = exp(2pi i ell n/N1)
    phase1 = np.exp(2j * np.pi * np.arange(n1)[:, None] * ell_idx1 / n1)  # (N1, N1-1)
    u1 = (phase3_1 @ phase1) / np.sqrt(n3 * n1)

    # U_2[m, ell] for m=1..N3-1, ell=1..N2-1
    n_arr2 = np.arange(n2)[None, :]
    ell_idx2 = np.arange(1, n2)[None, :]
    # phase offset for leg 2: exp(-2pi i m (n+N1)/N3)
    phase3_2 = np.exp(-2j * np.pi * m_idx * (n_arr2 + n1) / n3)  # (N3-1, N2)
    phase2 = np.exp(2j * np.pi * np.arange(n2)[:, None] * ell_idx2 / n2)  # (N2, N2-1)
    u2 = (phase3_2 @ phase2) / np.sqrt(n3 * n2)

    # xi[m] = (1/sqrt(N3)) sum_{n=0}^{N1-1} exp(-2pi i m n / N3)
    xi = np.sum(phase3_1, axis=1) / np.sqrt(n3)

    return u1, u2, xi


def dft_mode_metric(n_sites: int, alpha_prime: float = 1.0):
    """
    Mode metric in the DFT basis: diagonal with mu*omega_k for k=1..N-1.
    """
    k = np.arange(1, n_sites, dtype=float)
    weights = np.sin(np.pi * k / n_sites) / (np.pi * alpha_prime)
    return np.diag(weights)


def dft_neumann_matrix(n1: int, n2: int, alpha_prime: float = 1.0):
    """
    Extract the reduced sewn quadratic-form blocks in the DFT basis.

    Returns the DFT-basis blocks of the reduced constrained form.
    """
    n3 = n1 + n2
    m1 = dft_mode_metric(n1, alpha_prime)
    m2 = dft_mode_metric(n2, alpha_prime)
    m3 = dft_mode_metric(n3, alpha_prime)
    u1, u2, xi = dft_overlap_data(n1, n2)

    n11 = m1 + u1.conj().T @ m3 @ u1
    n12 = u1.conj().T @ m3 @ u2
    n22 = m2 + u2.conj().T @ m3 @ u2

    return {'N11': n11, 'N12': n12, 'N22': n22,
            'n1': n1, 'n2': n2, 'n3': n3,
            'M1': m1, 'M2': m2, 'M3': m3,
            'U1': u1, 'U2': u2, 'xi': xi}


def continuum_neumann_N_rs(m: int, n: int, alpha1: float, alpha2: float,
                            r: int, s: int) -> float:
    """
    Placeholder for the standard continuum three-string Neumann coefficient.

    This script does not yet implement that comparison, because the relevant
    lattice object for such a test is the full three-leg squeeze matrix rather
    than the reduced sewn quadratic-form blocks studied here.
    """
    alpha3 = alpha1 + alpha2
    if r == 1:
        alpha_r = alpha1
    else:
        alpha_r = alpha2

    if s == 1:
        alpha_s = alpha1
    else:
        alpha_s = alpha2

    # The continuum overlap U_r is computed from the integral
    # For now, compute numerically at high resolution
    # and compare with the discrete version
    pass  # placeholder


def print_dft_neumann_convergence():
    """Inspect oscillator-normalized reduced DFT blocks versus lattice size."""
    print("=" * 78)
    print("OSCILLATOR-NORMALIZED REDUCED DFT BLOCKS")
    print("=" * 78)
    print("Fixed ratio alpha1:alpha2 = 2:3")
    print(
        "Showing Ĝ^{rs}_{mn}(N) = G^{rs}_{mn}(N) / sqrt(M_{r,mm} M_{s,nn}) "
        "for the reduced sewn quadratic form."
    )
    print(
        "If these normalized entries grow logarithmically with N, that signals the "
        "branch-point normal-ordering divergence of the reduced form rather than a "
        "finite entrywise continuum Neumann limit."
    )
    print()

    for block, label in [('N11', 'Ĝ^{11}'), ('N12', 'Ĝ^{12}')]:
        print(f"--- Reduced block {label} ---")
        for mi, ni in [(0, 0), (1, 1), (2, 2), (0, 1)]:
            print(f"  (m={mi+1},n={ni+1}): ", end="")
            for scale in [4, 8, 16, 32, 64]:
                n1 = scale * 2
                n2 = scale * 3
                data = dft_neumann_matrix(n1, n2)
                mat = data[block]
                if mi < mat.shape[0] and ni < mat.shape[1]:
                    if block == 'N11':
                        denominator = np.sqrt(
                            data['M1'][mi, mi] * data['M1'][ni, ni]
                        )
                    else:
                        denominator = np.sqrt(
                            data['M1'][mi, mi] * data['M2'][ni, ni]
                        )
                    val = mat[mi, ni] / denominator
                    if abs(val.imag) < 1e-10 * max(abs(val.real), 1e-15):
                        print(f" N1={n1:4d}: {val.real:11.6f}", end="")
                    else:
                        print(f" N1={n1:4d}: {val:11.6f}", end="")
            print()
        print()

    # Check completeness in DFT basis
    print("--- DFT-basis completeness check ---")
    for scale in [4, 8, 16]:
        n1 = scale * 2
        n2 = scale * 3
        data = dft_neumann_matrix(n1, n2)
        u1, u2, xi = data['U1'], data['U2'], data['xi']
        n3 = n1 + n2
        xi_hat = xi * np.sqrt(n3 / (n1 * n2))
        comp = u1 @ u1.conj().T + u2 @ u2.conj().T + np.outer(xi_hat, xi_hat.conj())
        comp_err = np.max(np.abs(comp - np.eye(n3 - 1)))
        print(f"  N1={n1:4d}: |U1 U1† + U2 U2† + ξ̂ ξ̂† - I| = {comp_err:.2e}")
    print()

    # Check that N^{11} is Hermitian (should be for real-valued fields)
    print("--- Hermiticity check ---")
    for scale in [4, 8]:
        n1 = scale * 2
        n2 = scale * 3
        data = dft_neumann_matrix(n1, n2)
        n11 = data['N11']
        herm_err = np.max(np.abs(n11 - n11.conj().T))
        imag_diag = np.max(np.abs(np.diag(n11).imag))
        print(f"  N1={n1:4d}: |N11 - N11†| = {herm_err:.2e}, max|Im(diag)| = {imag_diag:.2e}")
    print()


def main():
    print_neumann_matrix_structure(8, 12)
    print_neumann_matrix_structure(32, 48)
    print_neumann_diagnostics()
    print_continuum_comparison(64, 96)
    print()
    print_dft_neumann_convergence()


if __name__ == "__main__":
    main()
