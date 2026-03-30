#!/usr/bin/env python3
r"""
Diagnostics for the formal fermionic cubic-overlap squeeze notation.

The fermionic overlap constraints in the main note are written in the reduced
real nonzero-mode basis as

    (M_F B + N_F B^\dagger) |V_kin^(F)> = 0.

For an ordinary finite-dimensional Grassmann squeezed state

    exp[1/2 B^\dagger S B^\dagger] |0>,

the naive inversion S = -M_F^{-1} N_F would have to be antisymmetric. This
script checks the relevant algebraic obstruction and the resulting antisymmetry
defect on representative joins.
"""

from __future__ import annotations

import numpy as np

import tachyon_check as tc


def fermionic_overlap_matrices(n1: int, n2: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the reduced-basis fermionic overlap matrices M_F and N_F."""
    n3 = n1 + n2
    d1, d2, d3 = n1 - 1, n2 - 1, n3 - 1
    u1, u2, _ = tc.overlap_data(n1, n2)

    c_theta = np.hstack([u1, u2, -np.eye(d3)])
    c_pi = np.zeros((d1 + d2, d1 + d2 + d3))
    c_pi[:d1, :d1] = np.eye(d1)
    c_pi[:d1, d1 + d2 :] = -u1.T
    c_pi[d1:, d1 : d1 + d2] = np.eye(d2)
    c_pi[d1:, d1 + d2 :] = -u2.T

    m_f = np.vstack([c_theta, -1j * c_pi])
    n_f = np.vstack([c_theta, +1j * c_pi])
    return m_f, n_f


def diagnostics(n1: int, n2: int) -> tuple[float, float]:
    """
    Return the algebraic obstruction and the antisymmetry defect of S_naive.

    For S_naive = -M_F^{-1} N_F to be antisymmetric, one needs

        M_F N_F^T + N_F M_F^T = 0.
    """
    m_f, n_f = fermionic_overlap_matrices(n1, n2)
    algebraic_defect = float(np.max(np.abs(m_f @ n_f.T + n_f @ m_f.T)))
    s_naive = -np.linalg.solve(m_f, n_f)
    antisym_defect = float(np.max(np.abs(s_naive + s_naive.T)))
    return algebraic_defect, antisym_defect


def main() -> None:
    print("=" * 78)
    print("FERMIONIC OVERLAP DIAGNOSTIC")
    print("=" * 78)
    print(
        "For the formal inversion S_F = -M_F^{-1} N_F, an ordinary Grassmann\n"
        "squeeze matrix would require M_F N_F^T + N_F M_F^T = 0 and hence\n"
        "S_F + S_F^T = 0."
    )
    print()
    print(f"{'N1':>4s} {'N2':>4s} {'|M N^T + N M^T|':>22s} {'|S + S^T|':>16s}")
    print("-" * 52)
    for n1, n2 in [(4, 6), (8, 12), (16, 24), (32, 48)]:
        algebraic_defect, antisym_defect = diagnostics(n1, n2)
        print(
            f"{n1:4d} {n2:4d} {algebraic_defect:22.9e} {antisym_defect:16.9e}"
        )
    print()
    print(
        "These numbers show that the direct inversion is only a formal shorthand\n"
        "for the overlap constraints, not a verified finite-N antisymmetric\n"
        "Grassmann squeeze matrix."
    )


if __name__ == "__main__":
    main()
