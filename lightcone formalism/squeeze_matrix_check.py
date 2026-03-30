#!/usr/bin/env python3
r"""
Neumann coefficient extraction and operator-level Lorentz diagnostics.

The cubic overlap state is a boundary state (position-space delta function),
so the naive squeeze matrix S = -M^{-1} N from the annihilation constraints
is ill-conditioned: the symplectic compatibility condition C_Q C_P^T = 0 fails,
with C_Q C_P^T = 2(U_1, U_2) != 0.

Instead, this script extracts the Neumann coefficients directly from
position-space Gaussian matrix elements using

    Nbar^{rs}_{mn} = -<0| a_m^{(r)} a_n^{(s)} |V_3> / <0|V_3>.

For incoming legs (r,s in {1,2}), this reduces to

    Nbar^{rs}_{mn} = -L_m^{(r)} G_T^{-1} (L_n^{(s)})^T

where L_m^{(r)} is the effective linear functional of the two-leg Gaussian.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tachyon_check as tc


# ---------------------------------------------------------------------------
# Symplectic compatibility diagnostic
# ---------------------------------------------------------------------------

def check_symplectic_compatibility(n1: int, n2: int) -> float:
    """
    Verify C_Q C_P^T = 2(U_1, U_2) != 0.

    This is the obstruction to the naive M^{-1}N squeeze matrix being symmetric.
    """
    n3 = n1 + n2
    d1, d2, d3 = n1 - 1, n2 - 1, n3 - 1
    d_total = d1 + d2 + d3

    u1, u2, xi = tc.overlap_data(n1, n2)

    c_q = np.hstack([u1, u2, -np.eye(d3)])
    c_p = np.zeros((d1 + d2, d_total))
    c_p[:d1, :d1] = np.eye(d1)
    c_p[:d1, d1 + d2:] = -u1.T
    c_p[d1:, d1:d1 + d2] = np.eye(d2)
    c_p[d1:, d1 + d2:] = -u2.T

    cross = c_q @ c_p.T
    expected = 2.0 * np.hstack([u1, u2])
    return float(np.max(np.abs(cross - expected)))


# ---------------------------------------------------------------------------
# Effective linear functionals for the Gaussian extraction
# ---------------------------------------------------------------------------

def _gaussian_data(n1: int, n2: int, alpha_prime: float = 1.0):
    """Shared Gaussian infrastructure."""
    n3 = n1 + n2
    d1, d2 = n1 - 1, n2 - 1
    m1 = tc.mode_metric(n1, alpha_prime)
    m2 = tc.mode_metric(n2, alpha_prime)
    m3 = tc.mode_metric(n3, alpha_prime)
    u1, u2, xi = tc.overlap_data(n1, n2)

    g_t = np.block([
        [m1 + u1.T @ m3 @ u1, u1.T @ m3 @ u2],
        [u2.T @ m3 @ u1, m2 + u2.T @ m3 @ u2],
    ])
    b_t = np.concatenate((u1.T @ m3 @ xi, u2.T @ m3 @ xi))
    gamma_t = float(xi.T @ m3 @ xi - b_t.T @ np.linalg.solve(g_t, b_t))
    g_inv = np.linalg.inv(g_t)

    return {
        "n1": n1, "n2": n2, "n3": n3, "d1": d1, "d2": d2,
        "m1": m1, "m2": m2, "m3": m3,
        "u1": u1, "u2": u2, "xi": xi,
        "g_t": g_t, "b_t": b_t, "gamma_t": gamma_t, "g_inv": g_inv,
    }


def effective_rows(n1: int, n2: int, alpha_prime: float = 1.0):
    r"""
    Build effective linear functionals L^{(r)}_m for r=1,2,3.

    For incoming legs:
        a_m^{(r)} Psi / Psi  =  L_m^{(r)} . Q  +  zeta_m^{(r)} y

    where Psi is the combined bra-ket Gaussian on the two-leg space.

    For the outgoing leg 3, the bra-side effective insertion is:
        a_k^{(3) dag} psi_0^{(3)} = sqrt(2 d_k) q_k^{(3)} psi_0^{(3)}

    which after the overlap gives
        L_k^{(3)} . Q + zeta_k^{(3)} y
    """
    data = _gaussian_data(n1, n2, alpha_prime)
    d1, d2 = data["d1"], data["d2"]
    g_t = data["g_t"]
    b_t = data["b_t"]
    m1, m2, m3 = data["m1"], data["m2"], data["m3"]
    u1, u2, xi = data["u1"], data["u2"], data["xi"]
    d_total = d1 + d2

    # Incoming leg 1: a_m acting on ket
    # a_m Psi/Psi = sqrt(d_m/2) Q_m - (1/sqrt(2 d_m)) (G_T Q + y B_T)_m
    d1_diag = np.diag(m1)  # mu_1 omega_m
    leg1_embed = np.hstack([np.eye(d1), np.zeros((d1, d2))])
    l1 = np.diag(np.sqrt(d1_diag / 2.0)) @ leg1_embed \
        - np.diag(1.0 / np.sqrt(2.0 * d1_diag)) @ g_t[:d1, :]
    z1 = -b_t[:d1] / np.sqrt(2.0 * d1_diag)

    # Incoming leg 2
    d2_diag = np.diag(m2)
    leg2_embed = np.hstack([np.zeros((d2, d1)), np.eye(d2)])
    l2 = np.diag(np.sqrt(d2_diag / 2.0)) @ leg2_embed \
        - np.diag(1.0 / np.sqrt(2.0 * d2_diag)) @ g_t[d1:, :]
    z2 = -b_t[d1:] / np.sqrt(2.0 * d2_diag)

    # Outgoing leg 3: bra-side insertion a_k^dag psi_0^{(3)} = sqrt(2d_k) q_k psi_0
    # After overlap: q_k^{(3)} = ((U1,U2) Q + xi y)_k
    d3_diag = np.diag(m3)
    u_stack = np.hstack([u1, u2])  # (d3, d1+d2)
    l3 = np.diag(np.sqrt(2.0 * d3_diag)) @ u_stack
    z3 = np.sqrt(2.0 * d3_diag) * xi

    return {"L1": l1, "L2": l2, "L3": l3,
            "z1": z1, "z2": z2, "z3": z3, **data}


def neumann_block(eff, r: int, s: int) -> np.ndarray:
    r"""
    Neumann coefficient block Nbar^{rs}_{mn} = -L_m^{(r)} G_T^{-1} L_n^{(s)}^T.

    This is the y^0 (connected, displacement-independent) part of
    <0|a_m^{(r)} a_n^{(s)}|V_3> / <0|V_3>.
    """
    lr = eff[f"L{r}"]
    ls = eff[f"L{s}"]
    g_inv = eff["g_inv"]
    return -(lr @ g_inv @ ls.T)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_symplectic_obstruction(pairs: list[tuple[int, int]]) -> None:
    print("=" * 78)
    print("SYMPLECTIC COMPATIBILITY OBSTRUCTION")
    print("=" * 78)
    print(
        "C_Q C_P^T should be zero for the naive M^{-1}N squeeze matrix to be\n"
        "symmetric. It is not: C_Q C_P^T = 2(U_1, U_2)."
    )
    print()
    print(f"{'N1':>4s} {'N2':>4s} {'|C_Q C_P^T - 2(U1,U2)|':>24s}")
    print("-" * 38)
    for n1, n2 in pairs:
        err = check_symplectic_compatibility(n1, n2)
        print(f"{n1:4d} {n2:4d} {err:24.3e}")
    print()


def print_neumann_symmetry(pairs: list[tuple[int, int]], ap: float) -> None:
    print("=" * 78)
    print("NEUMANN BLOCK SYMMETRY (GAUSSIAN EXTRACTION)")
    print("=" * 78)
    print(
        "Nbar^{rs} extracted from position-space Gaussian moments.\n"
        "Symmetry Nbar^{rs}_{mn} = Nbar^{sr}_{nm} is guaranteed by G_T^{-1} symmetry."
    )
    print()
    print(
        f"{'N1':>4s} {'N2':>4s} "
        f"{'|N11-N11^T|':>14s} {'|N12-N21^T|':>14s} "
        f"{'|N13-N31^T|':>14s} {'|N33-N33^T|':>14s}"
    )
    print("-" * 62)
    for n1, n2 in pairs:
        eff = effective_rows(n1, n2, ap)
        n11 = neumann_block(eff, 1, 1)
        n12 = neumann_block(eff, 1, 2)
        n21 = neumann_block(eff, 2, 1)
        n13 = neumann_block(eff, 1, 3)
        n31 = neumann_block(eff, 3, 1)
        n33 = neumann_block(eff, 3, 3)
        print(
            f"{n1:4d} {n2:4d} "
            f"{np.max(np.abs(n11 - n11.T)):14.3e} "
            f"{np.max(np.abs(n12 - n21.T)):14.3e} "
            f"{np.max(np.abs(n13 - n31.T)):14.3e} "
            f"{np.max(np.abs(n33 - n33.T)):14.3e}"
        )
    print()


def print_neumann_convergence(ap: float) -> None:
    print("=" * 78)
    print("NEUMANN COEFFICIENT CONVERGENCE (N1:N2 = 2:3)")
    print("=" * 78)
    print("Lowest entries of Nbar^{12} and Nbar^{13}:")
    print()
    print(
        f"{'N1':>6s} {'N2':>6s} "
        f"{'N12_00':>14s} {'N12_01':>14s} "
        f"{'N13_00':>14s} {'N33_00':>14s}"
    )
    print("-" * 68)
    for scale in [4, 8, 16, 32, 64]:
        n1 = 2 * scale
        n2 = 3 * scale
        eff = effective_rows(n1, n2, ap)
        n12 = neumann_block(eff, 1, 2)
        n13 = neumann_block(eff, 1, 3)
        n33 = neumann_block(eff, 3, 3)
        print(
            f"{n1:6d} {n2:6d} "
            f"{n12[0, 0]:14.9f} {n12[0, 1]:14.9f} "
            f"{n13[0, 0]:14.9f} {n33[0, 0]:14.9f}"
        )
    print()


def print_neumann_block_sample(n1: int, n2: int, ap: float) -> None:
    eff = effective_rows(n1, n2, ap)
    print("=" * 78)
    print(f"NEUMANN BLOCKS FOR (N1,N2) = ({n1},{n2})")
    print("=" * 78)

    for (r, s), label in [
        ((1, 2), "Nbar^{12}"), ((1, 1), "Nbar^{11}"),
        ((1, 3), "Nbar^{13}"), ((3, 3), "Nbar^{33}"),
    ]:
        blk = neumann_block(eff, r, s)
        nshow = min(5, blk.shape[0], blk.shape[1])
        print(f"\n  {label} ({blk.shape[0]}x{blk.shape[1]}), top-left {nshow}x{nshow}:")
        for i in range(nshow):
            row_str = "    "
            for j in range(nshow):
                row_str += f" {blk[i, j]:11.7f}"
            print(row_str)
    print()


# ---------------------------------------------------------------------------
# Hamiltonian matrix element in the one-excitation sector
# ---------------------------------------------------------------------------

def freq_weights(n_sites: int, alpha_prime: float = 1.0) -> np.ndarray:
    """omega_m / (2 p_r^+) = 2 alpha' sin(pi k / N) / N."""
    _, modes = tc.real_zero_sum_basis(n_sites)
    return 2.0 * alpha_prime * np.sin(math.pi * modes / n_sites) / n_sites


def print_hamiltonian_element(ap: float) -> None:
    print("=" * 78)
    print("ONE-EXCITATION HAMILTONIAN MATRIX ELEMENT")
    print("=" * 78)
    print(
        "H2_mn = -Nbar^{12}_{mn} (fw1_m + fw2_n)  where  fw_m = omega_m / (2 p^+)"
    )
    print()
    for scale in [4, 8, 16, 32]:
        n1 = 2 * scale
        n2 = 3 * scale
        eff = effective_rows(n1, n2, ap)
        n12 = neumann_block(eff, 1, 2)
        fw1 = freq_weights(n1, ap)
        fw2 = freq_weights(n2, ap)
        nshow = min(3, n12.shape[0], n12.shape[1])
        print(f"  (N1,N2) = ({n1},{n2}):")
        for m in range(nshow):
            for n in range(nshow):
                h2 = -n12[m, n] * (fw1[m] + fw2[n])
                print(f"    H2[{m},{n}] = {h2:14.9f}  (N12={n12[m,n]:.7f})")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_PAIRS = [(4, 6), (8, 12), (16, 24), (32, 48)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-prime", type=float, default=1.0)
    args = parser.parse_args()
    ap = args.alpha_prime

    print_symplectic_obstruction(DEFAULT_PAIRS)
    print_neumann_symmetry(DEFAULT_PAIRS, ap)
    print_neumann_block_sample(8, 12, ap)
    print_neumann_convergence(ap)
    print_hamiltonian_element(ap)


if __name__ == "__main__":
    main()
