#!/usr/bin/env python3
r"""
Diagnostics for the bosonic two-tachyon/one-massless cubic amplitude.

The setup matches the bosonic three-tachyon calculation in `tachyon_check.py`,
but now the outgoing leg carries the first oscillator-level insertion

    O_{1,N}^{IJ} = C_{1,N}^{-1} (X_1^I X_{-1}^J - C_{1,N} delta^{IJ}),

with the normalization fixed by the exact one-string vacuum two-point function
in the same finite-N conventions:

    C_{1,N} = < X_1 X_{-1} >_0 = pi alpha' / [2 N sin(pi/N)].

Here `X_{±1}` are the boundary Fourier coefficients with the branch-note
normalization

    X_m = (1/N) sum_n X_n exp[-2 pi i m n / N],

which in the main note are simply the nonzero Fourier coordinates divided by
sqrt(N): X_m = q_m / sqrt(N).

For outgoing leg 3 the discrete cubic overlap gives

    X_1^(3) = L_+ Q + zeta_+ y,
    X_-1^(3) = L_- Q + zeta_- y,

where `Q` is the stacked incoming real nonzero-mode variable and `y` is the
relative center-of-mass displacement. After the exact Gaussian integration over
Q and the Fourier transform in y, the matrix element takes the form

    < O_{1,N3}^{IJ} > = A_vac [ A_tr delta^{IJ} - B_rel q_rel^I q_rel^J ].

The script computes the reduced coefficients

    A_tr = ((Sigma_1 - C_1) + mu_+ mu_- / gamma_T) / C_1,
    B_rel = (mu_+ mu_-) / (C_1 gamma_T^2),

where

    Sigma_1 = L_+ G_T^{-1} L_-^T,
    mu_± = zeta_± - L_± G_T^{-1} B_T.

On the two-tachyon/one-massless shell, the relative momentum is fixed by the
longitudinal data to q_rel^2 = 4 / alpha', independent of alpha_1 / alpha_3.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tachyon_check as tc


@dataclass
class MasslessData:
    n1: int
    n2: int
    n3: int
    gamma_t: float
    c1_numeric: float
    c1_formula: float
    sigma_1: float
    trace_free_reduced: float
    trace_source_reduced: float
    a_trace_reduced: float
    b_rel_reduced: float
    mu_plus: complex
    mu_minus: complex
    q_rel_sq: float


DEFAULT_PAIRS = [(4, 4), (8, 12), (16, 24), (32, 48), (64, 96), (128, 192)]
RATIO_SCAN_FAMILIES = [(1, 3), (1, 2), (3, 5), (2, 3), (1, 1)]


def parse_pair(text: str) -> tuple[int, int]:
    pieces = text.split(",")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError(f"expected N1,N2 pair, got {text!r}")
    n1, n2 = (int(piece) for piece in pieces)
    if n1 < 2 or n2 < 2:
        raise argparse.ArgumentTypeError("require N1,N2 >= 2")
    return n1, n2


def first_harmonic_rows(n_sites: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return row vectors extracting X_{+1} and X_{-1} from the real zero-sum basis.

    In the main note's notation q_m = N^{-1/2} sum_n X_n exp[-2 pi i m n / N],
    while the older branch note writes X_m with an extra 1/sqrt(N). The rows
    below therefore implement X_{±1} = q_{±1} / sqrt(N) directly.
    """
    basis, _ = tc.real_zero_sum_basis(n_sites)
    sites = np.arange(n_sites, dtype=float)
    row_plus = np.exp(-2.0j * math.pi * sites / n_sites) / n_sites
    row_minus = np.exp(+2.0j * math.pi * sites / n_sites) / n_sites
    return row_plus @ basis, row_minus @ basis


def first_harmonic_covariance(n_sites: int, alpha_prime: float) -> tuple[float, float]:
    """
    Compute the exact free covariance C_{1,N} in two equivalent ways.

    The numeric value comes from the finite-N Gaussian in the real basis.
    The closed form uses mu omega_1 = sin(pi/N)/(pi alpha').
    """
    metric = tc.mode_metric(n_sites, alpha_prime)
    row_plus, row_minus = first_harmonic_rows(n_sites)
    numeric = 0.5 * float((row_plus @ np.linalg.solve(metric, row_minus)).real)
    formula = math.pi * alpha_prime / (2.0 * n_sites * math.sin(math.pi / n_sites))
    return numeric, formula


def on_shell_q_rel_sq_two_tachyons_one_massless(alpha_prime: float) -> float:
    """
    On-shell relative transverse momentum for T + T -> M.

    Writing p_1 = alpha_1 p_3 / alpha_3 + q_rel and
    p_2 = alpha_2 p_3 / alpha_3 - q_rel, the p_3-dependent terms cancel in
    lightcone energy conservation, leaving q_rel^2 = 4 / alpha'.
    """
    return 4.0 / alpha_prime


def compute_massless_data(n1: int, n2: int, alpha_prime: float = 1.0) -> MasslessData:
    n3 = n1 + n2

    m1 = tc.mode_metric(n1, alpha_prime)
    m2 = tc.mode_metric(n2, alpha_prime)
    m3 = tc.mode_metric(n3, alpha_prime)
    u1, u2, xi = tc.overlap_data(n1, n2)

    g_t = np.block(
        [
            [m1 + u1.T @ m3 @ u1, u1.T @ m3 @ u2],
            [u2.T @ m3 @ u1, m2 + u2.T @ m3 @ u2],
        ]
    )
    b_t = np.concatenate((u1.T @ m3 @ xi, u2.T @ m3 @ xi))
    solve_g_b = np.linalg.solve(g_t, b_t)
    gamma_t = float(xi.T @ m3 @ xi - b_t.T @ solve_g_b)

    c1_numeric, c1_formula = first_harmonic_covariance(n3, alpha_prime)
    row_plus, row_minus = first_harmonic_rows(n3)

    l_plus = np.concatenate((row_plus @ u1, row_plus @ u2))
    l_minus = np.concatenate((row_minus @ u1, row_minus @ u2))
    zeta_plus = row_plus @ xi
    zeta_minus = row_minus @ xi

    sigma_1 = float((l_plus @ np.linalg.solve(g_t, l_minus.T)).real)
    mu_plus = zeta_plus - l_plus @ solve_g_b
    mu_minus = zeta_minus - l_minus @ solve_g_b
    mu_prod = float((mu_plus * mu_minus).real)

    trace_free_reduced = (sigma_1 - c1_numeric) / c1_numeric
    trace_source_reduced = mu_prod / (c1_numeric * gamma_t)
    a_trace_reduced = trace_free_reduced + trace_source_reduced
    b_rel_reduced = mu_prod / (c1_numeric * gamma_t * gamma_t)

    return MasslessData(
        n1=n1,
        n2=n2,
        n3=n3,
        gamma_t=gamma_t,
        c1_numeric=c1_numeric,
        c1_formula=c1_formula,
        sigma_1=sigma_1,
        trace_free_reduced=float(trace_free_reduced),
        trace_source_reduced=float(trace_source_reduced),
        a_trace_reduced=float(a_trace_reduced),
        b_rel_reduced=float(b_rel_reduced),
        mu_plus=complex(mu_plus),
        mu_minus=complex(mu_minus),
        q_rel_sq=on_shell_q_rel_sq_two_tachyons_one_massless(alpha_prime),
    )


def print_covariance_check(alpha_prime: float) -> None:
    print("=" * 86)
    print("FIRST-HARMONIC VACUUM COVARIANCE")
    print("=" * 86)
    print(
        "C_{1,N} = <X_1 X_{-1}>_0 computed from the real-basis Gaussian and "
        "compared with pi alpha' / [2 N sin(pi/N)]."
    )
    print()
    print(f"{'N':>6s} {'numeric':>18s} {'formula':>18s} {'abs diff':>12s}")
    print("-" * 62)
    for n_sites in [4, 8, 16, 32, 64, 128]:
        numeric, formula = first_harmonic_covariance(n_sites, alpha_prime)
        print(
            f"{n_sites:6d} {numeric:18.12f} {formula:18.12f} "
            f"{abs(numeric - formula):12.3e}"
        )
    print()


def print_samples(pairs: list[tuple[int, int]], alpha_prime: float) -> None:
    print("=" * 122)
    print("BOSONIC TWO-TACHYON / ONE-MASSLESS DIAGNOSTIC")
    print("=" * 122)
    print(
        "Reduced coefficients in <O_{1,N3}^{IJ}> = A_vac [A_tr delta^{IJ} - "
        "B_rel q_rel^I q_rel^J] with the exact normalized insertion"
    )
    print(
        "O_{1,N}^{IJ} = C_{1,N}^{-1}(X_1^I X_{-1}^J - C_{1,N} delta^{IJ})."
    )
    print()
    print(
        f"{'N1':>4s} {'N2':>4s} {'N3':>4s} "
        f"{'C1':>12s} {'gamma_T':>12s} {'(Sigma-C)/C':>14s} "
        f"{'mu mu/(C gamma)':>16s} {'A_tr':>12s} {'B_rel':>12s}"
    )
    print("-" * 122)
    for n1, n2 in pairs:
        data = compute_massless_data(n1, n2, alpha_prime)
        print(
            f"{data.n1:4d} {data.n2:4d} {data.n3:4d} "
            f"{data.c1_numeric:12.9f} {data.gamma_t:12.9f} "
            f"{data.trace_free_reduced:14.9f} "
            f"{data.trace_source_reduced:16.9f} "
            f"{data.a_trace_reduced:12.9f} {data.b_rel_reduced:12.9f}"
        )
    print()
    print(f"On-shell q_rel^2 for T + T -> M: {4.0 / alpha_prime:.12f}")
    print()


def print_fixed_ratio_fit(alpha_prime: float) -> None:
    pairs = [(16, 24), (32, 48), (64, 96), (128, 192), (256, 384)]
    design = []
    target_a = []
    target_b = []
    print("=" * 86)
    print("FIXED-RATIO FAMILY N1:N2 = 2:3")
    print("=" * 86)
    print(f"{'N1':>6s} {'N2':>6s} {'A_tr':>16s} {'B_rel':>16s}")
    print("-" * 50)
    for n1, n2 in pairs:
        data = compute_massless_data(n1, n2, alpha_prime)
        design.append([1.0, 1.0 / n1])
        target_a.append(data.a_trace_reduced)
        target_b.append(data.b_rel_reduced)
        print(
            f"{n1:6d} {n2:6d} "
            f"{data.a_trace_reduced:16.12f} {data.b_rel_reduced:16.12f}"
        )
    coeff_a, _, _, _ = np.linalg.lstsq(np.array(design), np.array(target_a), rcond=None)
    coeff_b, _, _, _ = np.linalg.lstsq(np.array(design), np.array(target_b), rcond=None)
    print()
    print(
        "Least-squares fits on this family:"
        f"  A_tr ~= {coeff_a[0]:.9e} + {coeff_a[1]:.9e} / N1"
    )
    print(
        f"  B_rel ~= {coeff_b[0]:.9e} + {coeff_b[1]:.9e} / N1"
    )
    print()


def print_ratio_scan(scale: int, alpha_prime: float) -> None:
    print("=" * 86)
    print(f"RATIO SCAN AT N1 = {scale} a, N2 = {scale} b")
    print("=" * 86)
    print(f"{'lambda':>10s} {'N1 A_tr':>14s} {'B_rel':>14s}")
    print("-" * 42)
    for a, b in RATIO_SCAN_FAMILIES:
        n1 = scale * a
        n2 = scale * b
        data = compute_massless_data(n1, n2, alpha_prime)
        lam = n1 / (n1 + n2)
        print(
            f"{lam:10.6f} {n1 * data.a_trace_reduced:14.9f} "
            f"{data.b_rel_reduced:14.9f}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alpha-prime",
        type=float,
        default=1.0,
        help="target-space alpha' used in the lattice frequencies",
    )
    parser.add_argument(
        "--pair",
        type=parse_pair,
        action="append",
        help="specific N1,N2 pair to print; may be repeated",
    )
    parser.add_argument(
        "--skip-default-report",
        action="store_true",
        help="suppress the covariance table and the standard sample reports",
    )
    parser.add_argument(
        "--ratio-scan-scale",
        type=int,
        default=128,
        help="base scale used for the ratio scan families",
    )
    args = parser.parse_args()

    if args.pair:
        print_samples(args.pair, args.alpha_prime)
        return

    if not args.skip_default_report:
        print_covariance_check(args.alpha_prime)
        print_samples(DEFAULT_PAIRS, args.alpha_prime)
        print_fixed_ratio_fit(args.alpha_prime)
        print_ratio_scan(args.ratio_scan_scale, args.alpha_prime)


if __name__ == "__main__":
    main()
