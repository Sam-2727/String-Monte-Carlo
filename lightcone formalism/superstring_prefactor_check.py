#!/usr/bin/env python3
r"""
Diagnostics for the bosonic part of the discrete GS interaction-point prefactor.

The script works in the same independent real zero-sum basis used by
`tachyon_check.py`. For the minimal nearest-neighbor interaction-point stencils
of the note, it evaluates the exact finite-dimensional Gaussian coefficients
entering

    <0; p_1, p_2, p_3 | K_lat^I \widetilde K_lat^J | V_3^(B) >

and extracts the reduced tensor data

    delta^{IJ} A_delta(N) + q_rel^I q_rel^J B_qq(N),

up to the overall factors a^{-1} c_K \widetilde c_K and the common bosonic
vacuum amplitude.

This is not yet a full superstring amplitude computation. It isolates the
bosonic local-derivative part that must combine with the GS zero-mode
polynomial v_{IJ}(Lambda). It also contains the parity diagnostic explaining
why the raw first-order right-arc stencil forces eta_- = 0 and compares that
ansatz with second-order local alternatives.
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
class PrefactorData:
    n1: int
    n2: int
    n3: int
    gamma_t: float
    sigma_pm: float
    eta_plus: float
    eta_minus: float
    a_delta_reduced: float
    b_qq_reduced: float


def parse_pair(text: str) -> tuple[int, int]:
    pieces = text.split(",")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError(f"expected N1,N2 pair, got {text!r}")
    n1, n2 = (int(piece) for piece in pieces)
    if n1 < 2 or n2 < 2:
        raise argparse.ArgumentTypeError("require N1,N2 >= 2")
    return n1, n2


def local_stencil_rows(
    n1: int, n2: int, left_variant: str, right_variant: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return site-space row vectors for the two local bosonic prefactor stencils.

    Left arc, `minimal`: X_1^(1) - X_{I_+}.
    Right arc, `minimal`: X_{I_-} - X_{N_2-1}^(2), the choice used in the note.

    Two nearby alternatives and the one-sided second-order stencil already
    written in the note are provided as diagnostics showing that the vanishing
    of the right one-point coefficient is not a generic locality effect.
    """
    left = np.zeros(n1)
    if left_variant == "minimal":
        left[0] = -1.0
        left[1] = 1.0
    elif left_variant == "second_order":
        left[0] = -1.5
        left[1] = 2.0
        left[2] = -0.5
    else:
        raise ValueError(f"unknown left-arc stencil variant {left_variant!r}")

    right = np.zeros(n2)
    if right_variant == "minimal":
        right[0] = 1.0
        right[-1] = -1.0
    elif right_variant == "second_order":
        right[0] = 1.5
        right[-1] = -2.0
        right[-2] = 0.5
    elif right_variant == "forward_first":
        right[0] = -1.0
        right[1] = 1.0
    elif right_variant == "forward_last":
        right[-2] = -1.0
        right[-1] = 1.0
    else:
        raise ValueError(f"unknown right-arc stencil variant {right_variant!r}")

    return left, right


def one_sided_three_point_rows(
    n1: int, n2: int, left_t: float, right_t: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the general three-point one-sided local stencil family.

    On the left arc we act on (X_{I_+}, X_1, X_2) with coefficients

        (-1 + t_+, 1 - 2 t_+, t_+),

    while on the right arc we act on (X_{I_-}, X_{N_2-1}, X_{N_2-2}) with

        (1 + t_-, -1 - 2 t_-, t_-).

    These are the most general support-three one-sided stencils that:
      1. annihilate constants exactly, and
      2. differentiate linear profiles exactly.

    The first-order choices correspond to t_+ = t_- = 0. The standard
    one-sided second-order choices correspond to t_+ = -1/2 and t_- = +1/2.
    """
    if n1 < 3 or n2 < 3:
        raise ValueError("the three-point one-sided family requires N1,N2 >= 3")

    left = np.zeros(n1)
    left[0] = -1.0 + left_t
    left[1] = 1.0 - 2.0 * left_t
    left[2] = left_t

    right = np.zeros(n2)
    right[0] = 1.0 + right_t
    right[-1] = -1.0 - 2.0 * right_t
    right[-2] = right_t

    return left, right


def prefactor_data(
    n1: int,
    n2: int,
    alpha_prime: float = 1.0,
    left_variant: str = "minimal",
    right_variant: str = "minimal",
) -> PrefactorData:
    n3 = n1 + n2

    s1, _ = tc.real_zero_sum_basis(n1)
    s2, _ = tc.real_zero_sum_basis(n2)
    left_site, right_site = local_stencil_rows(n1, n2, left_variant, right_variant)
    left_mode = left_site @ s1
    right_mode = right_site @ s2

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
    gamma_t = float(xi.T @ m3 @ xi - b_t.T @ np.linalg.solve(g_t, b_t))

    left_full = np.concatenate((left_mode, np.zeros(n2 - 1)))
    right_full = np.concatenate((np.zeros(n1 - 1), right_mode))

    g_inv = np.linalg.inv(g_t)
    sigma_pm = float(left_full @ g_inv @ right_full)
    eta_plus = float(-(left_full @ g_inv @ b_t))
    eta_minus = float(-(right_full @ g_inv @ b_t))
    a_delta_reduced = float(sigma_pm + eta_plus * eta_minus / gamma_t)
    b_qq_reduced = float(-(eta_plus * eta_minus) / (gamma_t * gamma_t))

    return PrefactorData(
        n1=n1,
        n2=n2,
        n3=n3,
        gamma_t=gamma_t,
        sigma_pm=sigma_pm,
        eta_plus=eta_plus,
        eta_minus=eta_minus,
        a_delta_reduced=a_delta_reduced,
        b_qq_reduced=b_qq_reduced,
    )


def prefactor_data_three_point_family(
    n1: int,
    n2: int,
    left_t: float,
    right_t: float,
    alpha_prime: float = 1.0,
) -> PrefactorData:
    """Evaluate the prefactor data on the support-three one-sided stencil family."""
    n3 = n1 + n2

    s1, _ = tc.real_zero_sum_basis(n1)
    s2, _ = tc.real_zero_sum_basis(n2)
    left_site, right_site = one_sided_three_point_rows(n1, n2, left_t, right_t)
    left_mode = left_site @ s1
    right_mode = right_site @ s2

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
    gamma_t = float(xi.T @ m3 @ xi - b_t.T @ np.linalg.solve(g_t, b_t))

    left_full = np.concatenate((left_mode, np.zeros(n2 - 1)))
    right_full = np.concatenate((np.zeros(n1 - 1), right_mode))

    g_inv = np.linalg.inv(g_t)
    sigma_pm = float(left_full @ g_inv @ right_full)
    eta_plus = float(-(left_full @ g_inv @ b_t))
    eta_minus = float(-(right_full @ g_inv @ b_t))
    a_delta_reduced = float(sigma_pm + eta_plus * eta_minus / gamma_t)
    b_qq_reduced = float(-(eta_plus * eta_minus) / (gamma_t * gamma_t))

    return PrefactorData(
        n1=n1,
        n2=n2,
        n3=n3,
        gamma_t=gamma_t,
        sigma_pm=sigma_pm,
        eta_plus=eta_plus,
        eta_minus=eta_minus,
        a_delta_reduced=a_delta_reduced,
        b_qq_reduced=b_qq_reduced,
    )


def parity_diagnostics(n1: int, n2: int, alpha_prime: float = 1.0) -> tuple[float, float, float]:
    """
    Check the exact reflection symmetry responsible for eta_- = 0 for the
    minimal right-arc stencil.

    In the independent real bases, P = diag(P1, P2) is induced by reversing the
    site order separately on each incoming leg.
    """
    s1, _ = tc.real_zero_sum_basis(n1)
    s2, _ = tc.real_zero_sum_basis(n2)
    j1 = np.fliplr(np.eye(n1))
    j2 = np.fliplr(np.eye(n2))
    p1 = s1.T @ j1 @ s1
    p2 = s2.T @ j2 @ s2
    parity = np.block(
        [
            [p1, np.zeros((n1 - 1, n2 - 1))],
            [np.zeros((n2 - 1, n1 - 1)), p2],
        ]
    )

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

    comm_err = float(np.max(np.abs(parity @ g_t - g_t @ parity)))
    even_err = float(np.max(np.abs(parity @ b_t - b_t)))
    odd_err = float(np.max(np.abs(parity @ b_t + b_t)))
    return comm_err, even_err, odd_err


def print_samples() -> None:
    print("=" * 78)
    print("BOSONIC PART OF THE GS INTERACTION-POINT PREFACTOR")
    print("=" * 78)
    print(
        "Minimal right-arc stencil = X_{I_-} - X_{N_2-1}. "
        "Reduced coefficients are defined by"
    )
    print(
        "  <K_lat^I Ktilde_lat^J> / (a^{-1} c_K ctilde_K A_vac)"
        " = delta^{IJ} A_delta + q_rel^I q_rel^J B_qq"
    )
    print()
    print(
        f"{'N1':>4s} {'N2':>4s} {'N3':>4s} "
        f"{'gamma_T':>12s} {'Sigma':>12s} {'eta_+':>12s} "
        f"{'eta_-':>12s} {'A_delta':>12s} {'B_qq':>12s}"
    )
    print("-" * 92)
    for n1, n2 in [(4, 4), (8, 12), (16, 24), (32, 48)]:
        data = prefactor_data(n1, n2)
        print(
            f"{data.n1:4d} {data.n2:4d} {data.n3:4d} "
            f"{data.gamma_t:12.9f} {data.sigma_pm:12.9f} {data.eta_plus:12.9f} "
            f"{data.eta_minus:12.3e} {data.a_delta_reduced:12.9f} "
            f"{data.b_qq_reduced:12.3e}"
        )
    print()
    print(
        "On these samples eta_- is already at machine zero, so the minimal "
        "nearest-neighbor bosonic prefactor contributes only the delta^{IJ} "
        "piece and no q_rel^I q_rel^J term."
    )
    print()


def scan_minimal_right(max_n: int) -> None:
    max_abs_eta_minus = 0.0
    worst_pair: tuple[int, int] | None = None
    for n1 in range(4, max_n + 1):
        for n2 in range(4, max_n + 1):
            data = prefactor_data(n1, n2)
            if abs(data.eta_minus) > max_abs_eta_minus:
                max_abs_eta_minus = abs(data.eta_minus)
                worst_pair = (n1, n2)

    print("=" * 78)
    print("GRID SCAN OF THE MINIMAL RIGHT-ARC ONE-POINT COEFFICIENT")
    print("=" * 78)
    print(f"Grid: 4 <= N1,N2 <= {max_n}")
    print(f"max |eta_-| = {max_abs_eta_minus:.12e} at {worst_pair}")
    print()


def compare_right_variants(n1: int, n2: int) -> None:
    print("=" * 78)
    print(f"RIGHT-ARC STENCIL COMPARISON FOR (N1,N2)=({n1},{n2})")
    print("=" * 78)
    for variant in ["minimal", "second_order", "forward_first", "forward_last"]:
        data = prefactor_data(n1, n2, right_variant=variant)
        print(
            f"{variant:14s}  eta_- = {data.eta_minus: .12e}  "
            f"A_delta = {data.a_delta_reduced: .12e}  "
            f"B_qq = {data.b_qq_reduced: .12e}"
        )
    print()
    print(
        "The vanishing of eta_- is specific to the current minimal right-arc "
        "choice; nearby local site differences need not share it."
    )
    print()


def compare_orderings(n1: int, n2: int) -> None:
    print("=" * 78)
    print(f"LEFT/RIGHT STENCIL ORDERING COMPARISON FOR (N1,N2)=({n1},{n2})")
    print("=" * 78)
    cases = [
        ("minimal/minimal", "minimal", "minimal"),
        ("minimal/second", "minimal", "second_order"),
        ("second/second", "second_order", "second_order"),
        ("second/minimal", "second_order", "minimal"),
    ]
    for label, left_variant, right_variant in cases:
        data = prefactor_data(
            n1,
            n2,
            left_variant=left_variant,
            right_variant=right_variant,
        )
        print(
            f"{label:16s}  eta_+ = {data.eta_plus: .12e}  "
            f"eta_- = {data.eta_minus: .12e}  "
            f"A_delta = {data.a_delta_reduced: .12e}  "
            f"B_qq = {data.b_qq_reduced: .12e}"
        )
    print()
    print(
        "The essential obstruction is on the right arc, but a symmetric "
        "second-order choice on both arcs is the cleanest local candidate for "
        "subsequent supercharge-closure tests."
    )
    print()


def print_parity_explanation(n1: int, n2: int) -> None:
    comm_err, even_err, odd_err = parity_diagnostics(n1, n2)
    print("=" * 78)
    print(f"PARITY EXPLANATION FOR (N1,N2)=({n1},{n2})")
    print("=" * 78)
    print(
        "Incoming-leg site reversal induces a parity matrix P on the real "
        "mode variables. The exact discrete Gaussian satisfies:"
    )
    print(f"  max|P G_T - G_T P|   = {comm_err:.12e}")
    print(f"  max|P B_T - B_T|     = {even_err:.12e}")
    print(f"  max|P B_T + B_T|     = {odd_err:.12e}")
    print()
    print(
        "So G_T commutes with parity and B_T is parity-even. Therefore "
        "G_T^{-1} B_T is also parity-even. The minimal right-arc stencil "
        "X_{I_-} - X_{N_2-1} is parity-odd, which forces eta_- = 0 exactly."
    )
    print()


def print_second_order_asymptotics() -> None:
    """
    Track the fixed-ratio large-N behavior of the natural one-sided
    second-order right-arc stencil.

    This is the quantity relevant for deciding whether the restored
    q_rel^I q_rel^J term survives after the explicit overall a^{-1}
    prefactor is reinstated.
    """
    print("=" * 78)
    print("SECOND-ORDER RIGHT-ARC ASYMPTOTICS")
    print("=" * 78)
    print(
        "Fixed ratio N1:N2 = 2:3, so with alpha1 fixed one has a = alpha1/N1."
    )
    print(
        f"{'N1':>6s} {'N2':>6s} {'eta_-^(2)':>14s} {'Bqq^(2)':>14s} "
        f"{'N1*Bqq^(2)':>14s} {'sqrt(N1)*eta':>14s} {'A_delta^(2)':>14s}"
    )
    print("-" * 96)

    rows = []
    for scale in [8, 16, 32, 64, 128, 256, 512]:
        n1 = 2 * scale
        n2 = 3 * scale
        data = prefactor_data(n1, n2, right_variant="second_order")
        rows.append((n1, n2, data))
        print(
            f"{n1:6d} {n2:6d} {data.eta_minus:14.9f} {data.b_qq_reduced:14.9f} "
            f"{(n1 * data.b_qq_reduced):14.9f} {(math.sqrt(n1) * data.eta_minus):14.9f} "
            f"{data.a_delta_reduced:14.9f}"
        )

    xs = np.array([1.0 / n1 for n1, _, _ in rows])
    design = np.column_stack([np.ones_like(xs), xs])
    coeff_b = np.linalg.lstsq(
        design, np.array([n1 * data.b_qq_reduced for n1, _, data in rows]), rcond=None
    )[0]
    coeff_eta = np.linalg.lstsq(
        design,
        np.array([math.sqrt(n1) * data.eta_minus for n1, _, data in rows]),
        rcond=None,
    )[0]
    coeff_a = np.linalg.lstsq(
        design, np.array([data.a_delta_reduced for _, _, data in rows]), rcond=None
    )[0]

    print()
    print("Simple fixed-ratio fits:")
    print(f"  N1 * Bqq^(2)     ~= {coeff_b[0]:.9f} + {coeff_b[1]:.9f} / N1")
    print(f"  sqrt(N1) * eta_- ~= {coeff_eta[0]:.9f} + {coeff_eta[1]:.9f} / N1")
    print(f"  A_delta^(2)      ~= {coeff_a[0]:.9f} + {coeff_a[1]:.9f} / N1")
    print()
    print(
        "So for the second-order right-arc candidate the restored momentum-bilinear "
        "piece has Bqq^(2) = O(a), hence the full coefficient a^{-1} Bqq^(2) tends "
        "to a finite nonzero limit on this fixed-ratio family."
    )
    print()


def print_symmetric_second_order_asymptotics() -> None:
    print("=" * 78)
    print("SYMMETRIC SECOND-ORDER ASYMPTOTICS")
    print("=" * 78)
    print(
        "Both arcs use the natural one-sided second-order derivative stencils."
    )
    print(
        f"{'N1':>6s} {'N2':>6s} {'N1*Bqq':>14s} {'sqrt(N1)*eta_+':>16s} "
        f"{'sqrt(N1)*eta_-':>16s} {'A_delta':>14s}"
    )
    print("-" * 92)

    rows = []
    for scale in [8, 16, 32, 64, 128, 256]:
        n1 = 2 * scale
        n2 = 3 * scale
        data = prefactor_data(
            n1,
            n2,
            left_variant="second_order",
            right_variant="second_order",
        )
        rows.append((n1, n2, data))
        print(
            f"{n1:6d} {n2:6d} {(n1 * data.b_qq_reduced):14.9f} "
            f"{(math.sqrt(n1) * data.eta_plus):16.9f} "
            f"{(math.sqrt(n1) * data.eta_minus):16.9f} "
            f"{data.a_delta_reduced:14.9f}"
        )

    xs = np.array([1.0 / n1 for n1, _, _ in rows])
    design = np.column_stack([np.ones_like(xs), xs])
    coeff_b = np.linalg.lstsq(
        design, np.array([n1 * data.b_qq_reduced for n1, _, data in rows]), rcond=None
    )[0]
    coeff_ep = np.linalg.lstsq(
        design,
        np.array([math.sqrt(n1) * data.eta_plus for n1, _, data in rows]),
        rcond=None,
    )[0]
    coeff_em = np.linalg.lstsq(
        design,
        np.array([math.sqrt(n1) * data.eta_minus for n1, _, data in rows]),
        rcond=None,
    )[0]
    coeff_a = np.linalg.lstsq(
        design, np.array([data.a_delta_reduced for _, _, data in rows]), rcond=None
    )[0]
    print()
    print("Simple fixed-ratio fits:")
    print(f"  N1 * Bqq            ~= {coeff_b[0]:.9f} + {coeff_b[1]:.9f} / N1")
    print(f"  sqrt(N1) * eta_+    ~= {coeff_ep[0]:.9f} + {coeff_ep[1]:.9f} / N1")
    print(f"  sqrt(N1) * eta_-    ~= {coeff_em[0]:.9f} + {coeff_em[1]:.9f} / N1")
    print(f"  A_delta             ~= {coeff_a[0]:.9f} + {coeff_a[1]:.9f} / N1")
    print()


def print_ratio_samples(scale: int) -> None:
    print("=" * 78)
    print("SYMMETRIC SECOND-ORDER RATIO SAMPLES")
    print("=" * 78)
    print(
        "Large-N sample with both arcs using second-order stencils and"
        f" N1 = {scale} * a, N2 = {scale} * b."
    )
    print(
        f"{'a':>3s} {'b':>3s} {'lambda':>8s} {'N1*Bqq':>14s} "
        f"{'sqrt(N1)*eta_+':>16s} {'sqrt(N1)*eta_-':>16s} {'A_delta':>14s}"
    )
    print("-" * 96)
    for a, b in [(1, 3), (1, 2), (3, 5), (2, 3), (1, 1)]:
        n1 = scale * a
        n2 = scale * b
        data = prefactor_data(
            n1,
            n2,
            left_variant="second_order",
            right_variant="second_order",
        )
        lam = a / (a + b)
        print(
            f"{a:3d} {b:3d} {lam:8.5f} {(n1 * data.b_qq_reduced):14.9f} "
            f"{(math.sqrt(n1) * data.eta_plus):16.9f} "
            f"{(math.sqrt(n1) * data.eta_minus):16.9f} "
            f"{data.a_delta_reduced:14.9f}"
        )
    print()
    print(
        "The coefficients vary smoothly with lambda = alpha1 / alpha3, which is "
        "evidence that the symmetric second-order ansatz has a genuine "
        "ratio-dependent continuum profile rather than a single lucky fixed-ratio limit."
    )
    print()


def print_three_point_family_scan(n1: int, n2: int) -> None:
    """
    Sample the finite local support-three stencil family.

    This is the smallest honest family in which the bosonic prefactor test can
    distinguish:
      - the parity-odd right-arc first-order point t_- = 0,
      - the standard second-order point (t_+, t_-) = (-1/2, +1/2),
      - nearby local admixtures that supercharge closure may still need to fix.
    """
    print("=" * 78)
    print(f"THREE-POINT ONE-SIDED STENCIL FAMILY FOR (N1,N2)=({n1},{n2})")
    print("=" * 78)
    print(
        "Left stencil  = (-1+t_+, 1-2 t_+, t_+) on (I_+,1,2); "
        "right stencil = (1+t_-, -1-2 t_-, t_-) on (I_-,N2-1,N2-2)."
    )
    print(
        "Both annihilate constants and differentiate linear profiles exactly."
    )
    print(
        f"{'t_+':>8s} {'t_-':>8s} {'eta_+':>14s} {'eta_-':>14s} "
        f"{'N1*Bqq':>14s} {'A_delta':>14s}"
    )
    print("-" * 88)
    sample_points = [
        (0.0, 0.0),
        (0.0, 0.5),
        (-0.5, 0.5),
        (-0.25, 0.25),
        (-0.75, 0.75),
    ]
    for left_t, right_t in sample_points:
        data = prefactor_data_three_point_family(n1, n2, left_t, right_t)
        print(
            f"{left_t:8.3f} {right_t:8.3f} {data.eta_plus:14.9f} "
            f"{data.eta_minus:14.9f} {(n1 * data.b_qq_reduced):14.9f} "
            f"{data.a_delta_reduced:14.9f}"
        )
    print()
    print(
        "The bosonic three-graviton test rules out the parity-odd right point "
        "t_- = 0, but it does not by itself uniquely fix the remaining local "
        "coefficient. The standard second-order choice is the unique O(a^2) "
        "accurate member of this family."
    )
    print()


def print_three_point_family_asymptotics() -> None:
    """Track the large-N behavior of several representative family choices."""
    print("=" * 78)
    print("THREE-POINT FAMILY ASYMPTOTICS")
    print("=" * 78)
    print("Fixed ratio N1:N2 = 2:3.")
    families = [
        ("minimal/minimal", 0.0, 0.0),
        ("minimal/second", 0.0, 0.5),
        ("second/second", -0.5, 0.5),
        ("midpoint", -0.25, 0.25),
    ]
    header = f"{'family':>16s} {'lim N1*Bqq':>14s} {'lim sqrt(N1)eta_+':>18s} {'lim sqrt(N1)eta_-':>18s}"
    print(header)
    print("-" * len(header))
    for label, left_t, right_t in families:
        rows = []
        for scale in [8, 16, 32, 64, 128, 256]:
            n1 = 2 * scale
            n2 = 3 * scale
            data = prefactor_data_three_point_family(n1, n2, left_t, right_t)
            rows.append((n1, data))
        xs = np.array([1.0 / n1 for n1, _ in rows])
        design = np.column_stack([np.ones_like(xs), xs])
        coeff_b = np.linalg.lstsq(
            design, np.array([n1 * data.b_qq_reduced for n1, data in rows]), rcond=None
        )[0]
        coeff_ep = np.linalg.lstsq(
            design,
            np.array([math.sqrt(n1) * data.eta_plus for n1, data in rows]),
            rcond=None,
        )[0]
        coeff_em = np.linalg.lstsq(
            design,
            np.array([math.sqrt(n1) * data.eta_minus for n1, data in rows]),
            rcond=None,
        )[0]
        print(
            f"{label:16s} {coeff_b[0]:14.9f} {coeff_ep[0]:18.9f} {coeff_em[0]:18.9f}"
        )
    print()
    print(
        "Any right-arc choice with t_- != 0 restores a finite continuum "
        "momentum-bilinear coefficient. The supercharge algebra, not this "
        "bosonic test alone, must decide which local admixture is physical."
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-minimal-right-max-n",
        type=int,
        default=None,
        help="scan 4 <= N1,N2 <= MAX and report the largest |eta_-|",
    )
    parser.add_argument(
        "--compare-right-variants",
        type=parse_pair,
        default=None,
        metavar="N1,N2",
        help="compare the minimal right arc choice with two nearby local alternatives",
    )
    parser.add_argument(
        "--parity-explanation",
        type=parse_pair,
        default=None,
        metavar="N1,N2",
        help="print the exact parity diagnostic explaining eta_- = 0 for the minimal right arc",
    )
    parser.add_argument(
        "--second-order-asymptotics",
        action="store_true",
        help="print the fixed-ratio large-N behavior of the natural second-order right-arc stencil",
    )
    parser.add_argument(
        "--compare-orderings",
        type=parse_pair,
        default=None,
        metavar="N1,N2",
        help="compare minimal/minimal, minimal/second, and symmetric second-order local stencils",
    )
    parser.add_argument(
        "--symmetric-second-order-asymptotics",
        action="store_true",
        help="print the fixed-ratio large-N behavior when both arcs use second-order stencils",
    )
    parser.add_argument(
        "--ratio-samples",
        action="store_true",
        help="print large-N symmetric second-order samples across several split ratios",
    )
    parser.add_argument(
        "--ratio-sample-scale",
        type=int,
        default=128,
        help="base scale used by --ratio-samples; N1=scale*a and N2=scale*b",
    )
    parser.add_argument(
        "--three-point-family",
        type=parse_pair,
        default=None,
        metavar="N1,N2",
        help="sample the general support-three one-sided stencil family",
    )
    parser.add_argument(
        "--three-point-family-asymptotics",
        action="store_true",
        help="compare large-N limits for representative support-three one-sided families",
    )
    args = parser.parse_args()

    print_samples()
    if args.scan_minimal_right_max_n is not None:
        scan_minimal_right(args.scan_minimal_right_max_n)
    if args.compare_right_variants is not None:
        compare_right_variants(*args.compare_right_variants)
    if args.parity_explanation is not None:
        print_parity_explanation(*args.parity_explanation)
    if args.second_order_asymptotics:
        print_second_order_asymptotics()
    if args.compare_orderings is not None:
        compare_orderings(*args.compare_orderings)
    if args.symmetric_second_order_asymptotics:
        print_symmetric_second_order_asymptotics()
    if args.ratio_samples:
        print_ratio_samples(args.ratio_sample_scale)
    if args.three_point_family is not None:
        print_three_point_family_scan(*args.three_point_family)
    if args.three_point_family_asymptotics:
        print_three_point_family_asymptotics()


if __name__ == "__main__":
    main()
