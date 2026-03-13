#!/usr/bin/env python3
"""
Numerical bosonic three-tachyon check from the discrete cubic overlap.

This script works in an independent real zero-sum basis on each leg, computes
the Gaussian data G_T, B_T, C_T appearing in the note, and evaluates the
resulting on-shell quantity

    exp[- q_rel^2 / (2 gamma_T) ]

for sample joins N_3 = N_1 + N_2 with common lattice spacing.

The overall cubic normalization C_3^(B) is intentionally not fixed here. The
output is therefore the kinematic part that is already determined by the exact
discrete overlap, together with diagnostics for how the remaining normalization
factorizes across the three external legs.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class TachyonData:
    n1: int
    n2: int
    n3: int
    gamma_t: float
    q_rel_sq: float
    exponent: float
    log_prefactor_1d: float
    log_required_norm_noext: float
    completeness_error: float
    orthogonality_error: float


DEFAULT_PAIRS = [(2, 2), (3, 3), (4, 4), (8, 16)]
FIT_PAIRS = [
    (8, 8),
    (16, 16),
    (24, 24),
    (32, 32),
    (40, 40),
    (48, 48),
    (64, 64),
    (8, 16),
    (16, 32),
    (24, 48),
    (40, 80),
    (48, 96),
    (12, 18),
    (16, 24),
    (24, 36),
    (36, 54),
    (48, 72),
    (10, 20),
    (20, 40),
]


def real_zero_sum_basis(n_sites: int) -> tuple[np.ndarray, np.ndarray]:
    """Return an orthonormal real basis of the zero-sum subspace and mode labels."""
    n = np.arange(n_sites, dtype=float)
    columns: list[np.ndarray] = []
    modes: list[int] = []

    for k in range(1, (n_sites - 1) // 2 + 1):
        angle = 2.0 * math.pi * k * n / n_sites
        columns.append(math.sqrt(2.0 / n_sites) * np.cos(angle))
        columns.append(math.sqrt(2.0 / n_sites) * np.sin(angle))
        modes.extend([k, k])

    if n_sites % 2 == 0:
        columns.append(((-1.0) ** n) / math.sqrt(n_sites))
        modes.append(n_sites // 2)

    basis = np.column_stack(columns)
    mode_array = np.array(modes, dtype=int)

    eye_error = np.linalg.norm(basis.T @ basis - np.eye(n_sites - 1), ord=np.inf)
    zero_mode_error = np.linalg.norm(basis.T @ np.ones(n_sites), ord=np.inf)
    if eye_error > 1e-10 or zero_mode_error > 1e-10:
        raise RuntimeError(
            f"real_zero_sum_basis({n_sites}) failed orthonormality checks: "
            f"{eye_error=}, {zero_mode_error=}"
        )

    return basis, mode_array


def mode_metric(n_sites: int, alpha_prime: float) -> np.ndarray:
    """
    Diagonal metric M_r = (mu_r Omega_r)_phys in the real basis.

    For common lattice spacing a and alpha_r = a N_r one has
        mu_r omega_k = (1 / (pi alpha')) sin(pi k / N_r),
    which is independent of a.
    """
    _, modes = real_zero_sum_basis(n_sites)
    weights = np.sin(math.pi * modes / n_sites) / (math.pi * alpha_prime)
    return np.diag(weights)


def overlap_data(n1: int, n2: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return U_1, U_2, xi in the real physical basis."""
    n3 = n1 + n2
    s1, _ = real_zero_sum_basis(n1)
    s2, _ = real_zero_sum_basis(n2)
    s3, _ = real_zero_sum_basis(n3)

    p1 = np.zeros((n3, n1))
    p1[:n1, :] = np.eye(n1)
    p2 = np.zeros((n3, n2))
    p2[n1:, :] = np.eye(n2)

    u1 = s3.T @ p1 @ s1
    u2 = s3.T @ p2 @ s2

    beta = (n2 / n3) * (p1 @ np.ones(n1)) - (n1 / n3) * (p2 @ np.ones(n2))
    xi = s3.T @ beta
    return u1, u2, xi


def on_shell_q_rel_sq(n1: int, n2: int, alpha_prime: float) -> float:
    """
    On-shell relative transverse momentum for closed-string tachyons.

    Using alpha_r = a N_r and alpha_3 = alpha_1 + alpha_2, energy conservation
    gives
        q_rel^2 = 4/alpha' * (alpha_1^2 + alpha_1 alpha_2 + alpha_2^2)/alpha_3^2.
    The common lattice spacing a cancels.
    """
    n3 = n1 + n2
    return 4.0 * (n1 * n1 + n1 * n2 + n2 * n2) / (alpha_prime * n3 * n3)


def compute_tachyon_data(
    n1: int, n2: int, alpha_prime: float, d_perp: int
) -> TachyonData:
    n3 = n1 + n2

    m1 = mode_metric(n1, alpha_prime)
    m2 = mode_metric(n2, alpha_prime)
    m3 = mode_metric(n3, alpha_prime)
    u1, u2, xi = overlap_data(n1, n2)

    g_t = np.block(
        [
            [m1 + u1.T @ m3 @ u1, u1.T @ m3 @ u2],
            [u2.T @ m3 @ u1, m2 + u2.T @ m3 @ u2],
        ]
    )
    b_t = np.concatenate((u1.T @ m3 @ xi, u2.T @ m3 @ xi))
    c_t = float(xi.T @ m3 @ xi)
    gamma_t = float(c_t - b_t.T @ np.linalg.solve(g_t, b_t))
    q_rel_sq = on_shell_q_rel_sq(n1, n2, alpha_prime)
    exponent = q_rel_sq / (2.0 * gamma_t)

    # Standard real Gaussian ground-state normalization in one transverse direction.
    sign_g, logdet_g = np.linalg.slogdet(g_t)
    if sign_g <= 0:
        raise RuntimeError(f"G_T is not positive definite for ({n1}, {n2})")
    logdet_m = (
        np.linalg.slogdet(m1)[1] + np.linalg.slogdet(m2)[1] + np.linalg.slogdet(m3)[1]
    )
    n_q = (n1 - 1) + (n2 - 1)
    log_prefactor_1d = (
        # Product of the three normalized one-dimensional vacuum wavefunctions.
        0.25 * logdet_m
        - 0.25 * ((n1 - 1) + (n2 - 1) + (n3 - 1)) * math.log(math.pi)
        # Gaussian integral over the independent nonzero-mode variables Q.
        + 0.5 * n_q * math.log(2.0 * math.pi)
        - 0.5 * logdet_g
        # Residual Gaussian integral over the relative center-of-mass variable y.
        + 0.5 * math.log(2.0 * math.pi / gamma_t)
    )
    log_required_norm_noext = exponent - d_perp * log_prefactor_1d

    xi_hat = math.sqrt(n3 / (n1 * n2)) * xi
    completeness_matrix = u1 @ u1.T + u2 @ u2.T + np.outer(xi_hat, xi_hat)
    completeness_error = np.linalg.norm(
        completeness_matrix - np.eye(n3 - 1), ord=np.inf
    )
    orthogonality_error = max(
        np.linalg.norm(u1.T @ u1 - np.eye(n1 - 1), ord=np.inf),
        np.linalg.norm(u2.T @ u2 - np.eye(n2 - 1), ord=np.inf),
    )

    return TachyonData(
        n1=n1,
        n2=n2,
        n3=n3,
        gamma_t=gamma_t,
        q_rel_sq=q_rel_sq,
        exponent=exponent,
        log_prefactor_1d=log_prefactor_1d,
        log_required_norm_noext=log_required_norm_noext,
        completeness_error=float(completeness_error),
        orthogonality_error=float(orthogonality_error),
    )


def parse_pair(text: str) -> tuple[int, int]:
    pieces = text.split(",")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError(f"expected N1,N2 pair, got {text!r}")
    n1, n2 = (int(piece) for piece in pieces)
    if n1 < 2 or n2 < 2:
        raise argparse.ArgumentTypeError("require N1,N2 >= 2")
    return n1, n2


def fit_legwise_power_law(
    pairs: list[tuple[int, int]], alpha_prime: float, d_perp: int, min_n3: int
) -> None:
    rows = []
    for n1, n2 in pairs:
        if n1 + n2 < min_n3:
            continue
        data = compute_tachyon_data(n1, n2, alpha_prime, d_perp)
        n3 = n1 + n2
        x = n1 / n3
        rows.append((n1, n2, n3, x, data.log_required_norm_noext))

    if len(rows) < 3:
        raise RuntimeError("need at least three retained pairs for the fit")

    design = []
    target = []
    for _, _, n3, x, log_creq in rows:
        design.append([math.log(n3), math.log(x * (1.0 - x)), 1.0])
        target.append(log_creq)
    design_matrix = np.array(design)
    target_vector = np.array(target)

    coeffs, residuals, _, _ = np.linalg.lstsq(design_matrix, target_vector, rcond=None)
    a_scale, b_ratio, c_const = coeffs
    predictions = design_matrix @ coeffs
    errors = target_vector - predictions

    print(
        "Fit: log Creq ~= a log N3 + b log[(N1/N3)(N2/N3)] + c"
    )
    print(
        f"a = {a_scale:.9f}, b = {b_ratio:.9f}, c = {c_const:.9f}, "
        f"rmse = {math.sqrt(np.mean(errors * errors)):.9e}"
    )
    print(
        "Equivalent legwise form: "
        f"log Creq ~= {b_ratio:.9f} log N1 + {b_ratio:.9f} log N2 + "
        f"{(a_scale - 2.0 * b_ratio):.9f} log N3 + {c_const:.9f}"
    )
    if residuals.size:
        print(f"Residual sum of squares = {residuals[0]:.9e}")
    print("Retained pairs:")
    print(" N1  N2  N3   logCreq      fit        error")
    for (n1, n2, n3, _, log_creq), pred, err in zip(rows, predictions, errors):
        print(
            f"{n1:3d} {n2:3d} {n3:3d}"
            f"  {log_creq:10.6f}  {pred:10.6f}  {err: .3e}"
        )


def grid_pairs(min_n: int, max_n: int, max_n3: int | None) -> list[tuple[int, int]]:
    pairs = []
    for n1 in range(min_n, max_n + 1):
        for n2 in range(min_n, max_n + 1):
            if max_n3 is not None and n1 + n2 > max_n3:
                continue
            pairs.append((n1, n2))
    return pairs


def fit_exact_leg_factorization(
    pairs: list[tuple[int, int]],
    alpha_prime: float,
    d_perp: int,
    print_functions: bool,
) -> None:
    """
    Fit log Creq to the exact legwise ansatz

        log Creq(N1,N2) = f_in(N1) + f_in(N2) + f_out(N1+N2) + const.

    This is the most direct numerical test for whether the discrete three-point
    amplitude contains any irreducible three-body dependence.
    """
    rows = []
    for n1, n2 in pairs:
        data = compute_tachyon_data(n1, n2, alpha_prime, d_perp)
        n3 = n1 + n2
        rows.append((n1, n2, n3, data.log_required_norm_noext))
    solution = solve_exact_leg_factorization_from_rows(rows)
    coeffs = solution["coeffs"]
    errors = solution["errors"]
    incoming_list = solution["incoming_list"]
    outgoing_list = solution["outgoing_list"]
    incoming_min = solution["incoming_min"]
    incoming_functions = solution["incoming_functions"]
    outgoing_functions = solution["outgoing_functions"]
    predictions = solution["predictions"]

    print(
        "Exact legwise fit: log Creq ~= f_in(N1) + f_in(N2) + "
        "f_out(N1+N2) + const"
    )
    print(
        f"rmse = {math.sqrt(np.mean(errors * errors)):.9e}, "
        f"max_abs_err = {np.max(np.abs(errors)):.9e}"
    )
    print(
        f"Gauge choice: f_in({incoming_min}) = 0, "
        f"f_in({incoming_min + 1}) = 0, "
        f"f_out({2 * incoming_min}) = 0 over "
        f"{len(rows)} sampled joins."
    )
    print(
        "Residuals test only the gauge-invariant existence of the factorization; "
        "the individual functions depend on this reporting convention."
    )
    print("Worst residuals:")
    order = np.argsort(np.abs(errors))[::-1][:10]
    print(" N1  N2  N3   logCreq      fit        error")
    for idx in order:
        n1, n2, n3, log_creq = rows[idx]
        pred = predictions[idx]
        err = errors[idx]
        print(
            f"{n1:3d} {n2:3d} {n3:3d}"
            f"  {log_creq:10.6f}  {pred:10.6f}  {err: .3e}"
        )

    if print_functions:
        print("Canonical gauge-fixed incoming functions:")
        print(" N   f_in(N)")
        for n in incoming_list:
            print(f"{n:3d}  {incoming_functions[n]: .9f}")
        print("Canonical gauge-fixed outgoing functions:")
        print(" N   f_out(N)")
        for n in outgoing_list:
            print(f"{n:3d}  {outgoing_functions[n]: .9f}")
        print(f"Constant term: {solution['constant_term']: .9f}")


def solve_exact_leg_factorization_from_rows(
    rows: list[tuple[int, int, int, float]],
) -> dict[str, object]:
    """Solve the gauge-fixed exact legwise factorization problem."""
    incoming_sizes: set[int] = set()
    outgoing_sizes: set[int] = set()

    if not rows:
        raise RuntimeError("no rows supplied for exact leg-factorization fit")

    for n1, n2, n3, _ in rows:
        incoming_sizes.add(n1)
        incoming_sizes.add(n2)
        outgoing_sizes.add(n3)

    incoming_list = sorted(incoming_sizes)
    outgoing_list = sorted(outgoing_sizes)
    incoming_index = {n: i for i, n in enumerate(incoming_list)}
    outgoing_index = {
        n: i + len(incoming_list) for i, n in enumerate(outgoing_list)
    }
    const_index = len(incoming_list) + len(outgoing_list)

    design = []
    target = []
    for n1, n2, n3, log_creq in rows:
        row = np.zeros(const_index + 1)
        row[incoming_index[n1]] += 1.0
        row[incoming_index[n2]] += 1.0
        row[outgoing_index[n3]] += 1.0
        row[const_index] = 1.0
        design.append(row)
        target.append(log_creq)

    design_matrix = np.array(design)
    target_vector = np.array(target)

    # The split into incoming/outgoing pieces has a linear gauge freedom:
    #   f_in(n) -> f_in(n) + a n + b,   f_out(N) -> f_out(N) - a N - 2 b.
    # Fix it canonically for reporting by imposing
    #   f_in(n_min) = 0, f_in(n_min+1) = 0, f_out(2 n_min) = 0.
    incoming_min = incoming_list[0]
    gauge_rows = []
    for column in (
        incoming_index[incoming_min],
        incoming_index[incoming_min + 1],
        outgoing_index[2 * incoming_min],
    ):
        gauge_row = np.zeros(const_index + 1)
        gauge_row[column] = 1.0
        gauge_rows.append(gauge_row)
    gauge_matrix = np.array(gauge_rows)

    augmented_matrix = np.block(
        [
            [design_matrix.T @ design_matrix, gauge_matrix.T],
            [gauge_matrix, np.zeros((gauge_matrix.shape[0], gauge_matrix.shape[0]))],
        ]
    )
    augmented_rhs = np.concatenate(
        [design_matrix.T @ target_vector, np.zeros(gauge_matrix.shape[0])]
    )
    coeffs = np.linalg.solve(augmented_matrix, augmented_rhs)[: const_index + 1]
    predictions = design_matrix @ coeffs
    errors = target_vector - predictions

    incoming_functions = {n: coeffs[incoming_index[n]] for n in incoming_list}
    outgoing_functions = {n: coeffs[outgoing_index[n]] for n in outgoing_list}

    return {
        "coeffs": coeffs,
        "errors": errors,
        "predictions": predictions,
        "incoming_list": incoming_list,
        "outgoing_list": outgoing_list,
        "incoming_functions": incoming_functions,
        "outgoing_functions": outgoing_functions,
        "incoming_min": incoming_min,
        "constant_term": float(coeffs[const_index]),
    }


def factorization_errors_from_rows(
    rows: list[tuple[int, int, int, float]],
) -> tuple[float, float, np.ndarray]:
    """Return the RMS, max residual, and residual vector of the exact legwise fit."""
    errors = solve_exact_leg_factorization_from_rows(rows)["errors"]
    return (
        float(math.sqrt(np.mean(errors * errors))),
        float(np.max(np.abs(errors))),
        errors,
    )


def factorization_errors(
    pairs: list[tuple[int, int]], alpha_prime: float, d_perp: int
) -> tuple[float, float]:
    """Return the RMS and max residual of the exact legwise fit."""
    rows = []
    for n1, n2 in pairs:
        data = compute_tachyon_data(n1, n2, alpha_prime, d_perp)
        n3 = n1 + n2
        rows.append((n1, n2, n3, data.log_required_norm_noext))
    rmse, max_abs, _ = factorization_errors_from_rows(rows)
    return rmse, max_abs


def scan_transverse_dimension(
    pairs: list[tuple[int, int]],
    alpha_prime: float,
    d_perp_values: list[int],
) -> None:
    """Report the factorization residual as a function of D_perp."""
    print(" D_perp      rmse_factorization      max_abs_residual")
    for d_perp in d_perp_values:
        rmse, max_abs = factorization_errors(pairs, alpha_prime, d_perp)
        print(f"{d_perp:7d}  {rmse:20.12e}  {max_abs:20.12e}")


def scan_factorization_components(
    pairs: list[tuple[int, int]], alpha_prime: float, d_perp: int
) -> None:
    """
    Decompose the bosonic three-tachyon factorization test into its two pieces:

        exponent = q_rel^2 / (2 gamma_T),
        log_pref_1d = log kappa_1d.

    Neither piece factorizes by itself, but at D_perp = 24 their residuals
    cancel in the combination exponent - D_perp * log_pref_1d.
    """
    exponent_rows = []
    prefactor_rows = []
    combined_rows = []
    for n1, n2 in pairs:
        data = compute_tachyon_data(n1, n2, alpha_prime, d_perp)
        n3 = n1 + n2
        exponent_rows.append((n1, n2, n3, data.exponent))
        prefactor_rows.append((n1, n2, n3, data.log_prefactor_1d))
        combined_rows.append((n1, n2, n3, data.log_required_norm_noext))

    rmse_exp, max_exp, residual_exp = factorization_errors_from_rows(exponent_rows)
    rmse_pref, max_pref, residual_pref = factorization_errors_from_rows(prefactor_rows)
    rmse_comb, max_comb, residual_comb = factorization_errors_from_rows(combined_rows)
    cancellation = residual_exp - d_perp * residual_pref
    rmse_cancel = float(math.sqrt(np.mean(cancellation * cancellation)))
    max_cancel = float(np.max(np.abs(cancellation)))

    print("Component factorization test:")
    print(f"  D_perp = {d_perp}")
    print(
        f"  exponent residual:        rmse = {rmse_exp:.12e}, "
        f"max = {max_exp:.12e}"
    )
    print(
        f"  log_pref_1d residual:     rmse = {rmse_pref:.12e}, "
        f"max = {max_pref:.12e}"
    )
    print(
        f"  combined residual:        rmse = {rmse_comb:.12e}, "
        f"max = {max_comb:.12e}"
    )
    print(
        f"  exponent - D_perp*pref:   rmse = {rmse_cancel:.12e}, "
        f"max = {max_cancel:.12e}"
    )
    print(
        "  This last line should agree with the combined residual up to "
        "floating-point rounding."
    )


def fit_factorized_asymptotics(
    pairs: list[tuple[int, int]],
    alpha_prime: float,
    d_perp: int,
    incoming_start: int,
    outgoing_start: int,
) -> None:
    """
    Fit one canonical gauge-fixed representative of the exact legwise split to

        f(N) ~= c_log log N + c_1/N + c_2/N^2 + c_0 + c_lin N.

    The linear and constant pieces depend on the gauge convention, but the
    non-linear tail (logarithmic and inverse-power coefficients) is invariant
    under the allowed redefinitions

        f_in(N) -> f_in(N) + a N + b,
        f_out(N) -> f_out(N) - a N - 2 b.
    """
    rows = []
    for n1, n2 in pairs:
        data = compute_tachyon_data(n1, n2, alpha_prime, d_perp)
        n3 = n1 + n2
        rows.append((n1, n2, n3, data.log_required_norm_noext))

    solution = solve_exact_leg_factorization_from_rows(rows)
    incoming_functions = solution["incoming_functions"]
    outgoing_functions = solution["outgoing_functions"]
    incoming_list = solution["incoming_list"]
    outgoing_list = solution["outgoing_list"]

    def fit_one(
        functions: dict[int, float], labels: list[int], start: int
    ) -> tuple[np.ndarray, float, float, list[int]]:
        ns = [n for n in labels if n >= start]
        if len(ns) < 6:
            raise RuntimeError(
                f"need at least six retained points for asymptotic fit, got {len(ns)}"
            )
        design = np.array(
            [
                [math.log(n), 1.0 / n, 1.0 / (n * n), 1.0, float(n)]
                for n in ns
            ]
        )
        target = np.array([functions[n] for n in ns])
        coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        residuals = target - design @ coeffs
        return (
            coeffs,
            float(math.sqrt(np.mean(residuals * residuals))),
            float(np.max(np.abs(residuals))),
            ns,
        )

    def test_fixed_tail(
        functions: dict[int, float],
        labels: list[int],
        start: int,
        c_log: float,
        c_1: float,
        c_2: float,
    ) -> tuple[np.ndarray, float, float]:
        ns = [n for n in labels if n >= start]
        design = np.array([[1.0, float(n)] for n in ns])
        target = np.array(
            [
                functions[n]
                - c_log * math.log(n)
                - c_1 / n
                - c_2 / (n * n)
                for n in ns
            ]
        )
        coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        residuals = target - design @ coeffs
        return (
            coeffs,
            float(math.sqrt(np.mean(residuals * residuals))),
            float(np.max(np.abs(residuals))),
        )

    coeffs_in, rmse_in, max_in, ns_in = fit_one(
        incoming_functions, incoming_list, incoming_start
    )
    coeffs_out, rmse_out, max_out, ns_out = fit_one(
        outgoing_functions, outgoing_list, outgoing_start
    )

    print("Asymptotic fit for one canonical gauge-fixed leg-factor representative:")
    print(
        "  model: f(N) ~= c_log log N + c_1/N + c_2/N^2 + c_0 + c_lin N"
    )
    print(
        "  Gauge-invariant information lives in c_log, c_1, c_2. "
        "The c_0 and c_lin pieces depend on the reporting gauge."
    )
    print(
        f"  incoming fit range: N >= {incoming_start} ({len(ns_in)} points), "
        f"outgoing fit range: N >= {outgoing_start} ({len(ns_out)} points)"
    )
    print(
        "  incoming coefficients (c_log, c_1, c_2, c_0, c_lin): "
        + " ".join(f"{c:.12f}" for c in coeffs_in)
    )
    print(
        f"    rmse = {rmse_in:.12e}, max_abs = {max_in:.12e}"
    )
    print(
        "  outgoing coefficients (c_log, c_1, c_2, c_0, c_lin): "
        + " ".join(f"{c:.12f}" for c in coeffs_out)
    )
    print(
        f"    rmse = {rmse_out:.12e}, max_abs = {max_out:.12e}"
    )

    exact_c2 = math.pi * math.pi / 72.0
    fixed_in, rmse_in_fixed, max_in_fixed = test_fixed_tail(
        incoming_functions, incoming_list, incoming_start, 7.0, math.pi, exact_c2
    )
    fixed_out, rmse_out_fixed, max_out_fixed = test_fixed_tail(
        outgoing_functions, outgoing_list, outgoing_start, -5.0, -math.pi, exact_c2
    )
    print(
        "  fixed-tail test with c_log = 7, -5; c_1 = +/-pi; c_2 = pi^2/72:"
    )
    print(
        "    incoming residual after fitting only const + linear: "
        f"rmse = {rmse_in_fixed:.12e}, max_abs = {max_in_fixed:.12e}, "
        f"(c_0, c_lin) = ({fixed_in[0]:.12f}, {fixed_in[1]:.12f})"
    )
    print(
        "    outgoing residual after fitting only const + linear: "
        f"rmse = {rmse_out_fixed:.12e}, max_abs = {max_out_fixed:.12e}, "
        f"(c_0, c_lin) = ({fixed_out[0]:.12f}, {fixed_out[1]:.12f})"
    )

    combined_tail_values = []
    for n1, n2, n3, log_creq in rows:
        combined_tail = (
            7.0 * math.log(n1)
            + 7.0 * math.log(n2)
            - 5.0 * math.log(n3)
            + math.pi * (1.0 / n1 + 1.0 / n2 - 1.0 / n3)
            + exact_c2 * (1.0 / (n1 * n1) + 1.0 / (n2 * n2) + 1.0 / (n3 * n3))
        )
        combined_tail_values.append(log_creq - combined_tail)
    combined_tail_values = np.array(combined_tail_values)
    combined_constant = float(np.mean(combined_tail_values))
    combined_residuals = combined_tail_values - combined_constant
    print(
        "  full three-leg invariant tail test with "
        "7 log N1 + 7 log N2 - 5 log N3"
    )
    print(
        "    plus pi(1/N1 + 1/N2 - 1/N3) and (pi^2/72)(1/N1^2 + 1/N2^2 + 1/N3^2):"
    )
    print(
        f"    residual after fitting only one overall constant = "
        f"{combined_constant:.12f}: "
        f"rmse = {math.sqrt(np.mean(combined_residuals * combined_residuals)):.12e}, "
        f"max_abs = {np.max(np.abs(combined_residuals)):.12e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        action="append",
        type=parse_pair,
        help="join sizes N1,N2 to evaluate; may be repeated",
    )
    parser.add_argument("--alpha-prime", type=float, default=1.0)
    parser.add_argument(
        "--d-perp",
        type=int,
        default=24,
        help="transverse dimension used when reporting the required cubic normalization",
    )
    parser.add_argument(
        "--fit-legwise",
        action="store_true",
        help="fit the required normalization to a legwise power-law ansatz",
    )
    parser.add_argument(
        "--fit-min-n3",
        type=int,
        default=0,
        help="only keep pairs with N3 >= this value in --fit-legwise mode",
    )
    parser.add_argument(
        "--fit-factorized",
        action="store_true",
        help="fit the required normalization to exact incoming/outgoing leg functions",
    )
    parser.add_argument(
        "--factorized-min-n",
        type=int,
        default=4,
        help="smallest leg size used when constructing the factorized-fit grid",
    )
    parser.add_argument(
        "--factorized-max-n",
        type=int,
        default=20,
        help="largest leg size used when constructing the factorized-fit grid",
    )
    parser.add_argument(
        "--factorized-max-n3",
        type=int,
        default=40,
        help="largest joined size N3 kept in the factorized-fit grid",
    )
    parser.add_argument(
        "--print-factorized-functions",
        action="store_true",
        help="print one canonical gauge-fixed set of incoming/outgoing leg functions",
    )
    parser.add_argument(
        "--scan-d-perp",
        action="store_true",
        help="scan the exact factorization residual as a function of D_perp",
    )
    parser.add_argument(
        "--d-perp-min",
        type=int,
        default=20,
        help="smallest D_perp used in --scan-d-perp mode",
    )
    parser.add_argument(
        "--d-perp-max",
        type=int,
        default=28,
        help="largest D_perp used in --scan-d-perp mode",
    )
    parser.add_argument(
        "--scan-factorization-components",
        action="store_true",
        help="separately test the exponent and prefactor parts of the bosonic factorization",
    )
    parser.add_argument(
        "--fit-factorized-asymptotics",
        action="store_true",
        help="fit the canonical gauge-fixed leg factors to their large-N asymptotic form",
    )
    parser.add_argument(
        "--asymptotic-in-start",
        type=int,
        default=20,
        help="smallest incoming leg size retained in --fit-factorized-asymptotics mode",
    )
    parser.add_argument(
        "--asymptotic-out-start",
        type=int,
        default=40,
        help="smallest outgoing leg size retained in --fit-factorized-asymptotics mode",
    )
    args = parser.parse_args()

    if args.fit_legwise:
        fit_legwise_power_law(FIT_PAIRS, args.alpha_prime, args.d_perp, args.fit_min_n3)
        return
    if args.fit_factorized:
        pairs = grid_pairs(
            args.factorized_min_n,
            args.factorized_max_n,
            args.factorized_max_n3,
        )
        fit_exact_leg_factorization(
            pairs,
            args.alpha_prime,
            args.d_perp,
            args.print_factorized_functions,
        )
        return
    if args.scan_d_perp:
        pairs = grid_pairs(
            args.factorized_min_n,
            args.factorized_max_n,
            args.factorized_max_n3,
        )
        scan_transverse_dimension(
            pairs,
            args.alpha_prime,
            list(range(args.d_perp_min, args.d_perp_max + 1)),
        )
        return
    if args.scan_factorization_components:
        pairs = grid_pairs(
            args.factorized_min_n,
            args.factorized_max_n,
            args.factorized_max_n3,
        )
        scan_factorization_components(pairs, args.alpha_prime, args.d_perp)
        return
    if args.fit_factorized_asymptotics:
        pairs = grid_pairs(
            args.factorized_min_n,
            args.factorized_max_n,
            args.factorized_max_n3,
        )
        fit_factorized_asymptotics(
            pairs,
            args.alpha_prime,
            args.d_perp,
            args.asymptotic_in_start,
            args.asymptotic_out_start,
        )
        return

    pairs = args.pair or DEFAULT_PAIRS

    print(
        " N1  N2  N3        gamma_T        q_rel^2        q^2/(2gamma)"
        "    log_pref_1d    log Creq      comp.err      ortho.err"
    )
    for n1, n2 in pairs:
        data = compute_tachyon_data(n1, n2, args.alpha_prime, args.d_perp)
        print(
            f"{data.n1:3d} {data.n2:3d} {data.n3:3d}"
            f"  {data.gamma_t:13.9f}"
            f"  {data.q_rel_sq:13.9f}"
            f"  {data.exponent:14.9f}"
            f"  {data.log_prefactor_1d:13.9f}"
            f"  {data.log_required_norm_noext:11.6f}"
            f"  {data.completeness_error:11.2e}"
            f"  {data.orthogonality_error:11.2e}"
        )


if __name__ == "__main__":
    main()
