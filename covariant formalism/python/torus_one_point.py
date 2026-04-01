"""
torus_one_point.py

Numerical benchmark for the compact-boson torus one-point function
< :partial X partial X: >.

The script:

1. Computes the discretized disk one-point function using the lattice
   Gaussian described in the draft.
2. Uses ell_to_tau.py to map (L, l1, l2, l3) to the torus modulus tau and
   the normalized flat coordinate derivative \hat f(0).
3. Transforms the disk result to the flat torus coordinate used in the draft.
4. Compares against the analytic flat-torus formula.

This module follows the same lattice normalization and reduction convention as
compact_partition.py. In particular we use the same M, A', W, U, T, T'
notation and the same zero-puncture relation s3 = s2 - s1.

The analytic benchmark is written in Polchinski's flat torus coordinate
w \sim w + 2 pi \sim w + 2 pi tau, with alpha' = 1.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import mpmath as mp
import numpy as np

import compact_partition as cp
import ell_to_tau as elt


mp.mp.dps = 50
DEFAULT_ALPHA_PRIME = 1.0


@dataclass
class BenchmarkResult:
    L: int
    l1: int
    l2: int
    l3: int
    R: float
    alpha_prime: float
    tau: complex
    P1: complex
    fhat0: complex
    dw_dz0: complex
    schwarzian0: complex
    disk_numeric: complex
    disk_reconstructed: complex
    disk_exact_from_torus: complex
    torus_from_disk: complex
    torus_exact: complex
    disk_abs_error: float
    disk_rel_error: float
    disk_reconstruction_abs_error: float
    disk_reconstruction_rel_error: float
    torus_diagnostic_abs_error: float
    torus_diagnostic_rel_error: float
    torus_from_disk_oscillator: complex
    torus_from_disk_classical: complex
    torus_exact_oscillator: complex
    torus_exact_classical: complex
    torus_oscillator_diagnostic_abs_error: float
    torus_oscillator_diagnostic_rel_error: float
    torus_classical_diagnostic_abs_error: float
    torus_classical_diagnostic_rel_error: float


@dataclass
class LatticeGeometry:
    Mat: np.ndarray
    Aprime: np.ndarray
    W: np.ndarray
    U: np.ndarray
    T1: np.ndarray
    T2: np.ndarray
    T: np.ndarray
    Tp: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    segments: list


@dataclass
class NotebookFirstTermResult:
    L: int
    l1: int
    l2: int
    l3: int
    m: int
    n: int
    tau: complex
    tau2: float
    P1: complex
    f0: complex
    compact_trace: complex
    notebook_trace: complex
    notebook_value: float


@dataclass
class NotebookHolomorphicComparison:
    L: int
    l1: int
    l2: int
    l3: int
    alpha_prime: float
    tau: complex
    P1: complex
    fhat0: complex
    dw_dz0: complex
    schwarzian0: complex
    notebook_analytic: complex
    notebook_numerical: complex
    abs_error: float
    rel_error: float


@dataclass
class GeometryData:
    tau: complex
    P1: complex
    P2: complex
    f0: complex
    fhat0: complex
    dw_dz0: complex
    schwarzian0: complex


@dataclass
class DiscretizedDiskComponents:
    oscillator_disk: complex
    classical_disk: complex
    compact_trace_11: complex


@dataclass
class SelfConsistencyResult:
    L: int
    l1: int
    l2: int
    l3: int
    R: float
    alpha_prime: float
    tau: complex
    trace_normalization_abs_error: float
    trace_normalization_rel_error: float
    geometry_tau_abs_error: float
    geometry_fhat_abs_error: float
    schwarzian_fd_abs_error: float
    analytic_component_sum_abs_error: float
    analytic_component_sum_rel_error: float
    analytic_disk_transform_abs_error: float
    analytic_disk_transform_rel_error: float
    numerical_disk_component_sum_abs_error: float
    numerical_disk_component_sum_rel_error: float
    numerical_torus_transform_abs_error: float
    numerical_torus_transform_rel_error: float


def _full_one_form(f, z: complex) -> complex:
    """Return the full one-form coefficient f(z), not the split (singular, poly)."""
    value = f(z)
    if isinstance(value, tuple):
        singular, poly = value
        return complex(singular * poly)
    return complex(value)


def _singular_second_derivative_at_origin(f) -> complex:
    """
    Return s''(0) for the explicit singular factor in ell_to_tau.make_cyl_eqn_improved.

    For the genus-1 improved one-form we have

        f(z) = s(z) p(z),

    where p is a regular polynomial with coefficients stored in f.coeffs.
    The singular factor is either

        s(z) = Π_{a in {1, w1, w2}} (1 - a z^2)^(-1/3),

    or in the degenerate two-segment case used by the code,

        s(z) = Π_{a in {1, w1}} (1 - a z^2)^(-1/2).

    Since s'(0)=0, only s''(0) is needed for the Schwarzian at the origin.
    """
    L = getattr(f, "L", None)
    l1 = getattr(f, "l1", None)
    l2 = getattr(f, "l2", None)
    if L is None or l1 is None or l2 is None:
        raise ValueError("Need f.L, f.l1, and f.l2 to reconstruct the singular factor.")

    half = int(L) // 2
    l3 = half - int(l1) - int(l2)
    phase1 = 2.0 * np.pi * int(l1) / int(L)
    phase2 = 2.0 * np.pi * (int(l1) + int(l2)) / int(L)
    w1 = np.exp(-2j * phase1)
    w2 = np.exp(-2j * phase2)

    if l3 == 0:
        # s(z) = (1 - z^2)^(-1/2) (1 - w1 z^2)^(-1/2)
        return complex(1.0 + w1)

    # Generic three-factor case:
    # s(z) = Π_a (1 - a z^2)^(-1/3),  a in {1, w1, w2}
    return complex((2.0 / 3.0) * (1.0 + w1 + w2))


def _schwarzian_at_origin(f, h: float = 1.0e-4) -> complex:
    """
    Compute {u,z}|_{z=0}.

    For the improved genus-1 one-form returned by ell_to_tau.py, we know
    the full structure analytically:

        f(z) = s(z) p(z),

    with the regular polynomial coefficients stored in f.coeffs. We therefore
    compute the Schwarzian exactly from c0, c1, c2 and s''(0):

        f(0)  = c0,
        f'(0) = c1,
        f''(0)= s''(0) c0 + 2 c2.

    A finite-difference fallback is kept only for unexpected callables that do
    not provide the improved-form metadata.
    """
    coeffs = getattr(f, "coeffs", None)
    if coeffs is not None and hasattr(f, "L") and hasattr(f, "l1") and hasattr(f, "l2"):
        coeffs = np.asarray(coeffs, dtype=np.complex128)
        c0 = coeffs[0]
        c1 = coeffs[1] if coeffs.shape[0] > 1 else np.complex128(0.0)
        c2 = coeffs[2] if coeffs.shape[0] > 2 else np.complex128(0.0)
        sdd0 = _singular_second_derivative_at_origin(f)
        f0 = c0
        fp0 = c1
        fpp0 = sdd0 * c0 + 2.0 * c2
        return complex(fpp0 / f0 - 1.5 * (fp0 / f0) ** 2)

    f0 = _full_one_form(f, 0.0)
    fp = (_full_one_form(f, h) - _full_one_form(f, -h)) / (2.0 * h)
    fpp = (_full_one_form(f, h) - 2.0 * f0 + _full_one_form(f, -h)) / (h * h)
    return fpp / f0 - 1.5 * (fp / f0) ** 2


def _schwarzian_at_origin_finite_difference(f, h: float = 1.0e-4) -> complex:
    """Finite-difference Schwarzian used only as a diagnostic cross-check."""
    f0 = _full_one_form(f, 0.0)
    fp = (_full_one_form(f, h) - _full_one_form(f, -h)) / (2.0 * h)
    fpp = (_full_one_form(f, h) - 2.0 * f0 + _full_one_form(f, -h)) / (h * h)
    return fpp / f0 - 1.5 * (fp / f0) ** 2


def _segment_sums_from_w(W: np.ndarray, segments) -> np.ndarray:
    """
    Build the U matrix in the same notation as compact_partition.py:
        U[:, i] = sum of W columns in segment i.
    """
    n_red = W.shape[0]
    U = np.zeros((n_red, 3), dtype=np.complex128)
    for i, (start, end) in enumerate(segments):
        U[:, i] = W[:, start - 1:end].sum(axis=1)
    return U


def _prime_reduce_linear(M: np.ndarray) -> np.ndarray:
    """
    Reduce a linear coefficient M_i under the compact-partition convention

        s3 = s2 - s1.

    If sum_i M_i s_i is the original linear term, then after imposing the
    constraint it becomes sum_{i=1}^2 M'_i s_i with

        M'_1 = M_1 - M_3,
        M'_2 = M_2 + M_3.
    """
    dtype = np.result_type(M.dtype, np.float64)
    Mp = np.empty((M.shape[0], 2), dtype=dtype)
    Mp[:, 0] = M[:, 0] - M[:, 2]
    Mp[:, 1] = M[:, 1] + M[:, 2]
    return Mp


def _quadratic_form_on_grid(Q: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Evaluate s^T Q s on a mesh of independent winding numbers (s1, s2)."""
    return (
        Q[0, 0] * s1 * s1
        + Q[0, 1] * s1 * s2
        + Q[1, 0] * s2 * s1
        + Q[1, 1] * s2 * s2
    )


def _build_embedding_matrices(L: int, l1: int, l2: int):
    """
    Build the internal matrices that reconstruct the full boundary data from
    the reduced variables and the integer shift vector s = (s1, s2, s3).
    """
    half = L // 2
    n_red = half - 1
    segments, partner = cp._segment_maps(L, l1, l2)

    # Reduced variables with X(L/2) = -sum_{k=1}^{L/2-1} X(k).
    reduced_to_half = np.zeros((half, n_red), dtype=np.complex128)
    reduced_to_half[:n_red, :] = np.eye(n_red, dtype=np.complex128)
    reduced_to_half[n_red, :] = -1.0

    P = np.zeros((L, n_red), dtype=np.complex128)
    Q = np.zeros((L, 3), dtype=np.complex128)

    segment_id = np.empty(half, dtype=np.int64)
    for i, (start, end) in enumerate(segments):
        segment_id[start - 1:end] = i

    for j in range(half):
        P[j, :] = reduced_to_half[j, :]
        P[partner[j], :] = reduced_to_half[j, :]
        Q[partner[j], segment_id[j]] = 1.0

    return P, Q, segments


def _mode_quadratic_matrix(L: int, m: int, n: int) -> np.ndarray:
    """
    Full-boundary quadratic form for the holomorphic insertion X_m X_n.

    This matches the original Mathematica definition DirectCMatN[L, m, n]:
        (C_{m,n})_{k,l} = |m|! |n|! / L^2 * exp[-2 pi i (m k + n l)/L]
    with integer lattice sites k,l = 1,...,L.
    """
    idx = np.arange(1, L + 1, dtype=np.float64)
    phase_m = np.exp(-2j * np.pi * m * idx / L)
    phase_n = np.exp(-2j * np.pi * n * idx / L)
    prefactor = math.factorial(abs(m)) * math.factorial(abs(n)) / (L * L)
    return prefactor * np.outer(phase_m, phase_n)


def _lattice_setup(L: int, l1: int, l2: int):
    """
    Reusable lattice data in the notation of compact_partition.py.

    Mat, Aprime, W, U, T1, T2, T, Tp are defined exactly as in the compact
    partition-function code. P and Q are additional helper matrices used only
    to build the insertion-dependent tensors C11, Y11, E11.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        M = cp.direct_mat_n_fast(L)
        Aprime = cp.direct_red_traced_mat(L, l1, l2, M)
        W = cp.mat_w(L, l1, l2, M)
        T1 = cp.mat_t_first_part(L, l1, l2, M).astype(np.complex128)
        T2 = cp.mat_t_second_part(L, l1, l2, W, Aprime).astype(np.complex128)
        T = cp.symm(T1 - T2).astype(np.complex128)
    Tp = cp.mat_t_prime(T).astype(np.complex128)

    P, Q, segments = _build_embedding_matrices(L, l1, l2)
    U = _segment_sums_from_w(W.astype(np.complex128), segments)

    return LatticeGeometry(
        Mat=M.astype(np.complex128),
        Aprime=Aprime.astype(np.complex128),
        W=W.astype(np.complex128),
        U=U,
        T1=T1,
        T2=T2,
        T=T,
        Tp=Tp,
        P=P,
        Q=Q,
        segments=segments,
    )


def _geometry_data(L: int, l1: int, l2: int) -> GeometryData:
    """Shared geometry package built from ell_to_tau.py."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        f = elt.make_cyl_eqn_improved(L, l1, l2)
        P1, P2, _ = elt.periods_improved(L, l1, l2, f=f)
    tau = complex(P2 / P1)
    f0 = _full_one_form(f, 0.0)
    fhat0 = f0 / P1
    dw_dz0 = 2.0 * np.pi * fhat0
    schwarzian0 = _schwarzian_at_origin(f)
    return GeometryData(
        tau=complex(tau),
        P1=complex(P1),
        P2=complex(P2),
        f0=complex(f0),
        fhat0=complex(fhat0),
        dw_dz0=complex(dw_dz0),
        schwarzian0=complex(schwarzian0),
    )


def _geometry_data_notebook(L: int, l1: int, l2: int) -> GeometryData:
    """
    Geometry package matching onePointFunctionNew.nb literally.

    The notebook uses

        f      = CylEqnFastOptimized[l1, l2, l3, z]
        per    = PeriodsFastGivenF[l1, l2, l3, l1/2, l2/2, l3/2, f, z]
        perRed = per / per[[1]]
        fHat   = f / per[[1]]

    so the correct Python translation must preserve the half-integer midpoint
    inputs and use the non-improved polynomial one-form.
    """
    half = L // 2
    l3 = half - l1 - l2
    if l3 <= 0:
        raise ValueError(f"Need l3 > 0 for the notebook geometry, got l3={l3}")

    f = elt.make_cyl_eqn(L, l1, l2)
    P1, P2, _ = elt.periods_given_f(
        L, l1, l2, f,
        m1=l1 / 2,
        m2=l2 / 2,
        m3=l3 / 2,
    )
    tau = complex(P2 / P1)

    coeffs = np.asarray(f.coeffs, dtype=np.complex128)
    f0 = coeffs[0]
    f1 = coeffs[1] if coeffs.shape[0] > 1 else np.complex128(0.0)
    f2 = 2.0 * coeffs[2] if coeffs.shape[0] > 2 else np.complex128(0.0)
    fhat0 = f0 / P1
    dw_dz0 = 2.0 * np.pi * fhat0
    schwarzian0 = f2 / f0 - 1.5 * (f1 / f0) ** 2

    return GeometryData(
        tau=complex(tau),
        P1=complex(P1),
        P2=complex(P2),
        f0=complex(f0),
        fhat0=complex(fhat0),
        dw_dz0=complex(dw_dz0),
        schwarzian0=complex(schwarzian0),
    )


def _discretized_disk_components(L: int, l1: int, l2: int, R: float,
                                 n_winding: int = 8) -> DiscretizedDiskComponents:
    """
    Direct discretized disk oscillator/classical split.

    The oscillator part is the pure trace term, while the classical part
    contains the winding-sector average.
    """
    geom = _lattice_setup(L, l1, l2)
    dtype = np.complex128

    N11 = _mode_quadratic_matrix(L, 1, 1)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        C11 = geom.P.T @ N11 @ geom.P
        Y11 = geom.P.T @ N11 @ geom.Q
        E11 = geom.Q.T @ N11 @ geom.Q
        Up = _prime_reduce_linear(geom.U)
        Y11p = _prime_reduce_linear(Y11)
        E11p = cp.mat_t_prime(E11).astype(dtype)
        Tp = cp.mat_t_prime(geom.T).astype(dtype)

        AU = np.linalg.solve(geom.Aprime, Up)
        AC = np.linalg.solve(geom.Aprime, C11)
        Xi11p = E11p - 2.0 * (Y11p.T @ AU) + (Up.T @ AC @ AU)
        Xi11p = 0.5 * (Xi11p + Xi11p.T)

    sv = np.arange(-n_winding, n_winding + 1, dtype=np.float64)
    s1, s2 = np.meshgrid(sv, sv, indexing="ij")
    quad_t = _quadratic_form_on_grid(Tp, s1, s2)
    quad_xi = _quadratic_form_on_grid(Xi11p, s1, s2)
    weights = np.exp(-4.0 * np.pi * R * R * quad_t.real)
    classical_disk = (
        4.0 * np.pi * np.pi * R * R * np.sum(quad_xi * weights) / np.sum(weights)
    )
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        compact_trace_11 = np.trace(AC)
        oscillator_disk = 0.5 * np.pi * compact_trace_11

    return DiscretizedDiskComponents(
        oscillator_disk=complex(oscillator_disk),
        classical_disk=complex(classical_disk),
        compact_trace_11=complex(compact_trace_11),
    )


def discretized_disk_one_point(L: int, l1: int, l2: int, R: float,
                               n_winding: int = 8) -> complex:
    """
    Compute the normalized disk one-point function of :partial X partial X:
    in the discretized scheme.

    This follows the compact-partition notation as closely as possible.
    After building the full three-segment tensors, we impose s3 = s2 - s1 and
    work with the primed objects U', Y'_{1,1}, E'_{1,1}, Xi'_{1,1}, and T'.
    """
    comp = _discretized_disk_components(L, l1, l2, R, n_winding=n_winding)
    return complex(comp.oscillator_disk + comp.classical_disk)


def notebook_first_term_value(L: int, l1: int, l2: int, m: int = 1, n: int = 1
                              ) -> NotebookFirstTermResult:
    """
    Reproduce the Mathematica notebook benchmark quantity for the first term.

    In the Mathematica notebook benchmark, findTrCA returns

        { tau, analytic_test, numerical_test }

    and ``numerical_test`` is the raw reduced trace Tr[Cred . Ared^{-1}] in the
    older matrix convention. The current compact Python code uses an M/A
    convention smaller by a factor of 2, so

        tr_old = 0.5 * tr_compact.

    The new holomorphic notebook onePointFunctionNew.nb uses (m, n) = (1, 1)
    for the :partial X partial X: benchmark, so that is the default here.
    Pass n = -1 to reproduce the older mixed notebook convention instead.
    """
    l3 = L // 2 - l1 - l2
    if l3 <= 0:
        raise ValueError(f"Need l3 > 0, got l3={l3}")

    geom = _lattice_setup(L, l1, l2)
    Nmn = _mode_quadratic_matrix(L, m, n)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Cmn = geom.P.T @ Nmn @ geom.P
        compact_trace = np.trace(np.linalg.solve(geom.Aprime, Cmn))

    geo = _geometry_data_notebook(L, l1, l2)

    notebook_trace = 0.5 * compact_trace
    notebook_value = float(np.real(notebook_trace * abs(geo.P1 / geo.f0) ** 2))

    return NotebookFirstTermResult(
        L=L,
        l1=l1,
        l2=l2,
        l3=l3,
        m=m,
        n=n,
        tau=geo.tau,
        tau2=float(geo.tau.imag),
        P1=geo.P1,
        f0=geo.f0,
        compact_trace=complex(compact_trace),
        notebook_trace=complex(notebook_trace),
        notebook_value=notebook_value,
    )


def notebook_first_term_scan(k_start: int = 1, k_stop: int = 31, k_step: int = 2,
                             m: int = 1, n: int = 1):
    """
    Reproduce the old fixed-shape family

        findTrCA[18 k, k, k]

    as a list of pairs (tau2, notebook_value).
    """
    output = []
    for k in range(k_start, k_stop + 1, k_step):
        result = notebook_first_term_value(18 * k, k, k, m=m, n=n)
        output.append((result.tau2, result.notebook_value))
    return output


def notebook_first_term_scan_fixed_L(setL: int = 310, k_start: int = 1,
                                     k_stop: int = 31, k_step: int = 2,
                                     m: int = 1, n: int = 1):
    """
    Reproduce the onePointFunctionNew.nb family

        findTrCA[setL, 2 k, k, 1, AMat, CMat11]

    with fixed total lattice size and varying moduli.
    """
    output = []
    for k in range(k_start, k_stop + 1, k_step):
        result = notebook_first_term_value(setL, 2 * k, k, m=m, n=n)
        output.append(result)
    return output


def notebook_holomorphic_compare(L: int, l1: int, l2: int,
                                 alpha_prime: float = DEFAULT_ALPHA_PRIME,
                                 n_winding_analytic: int = 12
                                 ) -> NotebookHolomorphicComparison:
    """
    Compare the holomorphic notebook first-term analytic formula against the
    raw numerical trace in the same normalization.

    In onePointFunctionNew.nb, the numerical quantity is the old-convention
    trace Tr[Cred . Ared^{-1}]. With alpha' = 1, the matching analytic
    quantity is the disk oscillator one-point function divided by pi:

        analytic_notebook
          = ( (dw/dz)^2 < :dX dX: >_{T,osc} - {w,z}/12 ) / pi.
    """
    if abs(alpha_prime - 1.0) > 1.0e-14:
        raise ValueError(
            "The notebook holomorphic benchmark is currently implemented in "
            "the alpha' = 1 convention."
        )

    trace_result = notebook_first_term_value(L, l1, l2, m=1, n=1)
    oscillator_torus, _ = analytic_torus_components(
        trace_result.tau, 1.0, n_winding=n_winding_analytic,
        alpha_prime=alpha_prime
    )
    geo = _geometry_data_notebook(L, l1, l2)
    notebook_analytic = (
        geo.dw_dz0 * geo.dw_dz0 * oscillator_torus
        - alpha_prime * geo.schwarzian0 / 12.0
    ) / np.pi
    notebook_numerical = trace_result.notebook_trace
    abs_error = abs(notebook_numerical - notebook_analytic)
    rel_error = abs_error / abs(notebook_analytic)

    return NotebookHolomorphicComparison(
        L=L,
        l1=l1,
        l2=l2,
        l3=trace_result.l3,
        alpha_prime=alpha_prime,
        tau=trace_result.tau,
        P1=geo.P1,
        fhat0=geo.fhat0,
        dw_dz0=geo.dw_dz0,
        schwarzian0=geo.schwarzian0,
        notebook_analytic=complex(notebook_analytic),
        notebook_numerical=complex(notebook_numerical),
        abs_error=float(abs_error),
        rel_error=float(rel_error),
    )


def _analytic_torus_oscillator(tau: complex,
                               alpha_prime: float = DEFAULT_ALPHA_PRIME) -> complex:
    """Exact oscillator contribution in the flat torus coordinate w."""
    tau = complex(tau)
    tau2 = tau.imag
    tau_mp = mp.mpc(tau)
    q = mp.exp(mp.pi * 1j * tau_mp)

    theta1 = lambda z: mp.jtheta(1, z, q)
    theta1_prime = mp.diff(theta1, 0, 1)
    theta1_third = mp.diff(theta1, 0, 3)
    return complex(
        alpha_prime * complex(theta1_third / theta1_prime) / 24.0
        + alpha_prime / (8.0 * np.pi * tau2)
    )


def _analytic_torus_classical(tau: complex, R: float, n_winding: int = 10,
                              alpha_prime: float = DEFAULT_ALPHA_PRIME) -> complex:
    """Exact classical winding contribution in the flat torus coordinate w."""
    tau = complex(tau)
    tau2 = tau.imag
    sv = np.arange(-n_winding, n_winding + 1, dtype=np.float64)
    m_vals, n_vals = np.meshgrid(sv, sv, indexing="ij")
    arg = n_vals - m_vals * tau
    weights = np.exp(-(np.pi * R * R / (alpha_prime * tau2)) * np.abs(arg) ** 2)
    return complex(
        -R * R / (4.0 * tau2 * tau2)
        * np.sum(np.conjugate(arg) ** 2 * weights)
        / np.sum(weights)
    )


def discretized_torus_components(L: int, l1: int, l2: int, R: float,
                                 n_winding: int = 8,
                                 alpha_prime: float = DEFAULT_ALPHA_PRIME):
    """
    Numerical torus one-point components in the flat coordinate w.

    Returns
        (torus_oscillator, torus_classical, tau, P1, fhat0, dw_dz0, schwarzian0)
    where the oscillator/classical split matches the disk derivation:

        disk = disk_oscillator + disk_classical
             = (dw/dz)^2 (torus_oscillator + torus_classical)
               - alpha' * {w,z}/12.
    """
    disk_comp = _discretized_disk_components(L, l1, l2, R, n_winding=n_winding)
    geo = _geometry_data(L, l1, l2)

    torus_oscillator = (
        disk_comp.oscillator_disk + alpha_prime * geo.schwarzian0 / 12.0
    ) / (geo.dw_dz0 * geo.dw_dz0)
    torus_classical = disk_comp.classical_disk / (geo.dw_dz0 * geo.dw_dz0)
    return (
        complex(torus_oscillator),
        complex(torus_classical),
        geo.tau,
        geo.P1,
        geo.fhat0,
        geo.dw_dz0,
        geo.schwarzian0,
    )


def analytic_torus_one_point(tau: complex, R: float, n_winding: int = 10,
                             alpha_prime: float = DEFAULT_ALPHA_PRIME) -> complex:
    """
    Exact compact-boson one-point function of :partial X partial X:
    in the flat torus coordinate used in the draft.

    The note follows Polchinski's torus convention. If u has periods 1, tau
    and w = 2 pi u has periods 2 pi, 2 pi tau, then

        <T(u)> = -4 pi^2 * (theta_1'''(0|tau) / (6 theta_1'(0|tau))
                            + 1 / (8 pi tau_2))

    for the non-compact boson, and T = - :partial X partial X:.
    Dividing by (2 pi)^2 to pass from u to w therefore gives the oscillator
    part of < :partial_w X partial_w X: >.

    In code we evaluate theta_1 with mpmath.jtheta(1, z, q), which uses the
    same angular variable z as Polchinski/Mathematica JacobiTheta. That is why
    the coincident-limit term here is theta_1'''(0, q) / (24 theta_1'(0, q)),
    not theta_1''' / (24 pi^2 theta_1').

    Equivalently,

        < :partial_w X partial_w X: > = (1 / (2 pi i)) * partial_tau log Z

    in the flat torus coordinate w ~ w + 2 pi ~ w + 2 pi tau.
    """
    oscillator = _analytic_torus_oscillator(tau, alpha_prime=alpha_prime)
    classical = _analytic_torus_classical(
        tau, R, n_winding=n_winding, alpha_prime=alpha_prime
    )
    return complex(oscillator + classical)


def analytic_torus_components(tau: complex, R: float, n_winding: int = 10,
                              alpha_prime: float = DEFAULT_ALPHA_PRIME):
    """
    Exact torus one-point components in the flat coordinate w.

    Returns
        (torus_oscillator, torus_classical).
    """
    oscillator = _analytic_torus_oscillator(tau, alpha_prime=alpha_prime)
    classical = _analytic_torus_classical(
        tau, R, n_winding=n_winding, alpha_prime=alpha_prime
    )
    return complex(oscillator), complex(classical)


def analytic_disk_one_point(L: int, l1: int, l2: int, R: float,
                            n_winding: int = 10,
                            alpha_prime: float = DEFAULT_ALPHA_PRIME):
    """
    Analytic disk one-point function obtained by conformally transforming
    the flat-torus result.

    The torus formula is written in the flat coordinate w with periods
    2 pi and 2 pi tau. If u_raw(z) = int_0^z f_raw(z') dz' has periods
    (P1, P2), then w = 2 pi * u_raw / P1 and
        dw/dz|_{z=0} = 2 pi * fhat(0).
    """
    geo = _geometry_data(L, l1, l2)
    torus = analytic_torus_one_point(
        geo.tau, R, n_winding=n_winding, alpha_prime=alpha_prime
    )
    disk = geo.dw_dz0 * geo.dw_dz0 * torus - alpha_prime * geo.schwarzian0 / 12.0
    return (
        complex(disk),
        complex(torus),
        geo.tau,
        geo.P1,
        geo.fhat0,
        geo.dw_dz0,
        geo.schwarzian0,
    )


def benchmark_point(L: int, l1: int, l2: int, R: float,
                    n_winding_lattice: int = 8,
                    n_winding_analytic: int = 10,
                    alpha_prime: float = DEFAULT_ALPHA_PRIME) -> BenchmarkResult:
    """
    Run the full disk-to-torus benchmark for one choice of (L, l1, l2, R).

    The primary numerical observable is the direct discretized disk one-point
    function. We compare it against the exact torus formula mapped to the disk.

    Concretely, the main test is

        disk_numeric  vs.  disk_exact_from_torus,

    where disk_exact_from_torus is obtained from the exact flat-torus formula
    by the conformal transformation to the disk.

    We also map the numerical disk answer back to the flat torus using

        <dXdX>_T = ( <dXdX>_D + alpha' * {w,z}/12 ) / (dw/dz)^2

    This torus-from-disk quantity is only a diagnostic.
    The oscillator/classical torus split is also kept as a diagnostic,
    together with a check that it reconstructs the direct disk answer.
    """
    if abs(alpha_prime - 1.0) > 1.0e-14:
        raise ValueError(
            "The current lattice benchmark is normalized for alpha' = 1. "
            "Other values are not yet implemented consistently on the disk side."
        )

    l3 = L // 2 - l1 - l2
    if l3 <= 0:
        raise ValueError(f"Need l3 > 0, got l3={l3}")

    (
        torus_from_disk_oscillator,
        torus_from_disk_classical,
        tau,
        P1,
        fhat0,
        dw_dz0,
        schwarzian0,
    ) = discretized_torus_components(
        L, l1, l2, R, n_winding=n_winding_lattice, alpha_prime=alpha_prime
    )
    torus_from_components = torus_from_disk_oscillator + torus_from_disk_classical
    disk_reconstructed = (
        dw_dz0 * dw_dz0 * torus_from_components - alpha_prime * schwarzian0 / 12.0
    )
    disk_numeric = discretized_disk_one_point(
        L, l1, l2, R, n_winding=n_winding_lattice
    )
    torus_from_disk = (
        disk_numeric + alpha_prime * schwarzian0 / 12.0
    ) / (dw_dz0 * dw_dz0)

    disk_exact_from_torus, torus_exact, _, _, _, _, _ = analytic_disk_one_point(
        L, l1, l2, R, n_winding=n_winding_analytic, alpha_prime=alpha_prime
    )
    torus_exact_oscillator, torus_exact_classical = analytic_torus_components(
        tau, R, n_winding=n_winding_analytic, alpha_prime=alpha_prime
    )

    disk_abs_error = abs(disk_numeric - disk_exact_from_torus)
    disk_rel_error = disk_abs_error / max(abs(disk_exact_from_torus), 1.0e-14)
    disk_reconstruction_abs_error = abs(disk_numeric - disk_reconstructed)
    disk_reconstruction_rel_error = disk_reconstruction_abs_error / max(
        abs(disk_numeric), 1.0e-14
    )
    torus_diagnostic_abs_error = abs(torus_from_disk - torus_exact)
    torus_diagnostic_rel_error = torus_diagnostic_abs_error / max(
        abs(torus_exact), 1.0e-14
    )
    torus_oscillator_diagnostic_abs_error = abs(
        torus_from_disk_oscillator - torus_exact_oscillator
    )
    torus_oscillator_diagnostic_rel_error = torus_oscillator_diagnostic_abs_error / max(
        abs(torus_exact_oscillator), 1.0e-14
    )
    torus_classical_diagnostic_abs_error = abs(
        torus_from_disk_classical - torus_exact_classical
    )
    torus_classical_diagnostic_rel_error = torus_classical_diagnostic_abs_error / max(
        abs(torus_exact_classical), 1.0e-14
    )

    return BenchmarkResult(
        L=L,
        l1=l1,
        l2=l2,
        l3=l3,
        R=R,
        alpha_prime=alpha_prime,
        tau=tau,
        P1=P1,
        fhat0=fhat0,
        dw_dz0=dw_dz0,
        schwarzian0=schwarzian0,
        disk_numeric=disk_numeric,
        disk_reconstructed=disk_reconstructed,
        disk_exact_from_torus=disk_exact_from_torus,
        torus_from_disk=torus_from_disk,
        torus_exact=torus_exact,
        disk_abs_error=disk_abs_error,
        disk_rel_error=disk_rel_error,
        disk_reconstruction_abs_error=disk_reconstruction_abs_error,
        disk_reconstruction_rel_error=disk_reconstruction_rel_error,
        torus_diagnostic_abs_error=torus_diagnostic_abs_error,
        torus_diagnostic_rel_error=torus_diagnostic_rel_error,
        torus_from_disk_oscillator=torus_from_disk_oscillator,
        torus_from_disk_classical=torus_from_disk_classical,
        torus_exact_oscillator=torus_exact_oscillator,
        torus_exact_classical=torus_exact_classical,
        torus_oscillator_diagnostic_abs_error=torus_oscillator_diagnostic_abs_error,
        torus_oscillator_diagnostic_rel_error=torus_oscillator_diagnostic_rel_error,
        torus_classical_diagnostic_abs_error=torus_classical_diagnostic_abs_error,
        torus_classical_diagnostic_rel_error=torus_classical_diagnostic_rel_error,
    )


def compute_one_point(L: int, l1: int, l2: int, R: float,
                      n_winding_lattice: int = 8,
                      n_winding_analytic: int = 10,
                      alpha_prime: float = DEFAULT_ALPHA_PRIME):
    """
    Return the direct disk numerical result together with the exact torus
    formula mapped to the disk.

    Returns
        (disk_numeric, disk_exact_from_torus, tau)
    """
    result = benchmark_point(
        L=L,
        l1=l1,
        l2=l2,
        R=R,
        n_winding_lattice=n_winding_lattice,
        n_winding_analytic=n_winding_analytic,
        alpha_prime=alpha_prime,
    )
    return result.disk_numeric, result.disk_exact_from_torus, result.tau


def run_self_consistency_check(L: int, l1: int, l2: int, R: float,
                               n_winding_lattice: int = 8,
                               n_winding_analytic: int = 10,
                               alpha_prime: float = DEFAULT_ALPHA_PRIME
                               ) -> SelfConsistencyResult:
    """
    Run only exact code-identity checks.

    These checks do not ask whether the physics formula matches Mathematica;
    they only verify that the Python implementation is internally assembled in
    a consistent way.
    """
    if abs(alpha_prime - 1.0) > 1.0e-14:
        raise ValueError(
            "The current self-consistency checker is implemented in the "
            "alpha' = 1 convention."
        )

    l3 = L // 2 - l1 - l2
    if l3 <= 0:
        raise ValueError(f"Need l3 > 0, got l3={l3}")

    geo = _geometry_data(L, l1, l2)
    geo_again = _geometry_data(L, l1, l2)
    f = elt.make_cyl_eqn_improved(L, l1, l2)
    schwarzian_fd = _schwarzian_at_origin_finite_difference(f)

    disk_comp = _discretized_disk_components(L, l1, l2, R, n_winding=n_winding_lattice)
    disk_numeric = discretized_disk_one_point(L, l1, l2, R, n_winding=n_winding_lattice)
    numerical_disk_from_components = disk_comp.oscillator_disk + disk_comp.classical_disk

    torus_num_osc = (
        disk_comp.oscillator_disk + alpha_prime * geo.schwarzian0 / 12.0
    ) / (geo.dw_dz0 * geo.dw_dz0)
    torus_num_cl = disk_comp.classical_disk / (geo.dw_dz0 * geo.dw_dz0)
    torus_num_from_disk = (
        disk_numeric + alpha_prime * geo.schwarzian0 / 12.0
    ) / (geo.dw_dz0 * geo.dw_dz0)

    notebook_trace = 0.5 * disk_comp.compact_trace_11
    trace_target = disk_comp.oscillator_disk / np.pi

    analytic_total = analytic_torus_one_point(
        geo.tau, R, n_winding=n_winding_analytic, alpha_prime=alpha_prime
    )
    analytic_osc, analytic_cl = analytic_torus_components(
        geo.tau, R, n_winding=n_winding_analytic, alpha_prime=alpha_prime
    )
    analytic_disk, _, _, _, _, _, _ = analytic_disk_one_point(
        L, l1, l2, R, n_winding=n_winding_analytic, alpha_prime=alpha_prime
    )
    analytic_disk_from_components = (
        geo.dw_dz0 * geo.dw_dz0 * (analytic_osc + analytic_cl)
        - alpha_prime * geo.schwarzian0 / 12.0
    )

    trace_abs = abs(notebook_trace - trace_target)
    trace_rel = trace_abs / max(abs(trace_target), 1.0e-14)
    tau_abs = abs(geo.tau - geo_again.tau)
    fhat_abs = abs(geo.fhat0 - geo_again.fhat0)
    schwarz_abs = abs(geo.schwarzian0 - schwarzian_fd)
    analytic_sum_abs = abs(analytic_total - (analytic_osc + analytic_cl))
    analytic_sum_rel = analytic_sum_abs / max(abs(analytic_total), 1.0e-14)
    analytic_disk_abs = abs(analytic_disk - analytic_disk_from_components)
    analytic_disk_rel = analytic_disk_abs / max(abs(analytic_disk), 1.0e-14)
    numerical_disk_abs = abs(disk_numeric - numerical_disk_from_components)
    numerical_disk_rel = numerical_disk_abs / max(abs(disk_numeric), 1.0e-14)
    numerical_torus_abs = abs(torus_num_from_disk - (torus_num_osc + torus_num_cl))
    numerical_torus_rel = numerical_torus_abs / max(abs(torus_num_from_disk), 1.0e-14)

    return SelfConsistencyResult(
        L=L,
        l1=l1,
        l2=l2,
        l3=l3,
        R=R,
        alpha_prime=alpha_prime,
        tau=geo.tau,
        trace_normalization_abs_error=float(trace_abs),
        trace_normalization_rel_error=float(trace_rel),
        geometry_tau_abs_error=float(tau_abs),
        geometry_fhat_abs_error=float(fhat_abs),
        schwarzian_fd_abs_error=float(schwarz_abs),
        analytic_component_sum_abs_error=float(analytic_sum_abs),
        analytic_component_sum_rel_error=float(analytic_sum_rel),
        analytic_disk_transform_abs_error=float(analytic_disk_abs),
        analytic_disk_transform_rel_error=float(analytic_disk_rel),
        numerical_disk_component_sum_abs_error=float(numerical_disk_abs),
        numerical_disk_component_sum_rel_error=float(numerical_disk_rel),
        numerical_torus_transform_abs_error=float(numerical_torus_abs),
        numerical_torus_transform_rel_error=float(numerical_torus_rel),
    )


def _format_complex(z: complex) -> str:
    return f"{z.real:.12g} {z.imag:+.12g}i"


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _notebook_l1_scan(setL: int, k_step: int = 2) -> list[NotebookHolomorphicComparison]:
    half = setL // 2
    k_max = (half - 1) // 2
    return [
        notebook_holomorphic_compare(setL, k, k)
        for k in range(1, k_max + 1, k_step)
    ]


def _curve_path(xs, ys, x_min, x_max, y_min, y_max,
                left, top, width, height) -> str:
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0
    points = []
    for x, y in zip(xs, ys):
        sx = left + width * (x - x_min) / (x_max - x_min)
        sy = top + height * (1.0 - (y - y_min) / (y_max - y_min))
        points.append(f"{sx:.2f},{sy:.2f}")
    return "M " + " L ".join(points)


def write_notebook_l1_plot_svg(setL: int, output_path: str | Path,
                               k_step: int = 2) -> Path:
    """
    Plot the notebook-style L1 family:

        findTrCA[setL, k, k, 1, AMat, CMat11]

    using two panels for Re/Im parts of the analytic and numerical values.
    """
    data = _notebook_l1_scan(setL, k_step=k_step)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    xs = [item.l1 for item in data]
    ana_re = [item.notebook_analytic.real for item in data]
    num_re = [item.notebook_numerical.real for item in data]
    ana_im = [item.notebook_analytic.imag for item in data]
    num_im = [item.notebook_numerical.imag for item in data]

    width = 1100
    height = 820
    left = 110
    right = 40
    top_margin = 90
    bottom_margin = 80
    panel_gap = 80
    panel_width = width - left - right
    panel_height = (height - top_margin - bottom_margin - panel_gap) / 2.0

    x_min = min(xs)
    x_max = max(xs)

    def pad(lo, hi):
        if hi == lo:
            delta = 1.0
        else:
            delta = 0.08 * (hi - lo)
        return lo - delta, hi + delta

    re_min, re_max = pad(min(ana_re + num_re), max(ana_re + num_re))
    im_min, im_max = pad(min(ana_im + num_im), max(ana_im + num_im))

    top_panel = top_margin
    bot_panel = top_margin + panel_height + panel_gap

    analytic_color = "#1768ac"
    numerical_color = "#c44536"
    axis_color = "#222222"
    grid_color = "#d0d7de"
    bg_color = "#ffffff"

    def y_ticks(lo, hi, n=5):
        return [lo + (hi - lo) * i / (n - 1) for i in range(n)]

    x_ticks = [xs[0], xs[len(xs)//4], xs[len(xs)//2], xs[(3*len(xs))//4], xs[-1]]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg_color}"/>',
        '<style>',
        'text { font-family: "Helvetica", "Arial", sans-serif; fill: #111; }',
        '.title { font-size: 28px; font-weight: 700; }',
        '.subtitle { font-size: 16px; fill: #444; }',
        '.axis-label { font-size: 18px; font-weight: 600; }',
        '.tick { font-size: 14px; fill: #444; }',
        '.legend { font-size: 15px; }',
        '.panel-title { font-size: 20px; font-weight: 600; }',
        '</style>',
        f'<text class="title" x="{width/2:.1f}" y="40" text-anchor="middle">Notebook L1 Benchmark for :dX dX:</text>',
        f'<text class="subtitle" x="{width/2:.1f}" y="66" text-anchor="middle">setL = {setL}, x-axis k with l1 = l2 = k and l3 = L/2 - 2k</text>',
    ]

    for panel_top, y_lo, y_hi, panel_title, ana_vals, num_vals in [
        (top_panel, re_min, re_max, "Real Part", ana_re, num_re),
        (bot_panel, im_min, im_max, "Imaginary Part", ana_im, num_im),
    ]:
        lines.append(f'<text class="panel-title" x="{left}" y="{panel_top - 18:.1f}">{_svg_escape(panel_title)}</text>')
        lines.append(
            f'<rect x="{left}" y="{panel_top:.1f}" width="{panel_width:.1f}" height="{panel_height:.1f}" '
            f'fill="none" stroke="{axis_color}" stroke-width="1.5"/>'
        )

        for xt in x_ticks:
            x = left + panel_width * (xt - x_min) / (x_max - x_min)
            lines.append(
                f'<line x1="{x:.2f}" y1="{panel_top:.2f}" x2="{x:.2f}" y2="{panel_top + panel_height:.2f}" '
                f'stroke="{grid_color}" stroke-width="1"/>'
            )
            lines.append(
                f'<text class="tick" x="{x:.2f}" y="{panel_top + panel_height + 24:.2f}" text-anchor="middle">{xt}</text>'
            )

        for yt in y_ticks(y_lo, y_hi):
            y = panel_top + panel_height * (1.0 - (yt - y_lo) / (y_hi - y_lo))
            lines.append(
                f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{left + panel_width:.2f}" y2="{y:.2f}" '
                f'stroke="{grid_color}" stroke-width="1"/>'
            )
            lines.append(
                f'<text class="tick" x="{left - 12:.2f}" y="{y + 5:.2f}" text-anchor="end">{yt:.4f}</text>'
            )

        ana_path = _curve_path(
            xs, ana_vals, x_min, x_max, y_lo, y_hi,
            left, panel_top, panel_width, panel_height
        )
        num_path = _curve_path(
            xs, num_vals, x_min, x_max, y_lo, y_hi,
            left, panel_top, panel_width, panel_height
        )
        lines.append(f'<path d="{ana_path}" fill="none" stroke="{analytic_color}" stroke-width="3"/>')
        lines.append(f'<path d="{num_path}" fill="none" stroke="{numerical_color}" stroke-width="3" stroke-dasharray="10 7"/>')

    legend_x = width - 310
    legend_y = 46
    lines.extend([
        f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 55}" y2="{legend_y}" stroke="{analytic_color}" stroke-width="4"/>',
        f'<text class="legend" x="{legend_x + 68}" y="{legend_y + 5}">Analytic</text>',
        f'<line x1="{legend_x}" y1="{legend_y + 28}" x2="{legend_x + 55}" y2="{legend_y + 28}" stroke="{numerical_color}" stroke-width="4" stroke-dasharray="10 7"/>',
        f'<text class="legend" x="{legend_x + 68}" y="{legend_y + 33}">Numerical</text>',
        f'<text class="axis-label" x="{width/2:.1f}" y="{height - 22:.1f}" text-anchor="middle">k = l1 = l2</text>',
        f'<text class="axis-label" x="28" y="{height/2:.1f}" transform="rotate(-90 28 {height/2:.1f})" text-anchor="middle">One-point function value</text>',
        '</svg>',
    ])

    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _compact_classical_l1_scan(setL: int, R: float,
                               k_step: int = 2,
                               n_winding_lattice: int = 12,
                               n_winding_analytic: int = 24):
    half = setL // 2
    k_max = (half - 1) // 2
    rows = []
    for k in range(1, k_max + 1, k_step):
        disk_comp = _discretized_disk_components(
            setL, k, k, R, n_winding=n_winding_lattice
        )
        geo = _geometry_data(setL, k, k)
        exact_torus_classical = _analytic_torus_classical(
            geo.tau, R, n_winding=n_winding_analytic, alpha_prime=DEFAULT_ALPHA_PRIME
        )
        exact_disk_classical = geo.dw_dz0 * geo.dw_dz0 * exact_torus_classical
        rows.append(
            {
                "k": k,
                "tau": geo.tau,
                "numerical": complex(disk_comp.classical_disk),
                "analytic": complex(exact_disk_classical),
            }
        )
    return rows


def write_compact_classical_l1_plot_svg(setL: int, R: float,
                                        output_path: str | Path,
                                        k_step: int = 2,
                                        n_winding_lattice: int = 12,
                                        n_winding_analytic: int = 24) -> Path:
    """
    Plot only the compact classical correction along the L1 family

        l1 = l2 = k,  l3 = L/2 - 2k

    on the disk side. This isolates the winding/exponential part from the
    oscillator contribution.
    """
    data = _compact_classical_l1_scan(
        setL,
        R,
        k_step=k_step,
        n_winding_lattice=n_winding_lattice,
        n_winding_analytic=n_winding_analytic,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    xs = [item["k"] for item in data]
    ana_re = [item["analytic"].real for item in data]
    num_re = [item["numerical"].real for item in data]
    ana_im = [item["analytic"].imag for item in data]
    num_im = [item["numerical"].imag for item in data]

    width = 1100
    height = 820
    left = 110
    right = 40
    top_margin = 90
    bottom_margin = 80
    panel_gap = 80
    panel_width = width - left - right
    panel_height = (height - top_margin - bottom_margin - panel_gap) / 2.0

    x_min = min(xs)
    x_max = max(xs)

    def pad(lo, hi):
        if hi == lo:
            delta = 1.0
        else:
            delta = 0.08 * (hi - lo)
        return lo - delta, hi + delta

    re_min, re_max = pad(min(ana_re + num_re), max(ana_re + num_re))
    im_min, im_max = pad(min(ana_im + num_im), max(ana_im + num_im))

    top_panel = top_margin
    bot_panel = top_margin + panel_height + panel_gap

    analytic_color = "#1768ac"
    numerical_color = "#c44536"
    axis_color = "#222222"
    grid_color = "#d0d7de"
    bg_color = "#ffffff"

    def y_ticks(lo, hi, n=5):
        return [lo + (hi - lo) * i / (n - 1) for i in range(n)]

    x_ticks = [xs[0], xs[len(xs)//4], xs[len(xs)//2], xs[(3*len(xs))//4], xs[-1]]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{bg_color}"/>',
        '<style>',
        'text { font-family: "Helvetica", "Arial", sans-serif; fill: #111; }',
        '.title { font-size: 28px; font-weight: 700; }',
        '.subtitle { font-size: 16px; fill: #444; }',
        '.axis-label { font-size: 18px; font-weight: 600; }',
        '.tick { font-size: 14px; fill: #444; }',
        '.legend { font-size: 15px; }',
        '.panel-title { font-size: 20px; font-weight: 600; }',
        '</style>',
        f'<text class="title" x="{width/2:.1f}" y="40" text-anchor="middle">Compact Classical Correction on Disk</text>',
        f'<text class="subtitle" x="{width/2:.1f}" y="66" text-anchor="middle">setL = {setL}, R = {R}, x-axis k with l1 = l2 = k and l3 = L/2 - 2k</text>',
    ]

    for panel_top, y_lo, y_hi, panel_title, ana_vals, num_vals in [
        (top_panel, re_min, re_max, "Real Part", ana_re, num_re),
        (bot_panel, im_min, im_max, "Imaginary Part", ana_im, num_im),
    ]:
        lines.append(f'<text class="panel-title" x="{left}" y="{panel_top - 18:.1f}">{_svg_escape(panel_title)}</text>')
        lines.append(
            f'<rect x="{left}" y="{panel_top:.1f}" width="{panel_width:.1f}" height="{panel_height:.1f}" '
            f'fill="none" stroke="{axis_color}" stroke-width="1.5"/>'
        )

        for xt in x_ticks:
            x = left + panel_width * (xt - x_min) / (x_max - x_min)
            lines.append(
                f'<line x1="{x:.2f}" y1="{panel_top:.2f}" x2="{x:.2f}" y2="{panel_top + panel_height:.2f}" '
                f'stroke="{grid_color}" stroke-width="1"/>'
            )
            lines.append(
                f'<text class="tick" x="{x:.2f}" y="{panel_top + panel_height + 24:.2f}" text-anchor="middle">{xt}</text>'
            )

        for yt in y_ticks(y_lo, y_hi):
            y = panel_top + panel_height * (1.0 - (yt - y_lo) / (y_hi - y_lo))
            lines.append(
                f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{left + panel_width:.2f}" y2="{y:.2f}" '
                f'stroke="{grid_color}" stroke-width="1"/>'
            )
            lines.append(
                f'<text class="tick" x="{left - 12:.2f}" y="{y + 5:.2f}" text-anchor="end">{yt:.5f}</text>'
            )

        ana_path = _curve_path(
            xs, ana_vals, x_min, x_max, y_lo, y_hi,
            left, panel_top, panel_width, panel_height
        )
        num_path = _curve_path(
            xs, num_vals, x_min, x_max, y_lo, y_hi,
            left, panel_top, panel_width, panel_height
        )
        lines.append(f'<path d="{ana_path}" fill="none" stroke="{analytic_color}" stroke-width="3"/>')
        lines.append(f'<path d="{num_path}" fill="none" stroke="{numerical_color}" stroke-width="3" stroke-dasharray="10 7"/>')

    legend_x = width - 310
    legend_y = 46
    lines.extend([
        f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 55}" y2="{legend_y}" stroke="{analytic_color}" stroke-width="4"/>',
        f'<text class="legend" x="{legend_x + 68}" y="{legend_y + 5}">Analytic classical</text>',
        f'<line x1="{legend_x}" y1="{legend_y + 28}" x2="{legend_x + 55}" y2="{legend_y + 28}" stroke="{numerical_color}" stroke-width="4" stroke-dasharray="10 7"/>',
        f'<text class="legend" x="{legend_x + 68}" y="{legend_y + 33}">Numerical classical</text>',
        f'<text class="axis-label" x="{width/2:.1f}" y="{height - 22:.1f}" text-anchor="middle">k = l1 = l2</text>',
        f'<text class="axis-label" x="28" y="{height/2:.1f}" transform="rotate(-90 28 {height/2:.1f})" text-anchor="middle">Disk classical correction</text>',
        '</svg>',
    ])

    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the compact-boson torus one-point function < :partial X partial X: >."
    )
    parser.add_argument("--L", type=int, default=120, help="Even lattice size.")
    parser.add_argument("--l1", type=int, default=18, help="First Strebel length.")
    parser.add_argument("--l2", type=int, default=22, help="Second Strebel length.")
    parser.add_argument("--R", type=float, default=1.0, help="Compactification radius.")
    parser.add_argument(
        "--alpha-prime",
        type=float,
        default=DEFAULT_ALPHA_PRIME,
        help="Worldsheet normalization alpha'.",
    )
    parser.add_argument(
        "--nwind-lattice",
        type=int,
        default=8,
        help="Winding cutoff for the lattice sector sum.",
    )
    parser.add_argument(
        "--nwind-analytic",
        type=int,
        default=10,
        help="Winding cutoff for the analytic torus sum.",
    )
    parser.add_argument(
        "--decompose",
        action="store_true",
        help="Also print oscillator and classical pieces separately.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run internal code-consistency checks instead of the full benchmark.",
    )
    parser.add_argument(
        "--plot-notebook-l1",
        action="store_true",
        help="Write an SVG plot for the notebook L1 family with l1 = l2 = k.",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default="",
        help="Output path for the SVG plot written by --plot-notebook-l1.",
    )
    parser.add_argument(
        "--plot-step",
        type=int,
        default=2,
        help="Step size in k for the notebook L1 scan.",
    )
    parser.add_argument(
        "--plot-classical-l1",
        action="store_true",
        help="Write an SVG plot for the compact classical correction along l1 = l2 = k.",
    )
    args = parser.parse_args()

    if args.plot_notebook_l1:
        output = (
            Path(args.plot_output)
            if args.plot_output
            else Path(__file__).resolve().with_name(f"notebook_L1_setL{args.L}.svg")
        )
        output = write_notebook_l1_plot_svg(args.L, output, k_step=args.plot_step)
        print(f"Wrote notebook L1 plot to {output}")
        return

    if args.plot_classical_l1:
        output = (
            Path(args.plot_output)
            if args.plot_output
            else Path(__file__).resolve().with_name(
                f"classical_L1_setL{args.L}_R{str(args.R).replace('.', 'p')}.svg"
            )
        )
        output = write_compact_classical_l1_plot_svg(
            args.L,
            args.R,
            output,
            k_step=args.plot_step,
            n_winding_lattice=args.nwind_lattice,
            n_winding_analytic=args.nwind_analytic,
        )
        print(f"Wrote compact classical L1 plot to {output}")
        return

    if args.self_check:
        check = run_self_consistency_check(
            L=args.L,
            l1=args.l1,
            l2=args.l2,
            R=args.R,
            n_winding_lattice=args.nwind_lattice,
            n_winding_analytic=args.nwind_analytic,
            alpha_prime=args.alpha_prime,
        )
        print(
            f"Self-check for L={check.L}, l1={check.l1}, l2={check.l2}, "
            f"l3={check.l3}, R={check.R}, alpha'={check.alpha_prime}"
        )
        print(f"tau                      = {_format_complex(check.tau)}")
        print(f"trace normalization err  = {check.trace_normalization_rel_error:.6e}")
        print(f"geometry tau abs err     = {check.geometry_tau_abs_error:.6e}")
        print(f"geometry fhat abs err    = {check.geometry_fhat_abs_error:.6e}")
        print(f"Schwarzian fd abs err    = {check.schwarzian_fd_abs_error:.6e}")
        print(f"analytic split rel err   = {check.analytic_component_sum_rel_error:.6e}")
        print(f"analytic disk rel err    = {check.analytic_disk_transform_rel_error:.6e}")
        print(f"numerical disk rel err   = {check.numerical_disk_component_sum_rel_error:.6e}")
        print(f"numerical torus rel err  = {check.numerical_torus_transform_rel_error:.6e}")
        return

    result = benchmark_point(
        L=args.L,
        l1=args.l1,
        l2=args.l2,
        R=args.R,
        n_winding_lattice=args.nwind_lattice,
        n_winding_analytic=args.nwind_analytic,
        alpha_prime=args.alpha_prime,
    )

    print(
        f"L={result.L}, l1={result.l1}, l2={result.l2}, "
        f"l3={result.l3}, R={result.R}, alpha'={result.alpha_prime}"
    )
    print(f"tau                = {_format_complex(result.tau)}")
    print(f"P1                 = {_format_complex(result.P1)}")
    print(f"fhat(0)            = {_format_complex(result.fhat0)}")
    print(f"dw/dz(0)           = {_format_complex(result.dw_dz0)}")
    print(f"Schwarzian(0)      = {_format_complex(result.schwarzian0)}")
    print()
    print(f"disk numeric               = {_format_complex(result.disk_numeric)}")
    print(f"disk exact-from-torus      = {_format_complex(result.disk_exact_from_torus)}")
    print(f"disk abs error             = {result.disk_abs_error:.6e}")
    print(f"disk rel error             = {result.disk_rel_error:.6e}")
    print(f"disk from torus diagnostic = {_format_complex(result.disk_reconstructed)}")
    print(f"disk internal abs err      = {result.disk_reconstruction_abs_error:.6e}")
    print(f"disk internal rel err      = {result.disk_reconstruction_rel_error:.6e}")
    if args.decompose:
        print()
        print("torus exact               = "
              f"{_format_complex(result.torus_exact)}")
        print("torus from disk diagnostic= "
              f"{_format_complex(result.torus_from_disk)}")
        print(f"torus diagnostic abs err  = {result.torus_diagnostic_abs_error:.6e}")
        print(f"torus diagnostic rel err  = {result.torus_diagnostic_rel_error:.6e}")
        print()
        print("torus oscillator exact    = "
              f"{_format_complex(result.torus_exact_oscillator)}")
        print("torus oscillator diag     = "
              f"{_format_complex(result.torus_from_disk_oscillator)}")
        print(f"torus osc diag abs err    = {result.torus_oscillator_diagnostic_abs_error:.6e}")
        print(f"torus osc diag rel err    = {result.torus_oscillator_diagnostic_rel_error:.6e}")
        print()
        print("torus classical exact     = "
              f"{_format_complex(result.torus_exact_classical)}")
        print("torus classical diag      = "
              f"{_format_complex(result.torus_from_disk_classical)}")
        print(f"torus cls diag abs err    = {result.torus_classical_diagnostic_abs_error:.6e}")
        print(f"torus cls diag rel err    = {result.torus_classical_diagnostic_rel_error:.6e}")


if __name__ == "__main__":
    main()
