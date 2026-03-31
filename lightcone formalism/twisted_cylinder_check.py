#!/usr/bin/env python3
"""
Diagnostics for the sigma-twisted free cylinder kernels.

This file implements the first loop-side building block of the discrete-sigma
program:

1. the centered-frequency spectral shift operator R_N(phi),
2. the twisted bosonic site-basis Gaussian matrices A(T), B(T, phi),
3. the twisted fermionic transport matrix U(T, phi),
4. oscillator-sector diagnostics relevant for later loop integrands.

The implementation uses the DFT convention

    F_{kn} = N^{-1/2} exp(2 pi i k n / N),

so a geometric forward site shift by m sites corresponds to the spectral phase
exp(-2 pi i k m / N). With this sign choice, the spectral interpolation matches
the literal permutation when phi N is an integer.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tachyon_check as tc


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    return value


def dft_matrix(n_sites: int) -> np.ndarray:
    indices = np.arange(n_sites, dtype=float)
    return np.exp(2j * math.pi * np.outer(indices, indices) / n_sites) / math.sqrt(
        n_sites
    )


def centered_kappas(n_sites: int) -> np.ndarray:
    kappas = np.arange(n_sites, dtype=int)
    kappas[kappas > (n_sites - 1) // 2] -= n_sites
    return kappas


def lattice_frequencies(n_sites: int, lattice_spacing: float = 1.0) -> np.ndarray:
    modes = np.arange(n_sites, dtype=float)
    return (2.0 / lattice_spacing) * np.sin(math.pi * modes / n_sites)


def single_mode_mass(lattice_spacing: float = 1.0, alpha_prime: float = 1.0) -> float:
    return lattice_spacing / (2.0 * math.pi * alpha_prime)


def _omega_coth(omega: np.ndarray, schwinger_T: float) -> np.ndarray:
    values = np.empty_like(omega, dtype=float)
    nonzero = np.abs(omega) > 1.0e-14
    values[nonzero] = omega[nonzero] / np.tanh(omega[nonzero] * schwinger_T)
    values[~nonzero] = 1.0 / schwinger_T
    return values


def _omega_csch(omega: np.ndarray, schwinger_T: float) -> np.ndarray:
    values = np.empty_like(omega, dtype=float)
    nonzero = np.abs(omega) > 1.0e-14
    values[nonzero] = omega[nonzero] / np.sinh(omega[nonzero] * schwinger_T)
    values[~nonzero] = 1.0 / schwinger_T
    return values


def shift_operator(
    n_sites: int,
    phi: float,
) -> np.ndarray:
    """
    Spectral interpolation of a sigma translation by phi of the circumference.

    With the DFT convention used here, the exact integer shift m = phi N acts as
        (R_N(phi) X)_n = X_{n + m mod N},
    so the spectral phase is exp(-2 pi i kappa_k phi).
    """
    fourier = dft_matrix(n_sites)
    phases = np.exp(-2j * math.pi * centered_kappas(n_sites) * phi)
    return fourier.conj().T @ np.diag(phases) @ fourier


def exact_shift_permutation(n_sites: int, shift_sites: int) -> np.ndarray:
    permutation = np.zeros((n_sites, n_sites), dtype=float)
    shift = shift_sites % n_sites
    for row in range(n_sites):
        permutation[row, (row + shift) % n_sites] = 1.0
    return permutation


def bosonic_site_matrices(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    *,
    lattice_spacing: float = 1.0,
    alpha_prime: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    fourier = dft_matrix(n_sites)
    omega = lattice_frequencies(n_sites, lattice_spacing=lattice_spacing)
    mu = single_mode_mass(lattice_spacing=lattice_spacing, alpha_prime=alpha_prime)
    phases = np.exp(-2j * math.pi * centered_kappas(n_sites) * phi)
    a_diag = mu * _omega_coth(omega, schwinger_T)
    b_diag = mu * phases * _omega_csch(omega, schwinger_T)
    a_matrix = fourier.conj().T @ np.diag(a_diag) @ fourier
    b_matrix = fourier.conj().T @ np.diag(b_diag) @ fourier
    return a_matrix, b_matrix


def fermionic_transport_matrix(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    *,
    lattice_spacing: float = 1.0,
) -> np.ndarray:
    fourier = dft_matrix(n_sites)
    omega = lattice_frequencies(n_sites, lattice_spacing=lattice_spacing)
    phases = np.exp(-2j * math.pi * centered_kappas(n_sites) * phi)
    transport = np.exp(-omega * schwinger_T) * phases
    return fourier.conj().T @ np.diag(transport) @ fourier


def bosonic_trace_quadratic_oscillator(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    *,
    lattice_spacing: float = 1.0,
    alpha_prime: float = 1.0,
) -> np.ndarray:
    """
    Oscillator-sector quadratic form for the twisted vacuum trace.

    After setting X_f = X_i = X and removing the center-of-mass direction, the
    relevant complex-symmetric quadratic form is

        Q_osc = S^T (2 A - B - B^T) S,

    where S is the real zero-sum basis. Its real part controls Gaussian
    convergence on the real oscillator integration contour.
    """
    basis, _ = tc.real_zero_sum_basis(n_sites)
    a_matrix, b_matrix = bosonic_site_matrices(
        n_sites,
        schwinger_T,
        phi,
        lattice_spacing=lattice_spacing,
        alpha_prime=alpha_prime,
    )
    return basis.T @ (2.0 * a_matrix - b_matrix - b_matrix.T) @ basis


def bosonic_trace_real_eigenvalues(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    *,
    lattice_spacing: float = 1.0,
    alpha_prime: float = 1.0,
) -> np.ndarray:
    """Closed-form eigenvalues of Re Q_osc in the real oscillator basis."""
    _, modes = tc.real_zero_sum_basis(n_sites)
    mu = single_mode_mass(lattice_spacing=lattice_spacing, alpha_prime=alpha_prime)
    omega = (2.0 / lattice_spacing) * np.sin(math.pi * modes / n_sites)
    theta = 2.0 * math.pi * modes * phi
    return 2.0 * mu * (
        _omega_coth(omega, schwinger_T) - np.cos(theta) * _omega_csch(omega, schwinger_T)
    )


def bosonic_trace_normalization(
    n_sites: int,
    schwinger_T: float,
    *,
    lattice_spacing: float = 1.0,
    alpha_prime: float = 1.0,
) -> complex:
    _, modes = tc.real_zero_sum_basis(n_sites)
    mu = single_mode_mass(lattice_spacing=lattice_spacing, alpha_prime=alpha_prime)
    norm = 1.0 + 0.0j
    for mode in modes:
        omega = (2.0 / lattice_spacing) * math.sin(math.pi * mode / n_sites)
        norm *= cmath.sqrt(
            mu * omega / (2.0 * math.pi * math.sinh(omega * schwinger_T))
        )
    return norm


def bosonic_trace_factor_direct(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    *,
    lattice_spacing: float = 1.0,
    alpha_prime: float = 1.0,
) -> complex:
    q_osc = bosonic_trace_quadratic_oscillator(
        n_sites,
        schwinger_T,
        phi,
        lattice_spacing=lattice_spacing,
        alpha_prime=alpha_prime,
    )
    dim = q_osc.shape[0]
    norm = bosonic_trace_normalization(
        n_sites,
        schwinger_T,
        lattice_spacing=lattice_spacing,
        alpha_prime=alpha_prime,
    )
    return norm * (2.0 * math.pi) ** (dim / 2.0) / cmath.sqrt(np.linalg.det(q_osc))


def bosonic_trace_factor_closed(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    *,
    lattice_spacing: float = 1.0,
) -> complex:
    factor = 1.0 + 0.0j
    for mode in range(1, (n_sites - 1) // 2 + 1):
        omega = (2.0 / lattice_spacing) * math.sin(math.pi * mode / n_sites)
        theta = 2.0 * math.pi * mode * phi
        factor *= 1.0 / (2.0 * (math.cosh(omega * schwinger_T) - math.cos(theta)))

    if n_sites % 2 == 0:
        nyquist = n_sites // 2
        omega = (2.0 / lattice_spacing) * math.sin(math.pi * nyquist / n_sites)
        theta = math.pi * n_sites * phi
        factor *= 1.0 / cmath.sqrt(
            2.0 * (math.cosh(omega * schwinger_T) - cmath.exp(1j * theta))
        )

    return factor


def bosonic_trace_report(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    *,
    lattice_spacing: float = 1.0,
    alpha_prime: float = 1.0,
) -> dict[str, float]:
    q_osc = bosonic_trace_quadratic_oscillator(
        n_sites,
        schwinger_T,
        phi,
        lattice_spacing=lattice_spacing,
        alpha_prime=alpha_prime,
    )
    real_part = 0.5 * (q_osc.real + q_osc.real.T)
    sign, logdet_numeric = np.linalg.slogdet(real_part)
    if sign <= 0:
        raise RuntimeError(
            f"trace quadratic real part is not positive definite for "
            f"(N={n_sites}, T={schwinger_T}, phi={phi})"
        )
    q_closed_eigs = bosonic_trace_real_eigenvalues(
        n_sites,
        schwinger_T,
        phi,
        lattice_spacing=lattice_spacing,
        alpha_prime=alpha_prime,
    )
    logdet_closed = float(np.sum(np.log(q_closed_eigs)))
    direct_factor = bosonic_trace_factor_direct(
        n_sites,
        schwinger_T,
        phi,
        lattice_spacing=lattice_spacing,
        alpha_prime=alpha_prime,
    )
    closed_factor = bosonic_trace_factor_closed(
        n_sites,
        schwinger_T,
        phi,
        lattice_spacing=lattice_spacing,
    )
    trace_factor_scale = max(1.0, abs(closed_factor))
    return {
        "logdet_numeric": float(logdet_numeric),
        "logdet_closed_form": logdet_closed,
        "abs_error": float(abs(logdet_numeric - logdet_closed)),
        "min_real_eigenvalue": float(np.min(q_closed_eigs)),
        "trace_factor_direct": direct_factor,
        "trace_factor_closed": closed_factor,
        "trace_factor_abs_error": float(abs(direct_factor - closed_factor)),
        "trace_factor_rel_error": float(abs(direct_factor - closed_factor) / trace_factor_scale),
    }


def fermionic_transport_eigenvalue_report(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    *,
    lattice_spacing: float = 1.0,
) -> dict[str, float]:
    basis, modes = tc.real_zero_sum_basis(n_sites)
    transport = fermionic_transport_matrix(
        n_sites,
        schwinger_T,
        phi,
        lattice_spacing=lattice_spacing,
    )
    u_osc = basis.T @ transport @ basis
    eigvals = np.linalg.eigvals(u_osc)

    target_list: list[complex] = []
    counts: dict[int, int] = {}
    for mode in modes:
        counts[int(mode)] = counts.get(int(mode), 0) + 1

    for mode, multiplicity in sorted(counts.items()):
        omega = (2.0 / lattice_spacing) * math.sin(math.pi * mode / n_sites)
        phase = -2.0 * math.pi * mode * phi
        radius = math.exp(-omega * schwinger_T)
        if multiplicity == 2:
            target_list.extend(
                [
                    radius * complex(math.cos(phase), math.sin(phase)),
                    radius * complex(math.cos(-phase), math.sin(-phase)),
                ]
            )
        elif multiplicity == 1:
            if n_sites % 2 != 0 or mode != n_sites // 2:
                raise RuntimeError(
                    f"unexpected single multiplicity mode {mode} for N={n_sites}"
                )
            # The Nyquist mode is self-conjugate in site space but still picks up
            # the centered-frequency phase with kappa_{N/2} = -N/2.
            nyquist_phase = math.pi * n_sites * phi
            target_list.append(
                radius * complex(math.cos(nyquist_phase), math.sin(nyquist_phase))
            )
        else:
            raise RuntimeError(
                f"unexpected multiplicity {multiplicity} for mode {mode} at N={n_sites}"
            )

    target = np.array(target_list, dtype=complex)

    eigvals_sorted = np.array(sorted(eigvals, key=lambda z: (round(z.real, 14), round(z.imag, 14))))
    target_sorted = np.array(sorted(target, key=lambda z: (round(z.real, 14), round(z.imag, 14))))
    max_abs_error = float(np.max(np.abs(eigvals_sorted - target_sorted)))
    return {
        "max_abs_error": max_abs_error,
        "max_modulus": float(np.max(np.abs(eigvals))),
        "min_modulus": float(np.min(np.abs(eigvals))),
    }


def exact_shift_report(
    n_min: int = 3,
    n_max: int = 12,
) -> dict[str, float]:
    max_shift_error = 0.0
    max_cross_error = 0.0
    for n_sites in range(n_min, n_max + 1):
        for shift_sites in range(n_sites):
            phi = shift_sites / n_sites
            shift = shift_operator(n_sites, phi)
            permutation = exact_shift_permutation(n_sites, shift_sites)
            max_shift_error = max(
                max_shift_error,
                float(np.max(np.abs(shift - permutation))),
            )

            _, b_untwisted = bosonic_site_matrices(n_sites, 0.7, 0.0)
            _, b_twisted = bosonic_site_matrices(n_sites, 0.7, phi)
            max_cross_error = max(
                max_cross_error,
                float(np.max(np.abs(b_twisted - b_untwisted @ permutation))),
            )
    return {
        "max_shift_error": max_shift_error,
        "max_cross_error": max_cross_error,
    }


def generic_twist_reality_report(phi: float = 0.17) -> dict[str, float]:
    odd_ns = [5, 7, 9]
    even_ns = [6, 8, 10]
    odd_max_imag = 0.0
    even_min_imag = float("inf")
    odd_b_max_imag = 0.0
    even_b_min_imag = float("inf")
    for n_sites in odd_ns:
        shift = shift_operator(n_sites, phi)
        _, b_matrix = bosonic_site_matrices(n_sites, 0.7, phi)
        odd_max_imag = max(odd_max_imag, float(np.max(np.abs(shift.imag))))
        odd_b_max_imag = max(odd_b_max_imag, float(np.max(np.abs(b_matrix.imag))))
    for n_sites in even_ns:
        shift = shift_operator(n_sites, phi)
        _, b_matrix = bosonic_site_matrices(n_sites, 0.7, phi)
        even_min_imag = min(even_min_imag, float(np.max(np.abs(shift.imag))))
        even_b_min_imag = min(even_b_min_imag, float(np.max(np.abs(b_matrix.imag))))
    return {
        "odd_max_shift_imag": odd_max_imag,
        "odd_max_B_imag": odd_b_max_imag,
        "even_min_shift_imag": even_min_imag,
        "even_min_B_imag": even_b_min_imag,
    }


def oscillator_trace_report(
    *,
    n_values: list[int] | None = None,
    t_values: list[float] | None = None,
    phi_values: list[float] | None = None,
    lattice_spacing: float = 1.0,
    alpha_prime: float = 1.0,
) -> dict[str, object]:
    if n_values is None:
        n_values = [5, 6, 7, 8, 10, 12]
    if t_values is None:
        t_values = [0.05, 0.2, 0.7, 2.0]
    if phi_values is None:
        phi_values = [0.0, 0.17, 0.31, 0.5]

    rows: list[dict[str, float]] = []
    max_logdet_error = 0.0
    max_trace_factor_rel_error = 0.0
    min_real_eigenvalue = float("inf")
    max_transport_error = 0.0
    for n_sites in n_values:
        for schwinger_T in t_values:
            for phi in phi_values:
                trace_row = bosonic_trace_report(
                    n_sites,
                    schwinger_T,
                    phi,
                    lattice_spacing=lattice_spacing,
                    alpha_prime=alpha_prime,
                )
                transport_row = fermionic_transport_eigenvalue_report(
                    n_sites,
                    schwinger_T,
                    phi,
                    lattice_spacing=lattice_spacing,
                )
                max_logdet_error = max(max_logdet_error, trace_row["abs_error"])
                max_trace_factor_rel_error = max(
                    max_trace_factor_rel_error, trace_row["trace_factor_rel_error"]
                )
                min_real_eigenvalue = min(
                    min_real_eigenvalue, trace_row["min_real_eigenvalue"]
                )
                max_transport_error = max(
                    max_transport_error, transport_row["max_abs_error"]
                )
                rows.append(
                    {
                        "N": float(n_sites),
                        "T": float(schwinger_T),
                        "phi": float(phi),
                        "logdet_abs_error": trace_row["abs_error"],
                        "trace_factor_abs_error": trace_row["trace_factor_abs_error"],
                        "trace_factor_rel_error": trace_row["trace_factor_rel_error"],
                        "min_real_eigenvalue": trace_row["min_real_eigenvalue"],
                        "fermion_transport_abs_error": transport_row["max_abs_error"],
                    }
                )

    return {
        "rows": rows,
        "max_logdet_error": max_logdet_error,
        "max_trace_factor_rel_error": max_trace_factor_rel_error,
        "min_real_eigenvalue": min_real_eigenvalue,
        "max_fermion_transport_error": max_transport_error,
    }


def build_report() -> dict[str, object]:
    exact = exact_shift_report()
    reality = generic_twist_reality_report()
    trace = oscillator_trace_report()
    return {
        "exact_shift": exact,
        "generic_reality": reality,
        "oscillator_trace": trace,
        "all_checks_pass": bool(
            exact["max_shift_error"] < 1.0e-12
            and exact["max_cross_error"] < 1.0e-12
            and reality["odd_max_shift_imag"] < 1.0e-12
            and reality["odd_max_B_imag"] < 1.0e-12
            and reality["even_min_shift_imag"] > 1.0e-6
            and reality["even_min_B_imag"] > 1.0e-6
            and trace["max_logdet_error"] < 1.0e-12
            and trace["max_trace_factor_rel_error"] < 1.0e-12
            and trace["min_real_eigenvalue"] > 0.0
            and trace["max_fermion_transport_error"] < 1.0e-12
        ),
    }


def print_report(report: dict[str, object]) -> None:
    exact = report["exact_shift"]
    reality = report["generic_reality"]
    trace = report["oscillator_trace"]
    print("=" * 96)
    print("TWISTED CYLINDER CHECK")
    print("=" * 96)
    print(
        "Exact shifts: "
        f"max |R - P| = {exact['max_shift_error']:.3e}, "
        f"max |B(T,m/N) - B(T,0) P| = {exact['max_cross_error']:.3e}"
    )
    print(
        "Generic twist reality: "
        f"odd max Im R = {reality['odd_max_shift_imag']:.3e}, "
        f"odd max Im B = {reality['odd_max_B_imag']:.3e}, "
        f"even min max-Im R = {reality['even_min_shift_imag']:.3e}, "
        f"even min max-Im B = {reality['even_min_B_imag']:.3e}"
    )
    print(
        "Oscillator trace: "
        f"max logdet error = {trace['max_logdet_error']:.3e}, "
        f"max trace-factor rel error = {trace['max_trace_factor_rel_error']:.3e}, "
        f"min Re-eigenvalue = {trace['min_real_eigenvalue']:.3e}, "
        f"max fermion transport error = {trace['max_fermion_transport_error']:.3e}"
    )
    print()
    print(f"All checks pass: {report['all_checks_pass']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    report = build_report()
    print_report(report)

    if args.json_out is not None:
        output_path = Path(args.json_out)
        output_path.write_text(json.dumps(json_safe(report), indent=2) + "\n")
        print(f"\nWrote JSON report to {output_path}")


if __name__ == "__main__":
    main()
