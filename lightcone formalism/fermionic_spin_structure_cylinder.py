#!/usr/bin/env python3
"""
Fermionic one-cylinder oscillator factors with explicit spatial spin structure.

The earlier loop helpers only varied the coherent-state time-trace insertion
sign in the periodic oscillator sector,

    det(I + s U_periodic),  s = +/- 1,

which is useful but does not yet distinguish periodic and antiperiodic
fermions around the spatial cycle. This module adds that second binary datum:

1. spatial mode shift `nu = 0`   (Ramond-like / periodic oscillator sector),
2. spatial mode shift `nu = 1/2` (NS-like / antiperiodic oscillator sector),
3. time-trace insertion sign `s = +/- 1`.

The result is still an oscillator-sector building block. We have not yet added
the physical zero-mode degeneracy/phases or the final type-II GSO projection.
What this module does provide is an exact finite-N, site-basis implementation
of the four chiral oscillator spin structures that a later physical loop sum
must combine.
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

import single_cylinder_integrand as sci
import tachyon_check as tc


PERIODIC_SHIFT = 0.0
ANTIPERIODIC_SHIFT = 0.5
DEFAULT_N_GRID = (5, 6, 7, 8, 10, 16)
DEFAULT_T_GRID = (0.05, 0.2, 0.7, 2.0)
DEFAULT_PHI_GRID = (0.0, 0.17, 0.31, 0.5)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    return value


def _wrap_phase(phase: float) -> float:
    return ((phase + math.pi) % (2.0 * math.pi)) - math.pi


def sector_label(mode_shift: float, time_trace_sign: int) -> str:
    if time_trace_sign not in (-1, 1):
        raise ValueError("time_trace_sign must be +/-1")
    if abs(mode_shift - PERIODIC_SHIFT) < 1.0e-14:
        return "R" if time_trace_sign == 1 else "R_tilde"
    if abs(mode_shift - ANTIPERIODIC_SHIFT) < 1.0e-14:
        return "NS" if time_trace_sign == 1 else "NS_tilde"
    raise ValueError("only periodic and antiperiodic mode shifts are supported")


def centered_shifted_kappas(n_sites: int, mode_shift: float) -> np.ndarray:
    if abs(mode_shift - PERIODIC_SHIFT) < 1.0e-14:
        kappas = np.arange(n_sites, dtype=float)
        kappas[kappas > (n_sites - 1) // 2] -= n_sites
        return kappas

    kappas = np.arange(n_sites, dtype=float) + mode_shift
    kappas[kappas > n_sites / 2.0] -= n_sites
    return kappas


def shifted_fourier_matrix(n_sites: int, mode_shift: float) -> np.ndarray:
    modes = np.arange(n_sites, dtype=float) + mode_shift
    sites = np.arange(n_sites, dtype=float)
    return np.exp(2j * math.pi * np.outer(modes, sites) / n_sites) / math.sqrt(
        n_sites
    )


def fermionic_mode_frequencies(
    n_sites: int,
    mode_shift: float,
    *,
    lattice_spacing: float = 1.0,
) -> np.ndarray:
    kappas = centered_shifted_kappas(n_sites, mode_shift)
    return (2.0 / lattice_spacing) * np.sin(math.pi * np.abs(kappas) / n_sites)


def fermionic_transport_matrix(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    mode_shift: float,
    *,
    lattice_spacing: float = 1.0,
) -> np.ndarray:
    fourier = shifted_fourier_matrix(n_sites, mode_shift)
    kappas = centered_shifted_kappas(n_sites, mode_shift)
    omega = fermionic_mode_frequencies(
        n_sites,
        mode_shift,
        lattice_spacing=lattice_spacing,
    )
    phases = np.exp(-2j * math.pi * kappas * phi)
    return fourier.conj().T @ np.diag(np.exp(-omega * schwinger_T) * phases) @ fourier


def oscillator_basis(n_sites: int, mode_shift: float) -> np.ndarray:
    if abs(mode_shift - PERIODIC_SHIFT) < 1.0e-14:
        basis, _ = tc.real_zero_sum_basis(n_sites)
        return basis.astype(complex)
    return np.eye(n_sites, dtype=complex)


def fermionic_oscillator_transport(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    mode_shift: float,
    *,
    lattice_spacing: float = 1.0,
) -> np.ndarray:
    basis = oscillator_basis(n_sites, mode_shift)
    transport = fermionic_transport_matrix(
        n_sites,
        schwinger_T,
        phi,
        mode_shift,
        lattice_spacing=lattice_spacing,
    )
    return basis.conj().T @ transport @ basis


def fermionic_transport_eigenvalues(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    mode_shift: float,
    *,
    lattice_spacing: float = 1.0,
) -> list[complex]:
    kappas = centered_shifted_kappas(n_sites, mode_shift)
    omega = fermionic_mode_frequencies(
        n_sites,
        mode_shift,
        lattice_spacing=lattice_spacing,
    )
    eigenvalues = []
    for kappa, omega_k in zip(kappas, omega, strict=True):
        if abs(kappa) < 1.0e-14:
            continue
        eigenvalues.append(cmath.exp(-omega_k * schwinger_T - 2j * math.pi * kappa * phi))
    return eigenvalues


def fermionic_trace_factor_direct(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    mode_shift: float,
    time_trace_sign: int,
    *,
    lattice_spacing: float = 1.0,
) -> complex:
    if time_trace_sign not in (-1, 1):
        raise ValueError("time_trace_sign must be +/-1")
    transport = fermionic_oscillator_transport(
        n_sites,
        schwinger_T,
        phi,
        mode_shift,
        lattice_spacing=lattice_spacing,
    )
    identity = np.eye(transport.shape[0], dtype=complex)
    return complex(np.linalg.det(identity + time_trace_sign * transport))


def fermionic_trace_factor_closed(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    mode_shift: float,
    time_trace_sign: int,
    *,
    lattice_spacing: float = 1.0,
) -> complex:
    if time_trace_sign not in (-1, 1):
        raise ValueError("time_trace_sign must be +/-1")
    factor = 1.0 + 0.0j
    for eigenvalue in fermionic_transport_eigenvalues(
        n_sites,
        schwinger_T,
        phi,
        mode_shift,
        lattice_spacing=lattice_spacing,
    ):
        factor *= 1.0 + time_trace_sign * eigenvalue
    return factor


def fermionic_trace_factor_log_polar(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    mode_shift: float,
    time_trace_sign: int,
    *,
    lattice_spacing: float = 1.0,
) -> tuple[float, float]:
    if time_trace_sign not in (-1, 1):
        raise ValueError("time_trace_sign must be +/-1")
    logabs = 0.0
    phase = 0.0
    for eigenvalue in fermionic_transport_eigenvalues(
        n_sites,
        schwinger_T,
        phi,
        mode_shift,
        lattice_spacing=lattice_spacing,
    ):
        factor = 1.0 + time_trace_sign * eigenvalue
        logabs += math.log(abs(factor))
        phase += cmath.phase(factor)
    return (float(logabs), float(_wrap_phase(phase)))


def single_sector_report(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    mode_shift: float,
    time_trace_sign: int,
    *,
    lattice_spacing: float = 1.0,
) -> dict[str, Any]:
    direct = fermionic_trace_factor_direct(
        n_sites,
        schwinger_T,
        phi,
        mode_shift,
        time_trace_sign,
        lattice_spacing=lattice_spacing,
    )
    closed = fermionic_trace_factor_closed(
        n_sites,
        schwinger_T,
        phi,
        mode_shift,
        time_trace_sign,
        lattice_spacing=lattice_spacing,
    )
    logabs, phase = fermionic_trace_factor_log_polar(
        n_sites,
        schwinger_T,
        phi,
        mode_shift,
        time_trace_sign,
        lattice_spacing=lattice_spacing,
    )
    from_log = cmath.exp(logabs + 1j * phase)
    rel_error = abs(direct - closed) / max(1.0, abs(closed))
    log_polar_rel_error = abs(from_log - closed) / max(1.0, abs(closed))
    return {
        "label": sector_label(mode_shift, time_trace_sign),
        "N": n_sites,
        "T": schwinger_T,
        "phi": phi,
        "mode_shift": mode_shift,
        "time_trace_sign": time_trace_sign,
        "direct": direct,
        "closed": closed,
        "closed_rel_error": float(rel_error),
        "log_polar": {"logabs": logabs, "phase": phase},
        "log_polar_rel_error": float(log_polar_rel_error),
        "oscillator_dimension": int(
            fermionic_oscillator_transport(
                n_sites,
                schwinger_T,
                phi,
                mode_shift,
                lattice_spacing=lattice_spacing,
            ).shape[0]
        ),
        "min_frequency": float(
            np.min(
                [
                    omega
                    for kappa, omega in zip(
                        centered_shifted_kappas(n_sites, mode_shift),
                        fermionic_mode_frequencies(
                            n_sites,
                            mode_shift,
                            lattice_spacing=lattice_spacing,
                        ),
                        strict=True,
                    )
                    if abs(kappa) > 1.0e-14
                ]
            )
        ),
    }


def default_scan() -> dict[str, Any]:
    rows = []
    max_rel_error = 0.0
    max_log_polar_rel_error = 0.0
    max_periodic_compatibility_error = 0.0
    min_antiperiodic_frequency = math.inf

    for n_sites in DEFAULT_N_GRID:
        for schwinger_T in DEFAULT_T_GRID:
            for phi in DEFAULT_PHI_GRID:
                for mode_shift in (PERIODIC_SHIFT, ANTIPERIODIC_SHIFT):
                    for time_trace_sign in (1, -1):
                        row = single_sector_report(
                            n_sites,
                            schwinger_T,
                            phi,
                            mode_shift,
                            time_trace_sign,
                        )
                        max_rel_error = max(max_rel_error, row["closed_rel_error"])
                        max_log_polar_rel_error = max(
                            max_log_polar_rel_error, row["log_polar_rel_error"]
                        )
                        if abs(mode_shift - PERIODIC_SHIFT) < 1.0e-14:
                            direct_old = sci.fermionic_trace_factor_direct(
                                n_sites,
                                schwinger_T,
                                phi,
                                time_trace_sign,
                            )
                            closed_old = sci.fermionic_trace_factor_closed(
                                n_sites,
                                schwinger_T,
                                phi,
                                time_trace_sign,
                            )
                            compatibility_error = max(
                                abs(row["direct"] - direct_old),
                                abs(row["closed"] - closed_old),
                            ) / max(1.0, abs(closed_old))
                            max_periodic_compatibility_error = max(
                                max_periodic_compatibility_error, compatibility_error
                            )
                        else:
                            min_antiperiodic_frequency = min(
                                min_antiperiodic_frequency, row["min_frequency"]
                            )
                        rows.append(
                            {
                                "label": row["label"],
                                "N": n_sites,
                                "T": schwinger_T,
                                "phi": phi,
                                "closed_rel_error": row["closed_rel_error"],
                                "log_polar_rel_error": row["log_polar_rel_error"],
                                "oscillator_dimension": row["oscillator_dimension"],
                                "min_frequency": row["min_frequency"],
                            }
                        )

    return {
        "rows": rows,
        "max_rel_error": float(max_rel_error),
        "max_log_polar_rel_error": float(max_log_polar_rel_error),
        "max_periodic_compatibility_error": float(max_periodic_compatibility_error),
        "min_antiperiodic_frequency": float(min_antiperiodic_frequency),
        "pass": (
            max_rel_error < 1.0e-12
            and max_log_polar_rel_error < 1.0e-12
            and max_periodic_compatibility_error < 1.0e-12
            and min_antiperiodic_frequency > 0.0
        ),
    }


def print_summary(report: dict[str, Any]) -> None:
    print("=" * 96)
    print("FERMIONIC SPIN-STRUCTURE CYLINDER CHECK")
    print("=" * 96)
    print(
        f"max direct/closed rel error = {report['max_rel_error']:.3e}, "
        f"max log-polar rel error = {report['max_log_polar_rel_error']:.3e}, "
        f"periodic compatibility error = {report['max_periodic_compatibility_error']:.3e}"
    )
    print(
        "min antiperiodic oscillator frequency = "
        f"{report['min_antiperiodic_frequency']:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    report = default_scan()
    print_summary(report)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
