#!/usr/bin/env python3
"""
Stable pre-GSO Bose-Fermi oscillator-ratio diagnostics on a single cylinder.

The raw `prototype_ratio` in `single_cylinder_integrand.py` is useful at small
N, but it quickly underflows or overflows on a broader scan because the bosonic
and fermionic trace factors are each raised to powers. This helper works in
log-polar form instead:

    R_sector = (Z_F,left)^8 (Z_F,right)^8 / (Z_B)^8

for a chosen pair of spin-sign sectors. The resulting log-magnitude and phase
are finite even where the raw complex ratio is not.

This is intentionally a pre-GSO diagnostic. The point is not to claim
supersymmetric cancellation yet, but to quantify how far each unsummed sector is
from the target ratio 1 and thereby make the need for the spin-structure/GSO
sum explicit.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import single_cylinder_integrand as sci


DEFAULT_N_GRID = (8, 16, 32, 64, 128)
DEFAULT_T_GRID = (0.2, 0.7, 2.0)
DEFAULT_PHI_GRID = (0.0, 0.17, 0.31, 0.5)
DEFAULT_SECTORS = ((1, 1), (1, -1), (-1, -1))


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


def _complex_log_polar(value: complex) -> tuple[float, float]:
    return (math.log(abs(value)), cmath.phase(value))


def bosonic_one_coordinate_log_polar_closed(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    *,
    lattice_spacing: float = 1.0,
) -> tuple[float, float]:
    logabs = 0.0
    phase = 0.0
    for mode in range(1, (n_sites - 1) // 2 + 1):
        omega = (2.0 / lattice_spacing) * math.sin(math.pi * mode / n_sites)
        theta = 2.0 * math.pi * mode * phi
        denom = 2.0 * (math.cosh(omega * schwinger_T) - math.cos(theta))
        logabs -= math.log(abs(denom))
    if n_sites % 2 == 0:
        nyquist = n_sites // 2
        omega = (2.0 / lattice_spacing) * math.sin(math.pi * nyquist / n_sites)
        theta = math.pi * n_sites * phi
        denom = 2.0 * (math.cosh(omega * schwinger_T) - cmath.exp(1j * theta))
        logabs -= 0.5 * math.log(abs(denom))
        phase -= 0.5 * cmath.phase(denom)
    return (logabs, _wrap_phase(phase))


def fermionic_one_component_log_polar_closed(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    spin_sign: int,
    *,
    lattice_spacing: float = 1.0,
) -> tuple[float, float]:
    if spin_sign not in (-1, 1):
        raise ValueError("spin_sign must be ±1")
    logabs = 0.0
    phase = 0.0
    for mode in range(1, (n_sites - 1) // 2 + 1):
        omega = (2.0 / lattice_spacing) * math.sin(math.pi * mode / n_sites)
        lam = math.exp(-omega * schwinger_T)
        theta = 2.0 * math.pi * mode * phi
        factor = 1.0 + 2.0 * spin_sign * lam * math.cos(theta) + lam * lam
        logabs += math.log(abs(factor))
    if n_sites % 2 == 0:
        nyquist = n_sites // 2
        omega = (2.0 / lattice_spacing) * math.sin(math.pi * nyquist / n_sites)
        factor = 1.0 + spin_sign * cmath.exp(
            -omega * schwinger_T + 1j * math.pi * n_sites * phi
        )
        logabs += math.log(abs(factor))
        phase += cmath.phase(factor)
    return (logabs, _wrap_phase(phase))


def sector_ratio_log_polar(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    left_spin_sign: int,
    right_spin_sign: int,
    *,
    lattice_spacing: float = 1.0,
    d_perp: int = 8,
    spinor_components: int = 8,
) -> dict[str, Any]:
    bos_logabs, bos_phase = bosonic_one_coordinate_log_polar_closed(
        n_sites,
        schwinger_T,
        phi,
        lattice_spacing=lattice_spacing,
    )
    left_logabs, left_phase = fermionic_one_component_log_polar_closed(
        n_sites,
        schwinger_T,
        phi,
        left_spin_sign,
        lattice_spacing=lattice_spacing,
    )
    right_logabs, right_phase = fermionic_one_component_log_polar_closed(
        n_sites,
        schwinger_T,
        phi,
        right_spin_sign,
        lattice_spacing=lattice_spacing,
    )

    ratio_logabs = spinor_components * (left_logabs + right_logabs) - d_perp * bos_logabs
    ratio_phase = _wrap_phase(
        spinor_components * (left_phase + right_phase) - d_perp * bos_phase
    )
    distance_to_one = float(math.hypot(ratio_logabs, ratio_phase))
    ratio_complex = None
    if abs(ratio_logabs) < 700.0:
        ratio_complex = cmath.exp(ratio_logabs + 1j * ratio_phase)

    return {
        "N": n_sites,
        "T": schwinger_T,
        "phi": phi,
        "left_spin_sign": left_spin_sign,
        "right_spin_sign": right_spin_sign,
        "log_abs_ratio": float(ratio_logabs),
        "phase_ratio": float(ratio_phase),
        "distance_to_one": distance_to_one,
        "ratio_complex_if_safe": ratio_complex,
    }


def default_scan() -> dict[str, Any]:
    rows = []
    closest = None
    largest = None
    for n_sites in DEFAULT_N_GRID:
        for schwinger_T in DEFAULT_T_GRID:
            for phi in DEFAULT_PHI_GRID:
                for left_spin_sign, right_spin_sign in DEFAULT_SECTORS:
                    row = sector_ratio_log_polar(
                        n_sites,
                        schwinger_T,
                        phi,
                        left_spin_sign,
                        right_spin_sign,
                    )
                    rows.append(row)
                    if closest is None or row["distance_to_one"] < closest["distance_to_one"]:
                        closest = row
                    if largest is None or abs(row["log_abs_ratio"]) > abs(largest["log_abs_ratio"]):
                        largest = row
    return {
        "rows": rows,
        "closest_to_cancellation": closest,
        "largest_log_magnitude": largest,
    }


def print_summary(summary: dict[str, Any]) -> None:
    closest = summary["closest_to_cancellation"]
    largest = summary["largest_log_magnitude"]
    print("=" * 96)
    print("PRE-GSO BOSE-FERMI CANCELLATION SCAN")
    print("=" * 96)
    print("Sampled pre-GSO sectors do not approach ratio 1 on the default grid.")
    print(
        "Closest sampled sector to unit ratio:"
        f" N={closest['N']}, T={closest['T']}, phi={closest['phi']},"
        f" spins=({closest['left_spin_sign']},{closest['right_spin_sign']}),"
        f" log|R|={closest['log_abs_ratio']:.6f},"
        f" phase={closest['phase_ratio']:.6f},"
        f" distance={closest['distance_to_one']:.6f}"
    )
    print(
        "Largest sampled log-magnitude:"
        f" N={largest['N']}, T={largest['T']}, phi={largest['phi']},"
        f" spins=({largest['left_spin_sign']},{largest['right_spin_sign']}),"
        f" log|R|={largest['log_abs_ratio']:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="optional path for a JSON dump",
    )
    args = parser.parse_args()
    summary = default_scan()
    print_summary(summary)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(json_safe(summary), indent=2))
        print()
        print(f"Wrote JSON summary to {args.json_out}")


if __name__ == "__main__":
    main()
