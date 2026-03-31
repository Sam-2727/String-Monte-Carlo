#!/usr/bin/env python3
"""
Single-cylinder oscillator trace prototype for the loop-side numerics.

This is not yet a full genus-one amplitude. It is the first explicit loop-side
integrand ingredient built from the now-tested twisted free propagator:

1. bosonic oscillator trace factor on one twisted cylinder,
2. fermionic coherent-state trace factors for a chosen spin-sign sector,
3. combined chiral trace data that can later be inserted into multi-edge sewing.

The emphasis here is reproducibility of the exact finite-N building blocks, not
yet physical GSO-summed normalization.
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
import twisted_cylinder_check as tcc


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


def fermionic_oscillator_transport(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    *,
    lattice_spacing: float = 1.0,
) -> np.ndarray:
    basis, _ = tc.real_zero_sum_basis(n_sites)
    transport = tcc.fermionic_transport_matrix(
        n_sites,
        schwinger_T,
        phi,
        lattice_spacing=lattice_spacing,
    )
    return basis.T @ transport @ basis


def fermionic_trace_factor_direct(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    spin_sign: int,
    *,
    lattice_spacing: float = 1.0,
) -> complex:
    if spin_sign not in {-1, 1}:
        raise ValueError("spin_sign must be ±1")
    transport = fermionic_oscillator_transport(
        n_sites,
        schwinger_T,
        phi,
        lattice_spacing=lattice_spacing,
    )
    identity = np.eye(transport.shape[0], dtype=complex)
    return complex(np.linalg.det(identity + spin_sign * transport))


def fermionic_trace_factor_closed(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    spin_sign: int,
    *,
    lattice_spacing: float = 1.0,
) -> complex:
    if spin_sign not in {-1, 1}:
        raise ValueError("spin_sign must be ±1")

    factor = 1.0 + 0.0j
    for mode in range(1, (n_sites - 1) // 2 + 1):
        omega = (2.0 / lattice_spacing) * math.sin(math.pi * mode / n_sites)
        lam = math.exp(-omega * schwinger_T)
        theta = 2.0 * math.pi * mode * phi
        factor *= 1.0 + 2.0 * spin_sign * lam * math.cos(theta) + lam * lam

    if n_sites % 2 == 0:
        nyquist = n_sites // 2
        omega = (2.0 / lattice_spacing) * math.sin(math.pi * nyquist / n_sites)
        factor *= 1.0 + spin_sign * cmath.exp(
            -omega * schwinger_T + 1j * math.pi * n_sites * phi
        )

    return factor


def cylinder_trace_data(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    *,
    lattice_spacing: float = 1.0,
    alpha_prime: float = 1.0,
    d_perp: int = 8,
    spinor_components: int = 8,
    left_spin_sign: int = 1,
    right_spin_sign: int = 1,
) -> dict[str, Any]:
    bosonic_one = tcc.bosonic_trace_factor_direct(
        n_sites,
        schwinger_T,
        phi,
        lattice_spacing=lattice_spacing,
        alpha_prime=alpha_prime,
    )
    bosonic_one_closed = tcc.bosonic_trace_factor_closed(
        n_sites,
        schwinger_T,
        phi,
        lattice_spacing=lattice_spacing,
    )
    fermion_left = fermionic_trace_factor_direct(
        n_sites,
        schwinger_T,
        phi,
        left_spin_sign,
        lattice_spacing=lattice_spacing,
    )
    fermion_right = fermionic_trace_factor_direct(
        n_sites,
        schwinger_T,
        phi,
        right_spin_sign,
        lattice_spacing=lattice_spacing,
    )
    fermion_left_closed = fermionic_trace_factor_closed(
        n_sites,
        schwinger_T,
        phi,
        left_spin_sign,
        lattice_spacing=lattice_spacing,
    )
    fermion_right_closed = fermionic_trace_factor_closed(
        n_sites,
        schwinger_T,
        phi,
        right_spin_sign,
        lattice_spacing=lattice_spacing,
    )

    bosonic_total = bosonic_one ** d_perp
    fermionic_total = (fermion_left ** spinor_components) * (
        fermion_right ** spinor_components
    )
    prototype_ratio = fermionic_total / bosonic_total

    return {
        "parameters": {
            "N": n_sites,
            "T": schwinger_T,
            "phi": phi,
            "lattice_spacing": lattice_spacing,
            "alpha_prime": alpha_prime,
            "d_perp": d_perp,
            "spinor_components": spinor_components,
            "left_spin_sign": left_spin_sign,
            "right_spin_sign": right_spin_sign,
        },
        "bosonic_one_coordinate": bosonic_one,
        "bosonic_one_coordinate_closed": bosonic_one_closed,
        "bosonic_one_coordinate_abs_error": float(abs(bosonic_one - bosonic_one_closed)),
        "bosonic_one_coordinate_rel_error": float(
            abs(bosonic_one - bosonic_one_closed) / max(1.0, abs(bosonic_one_closed))
        ),
        "fermion_left_component": fermion_left,
        "fermion_left_component_closed": fermion_left_closed,
        "fermion_left_component_abs_error": float(abs(fermion_left - fermion_left_closed)),
        "fermion_left_component_rel_error": float(
            abs(fermion_left - fermion_left_closed) / max(1.0, abs(fermion_left_closed))
        ),
        "fermion_right_component": fermion_right,
        "fermion_right_component_closed": fermion_right_closed,
        "fermion_right_component_abs_error": float(abs(fermion_right - fermion_right_closed)),
        "fermion_right_component_rel_error": float(
            abs(fermion_right - fermion_right_closed) / max(1.0, abs(fermion_right_closed))
        ),
        "bosonic_total": bosonic_total,
        "fermionic_total": fermionic_total,
        "prototype_ratio": prototype_ratio,
    }


def default_scan() -> dict[str, Any]:
    rows = []
    max_bosonic_rel_error = 0.0
    max_fermionic_rel_error = 0.0
    for n_sites in [5, 6, 7, 8, 10]:
        for schwinger_T in [0.05, 0.2, 0.7, 2.0]:
            for phi in [0.0, 0.17, 0.31, 0.5]:
                for left_spin, right_spin in [(1, 1), (1, -1), (-1, -1)]:
                    row = cylinder_trace_data(
                        n_sites,
                        schwinger_T,
                        phi,
                        left_spin_sign=left_spin,
                        right_spin_sign=right_spin,
                    )
                    max_bosonic_rel_error = max(
                        max_bosonic_rel_error, row["bosonic_one_coordinate_rel_error"]
                    )
                    max_fermionic_rel_error = max(
                        max_fermionic_rel_error,
                        row["fermion_left_component_rel_error"],
                        row["fermion_right_component_rel_error"],
                    )
                    rows.append(
                        {
                            "N": n_sites,
                            "T": schwinger_T,
                            "phi": phi,
                            "left_spin_sign": left_spin,
                            "right_spin_sign": right_spin,
                            "bosonic_abs_error": row["bosonic_one_coordinate_abs_error"],
                            "bosonic_rel_error": row["bosonic_one_coordinate_rel_error"],
                            "fermion_left_abs_error": row["fermion_left_component_abs_error"],
                            "fermion_left_rel_error": row["fermion_left_component_rel_error"],
                            "fermion_right_abs_error": row["fermion_right_component_abs_error"],
                            "fermion_right_rel_error": row["fermion_right_component_rel_error"],
                        }
                    )

    return {
        "rows": rows,
        "max_bosonic_rel_error": max_bosonic_rel_error,
        "max_fermionic_rel_error": max_fermionic_rel_error,
        "pass": max_bosonic_rel_error < 1.0e-12 and max_fermionic_rel_error < 1.0e-12,
    }


def print_report(report: dict[str, Any]) -> None:
    if "parameters" in report:
        print("=" * 96)
        print("SINGLE CYLINDER TRACE PROTOTYPE")
        print("=" * 96)
        print(
            f"N={report['parameters']['N']} "
            f"T={report['parameters']['T']:.6f} "
            f"phi={report['parameters']['phi']:.6f} "
            f"spins=({report['parameters']['left_spin_sign']},"
            f"{report['parameters']['right_spin_sign']})"
        )
        print(
            "bosonic one-coordinate: "
            f"direct={report['bosonic_one_coordinate']}, "
            f"closed={report['bosonic_one_coordinate_closed']}, "
            f"err={report['bosonic_one_coordinate_abs_error']:.3e}"
        )
        print(
            "fermion left/right component: "
            f"left err={report['fermion_left_component_abs_error']:.3e}, "
            f"right err={report['fermion_right_component_abs_error']:.3e}"
        )
        print(f"prototype ratio = {report['prototype_ratio']}")
        return

    print("=" * 96)
    print("SINGLE CYLINDER TRACE PROTOTYPE SCAN")
    print("=" * 96)
    print(
        f"max bosonic rel error={report['max_bosonic_rel_error']:.3e}, "
        f"max fermionic rel error={report['max_fermionic_rel_error']:.3e}, "
        f"pass={report['pass']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["single", "scan"], default="scan")
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--T", type=float, default=0.7)
    parser.add_argument("--phi", type=float, default=0.17)
    parser.add_argument("--left-spin-sign", type=int, default=1)
    parser.add_argument("--right-spin-sign", type=int, default=1)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "single":
        report = cylinder_trace_data(
            args.N,
            args.T,
            args.phi,
            left_spin_sign=args.left_spin_sign,
            right_spin_sign=args.right_spin_sign,
        )
    else:
        report = default_scan()

    print_report(report)

    if args.json_out is not None:
        output_path = Path(args.json_out)
        output_path.write_text(json.dumps(json_safe(report), indent=2) + "\n")
        print(f"\nWrote JSON report to {output_path}")


if __name__ == "__main__":
    main()
