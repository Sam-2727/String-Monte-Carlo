#!/usr/bin/env python3
"""
Oscillator-sector GSO sign scan built from explicit spatial spin structures.

This is still not the final physical type-II one-loop integrand. The missing
pieces are the full zero-mode factors and any remaining normalization/phases.
What this module does is narrower and useful:

1. build the four chiral oscillator sectors (`NS`, `NS_tilde`, `R`, `R_tilde`)
   from explicit finite-N spin-structure traces,
2. form candidate chiral sums with configurable +/- coefficients,
3. compare the resulting closed-string oscillator sum against the bosonic
   one-cylinder factor `Z_B^8`,
4. check whether a sign choice alone can drive the expected cancellation.

Numerically, the answer is no on the sampled grid: once the true spatial
spin-structure sectors are included, a pure sign choice is still far from
enough. That means the remaining loop gap is not bookkeeping alone.
"""

from __future__ import annotations

import argparse
import cmath
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import fermionic_spin_structure_cylinder as fssc
import twisted_cylinder_check as tcc


DEFAULT_N_GRID = (8, 16, 32, 64)
DEFAULT_T_GRID = (0.2, 0.7, 2.0)
DEFAULT_PHI_GRID = (0.0, 0.17, 0.31, 0.5)
DEFAULT_STANDARD_PATTERN = {
    "NS": 1,
    "NS_tilde": -1,
    "R": -1,
    "R_tilde": 1,
}


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


def sector_order() -> tuple[str, ...]:
    return ("NS", "NS_tilde", "R", "R_tilde")


def pattern_label(coeffs: dict[str, int]) -> str:
    return ",".join(f"{name}:{coeffs[name]:+d}" for name in sector_order())


def all_chiral_patterns() -> list[dict[str, int]]:
    patterns = []
    for ns_tilde, r_sector, r_tilde in itertools.product((1, -1), repeat=3):
        pattern = {
            "NS": 1,
            "NS_tilde": ns_tilde,
            "R": r_sector,
            "R_tilde": r_tilde,
        }
        patterns.append(pattern)
    return patterns


def sector_component_log_polar(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    label: str,
) -> tuple[float, float]:
    if label == "NS":
        return fssc.fermionic_trace_factor_log_polar(
            n_sites, schwinger_T, phi, fssc.ANTIPERIODIC_SHIFT, 1
        )
    if label == "NS_tilde":
        return fssc.fermionic_trace_factor_log_polar(
            n_sites, schwinger_T, phi, fssc.ANTIPERIODIC_SHIFT, -1
        )
    if label == "R":
        return fssc.fermionic_trace_factor_log_polar(
            n_sites, schwinger_T, phi, fssc.PERIODIC_SHIFT, 1
        )
    if label == "R_tilde":
        return fssc.fermionic_trace_factor_log_polar(
            n_sites, schwinger_T, phi, fssc.PERIODIC_SHIFT, -1
        )
    raise ValueError(f"unknown sector label {label}")


def sector_component_direct(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    label: str,
) -> complex:
    if label == "NS":
        return fssc.fermionic_trace_factor_direct(
            n_sites, schwinger_T, phi, fssc.ANTIPERIODIC_SHIFT, 1
        )
    if label == "NS_tilde":
        return fssc.fermionic_trace_factor_direct(
            n_sites, schwinger_T, phi, fssc.ANTIPERIODIC_SHIFT, -1
        )
    if label == "R":
        return fssc.fermionic_trace_factor_direct(
            n_sites, schwinger_T, phi, fssc.PERIODIC_SHIFT, 1
        )
    if label == "R_tilde":
        return fssc.fermionic_trace_factor_direct(
            n_sites, schwinger_T, phi, fssc.PERIODIC_SHIFT, -1
        )
    raise ValueError(f"unknown sector label {label}")


def _stable_complex_sum(terms: list[tuple[int, float, float]]) -> tuple[float, float]:
    nonzero_terms = [term for term in terms if term[0] != 0]
    if not nonzero_terms:
        return (-math.inf, 0.0)
    max_logabs = max(logabs for _, logabs, _ in nonzero_terms)
    scaled = 0.0 + 0.0j
    for coeff, logabs, phase in nonzero_terms:
        scaled += coeff * cmath.exp((logabs - max_logabs) + 1j * phase)
    if abs(scaled) < 1.0e-300:
        return (-math.inf, 0.0)
    return (
        float(max_logabs + math.log(abs(scaled))),
        float(_wrap_phase(cmath.phase(scaled))),
    )


def chiral_sum_log_polar(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    coeffs: dict[str, int],
    *,
    spinor_components: int = 8,
) -> tuple[float, float]:
    terms = []
    for label in sector_order():
        logabs_component, phase_component = sector_component_log_polar(
            n_sites,
            schwinger_T,
            phi,
            label,
        )
        terms.append(
            (
                coeffs[label],
                spinor_components * logabs_component,
                _wrap_phase(spinor_components * phase_component),
            )
        )
    return _stable_complex_sum(terms)


def chiral_sum_direct(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    coeffs: dict[str, int],
    *,
    spinor_components: int = 8,
) -> complex:
    total = 0.0 + 0.0j
    for label in sector_order():
        total += coeffs[label] * (
            sector_component_direct(n_sites, schwinger_T, phi, label) ** spinor_components
        )
    return total


def bosonic_total_log_polar(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    *,
    d_perp: int = 8,
) -> tuple[float, float]:
    one_coordinate = tcc.bosonic_trace_factor_closed(n_sites, schwinger_T, phi)
    return (
        float(d_perp * math.log(abs(one_coordinate))),
        float(_wrap_phase(d_perp * cmath.phase(one_coordinate))),
    )


def pattern_distance_report(
    n_sites: int,
    schwinger_T: float,
    phi: float,
    coeffs: dict[str, int],
    *,
    spinor_components: int = 8,
    d_perp: int = 8,
) -> dict[str, Any]:
    chiral_logabs, chiral_phase = chiral_sum_log_polar(
        n_sites,
        schwinger_T,
        phi,
        coeffs,
        spinor_components=spinor_components,
    )
    bos_logabs, bos_phase = bosonic_total_log_polar(
        n_sites,
        schwinger_T,
        phi,
        d_perp=d_perp,
    )

    if math.isinf(chiral_logabs) and chiral_logabs < 0.0:
        ratio_logabs = -math.inf
        ratio_phase = 0.0
        distance = math.inf
        ratio_if_safe = 0.0 + 0.0j
    else:
        ratio_logabs = 2.0 * chiral_logabs - bos_logabs
        ratio_phase = _wrap_phase(2.0 * chiral_phase - bos_phase)
        distance = float(math.hypot(ratio_logabs, ratio_phase))
        ratio_if_safe = None
        if abs(ratio_logabs) < 700.0:
            ratio_if_safe = cmath.exp(ratio_logabs + 1j * ratio_phase)

    return {
        "N": n_sites,
        "T": schwinger_T,
        "phi": phi,
        "coeffs": coeffs,
        "label": pattern_label(coeffs),
        "ratio_logabs": float(ratio_logabs),
        "ratio_phase": float(ratio_phase),
        "distance_to_one": distance,
        "ratio_if_safe": ratio_if_safe,
    }


def default_scan() -> dict[str, Any]:
    patterns = all_chiral_patterns()
    rows = []
    best_by_max = None
    standard_summary = None

    for coeffs in patterns:
        sample_rows = []
        max_distance = 0.0
        min_distance = math.inf
        closest = None
        for n_sites in DEFAULT_N_GRID:
            for schwinger_T in DEFAULT_T_GRID:
                for phi in DEFAULT_PHI_GRID:
                    row = pattern_distance_report(n_sites, schwinger_T, phi, coeffs)
                    sample_rows.append(row)
                    max_distance = max(max_distance, row["distance_to_one"])
                    if row["distance_to_one"] < min_distance:
                        min_distance = row["distance_to_one"]
                        closest = row
        summary = {
            "coeffs": coeffs,
            "label": pattern_label(coeffs),
            "max_distance_to_one": float(max_distance),
            "closest_distance_to_one": float(min_distance),
            "closest_sample": closest,
        }
        rows.append(summary)
        if best_by_max is None or summary["max_distance_to_one"] < best_by_max["max_distance_to_one"]:
            best_by_max = summary
        if coeffs == DEFAULT_STANDARD_PATTERN:
            standard_summary = summary

    if standard_summary is None:
        raise RuntimeError("failed to locate the default standard pattern in the scan")

    return {
        "pattern_summaries": rows,
        "best_pattern_by_max_distance": best_by_max,
        "standard_pattern": standard_summary,
    }


def direct_sample_check() -> dict[str, Any]:
    coeffs = DEFAULT_STANDARD_PATTERN
    n_sites = 8
    schwinger_T = 0.7
    phi = 0.31
    direct = chiral_sum_direct(n_sites, schwinger_T, phi, coeffs)
    logabs, phase = chiral_sum_log_polar(n_sites, schwinger_T, phi, coeffs)
    reconstructed = cmath.exp(logabs + 1j * phase)
    abs_error = abs(direct - reconstructed)
    rel_error = abs_error / max(1.0, abs(direct))
    return {
        "coeffs": coeffs,
        "N": n_sites,
        "T": schwinger_T,
        "phi": phi,
        "direct": direct,
        "reconstructed": reconstructed,
        "abs_error": float(abs_error),
        "rel_error": float(rel_error),
    }


def print_summary(report: dict[str, Any]) -> None:
    best = report["best_pattern_by_max_distance"]
    standard = report["standard_pattern"]
    print("=" * 96)
    print("OSCILLATOR SPIN-STRUCTURE GSO SIGN SCAN")
    print("=" * 96)
    print(
        "Best sign pattern by worst-case distance to unit boson/fermion ratio: "
        f"{best['label']}"
    )
    print(
        f"  max distance = {best['max_distance_to_one']:.6f}, "
        f"closest sampled distance = {best['closest_distance_to_one']:.6f}"
    )
    print(
        "Standard pattern "
        f"{standard['label']}: max distance = {standard['max_distance_to_one']:.6f}, "
        f"closest sampled distance = {standard['closest_distance_to_one']:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    report = default_scan()
    report["direct_sample_check"] = direct_sample_check()
    print_summary(report)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
