#!/usr/bin/env python3
"""
Regression tests for the twisted free-cylinder building blocks.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import twisted_cylinder_check as tcc


@functools.lru_cache(maxsize=1)
def cached_report() -> dict[str, object]:
    return tcc.build_report()


def test_exact_shift_recovery() -> dict[str, object]:
    report = cached_report()["exact_shift"]
    max_error = max(report["max_shift_error"], report["max_cross_error"])
    return {
        "test": "exact_shift_recovery",
        "max_shift_error": report["max_shift_error"],
        "max_cross_error": report["max_cross_error"],
        "pass": max_error < 1.0e-12,
    }


def test_generic_twist_reality_pattern() -> dict[str, object]:
    report = cached_report()["generic_reality"]
    ok = (
        report["odd_max_shift_imag"] < 1.0e-12
        and report["odd_max_B_imag"] < 1.0e-12
        and report["even_min_shift_imag"] > 1.0e-6
        and report["even_min_B_imag"] > 1.0e-6
    )
    return {
        "test": "generic_twist_reality_pattern",
        "odd_max_shift_imag": report["odd_max_shift_imag"],
        "odd_max_B_imag": report["odd_max_B_imag"],
        "even_min_shift_imag": report["even_min_shift_imag"],
        "even_min_B_imag": report["even_min_B_imag"],
        "pass": ok,
    }


def test_oscillator_trace_positivity_and_closed_form() -> dict[str, object]:
    report = cached_report()["oscillator_trace"]
    return {
        "test": "oscillator_trace_positivity_and_closed_form",
        "max_logdet_error": report["max_logdet_error"],
        "max_trace_factor_rel_error": report["max_trace_factor_rel_error"],
        "min_real_eigenvalue": report["min_real_eigenvalue"],
        "pass": (
            report["max_logdet_error"] < 1.0e-12
            and report["max_trace_factor_rel_error"] < 1.0e-12
            and report["min_real_eigenvalue"] > 0.0
        ),
    }


def test_fermionic_transport_spectrum() -> dict[str, object]:
    report = cached_report()["oscillator_trace"]
    return {
        "test": "fermionic_transport_spectrum",
        "max_abs_error": report["max_fermion_transport_error"],
        "pass": report["max_fermion_transport_error"] < 1.0e-12,
    }


def run_all_tests() -> dict[str, object]:
    results = {
        "exact_shift_recovery": test_exact_shift_recovery(),
        "generic_twist_reality_pattern": test_generic_twist_reality_pattern(),
        "oscillator_trace_positivity_and_closed_form": test_oscillator_trace_positivity_and_closed_form(),
        "fermionic_transport_spectrum": test_fermionic_transport_spectrum(),
    }

    print("=" * 96)
    print("TWISTED CYLINDER TESTS")
    print("=" * 96)
    for name, result in results.items():
        status = "PASS" if result.get("pass") else "FAIL"
        scalar_parts = []
        for key, value in result.items():
            if key in {"test", "pass"}:
                continue
            if isinstance(value, float):
                scalar_parts.append(f"{key}={value:.3e}")
            else:
                scalar_parts.append(f"{key}={value}")
        print(f"{name:44s} [{status}] " + ", ".join(scalar_parts))

    passed = sum(1 for result in results.values() if result.get("pass"))
    total = len(results)
    print()
    print(f"Summary: {passed}/{total} passed")
    return results


if __name__ == "__main__":
    run_all_tests()
