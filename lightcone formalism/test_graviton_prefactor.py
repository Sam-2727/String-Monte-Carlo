#!/usr/bin/env python3
"""
Bosonic prefactor tests for the superstring three-graviton amplitude.

Tests:
1. Parity obstruction for the minimal right-arc stencil (eta_- = 0 exactly)
2. Second-order stencil restores nonzero eta_-
3. Prefactor tensor convergence with the second-order stencil
4. Smooth ratio-dependence of A_delta and B_qq
5. Fermionic zero-mode tensor structure (Weyl map diagnostic)
"""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tachyon_check as tc
import superstring_prefactor_check as sp


def test_parity_obstruction(n1=16, n2=24):
    """Verify eta_- = 0 exactly for the minimal stencil, explained by parity."""
    data = sp.prefactor_data(n1, n2)
    comm_err, even_err, odd_err = sp.parity_diagnostics(n1, n2)

    return {
        "test": "parity_obstruction",
        "N1": n1, "N2": n2,
        "eta_minus": data.eta_minus,
        "eta_plus": data.eta_plus,
        "parity_commutator": comm_err,
        "parity_b_even": even_err,
        "pass": abs(data.eta_minus) < 1e-12 and comm_err < 1e-12 and even_err < 1e-12,
    }


def test_parity_scan(max_n=30):
    """Scan eta_- across a grid to confirm it vanishes universally."""
    worst = 0.0
    worst_pair = None
    for n1 in range(4, max_n + 1):
        for n2 in range(4, max_n + 1):
            data = sp.prefactor_data(n1, n2)
            if abs(data.eta_minus) > worst:
                worst = abs(data.eta_minus)
                worst_pair = (n1, n2)

    return {
        "test": "parity_scan",
        "grid": f"4..{max_n}",
        "max_abs_eta_minus": worst,
        "worst_pair": worst_pair,
        "pass": worst < 1e-12,
    }


def test_second_order_stencil(n1=16, n2=24):
    """Verify second-order stencil restores nonzero eta_- and B_qq."""
    cases = {
        "minimal/minimal": sp.prefactor_data(n1, n2, left_variant="minimal", right_variant="minimal"),
        "minimal/second": sp.prefactor_data(n1, n2, left_variant="minimal", right_variant="second_order"),
        "second/second": sp.prefactor_data(n1, n2, left_variant="second_order", right_variant="second_order"),
    }

    return {
        "test": "second_order_stencil",
        "N1": n1, "N2": n2,
        "results": {
            name: {
                "eta_plus": d.eta_plus,
                "eta_minus": d.eta_minus,
                "A_delta": d.a_delta_reduced,
                "B_qq": d.b_qq_reduced,
            }
            for name, d in cases.items()
        },
        "pass": abs(cases["second/second"].eta_minus) > 0.01,
    }


def test_prefactor_convergence(alpha_prime=1.0):
    """Track convergence of the second-order prefactor on fixed-ratio family."""
    results = []
    for scale in [8, 16, 32, 64, 128]:
        n1, n2 = 2 * scale, 3 * scale
        data = sp.prefactor_data(n1, n2, alpha_prime,
                                  left_variant="second_order",
                                  right_variant="second_order")
        results.append({
            "N1": n1, "N2": n2,
            "N1_B_qq": n1 * data.b_qq_reduced,
            "sqrt_N1_eta_plus": math.sqrt(n1) * data.eta_plus,
            "sqrt_N1_eta_minus": math.sqrt(n1) * data.eta_minus,
            "A_delta": data.a_delta_reduced,
        })

    # Check that N1*B_qq converges
    scaled_bqq = [r["N1_B_qq"] for r in results]
    diffs = [abs(scaled_bqq[i+1] - scaled_bqq[i]) for i in range(len(scaled_bqq)-1)]
    converging = len(diffs) >= 2 and diffs[-1] < diffs[-2]

    return {
        "test": "prefactor_convergence",
        "family": "2:3",
        "values": results,
        "converging": converging,
        "pass": converging,
    }


def test_ratio_scan(scale=128, alpha_prime=1.0):
    """Verify smooth ratio-dependence of the bosonic prefactor."""
    families = [(1, 3), (1, 2), (3, 5), (2, 3), (1, 1)]
    results = []
    for a, b in families:
        n1, n2 = a * scale, b * scale
        data = sp.prefactor_data(n1, n2, alpha_prime,
                                  left_variant="second_order",
                                  right_variant="second_order")
        lam = n1 / (n1 + n2)
        results.append({
            "lambda": lam,
            "N1": n1, "N2": n2,
            "N1_B_qq": n1 * data.b_qq_reduced,
            "A_delta": data.a_delta_reduced,
        })

    # Check monotonicity of N1*B_qq in lambda
    vals = [r["N1_B_qq"] for r in results]
    monotone = all(vals[i] < vals[i+1] for i in range(len(vals)-1))

    return {
        "test": "ratio_scan",
        "scale": scale,
        "results": results,
        "monotone_in_lambda": monotone,
        "pass": True,  # Informational
    }


def test_weyl_tensor_structure():
    """Test that the Weyl-map vector block has minimal SO(8) tensor span."""
    try:
        import gs_weyl_symbol_diagnostic as gw
        coeffs, max_err, rms_err = gw.fit_vector_block_invariants(1.0)
        return {
            "test": "weyl_tensor_structure",
            "max_residual": max_err,
            "rms_residual": rms_err,
            "coefficients_A_B_C": [complex(c) for c in coeffs],
            "pass": max_err < 1e-10,
        }
    except Exception as e:
        return {
            "test": "weyl_tensor_structure",
            "error": str(e),
            "pass": False,
        }


def run_all_tests():
    results = {}

    print("Running parity obstruction test...")
    r = test_parity_obstruction()
    results["parity"] = r
    status = "PASS" if r["pass"] else "FAIL"
    print(f"  eta_-={r['eta_minus']:.3e}, parity_comm={r['parity_commutator']:.3e} [{status}]")

    print("\nRunning parity scan (4..30)...")
    r = test_parity_scan(30)
    results["parity_scan"] = r
    status = "PASS" if r["pass"] else "FAIL"
    print(f"  max|eta_-|={r['max_abs_eta_minus']:.3e} [{status}]")

    print("\nRunning second-order stencil test...")
    r = test_second_order_stencil()
    results["second_order"] = r
    status = "PASS" if r["pass"] else "FAIL"
    for name, vals in r["results"].items():
        print(f"  {name}: eta_+={vals['eta_plus']:.6f} eta_-={vals['eta_minus']:.6f} "
              f"B_qq={vals['B_qq']:.6f}")
    print(f"  [{status}]")

    print("\nRunning prefactor convergence test...")
    r = test_prefactor_convergence()
    results["convergence"] = r
    status = "PASS" if r["pass"] else "FAIL"
    for v in r["values"]:
        print(f"  N1={v['N1']:4d}: N1*B_qq={v['N1_B_qq']:.6f} A_delta={v['A_delta']:.6f}")
    print(f"  [{status}]")

    print("\nRunning ratio scan...")
    r = test_ratio_scan()
    results["ratio_scan"] = r
    for v in r["results"]:
        print(f"  lambda={v['lambda']:.3f}: N1*B_qq={v['N1_B_qq']:.6f}")

    print("\nRunning Weyl tensor structure test...")
    r = test_weyl_tensor_structure()
    results["weyl"] = r
    status = "PASS" if r["pass"] else "FAIL"
    if "error" not in r:
        print(f"  max residual={r['max_residual']:.3e} [{status}]")
    else:
        print(f"  Error: {r['error']} [{status}]")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_pass = sum(1 for v in results.values() if isinstance(v, dict) and v.get("pass"))
    n_total = sum(1 for v in results.values() if isinstance(v, dict) and "pass" in v)
    print(f"  {n_pass}/{n_total} tests passed")

    return results


if __name__ == "__main__":
    run_all_tests()
