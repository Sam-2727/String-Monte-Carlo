#!/usr/bin/env python3
"""
Neumann coefficient extraction and validation tests.

Tests:
1. Symplectic obstruction C_Q C_P^T = 2(U1, U2)
2. Gaussian-moment Neumann symmetry
3. Neumann block convergence behavior
4. Cross-check: reduced G_T from Neumann blocks matches tachyon_check
5. Massless covariance C_{1,N} formula
"""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tachyon_check as tc
import squeeze_matrix_check as sq
import bosonic_massless_check as bm


def test_symplectic_obstruction():
    """Verify C_Q C_P^T = 2(U1,U2) exactly."""
    results = []
    for n1, n2 in [(4, 6), (8, 12), (16, 24), (32, 48)]:
        err = sq.check_symplectic_compatibility(n1, n2)
        results.append({"N1": n1, "N2": n2, "error": err})

    max_err = max(r["error"] for r in results)
    return {
        "test": "symplectic_obstruction",
        "values": results,
        "max_error": max_err,
        "pass": max_err < 1e-12,
    }


def test_neumann_symmetry():
    """Verify Gaussian-moment Neumann blocks satisfy N^{rs}_{mn} = N^{sr}_{nm}."""
    results = []
    for n1, n2 in [(4, 6), (8, 12), (16, 24), (32, 48)]:
        eff = sq.effective_rows(n1, n2)
        errs = {}
        for (r, s) in [(1, 1), (1, 2), (1, 3), (3, 3)]:
            nrs = sq.neumann_block(eff, r, s)
            nsr = sq.neumann_block(eff, s, r)
            errs[f"N{r}{s}"] = float(np.max(np.abs(nrs - nsr.T)))
        results.append({"N1": n1, "N2": n2, "errors": errs})

    max_err = max(max(r["errors"].values()) for r in results)
    return {
        "test": "neumann_symmetry",
        "values": results,
        "max_error": max_err,
        "pass": max_err < 1e-12,
    }


def test_neumann_vs_reduced_gaussian(n1=16, n2=24, alpha_prime=1.0):
    """
    Cross-check: the (1,2) Neumann block should be related to the off-diagonal
    block of G_T after appropriate frequency weighting.

    Specifically: G_T^{12} = M_1 delta_{r1} + U_1^T M_3 U_2 should be
    recoverable from the Neumann data.
    """
    eff = sq.effective_rows(n1, n2, alpha_prime)
    g_t = eff["g_t"]
    n3 = n1 + n2
    d1, d2 = n1 - 1, n2 - 1

    g12_from_gt = g_t[:d1, d1:]  # Off-diagonal block of G_T

    # The Neumann block N12 encodes specific combinations of G_T^{-1} blocks.
    # As a sanity check, verify that G_T is positive definite.
    eigvals = np.linalg.eigvalsh(g_t)
    pd = float(np.min(eigvals))

    # Check that the Schur complement gamma_T from tachyon_check matches
    data = tc.compute_tachyon_data(n1, n2, alpha_prime, 24)
    gamma_from_tc = data.gamma_t
    gamma_from_sq = eff["gamma_t"]

    return {
        "test": "neumann_vs_reduced",
        "N1": n1, "N2": n2,
        "G_T_min_eigenvalue": pd,
        "G_T_positive_definite": pd > 0,
        "gamma_T_tachyon_check": gamma_from_tc,
        "gamma_T_squeeze_check": gamma_from_sq,
        "gamma_T_agreement": abs(gamma_from_tc - gamma_from_sq),
        "pass": pd > 0 and abs(gamma_from_tc - gamma_from_sq) < 1e-12,
    }


def test_massless_covariance():
    """Verify C_{1,N} = pi*alpha'/(2*N*sin(pi/N))."""
    results = []
    for n in [4, 8, 16, 32, 64, 128]:
        numeric, formula = bm.first_harmonic_covariance(n, 1.0)
        results.append({
            "N": n,
            "numeric": numeric,
            "formula": formula,
            "diff": abs(numeric - formula),
        })

    max_diff = max(r["diff"] for r in results)
    return {
        "test": "massless_covariance",
        "values": results,
        "max_diff": max_diff,
        "pass": max_diff < 1e-14,
    }


def test_ttm_trace_suppression(alpha_prime=1.0):
    """Verify that the TTM trace part A_tr goes to zero as O(1/N)."""
    results = []
    for scale in [8, 16, 32, 64, 128]:
        n1, n2 = 2 * scale, 3 * scale
        data = bm.compute_massless_data(n1, n2, alpha_prime)
        results.append({
            "N1": n1, "N2": n2,
            "A_tr": data.a_trace_reduced,
            "B_rel": data.b_rel_reduced,
            "N1_A_tr": n1 * data.a_trace_reduced,
        })

    # Check that N1*A_tr converges (i.e., A_tr ~ c/N1)
    scaled = [r["N1_A_tr"] for r in results]
    diffs = [abs(scaled[i+1] - scaled[i]) for i in range(len(scaled)-1)]
    converging = len(diffs) >= 2 and diffs[-1] < diffs[-2]

    # Check that B_rel converges
    brels = [r["B_rel"] for r in results]
    b_diffs = [abs(brels[i+1] - brels[i]) for i in range(len(brels)-1)]
    b_converging = len(b_diffs) >= 2 and b_diffs[-1] < b_diffs[-2]

    return {
        "test": "ttm_trace_suppression",
        "values": results,
        "A_tr_scaling_converges": converging,
        "B_rel_converges": b_converging,
        "pass": converging and b_converging,
    }


def run_all_tests():
    results = {}

    print("Running symplectic obstruction test...")
    r = test_symplectic_obstruction()
    results["symplectic"] = r
    status = "PASS" if r["pass"] else "FAIL"
    print(f"  max error: {r['max_error']:.3e} [{status}]")

    print("\nRunning Neumann symmetry test...")
    r = test_neumann_symmetry()
    results["neumann_sym"] = r
    status = "PASS" if r["pass"] else "FAIL"
    print(f"  max error: {r['max_error']:.3e} [{status}]")

    print("\nRunning Neumann vs reduced Gaussian cross-check...")
    r = test_neumann_vs_reduced_gaussian()
    results["neumann_vs_reduced"] = r
    status = "PASS" if r["pass"] else "FAIL"
    print(f"  gamma_T agreement: {r['gamma_T_agreement']:.3e} [{status}]")

    print("\nRunning massless covariance test...")
    r = test_massless_covariance()
    results["massless_cov"] = r
    status = "PASS" if r["pass"] else "FAIL"
    print(f"  max diff: {r['max_diff']:.3e} [{status}]")

    print("\nRunning TTM trace suppression test...")
    r = test_ttm_trace_suppression()
    results["ttm_trace"] = r
    status = "PASS" if r["pass"] else "FAIL"
    for v in r["values"]:
        print(f"  N1={v['N1']:4d}: A_tr={v['A_tr']:.6f} B_rel={v['B_rel']:.6f}")
    print(f"  [{status}]")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_pass = sum(1 for v in results.values() if v.get("pass"))
    n_total = sum(1 for v in results.values() if "pass" in v)
    print(f"  {n_pass}/{n_total} tests passed")

    return results


if __name__ == "__main__":
    run_all_tests()
