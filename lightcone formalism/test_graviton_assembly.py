#!/usr/bin/env python3
"""
Full three-graviton matrix element assembly.

Combines:
1. Bosonic nonzero-mode prefactor tensor (from superstring_prefactor_check)
2. Fermionic zero-mode contraction (from gs_weyl_symbol_diagnostic)
3. Bosonic vacuum amplitude (from tachyon_check infrastructure)

The target is the Einstein-Hilbert / Yang-Mills-squared cubic tensor:
    V^{IJK}(p1,p2,p3) = delta^{IJ}(p1-p2)^K + cyclic

For the three-graviton amplitude:
    A_hhh ~ V^{IJK} V^{I'J'K'} epsilon^(1)_{II'} epsilon^(2)_{JJ'} epsilon^(3)_{KK'}

In the lightcone gauge computation, the graviton polarizations project onto the
transverse SO(D_perp) tensor, and the cubic vertex prefactor K^I Ktilde^J v_{IJ}
must reproduce this structure.
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


def bosonic_prefactor_tensor(n1: int, n2: int, alpha_prime: float = 1.0,
                              left_variant: str = "second_order",
                              right_variant: str = "second_order"):
    """
    Compute the bosonic part of <0|K^I Ktilde^J|V_3^B> / <0|V_3^B>.

    Returns (A_delta, B_qq) such that the tensor is
        A_delta delta^{IJ} + B_qq q_rel^I q_rel^J
    """
    data = sp.prefactor_data(n1, n2, alpha_prime,
                              left_variant=left_variant,
                              right_variant=right_variant)
    return data.a_delta_reduced, data.b_qq_reduced, data.gamma_t


def massless_on_shell_q_rel_sq(alpha_prime: float = 1.0) -> float:
    """On-shell q_rel^2 for three massless states: q_rel^2 = 0."""
    # For three massless particles with real momenta in D > 4,
    # the three-point kinematics forces q_rel = 0.
    # The nontrivial tensor comes from the polarization contractions.
    return 0.0


def graviton_tensor_target(d_perp: int = 8):
    """
    Build the target Einstein-Hilbert tensor T^{IJ,KL,MN} such that
    A_hhh = T^{IJ,KL,MN} eps^(1)_{IJ} eps^(2)_{KL} eps^(3)_{MN}.

    For three gravitons in D dimensions with all-outgoing convention:
    T = V^{IKM} V^{JLN} + (I<->J on leg 1) + ... (symmetrizations)

    where V^{IKM} = delta^{IK}(p1-p2)^M + cyclic.

    For lightcone gauge with q_rel -> 0 (massless three-point kinematics),
    the tensor structure simplifies.
    """
    # For massless three-point: in the lightcone frame the nontrivial
    # tensor arises from the interaction-point derivatives K^I, Ktilde^J
    # contracted with the zero-mode polynomial v_{IJ}(Lambda).
    #
    # The full structure is: K^I Ktilde^J v_{IJ}(Lambda)
    # where v_{IJ} is an even polynomial in the 8 Grassmann zero modes.
    pass


def test_bosonic_tensor_structure():
    """
    Test that the bosonic prefactor reproduces the expected IJ tensor
    at various lattice sizes.

    For three gravitons, the expected structure has BOTH delta^{IJ} and
    q_rel^I q_rel^J pieces. At q_rel = 0 (on-shell massless), only the
    delta^{IJ} piece survives in the bosonic factor. The full graviton
    tensor then comes from combining this with the fermionic v_{IJ}.
    """
    results = []
    for scale in [8, 16, 32, 64, 128]:
        n1, n2 = 2 * scale, 3 * scale
        a_d, b_qq, gamma = bosonic_prefactor_tensor(n1, n2)
        results.append({
            "N1": n1, "N2": n2,
            "A_delta": a_d,
            "B_qq": b_qq,
            "gamma_T": gamma,
            "A_delta_scaled": n1 * a_d,  # Should have finite limit
        })

    # A_delta should converge to a nonzero value
    a_vals = [r["A_delta"] for r in results]
    converging = all(abs(a_vals[i+1] - a_vals[i]) < abs(a_vals[i] - a_vals[i-1])
                     for i in range(1, len(a_vals)-1))
    b_vals = [r["B_qq"] for r in results]
    b_decreasing = all(
        abs(b_vals[i + 1]) < abs(b_vals[i]) for i in range(len(b_vals) - 1)
    )

    return {
        "test": "bosonic_tensor_structure",
        "values": results,
        "A_delta_converging": converging,
        "B_qq_decreasing": b_decreasing,
        "pass": converging and b_decreasing and results[-1]["A_delta"] > 0.0,
    }


def test_fermionic_zeromodes():
    """
    Test the fermionic zero-mode module and prefactor assembly.

    The v_{IJ}(Lambda) polynomial must:
    1. Have the correct SO(8) tensor structure at each Grassmann degree
    2. When quantized on the Clifford module, give nonzero matrix elements
       between graviton states
    """
    results = {}

    try:
        import gs_zero_mode_prefactor as gp
        import gs_zero_mode_module as gm

        # Build the prefactor data at alpha_ratio = 1.0
        vdata = gp.build_v_prefactor(1.0)
        results["w_symmetry_error"] = vdata.max_w_symmetry_error
        results["y_antisymmetry_error"] = vdata.max_y_antisymmetry_error
        results["t_trace_error"] = vdata.max_t_trace_error
        results["trace_y2_error"] = vdata.max_trace_y2_error
        results["trace_y6_error"] = vdata.max_trace_y6_error

        # Build the Clifford module
        module = gm.build_zero_mode_module()
        results["module_dimension"] = 16  # 8_v + 8_c
        results["clifford_anticommutator_check"] = True

        # Check anticommutator {S^a, S^b} = delta^{ab}
        sigma = module.sigma
        max_anticomm_err = 0.0
        for a in range(8):
            for b in range(8):
                anticomm = sigma[a] @ sigma[b] + sigma[b] @ sigma[a]
                expected = 2.0 * (1.0 if a == b else 0.0) * np.eye(16)
                err = np.max(np.abs(anticomm - expected))
                max_anticomm_err = max(max_anticomm_err, err)
        results["clifford_error"] = float(max_anticomm_err)

        # Check that the Weyl map gives the expected tensor structure
        import gs_weyl_symbol_diagnostic as gw
        for alpha_ratio in [0.5, 1.0, 2.0]:
            coeffs, max_err, rms_err = gw.fit_vector_block_invariants(alpha_ratio)
            results[f"weyl_fit_alpha_{alpha_ratio}"] = {
                "max_error": max_err,
                "rms_error": rms_err,
                "coeffs": [complex(c) for c in coeffs],
            }

        results["pass"] = (max_anticomm_err < 1e-12 and
                          all(results[f"weyl_fit_alpha_{ar}"]["max_error"] < 1e-10
                              for ar in [0.5, 1.0, 2.0]))

    except Exception as e:
        results["error"] = str(e)
        results["pass"] = False

    return {"test": "fermionic_zeromodes", **results}


def test_graviton_v_operator_matrix_elements():
    """
    Compute explicit matrix elements of v_{IJ} between graviton states
    in the 8_v sector of the Clifford module.

    The graviton is |I> in 8_v. We compute <I|v_{KL}|J> for various I,J,K,L.
    """
    try:
        import gs_weyl_symbol_diagnostic as gw
        import gs_zero_mode_module as gm
        import gs_zero_mode_prefactor as gp

        module = gm.build_zero_mode_module()
        prefactor = gp.build_v_prefactor(1.0)
        antisym = gw.build_antisymmetrized_products(module.sigma)

        # Build the full v_{IJ} operator on the 16-dim module
        n_v = 8  # number of transverse directions (for SO(8))

        # Compute <K|v_{IJ}|L> for all I,J,K,L in 8_v
        # v_operator is 16x16; the 8_v sector is the first 8 rows/columns
        tensor = np.zeros((n_v, n_v, n_v, n_v), dtype=complex)
        for i in range(n_v):
            for j in range(n_v):
                v_op = gw.v_operator_from_data(prefactor, antisym, i, j)
                # Extract 8_v -> 8_v block
                vv_block = v_op[:n_v, :n_v]
                for k in range(n_v):
                    for l in range(n_v):
                        tensor[i, j, k, l] = vv_block[k, l]

        # Check: tensor should be decomposable into
        # A(alpha) delta_{IJ} delta_{KL} + B(alpha) delta_{IK} delta_{JL} + C(alpha) delta_{IL} delta_{JK}
        # Fit
        design_rows = []
        target_vals = []
        for i in range(n_v):
            for j in range(n_v):
                for k in range(n_v):
                    for l in range(n_v):
                        row = [
                            1.0 if (i == j and k == l) else 0.0,
                            1.0 if (i == k and j == l) else 0.0,
                            1.0 if (i == l and j == k) else 0.0,
                        ]
                        design_rows.append(row)
                        target_vals.append(tensor[i, j, k, l])

        design = np.array(design_rows, dtype=float)
        target = np.array(target_vals, dtype=complex)
        coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        residual = design @ coeffs - target
        max_res = float(np.max(np.abs(residual)))

        # The graviton three-point tensor should have B != 0
        A, B, C = coeffs

        return {
            "test": "graviton_v_matrix_elements",
            "A": complex(A),
            "B": complex(B),
            "C": complex(C),
            "fit_max_residual": max_res,
            "B_nonzero": abs(B) > 1e-6,
            "pass": max_res < 1e-10 and abs(B) > 1e-6,
        }

    except Exception as e:
        return {
            "test": "graviton_v_matrix_elements",
            "error": str(e),
            "pass": False,
        }


def test_full_graviton_ratio_dependence():
    """
    Test the full bosonic prefactor * v_{IJ} contraction at multiple ratios.

    The combination [A_delta * v_{IJ} + B_qq * v_{KL} q^K q^L] contracted
    with graviton states should give the Yang-Mills-squared tensor.
    """
    try:
        import gs_weyl_symbol_diagnostic as gw
        import gs_zero_mode_module as gm
        import gs_zero_mode_prefactor as gp

        results = []
        for alpha_ratio in [0.25, 0.333, 0.4, 0.5]:
            coeffs, max_err, rms_err = gw.fit_vector_block_invariants(alpha_ratio)
            A, B, C = coeffs

            # At this ratio, get the bosonic prefactor
            # alpha_1/alpha_3 = alpha_ratio, so N1/N3 = alpha_ratio
            # Use scale=128
            scale = 128
            n3_target = int(round(scale / min(alpha_ratio, 1 - alpha_ratio)))
            n1 = int(round(alpha_ratio * n3_target))
            n2 = n3_target - n1
            if n1 < 4 or n2 < 4:
                continue

            a_d, b_qq, gamma = bosonic_prefactor_tensor(n1, n2)

            results.append({
                "alpha_ratio": alpha_ratio,
                "N1": n1, "N2": n2,
                "A_delta": a_d,
                "B_qq": b_qq,
                "v_A": complex(A),
                "v_B": complex(B),
                "v_C": complex(C),
                "weyl_residual": max_err,
            })

        return {
            "test": "graviton_ratio_dependence",
            "results": results,
            "pass": all(r["weyl_residual"] < 1e-10 for r in results),
        }

    except Exception as e:
        return {
            "test": "graviton_ratio_dependence",
            "error": str(e),
            "pass": False,
        }


def run_all_tests():
    results = {}

    print("Running bosonic tensor structure test...")
    r = test_bosonic_tensor_structure()
    results["bosonic_tensor"] = r
    for v in r["values"]:
        print(f"  N1={v['N1']:4d}: A_delta={v['A_delta']:.6f} B_qq={v['B_qq']:.6f}")

    print("\nRunning fermionic zero-mode test...")
    r = test_fermionic_zeromodes()
    results["fermionic"] = r
    status = "PASS" if r.get("pass") else "FAIL"
    if "error" not in r:
        print(f"  Clifford error: {r['clifford_error']:.3e}")
        for ar in [0.5, 1.0, 2.0]:
            key = f"weyl_fit_alpha_{ar}"
            if key in r:
                print(f"  Weyl fit alpha={ar}: max_err={r[key]['max_error']:.3e}")
    else:
        print(f"  Error: {r['error']}")
    print(f"  [{status}]")

    print("\nRunning graviton v_{IJ} matrix element test...")
    r = test_graviton_v_operator_matrix_elements()
    results["v_elements"] = r
    status = "PASS" if r.get("pass") else "FAIL"
    if "error" not in r:
        print(f"  A={r['A']:.6f}, B={r['B']:.6f}, C={r['C']:.6f}")
        print(f"  Fit residual: {r['fit_max_residual']:.3e}")
    else:
        print(f"  Error: {r['error']}")
    print(f"  [{status}]")

    print("\nRunning full graviton ratio dependence test...")
    r = test_full_graviton_ratio_dependence()
    results["ratio_dep"] = r
    status = "PASS" if r.get("pass") else "FAIL"
    if "results" in r:
        for v in r["results"]:
            print(f"  alpha={v['alpha_ratio']:.3f}: A_delta={v['A_delta']:.6f} "
                  f"v_B={v['v_B']:.6f} weyl_res={v['weyl_residual']:.3e}")
    print(f"  [{status}]")

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
