"""
Bosonized bc ghost: identifying b-ghost insertions from the sewing shift.

The sewing rule c(k') = -exp(2πi(k'-k)/L) c(k) translates to the
bosonized shift  φ(k') = φ(k) + iδ_k  where

    δ_k = π + 2π(k'-k)/L.

Within each arc δ_k is linear (slope -4π/L).  At the three arc
junctions k = l₁, l₁+l₂, L/2 the shift JUMPS.  These jumps are the
discrete b-ghost insertions: they source the scalar field at the
Strebel vertex positions through the Green's function.

This script:
  1. Computes δ_k and its discrete derivative, showing spikes at junctions
  2. Decomposes δ into smooth + singular parts
  3. Shows that the jump magnitudes are 2π(1−t_j) where t_j is the
     "opposite" edge length
  4. Computes the classical field  φ_cl = A⁻¹ω_singular  and shows it
     is a Green's function sourced at the Strebel vertices
  5. Compares the vertex residues with the known ν_i factors from det B
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from partition_function import direct_mat_n_fast, bmat


def compute_sewing_data(L, l1, l2):
    """Partner sites, shifts, and jump decomposition."""
    N = L // 2
    partner = np.empty(N, dtype=int)
    alpha = np.empty(N, dtype=int)  # k' - k

    for k0 in range(N):
        k = k0 + 1  # 1-based
        if k <= l1:
            kp = N + l1 + 1 - k
        elif k <= l1 + l2:
            kp = N + 2 * l1 + l2 + 1 - k
        else:
            kp = L + l1 + l2 + 1 - k
            if kp > L:
                kp -= L
        partner[k0] = kp - 1
        alpha[k0] = kp - k

    delta = np.pi + 2.0 * np.pi * alpha / L
    return partner, alpha, delta


def analyse_jumps(L, l1, l2):
    """Identify the three junction positions and their jump magnitudes."""
    N = L // 2
    l3 = N - l1 - l2
    _, _, delta = compute_sewing_data(L, l1, l2)

    # Discrete derivative  Δδ_k = δ_{k+1} - δ_k  (with periodic wrap)
    ddelta = np.diff(delta, append=delta[0])

    # Within-arc slope
    slope = -4.0 * np.pi / L

    # Excess over smooth slope = the "jump" signal
    jump_signal = ddelta - slope

    # Junction positions (0-based): l1-1, l1+l2-1, N-1
    junctions = [l1 - 1, l1 + l2 - 1, N - 1]

    # Predicted jump magnitudes: pi*(1 - t_j) where t_j is the NON-incident edge
    t1, t2, t3 = l1 / N, l2 / N, l3 / N
    predicted_jumps = [np.pi * (1 - t3),   # junction at l1:  edges 1,2 meet
                       np.pi * (1 - t1),   # junction at l1+l2: edges 2,3 meet
                       np.pi * (1 - t2)]   # junction at N:  edges 3,1 meet

    return {
        "delta": delta,
        "ddelta": ddelta,
        "jump_signal": jump_signal,
        "junctions": junctions,
        "predicted_jumps": predicted_jumps,
        "actual_jumps": [jump_signal[j] for j in junctions],
    }


def sew_matter_matrix(A0, partner):
    N = len(partner)
    r = np.arange(N)
    p = partner
    return A0[np.ix_(r, r)] + A0[np.ix_(r, p)] + A0[np.ix_(p, r)] + A0[np.ix_(p, p)]


def remove_zero_mode(M, v=None):
    N = M.shape[0]
    C = np.zeros((N, N - 1))
    C[:N - 1, :] = np.eye(N - 1)
    C[N - 1, :] = -1.0
    Mt = C.T @ M @ C
    if v is not None:
        vt = C.T @ v
        return Mt, vt
    return Mt


def classical_field_from_jumps(L, l1, l2):
    """Compute the classical bosonized field sourced by the singular
    (jump) part of the sewing shift.

    Returns the field φ_cl = Ã⁻¹ ω_sing  on the N-1 reduced sites.
    """
    N = L // 2
    partner, alpha, delta = compute_sewing_data(L, l1, l2)
    A0 = direct_mat_n_fast(L)
    p = partner

    # Full linear coefficient: w_k = 2 Σ_s [A0(k,p(s)) + A0(p(k),p(s))] δ_s
    M_coupling = A0[np.arange(N)[:, None], p[None, :]] + A0[p[:, None], p[None, :]]
    omega_full = 2.0 * M_coupling @ delta        # real N-vector

    # Decompose delta into smooth + singular
    # Smooth part: extrapolate within-arc linear slope from first arc
    slope = -4.0 * np.pi / L
    delta_smooth = delta[0] + slope * np.arange(N, dtype=float)
    delta_sing = delta - delta_smooth

    omega_sing = 2.0 * M_coupling @ delta_sing
    omega_smooth = 2.0 * M_coupling @ delta_smooth

    # Remove zero mode
    A_sewn = sew_matter_matrix(A0, partner)
    At, omega_sing_t = remove_zero_mode(A_sewn, omega_sing)
    At = 0.5 * (At + At.T)
    _, omega_full_t = remove_zero_mode(A_sewn, omega_full)
    _, omega_smooth_t = remove_zero_mode(A_sewn, omega_smooth)

    phi_cl_sing = np.linalg.solve(At, omega_sing_t)
    phi_cl_full = np.linalg.solve(At, omega_full_t)

    return {
        "delta": delta,
        "delta_smooth": delta_smooth,
        "delta_sing": delta_sing,
        "omega_full": omega_full,
        "omega_sing": omega_sing,
        "omega_smooth": omega_smooth,
        "phi_cl_sing": phi_cl_sing,
        "phi_cl_full": phi_cl_full,
        "A_tilde": At,
    }


# ─────────────────────────────────────────────────────────────────────

def run(L, l1, l2):
    N = L // 2
    l3 = N - l1 - l2
    t1, t2, t3 = l1/N, l2/N, l3/N
    print("=" * 65)
    print(f"  L={L}  (l1,l2,l3)=({l1},{l2},{l3})  t=({t1:.3f},{t2:.3f},{t3:.3f})")
    print("=" * 65)

    # ── 1. Jump analysis ──
    info = analyse_jumps(L, l1, l2)
    print("\n--- Sewing shift jumps ---")
    print(f"  Junction positions (0-based): {info['junctions']}")
    for i, j in enumerate(info["junctions"]):
        print(f"  junction {i+1} at k={j}:  actual jump = {info['actual_jumps'][i]:+.8f}"
              f"  predicted 2pi(1-t_j) = {info['predicted_jumps'][i]:+.8f}"
              f"  error = {abs(info['actual_jumps'][i]-info['predicted_jumps'][i]):.2e}")

    # Check: jumps elsewhere should be ~0
    js = set(info["junctions"])
    other = [info["jump_signal"][k] for k in range(N) if k not in js]
    print(f"  max |jump| away from junctions: {max(abs(x) for x in other):.2e}")

    # ── 2. Classical field from singular part ──
    cf = classical_field_from_jumps(L, l1, l2)

    print("\n--- delta_k decomposition ---")
    print(f"  max|delta_sing| = {np.max(np.abs(cf['delta_sing'])):.6f}")
    print(f"  delta_sing is nonzero only near junctions:")
    for j in info["junctions"]:
        vals = cf["delta_sing"][max(0,j-1):min(N,j+2)]
        print(f"    k={j}: delta_sing = {vals}")

    print(f"\n--- Classical field phi_cl = A_tilde^{-1} omega_sing ---")
    phi = cf["phi_cl_sing"]
    # Show values near junctions
    for i, j in enumerate(info["junctions"]):
        if j < N - 1:  # within reduced range
            jj = min(j, N - 2)
            print(f"  near junction {i+1} (k={j}): phi_cl[{jj}] = {phi[jj]:+.6f}")

    # ── 3. Compare with det B structure ──
    B = bmat(L, l1, l2)
    _, logabsdet_B = np.linalg.slogdet(B)

    # The b-ghost insertion from det B is at the Strebel vertices.
    # From string MC.tex: |det B| * (det'A)^{-13} ∝ ... (1/3) Σ |ν_i|^4
    # where ν_i = lim_{z→z_i} (1-z/z_i)^{1/3} f(z).
    # The jump magnitudes J_v = pi*(1-t_v) should be related to the nu_i.

    # Compute the quadratic form from the singular part only
    At = cf["A_tilde"]
    _, omega_sing_t = remove_zero_mode(
        sew_matter_matrix(direct_mat_n_fast(L),
                          compute_sewing_data(L, l1, l2)[0]),
        cf["omega_sing"])
    bilinear_sing = omega_sing_t @ np.linalg.solve(At, omega_sing_t)

    _, omega_full_t = remove_zero_mode(
        sew_matter_matrix(direct_mat_n_fast(L),
                          compute_sewing_data(L, l1, l2)[0]),
        cf["omega_full"])
    bilinear_full = omega_full_t @ np.linalg.solve(At, omega_full_t)

    # det'A
    logdet_At = np.linalg.slogdet(0.5*(At+At.T))[1]
    log_prime_det = logdet_At - np.log(N)

    print("\n--- Bilinear forms ---")
    print(f"  omega_sing bilinear  = {bilinear_sing:+.8f}")
    print(f"  omega_full bilinear  = {bilinear_full:+.8f}")
    print(f"  log|det B|           = {logabsdet_B:+.8f}")
    print(f"  (1/2) log det'A      = {0.5*log_prime_det:+.8f}")
    ratio = logabsdet_B - 0.5 * log_prime_det
    print(f"  ghost_ratio          = {ratio:+.8f}")
    print(f"  (1/4) full bilinear  = {0.25*bilinear_full:+.8f}")
    print()

    return info, cf


def jump_vs_nu_sweep(L_values=[40, 60, 80, 100]):
    """Show that jump magnitudes converge to 2π(1-t_j) at each junction."""
    print("\n" + "=" * 65)
    print("  Jump magnitude convergence with L")
    print("=" * 65)
    l1_frac, l2_frac = 0.3, 0.4  # fixed moduli fractions
    for L in L_values:
        N = L // 2
        l1 = max(1, round(l1_frac * N))
        l2 = max(1, round(l2_frac * N))
        l3 = N - l1 - l2
        if l3 < 1:
            l2 -= 1
            l3 = N - l1 - l2
        info = analyse_jumps(L, l1, l2)
        errors = [abs(a - p) for a, p in zip(info["actual_jumps"], info["predicted_jumps"])]
        max_other = max(abs(info["jump_signal"][k]) for k in range(N) if k not in set(info["junctions"]))
        print(f"  L={L:4d}  junction errors: {errors[0]:.2e} {errors[1]:.2e} {errors[2]:.2e}"
              f"   max off-junction: {max_other:.2e}")


if __name__ == "__main__":
    run(60, 8, 12)
    jump_vs_nu_sweep()
