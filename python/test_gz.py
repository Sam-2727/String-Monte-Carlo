"""
Compute g(z) = det B^{c(z)} / det B^{(1)} on |z|=1 and compare with 1/f(z).

Instead of building g(z) as a power series, directly construct B^{c(z)}
by replacing column 0 with the sum over all modes weighted by z^{1-n}.
"""
import numpy as np
import matplotlib.pyplot as plt
from partition_function import bmat
from ell_to_tau import make_cyl_eqn_improved, make_cyl_eqn


def mapped_site(k, l1, l2, L):
    """Return the mapped site s(k) for k in 1..L/2."""
    half = L // 2
    l3 = half - l1 - l2
    if k <= l1:
        return half + l1 + 1 - k
    elif k <= l1 + l2:
        return half + 2*l1 + l2 + 1 - k
    else:
        return L + l1 + l2 + 1 - k


def compute_gz_direct(L, l1, l2, z_pts):
    """
    For each z on |z|=1, build the matrix B^{c(z)} directly and compute
    det B^{c(z)} / det B^{(1)}.

    c(z) = sum_{n <= 1} c_n z^{1-n}, so the first column of B^{c(z)} is:
    C(z,k) = sum_{n=-(L/2-1)}^{1} z^{1-n} * [exp(2pi i n k/L) - exp(2pi i n s(k)/L)]
    """
    half = L // 2

    # Standard B matrix
    B_std = bmat(L, l1, l2)
    det_std = np.linalg.det(B_std)

    # Precompute mapped sites
    sites = np.arange(1, half + 1)
    mapped = np.array([mapped_site(k, l1, l2, L) for k in sites])

    # Mode numbers: n from -(L/2-1) to 1
    modes = np.arange(-(half - 1), 2)  # -(L/2-1), ..., 0, 1

    # Precompute exp(2pi i n k / L) and exp(2pi i n s(k) / L) for all n, k
    twopi_i_over_L = 2j * np.pi / L
    # exp_nk[i, j] = exp(2pi i modes[i] * sites[j] / L)
    exp_nk = np.exp(twopi_i_over_L * np.outer(modes, sites))
    exp_ns = np.exp(twopi_i_over_L * np.outer(modes, mapped))
    diff_exp = exp_nk - exp_ns  # shape: (n_modes, half)

    results = np.empty(len(z_pts), dtype=np.complex128)

    for idx, z in enumerate(z_pts):
        # z^{1-n} for each mode n
        powers = z ** (1 - modes)  # shape: (n_modes,)

        # First column: C(z,k) = sum_n z^{1-n} * diff_exp[n, k]
        new_col = powers @ diff_exp  # shape: (half,)

        # Build modified B matrix
        B_mod = B_std.copy()
        B_mod[:, 0] = new_col  # Replace column 0 (mode m=1)

        det_mod = np.linalg.det(B_mod)
        results[idx] = det_mod / det_std

    return results


def compute_fz_improved(L, l1, l2, z_pts):
    """Compute f(z) using make_cyl_eqn_improved."""
    f = make_cyl_eqn_improved(L, l1, l2)
    vals = []
    for z in z_pts:
        try:
            s, p = f(z)
            vals.append(s * p)
        except:
            vals.append(np.nan + 1j*np.nan)
    return np.array(vals)


def compute_fz_basic(L, l1, l2, z_pts):
    """Compute f(z) using make_cyl_eqn (non-improved)."""
    f = make_cyl_eqn(L, l1, l2)
    vals = np.array([f(z) for z in z_pts])
    return vals


def main():
    # Parameters - use moderate L
    L = 78
    l1, l2 = 7, 11
    l3 = L // 2 - l1 - l2
    print(f"L={L}, l1={l1}, l2={l2}, l3={l3}")

    # Points on the unit circle, avoiding exact vertex points
    N_pts = 300
    thetas = np.linspace(0.01, 2*np.pi - 0.01, N_pts)
    z_pts = np.exp(1j * thetas)

    print("Computing g(z) directly...")
    gz = compute_gz_direct(L, l1, l2, z_pts)

    print("Computing f(z) (improved)...")
    fz_imp = compute_fz_improved(L, l1, l2, z_pts)
    inv_fz_imp = 1.0 / fz_imp

    print("Computing f(z) (basic)...")
    fz_basic = compute_fz_basic(L, l1, l2, z_pts)
    inv_fz_basic = 1.0 / fz_basic

    # Plot comparison with improved f
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'g(z) vs 1/f(z) on |z|=1, L={L}, l1={l1}, l2={l2}, l3={l3}')

    # Real parts
    axes[0,0].plot(thetas/np.pi, gz.real, 'b-', label='Re g(z)', lw=1.5)
    axes[0,0].plot(thetas/np.pi, inv_fz_imp.real, 'r--', label='Re 1/f_imp(z)', lw=1)
    axes[0,0].plot(thetas/np.pi, inv_fz_basic.real, 'g:', label='Re 1/f_basic(z)', lw=1)
    axes[0,0].legend(fontsize=8)
    axes[0,0].set_title('Real part')
    axes[0,0].set_xlabel('θ/π')

    # Imag parts
    axes[0,1].plot(thetas/np.pi, gz.imag, 'b-', label='Im g(z)', lw=1.5)
    axes[0,1].plot(thetas/np.pi, inv_fz_imp.imag, 'r--', label='Im 1/f_imp(z)', lw=1)
    axes[0,1].plot(thetas/np.pi, inv_fz_basic.imag, 'g:', label='Im 1/f_basic(z)', lw=1)
    axes[0,1].legend(fontsize=8)
    axes[0,1].set_title('Imaginary part')
    axes[0,1].set_xlabel('θ/π')

    # Magnitude
    axes[1,0].plot(thetas/np.pi, np.abs(gz), 'b-', label='|g(z)|', lw=1.5)
    axes[1,0].plot(thetas/np.pi, np.abs(inv_fz_imp), 'r--', label='|1/f_imp(z)|', lw=1)
    axes[1,0].plot(thetas/np.pi, np.abs(inv_fz_basic), 'g:', label='|1/f_basic(z)|', lw=1)
    axes[1,0].legend(fontsize=8)
    axes[1,0].set_title('Magnitude')
    axes[1,0].set_xlabel('θ/π')

    # Ratio g(z) * f(z) - should be constant if g = 1/f
    ratio_imp = gz * fz_imp
    ratio_basic = gz * fz_basic
    axes[1,1].plot(thetas/np.pi, np.abs(ratio_imp), 'r-', label='|g(z)*f_imp(z)|', lw=1)
    axes[1,1].plot(thetas/np.pi, np.abs(ratio_basic), 'g-', label='|g(z)*f_basic(z)|', lw=1)
    axes[1,1].legend(fontsize=8)
    axes[1,1].set_title('|g(z) * f(z)| (should be constant)')
    axes[1,1].set_xlabel('θ/π')

    # Mark vertex locations
    phase1 = 2*np.pi * l1 / L
    phase2 = 2*np.pi * (l1 + l2) / L
    vtx = [0, np.pi, phase1, phase1+np.pi, phase2, phase2+np.pi]
    for ax in axes.flat:
        for v in vtx:
            ax.axvline(v/np.pi, color='gray', alpha=0.3, ls=':')

    plt.tight_layout()
    plt.savefig('gz_vs_inv_fz.png', dpi=150)
    print("Saved gz_vs_inv_fz.png")

    # Print sample ratios
    print("\nSample g(z)*f(z) values (should be constant):")
    print("  improved f:")
    for i in range(0, N_pts, 30):
        print(f"    θ/π={thetas[i]/np.pi:.3f}: g*f = {ratio_imp[i]:.6f}")
    print("  basic f:")
    for i in range(0, N_pts, 30):
        print(f"    θ/π={thetas[i]/np.pi:.3f}: g*f = {ratio_basic[i]:.6f}")


if __name__ == '__main__':
    main()
