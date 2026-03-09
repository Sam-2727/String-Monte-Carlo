"""
Compare |g(z)| vs |1/f(z)| in the bulk (small |z|).
"""
import numpy as np
from partition_function import bmat
from ell_to_tau import make_cyl_eqn_improved, make_cyl_eqn


def det_ratio(L, l1, l2, n):
    """
    Compute det B^{(n)} / det B^{(1)} by replacing column 0 of bmat.
    """
    B = bmat(L, l1, l2)
    det1 = np.linalg.det(B)

    # Build column for mode n using same convention as bmat
    half = L // 2
    twopi_i_over_L = 2j * np.pi / L
    m_arr = np.array([n], dtype=np.float64)  # single mode

    new_col = np.zeros(half, dtype=np.complex128)
    idx = 0
    for k in range(1, l1 + 1):
        sk = half + l1 + 1 - k
        new_col[idx] = np.exp(twopi_i_over_L * n * k) - np.exp(twopi_i_over_L * n * sk)
        idx += 1
    for k in range(l1 + 1, l1 + l2 + 1):
        sk = half + 2*l1 + l2 + 1 - k
        new_col[idx] = np.exp(twopi_i_over_L * n * k) - np.exp(twopi_i_over_L * n * sk)
        idx += 1
    for k in range(l1 + l2 + 1, half + 1):
        sk = L + l1 + l2 + 1 - k
        new_col[idx] = np.exp(twopi_i_over_L * n * k) - np.exp(twopi_i_over_L * n * sk)
        idx += 1

    B2 = B.copy()
    B2[:, 0] = new_col
    det_n = np.linalg.det(B2)
    return det_n / det1


def main():
    L = 30
    l1, l2 = 3, 5
    half = L // 2
    print(f"L={L}, l1={l1}, l2={l2}, l3={half-l1-l2}")

    # Sanity check: n=1 should give ratio = 1
    r1 = det_ratio(L, l1, l2, 1)
    print(f"det B^(1)/det B^(1) = {r1:.10f} (should be 1)")

    # Compute ratios for negative n
    ratios = {}
    print("\nRatios |det B^(-n) / det B^(1)|:")
    for n in range(1, half):
        r = det_ratio(L, l1, l2, -n)
        if abs(r) > 1e-12:
            ratios[n] = r
            print(f"  n={n:3d}: |ratio| = {abs(r):.8f}")

    # Compare |g(z)| vs |1/f(z)| for real z in (0, 1)
    f_basic = make_cyl_eqn(L, l1, l2)
    f_improved = make_cyl_eqn_improved(L, l1, l2)

    print(f"\n{'z':>6s} | {'|g(z)|':>12s} | {'|1/f_basic|':>12s} | {'|1/f_impr|':>12s} | {'|g/ginv_b|':>12s}")
    print("-" * 72)

    for z in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        gz = 1.0 + sum(r * z**(1+n) for n, r in ratios.items())

        fz_b = f_basic(z)
        s, p = f_improved(z)
        fz_i = s * p

        print(f"{z:6.2f} | {abs(gz):12.8f} | {1/abs(fz_b):12.8f} | {1/abs(fz_i):12.8f} | {abs(gz)*abs(fz_b):12.8f}")


if __name__ == '__main__':
    main()
