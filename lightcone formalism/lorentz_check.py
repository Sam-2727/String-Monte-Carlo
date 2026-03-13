#!/usr/bin/env python3
"""
Discrete Lorentz invariance check for the lightcone Mandelstam vertex.

Computes:
1. Zero-point energy mismatch at finite N and its continuum limit
2. Discrete Neumann coefficients and their convergence
3. Bond-locality verification
4. Forward vs backward arc difference comparison

All quantities are checked against the known continuum values.
"""

import numpy as np
from numpy.linalg import inv, det
from scipy.linalg import block_diag

# ============================================================
# 1. Zero-point energy mismatch
# ============================================================

def lattice_freq(m, N, a=1.0):
    """Lattice frequency omega_m = (2/a) sin(pi m / N)."""
    return (2.0 / a) * np.sin(np.pi * m / N)


def zero_point_sum(N, a=1.0):
    """Sum_{m=1}^{N-1} omega_m = (2/a) cot(pi/(2N))."""
    ms = np.arange(1, N)
    return np.sum(lattice_freq(ms, N, a))


def zero_point_sum_analytic(N, a=1.0):
    """Closed form: (2/a) cot(pi/(2N))."""
    return (2.0 / a) / np.tan(np.pi / (2.0 * N))


def zpe_mismatch(N1, N2, a=1.0):
    """
    Weighted zero-point energy mismatch:
      Delta_E0 = E0(N3)/(2p3+) - E0(N1)/(2p1+) - E0(N2)/(2p2+)

    Using p_r^+ = a*N_r / (2*alpha'), set alpha'=1 for simplicity.
    Then 1/(2p_r^+) = 1/(a*N_r).

    Returns the mismatch per transverse dimension (factor D_perp omitted).
    """
    N3 = N1 + N2
    alpha_prime = 1.0

    E1 = zero_point_sum_analytic(N1, a) / 2.0  # factor 1/2 in E0 = (D_perp/2) sum omega
    E2 = zero_point_sum_analytic(N2, a) / 2.0
    E3 = zero_point_sum_analytic(N3, a) / 2.0

    p1_plus = a * N1 / (2.0 * alpha_prime)
    p2_plus = a * N2 / (2.0 * alpha_prime)
    p3_plus = a * N3 / (2.0 * alpha_prime)

    return E3 / (2.0 * p3_plus) - E1 / (2.0 * p1_plus) - E2 / (2.0 * p2_plus)


def renormalized_zpe_mismatch(N1, N2, a=1.0):
    """
    Renormalized weighted three-string zero-point mismatch per transverse
    dimension.

    The raw mismatch contains the universal weighted vacuum-energy divergence

        -2 / (pi a^2),

    so the finite quantity relevant for the continuum comparison is obtained by
    adding back that universal term.
    """
    return zpe_mismatch(N1, N2, a) + 2.0 / (np.pi * a * a)


def zpe_mismatch_continuum(alpha1, alpha2, alpha_prime=1.0):
    """
    Continuum zero-point energy mismatch (per transverse dimension,
    both chiralities, using zeta regularization).

    = -(pi / 6) * [1/alpha3 - 1/alpha1 - 1/alpha2] / (alpha_r / alpha')
    = (pi * alpha' / 3) * (1/alpha1^2 + 1/alpha2^2 - 1/alpha3^2)

    Wait, let me recompute. Per transverse dimension, the zero-point energy
    on a closed string of circumference alpha is:
      E_0 = sum_{n=1}^{infty} omega_n = (2pi/alpha) * zeta(-1) = -pi/(6*alpha)

    (This includes both chiralities: L+R each contribute pi/(12*alpha),
     total = -pi/(6*alpha).)

    Weighted: E_0/(2p^+) = [-pi/(6*alpha)] / [alpha/alpha'] = -pi*alpha'/(6*alpha^2)

    Mismatch (leg3 minus leg1 minus leg2):
      Delta = -pi*alpha'/(6) * [1/alpha3^2 - 1/alpha1^2 - 1/alpha2^2]
            = pi*alpha'/(6) * [1/alpha1^2 + 1/alpha2^2 - 1/alpha3^2]
    """
    alpha3 = alpha1 + alpha2
    return (np.pi * alpha_prime / 6.0) * (
        1.0 / alpha1**2 + 1.0 / alpha2**2 - 1.0 / alpha3**2
    )


def test_zpe_convergence():
    """
    Test that the renormalized finite part of the discrete weighted ZPE
    mismatch converges to the continuum value.
    """
    print("=" * 70)
    print("1. RENORMALIZED ZERO-POINT MISMATCH: CONVERGENCE TO CONTINUUM")
    print("=" * 70)

    # Test with alpha1/alpha2 = 2/3 (i.e., N1/N2 = 2/3)
    ratio = 2.0 / 3.0
    alpha_prime = 1.0

    # Continuum values at different alpha1
    alpha1_ref = 1.0
    alpha2_ref = alpha1_ref / ratio  # = 1.5
    continuum_val = zpe_mismatch_continuum(alpha1_ref, alpha2_ref, alpha_prime)

    print(f"\nRatio alpha1/alpha2 = {ratio:.4f}")
    print(f"Continuum mismatch (per D_perp): {continuum_val:.10f}")
    print(
        f"\n{'N1':>6s} {'N2':>6s} {'N3':>6s}   "
        f"{'Raw':>14s}  {'Renorm.':>14s}  {'Continuum':>14s}  {'Rel. err':>12s}"
    )
    print("-" * 92)

    for scale in [5, 10, 20, 50, 100, 200, 500, 1000]:
        N1 = 2 * scale
        N2 = 3 * scale
        N3 = N1 + N2
        a = alpha1_ref / N1  # so alpha1 = a*N1 = alpha1_ref

        raw_val = zpe_mismatch(N1, N2, a)
        renorm_val = renormalized_zpe_mismatch(N1, N2, a)
        rel_err = (
            abs(renorm_val - continuum_val) / abs(continuum_val)
            if continuum_val != 0
            else 0
        )
        print(
            f"{N1:6d} {N2:6d} {N3:6d}   "
            f"{raw_val:14.10f}  {renorm_val:14.10f}  "
            f"{continuum_val:14.10f}  {rel_err:12.2e}"
        )

    print()


# ============================================================
# 2. Discrete Neumann coefficients
# ============================================================

def build_overlap_matrices(N1, N2):
    """
    Build the truncated DFT overlap matrices U1, U2, xi.

    U_r = hat{F}_3 P_r hat{F}_r^dagger  (both (N3-1) x (N_r-1))
    xi = hat{F}_3 P_1 1_{N1}  ((N3-1) vector)

    where hat{F}_r is the truncated DFT (zero-mode row removed),
    P_1 = (I_{N1}; 0_{N2 x N1}), P_2 = (0_{N1 x N2}; I_{N2}).
    """
    N3 = N1 + N2

    # Full DFT matrices (symmetric: F_{kn} = (1/sqrt(N)) exp(2pi i k n / N))
    def dft_matrix(N):
        ns = np.arange(N)
        ks = np.arange(N)
        return np.exp(2j * np.pi * np.outer(ks, ns) / N) / np.sqrt(N)

    F1 = dft_matrix(N1)
    F2 = dft_matrix(N2)
    F3 = dft_matrix(N3)

    # Truncated: remove zero-mode row (k=0)
    Fhat1 = F1[1:, :]   # (N1-1) x N1
    Fhat2 = F2[1:, :]   # (N2-1) x N2
    Fhat3 = F3[1:, :]   # (N3-1) x N3

    # Embedding matrices P1, P2
    # P1: N3 x N1, first N1 rows are I, rest 0
    # P2: N3 x N2, first N1 rows are 0, last N2 rows are I
    P1 = np.zeros((N3, N1))
    P1[:N1, :] = np.eye(N1)

    P2 = np.zeros((N3, N2))
    P2[N1:, :] = np.eye(N2)

    # U_r = Fhat3 @ P_r @ Fhat_r^dagger
    U1 = Fhat3 @ P1 @ Fhat1.conj().T  # (N3-1) x (N1-1)
    U2 = Fhat3 @ P2 @ Fhat2.conj().T  # (N3-1) x (N2-1)

    # Affine vector xi = Fhat3 @ P1 @ 1_{N1}
    ones_N1 = np.ones(N1)
    xi = Fhat3 @ P1 @ ones_N1  # (N3-1) vector

    return U1, U2, xi


def build_neumann_coefficients(N1, N2, a=1.0):
    """
    Build the discrete Neumann coefficient matrix bar{N}^{rs}_{mn}.

    The vertex quadratic form in oscillator modes is determined by
    the overlap and the frequency matrices. The Neumann coefficients
    encode the constrained quadratic form after imposing the overlap.

    Returns N_bar as a dict with keys (r,s) for r,s in {1,2,3},
    each value being the coefficient matrix.
    """
    N3 = N1 + N2
    U1, U2, xi = build_overlap_matrices(N1, N2)

    # Frequency arrays for each leg (nonzero modes)
    def omega_array(N):
        ms = np.arange(1, N)
        return (2.0 / a) * np.sin(np.pi * ms / N)

    omega1 = omega_array(N1)
    omega2 = omega_array(N2)
    omega3 = omega_array(N3)

    # mu_r = a / (2 pi alpha'), set alpha' = 1
    mu1 = a / (2.0 * np.pi)
    mu2 = a / (2.0 * np.pi)
    mu3 = a / (2.0 * np.pi)

    # Diagonal matrices sqrt(mu * omega) and 1/sqrt(mu * omega)
    def sqrt_mu_omega(mu, omega):
        return np.sqrt(mu * omega)

    smo1 = sqrt_mu_omega(mu1, omega1)  # (N1-1) array
    smo2 = sqrt_mu_omega(mu2, omega2)
    smo3 = sqrt_mu_omega(mu3, omega3)

    # The overlap constraint in oscillator language:
    # M * a + N * a^dagger = 0
    # where a = (a^(1), a^(2), a^(3))^T
    #
    # The Neumann coefficients come from solving this constraint.
    # In the bosonic vertex, the squeeze matrix S = -M^{-1} N gives
    # the Neumann coefficient matrix (after proper normalization).
    #
    # For the position-space overlap q^(3) = U1 q^(1) + U2 q^(2) + xi*y,
    # the overlap constraint in (q, pi) variables is:
    #   C_q = (U1, U2, -I)   [position constraint]
    #   C_pi = (I, 0, -U1^dag; 0, I, -U2^dag)  [momentum constraint]
    #
    # Converting to oscillator variables:
    #   a_m = sqrt(mu*omega/2) q_m + i/(sqrt(2*mu*omega)) pi_m
    #   a_m^dag = sqrt(mu*omega/2) q_{N-m} - i/(sqrt(2*mu*omega)) pi_{N-m}
    #
    # For the annihilation constraints on the vertex:
    #   (M_B a + N_B a^dag)|V> = 0
    #
    # The Neumann coefficients are related to S = -M_B^{-1} N_B.

    # Build M_B and N_B matrices following the note's conventions.
    # For the constrained system, the overlap equations in Fourier modes are:
    #   q_m^(3) = sum_n [U1]_{mn} q_n^(1) + sum_n [U2]_{mn} q_n^(2) + xi_m y
    #   pi_m^(1) = sum_n [U1^dag]_{mn} pi_n^(3)
    #   pi_m^(2) = sum_n [U2^dag]_{mn} pi_n^(3)
    #
    # Converting q, pi to a, a^dag:
    #   q_m = (a_m + a_{N-m}^dag) / sqrt(2 mu omega_m)
    #   pi_m = -i sqrt(mu omega_m / 2) (a_m - a_{N-m}^dag)

    # For the Neumann coefficient comparison, we can directly compute the
    # constrained quadratic form in the Fourier basis.
    # The quadratic form at the vertex (position space) is:
    #   Q = sum_r sum_n mu_r omega_{m(n)}^2 |q_m^(r)|^2
    # subject to the overlap constraint.
    #
    # After imposing the constraint, the quadratic form in (q^(1), q^(2)) is:
    #   Q_constrained = sum terms involving U1, U2, omega matrices

    # The Neumann coefficients are most directly extracted by computing
    # the quadratic form in the Fourier basis after imposing the overlap.

    # The constrained oscillator quadratic form:
    # Q = (1/2) sum_{r,s} sum_{m,n} q_m^(r) Nbar^{rs}_{mn} q_n^(s)

    # From the overlap: q^(3) = U1 q^(1) + U2 q^(2) (ignoring affine part)
    # So the bond part of the Hamiltonian for leg 3 becomes:
    #   Omega3^2 |q^(3)|^2 = (U1 q1 + U2 q2)^dag Omega3^2 (U1 q1 + U2 q2)

    # The full constrained quadratic form (kinetic part cancels on overlap):
    # Only the potential energy (bond) part matters for the Neumann coefficients:
    #   V = (1/2) [q1^dag Omega1^2 q1 + q2^dag Omega2^2 q2 + (U1 q1 + U2 q2)^dag Omega3^2 (U1 q1 + U2 q2)]
    # Wait, this overcounts. On the overlap, q^(3) is determined, so:
    #   V_constrained = V with q^(3) = U1 q^(1) + U2 q^(2)

    # This is a stability diagnostic rather than a convention-matched extraction
    # of the standard continuum \bar N^{rs}_{mn}. The common factor
    # mu = a / (2 pi alpha') has been stripped off, so the entries below are the
    # constrained quadratic-form coefficients in the q-variables, proportional to
    # the conventional Neumann matrices but not yet identical to them.
    O1sq = np.diag(omega1**2)
    O2sq = np.diag(omega2**2)
    O3sq = np.diag(omega3**2)

    # Full frequency matrix (block diagonal):
    M_full = block_diag(O1sq, O2sq, O3sq)

    # Overlap constraint: C @ q_full = 0 where q_full = (q1, q2, q3)
    # C = (U1, U2, -I_{N3-1})
    C = np.hstack([U1, U2, -np.eye(N3 - 1)])

    # The constrained quadratic form in the (q1, q2) independent variables:
    # q3 = U1 q1 + U2 q2
    # Substitute into the full quadratic form:
    # q_full = (q1, q2, U1 q1 + U2 q2) = (I 0; 0 I; U1 U2) @ (q1; q2)

    n_indep = (N1 - 1) + (N2 - 1)
    Substitution = np.zeros((N1 - 1 + N2 - 1 + N3 - 1, n_indep), dtype=complex)
    Substitution[:N1 - 1, :N1 - 1] = np.eye(N1 - 1)
    Substitution[N1 - 1:N1 - 1 + N2 - 1, N1 - 1:] = np.eye(N2 - 1)
    Substitution[N1 - 1 + N2 - 1:, :N1 - 1] = U1
    Substitution[N1 - 1 + N2 - 1:, N1 - 1:] = U2

    # Constrained quadratic form:
    Q_constrained = Substitution.conj().T @ M_full @ Substitution

    # Extract blocks:
    Nbar11 = Q_constrained[:N1 - 1, :N1 - 1]
    Nbar12 = Q_constrained[:N1 - 1, N1 - 1:]
    Nbar21 = Q_constrained[N1 - 1:, :N1 - 1]
    Nbar22 = Q_constrained[N1 - 1:, N1 - 1:]

    return {
        (1, 1): Nbar11,
        (1, 2): Nbar12,
        (2, 1): Nbar21,
        (2, 2): Nbar22,
    }


def continuum_neumann_11(m, n, alpha1, alpha2):
    """
    Continuum Neumann coefficient Nbar^{11}_{mn} for the cubic vertex.

    Standard result (see e.g. Mandelstam 1973, or LeClair et al.):

    Nbar^{rr}_{mn} = -(1/sqrt(mn)) * [sum over terms from the Mandelstam map]

    For general alpha, the closed-form expressions involve the
    Mandelstam mapping rho = alpha1 ln(z - z_1) + alpha2 ln(z - z_2).
    For the simple case alpha1 = alpha2 = alpha3/2, things simplify.

    Here we use the known asymptotic forms for comparison.
    The simplest check is the diagonal coefficient Nbar^{rr}_{nn}:

    Nbar^{11}_{nn} ≈ omega_n^(1)^2 + (corrections from sewing)

    Actually, the exact continuum formulas are complicated. Let me just
    compare low-lying coefficients numerically at different N.
    """
    # For a more concrete comparison, we'll track the convergence of
    # discrete coefficients as N increases.
    pass


def test_neumann_convergence():
    """
    Track representative low-mode entries of the constrained quadratic form.

    This is a finite-N stability diagnostic, not yet a convention-matched
    continuum comparison of Neumann coefficients.
    """
    print("=" * 70)
    print("2. LOW-MODE CONSTRAINED QUADRATIC ENTRIES")
    print("=" * 70)

    # Use ratio N1:N2 = 2:3
    ratio_N1 = 2
    ratio_N2 = 3

    # Track low-lying coefficients
    scales = [5, 10, 20, 50, 100]

    print(f"\nTracking Q_constrained[(1,1)]_{{1,1}} (lowest diagonal Neumann coeff)")
    print(f"{'Scale':>6s}  {'N1':>4s}  {'N2':>4s}  {'Re[Nbar11_11]':>16s}  {'Im[Nbar11_11]':>16s}  {'Change':>12s}")
    print("-" * 70)

    prev_val = None
    for scale in scales:
        N1 = ratio_N1 * scale
        N2 = ratio_N2 * scale

        Nbar = build_neumann_coefficients(N1, N2)
        val = Nbar[(1, 1)][0, 0]

        change = abs(val - prev_val) if prev_val is not None else float('nan')
        prev_val = val

        print(f"{scale:6d}  {N1:4d}  {N2:4d}  {val.real:16.10f}  {val.imag:16.10f}  {change:12.2e}")

    print(f"\nTracking Q_constrained[(1,2)]_{{1,1}} (lowest off-diagonal coeff)")
    print(f"{'Scale':>6s}  {'N1':>4s}  {'N2':>4s}  {'Re[Nbar12_11]':>16s}  {'Im[Nbar12_11]':>16s}  {'Change':>12s}")
    print("-" * 70)

    prev_val = None
    for scale in scales:
        N1 = ratio_N1 * scale
        N2 = ratio_N2 * scale

        Nbar = build_neumann_coefficients(N1, N2)
        val = Nbar[(1, 2)][0, 0]

        change = abs(val - prev_val) if prev_val is not None else float('nan')
        prev_val = val

        print(f"{scale:6d}  {N1:4d}  {N2:4d}  {val.real:16.10f}  {val.imag:16.10f}  {change:12.2e}")

    print()
    print(
        "These values show finite-N stabilization of representative quadratic "
        "data, but they are not by themselves a proof of convention-matched "
        "convergence to the continuum Neumann coefficients."
    )
    print()


# ============================================================
# 3. Bond-locality verification
# ============================================================

def test_bond_locality():
    """
    Verify that the Hamiltonian mismatch on the overlap is join-local.

    For random X^(1), X^(2), compute X^(3) = P1 X^(1) + P2 X^(2),
    then verify that the bond mismatch equals Delta_I^bond.
    """
    print("=" * 70)
    print("3. BOND-LOCALITY VERIFICATION")
    print("=" * 70)

    N1, N2 = 7, 5
    N3 = N1 + N2
    np.random.seed(42)

    X1 = np.random.randn(N1)
    X2 = np.random.randn(N2)
    X3 = np.concatenate([X1, X2])  # P1 X1 + P2 X2

    # Bond sums
    def bond_sum(X, N):
        return sum((X[(n + 1) % N] - X[n])**2 for n in range(N))

    B1 = bond_sum(X1, N1)
    B2 = bond_sum(X2, N2)
    B3 = bond_sum(X3, N3)

    Delta_raw = B1 + B2 - B3

    # Expected: Delta = 2 * (X_{I+} - X_{I-}) * (X_{N2-1}^(2) - X_{N1-1}^(1))
    # where I+ = X_0^(1) = X_0^(3), I- = X_0^(2) = X_{N1}^(3)
    X_Iplus = X1[0]
    X_Iminus = X2[0]
    delta_I = X_Iplus - X_Iminus

    Delta_formula = 2.0 * delta_I * (X2[N2 - 1] - X1[N1 - 1])

    print(f"\nN1={N1}, N2={N2}, N3={N3}")
    print(f"Bond sum mismatch (raw):    {Delta_raw:.12f}")
    print(f"Bond formula (2*delta_I*...): {Delta_formula:.12f}")
    print(f"Match: {np.isclose(Delta_raw, Delta_formula)}")

    # Also verify the note's rewriting with backward differences:
    # Delta = 2 delta_I * (nabla+^bwd - nabla-^bwd - delta_I)
    nabla_plus_bwd = X_Iplus - X1[N1 - 1]   # X_{I+} - X_{N1-1}^(1)
    nabla_minus_bwd = X_Iminus - X2[N2 - 1]  # X_{I-} - X_{N2-1}^(2)

    Delta_rewrite = 2.0 * delta_I * (nabla_plus_bwd - nabla_minus_bwd - delta_I)
    print(f"Bond formula (rewrite):     {Delta_rewrite:.12f}")
    print(f"Match: {np.isclose(Delta_raw, Delta_rewrite)}")

    print()


# ============================================================
# 4. Forward vs backward arc differences
# ============================================================

def test_forward_backward():
    """
    Compare forward and backward arc differences at the interaction point
    and show their convergence in the continuum limit.

    Forward: nabla+ X_{I+}^(1) = X_1^(1) - X_{I+}
    Backward: nabla+^bwd X_{I+}^(1) = X_{I+} - X_{N1-1}^(1)

    On a smooth field, these should agree up to O(a) corrections.
    """
    print("=" * 70)
    print("4. FORWARD VS BACKWARD ARC DIFFERENCES")
    print("=" * 70)

    # Use a smooth field: X_n = cos(2*pi*n/N) + sin(4*pi*n/N)
    print("\nSmooth field: X_n = cos(2*pi*n/N) + 0.5*sin(4*pi*n/N)")
    print(f"{'N':>6s}  {'Forward':>14s}  {'Backward':>14s}  {'Ratio':>10s}  {'Diff':>14s}")
    print("-" * 65)

    for N in [10, 20, 50, 100, 200, 500, 1000]:
        ns = np.arange(N)
        X = np.cos(2 * np.pi * ns / N) + 0.5 * np.sin(4 * np.pi * ns / N)

        fwd = X[1] - X[0]
        bwd = X[0] - X[N - 1]

        print(f"{N:6d}  {fwd:14.10f}  {bwd:14.10f}  {fwd / bwd if bwd != 0 else float('inf'):10.6f}  {abs(fwd - bwd):14.2e}")

    print("\nKey observation: forward and backward differences converge to the same")
    print("value (∂X at the interaction point) as N → ∞, differing by O(1/N) = O(a).")
    print()


# ============================================================
# 5. Completeness relation check
# ============================================================

def test_completeness():
    """
    Verify U1 U1^dag + U2 U2^dag + xi_hat xi_hat^dag = I_{N3-1}.
    """
    print("=" * 70)
    print("5. OVERLAP COMPLETENESS RELATION")
    print("=" * 70)

    for N1, N2 in [(3, 4), (5, 7), (10, 15), (20, 30)]:
        N3 = N1 + N2
        U1, U2, xi = build_overlap_matrices(N1, N2)

        # The exact identity ||xi||^2 = N1 N2 / N3 matches the note's
        # normalization \hat\xi = sqrt(N3/(N1 N2)) xi.
        xi_hat = np.sqrt(N3 / (N1 * N2)) * xi

        # Completeness: U1 U1^dag + U2 U2^dag + xi_hat xi_hat^dag = I
        LHS = U1 @ U1.conj().T + U2 @ U2.conj().T + np.outer(xi_hat, xi_hat.conj())
        RHS = np.eye(N3 - 1)

        err = np.max(np.abs(LHS - RHS))
        print(f"N1={N1:3d}, N2={N2:3d}: max|U1U1†+U2U2†+ξ̂ξ̂†-I| = {err:.2e}")

    print()


# ============================================================
# 6. Zero-point energy: detailed convergence analysis
# ============================================================

def test_zpe_detailed():
    """
    Detailed analysis of the renormalized weighted zero-point mismatch.
    """
    print("=" * 70)
    print("6. RENORMALIZED ZPE MISMATCH: DIVERGENT AND FINITE PARTS")
    print("=" * 70)

    # For fixed alpha1 = 1.0, alpha2 = 1.5:
    alpha1 = 1.0
    alpha2 = 1.5
    alpha3 = alpha1 + alpha2
    alpha_prime = 1.0

    continuum = zpe_mismatch_continuum(alpha1, alpha2, alpha_prime)

    print(f"\nalpha1={alpha1}, alpha2={alpha2}, alpha3={alpha3}")
    print(f"Continuum finite mismatch: {continuum:.10f}")

    # Asymptotic expansion: cot(pi/(2N))/N = 2/pi - pi/(6N^2) - pi^3/(360N^4) - ...
    # Leading term: (2/pi) [1/N3 - 1/N1 - 1/N2] * 1/a^2
    # which in terms of alpha: (2/pi) [1/alpha3 - 1/alpha1 - 1/alpha2] / a
    # This is divergent as a → 0. It's the cosmological constant.

    # The FINITE part comes from the pi/(6N^2) term:
    # (pi/6) [1/N1^2 + 1/N2^2 - 1/N3^2] * 1/a^2
    # = (pi/6) [1/alpha1^2 + 1/alpha2^2 - 1/alpha3^2]  (independent of a!)

    finite_pred = (np.pi / 6.0) * (1.0/alpha1**2 + 1.0/alpha2**2 - 1.0/alpha3**2)

    print(f"\nPredicted finite part (pi/6 * [...]): {finite_pred:.10f}")
    print(f"Continuum value:                      {continuum:.10f}")
    print(f"Match: {np.isclose(finite_pred, continuum)}")

    # Track convergence rate
    print(
        f"\n{'N1':>6s} {'Raw':>14s} {'Renorm.':>14s} {'Error':>14s} {'Error*N1^2':>14s}"
    )
    print("-" * 76)

    for scale in [10, 20, 50, 100, 200, 500, 1000]:
        N1 = scale
        N2 = int(alpha2 / alpha1 * N1)
        a = alpha1 / N1

        raw = zpe_mismatch(N1, N2, a)
        renorm = renormalized_zpe_mismatch(N1, N2, a)
        err = renorm - continuum

        print(
            f"{N1:6d} {raw:14.10f} {renorm:14.10f} "
            f"{err:14.2e} {err * N1**2:14.6f}"
        )

    print("\nThe error × N1² approaches a constant, confirming O(1/N²) = O(a²) convergence.")
    print()


# ============================================================
# 7. The Lorentz anomaly: critical dimension check
# ============================================================

def test_critical_dimension():
    """
    Verify that the lattice zero-point energy reproduces the continuum
    intercept relation a_L = D_perp / 24, hence D = 26 for the bosonic
    closed string.

    For one transverse dimension and both chiralities,
        E_0(N, a) = (1 / a) cot(pi / (2N))
                  = 2N / (pi a) - pi / (6 a N) + O(N^{-3} / a).
    With alpha = a N fixed, the first term is the removable linear
    divergence, while the finite piece is -pi / (6 alpha). Dividing by two
    gives the one-chirality result -pi / (12 alpha), which implies
        a_L = D_perp / 24.
    """
    print("=" * 70)
    print("7. CRITICAL DIMENSION FROM LATTICE ZERO-POINT ENERGY")
    print("=" * 70)

    print("\nExtracting the intercept from the discrete zero-point energy:")
    print("(using zeta-function subtraction of the linearly divergent piece)")
    print()

    print(f"{'N':>6s}  {'Sum':>16s}  {'Linear div.':>16s}  {'Finite part':>16s}  {'Pred. -pi/(6alpha)':>18s}")
    print("-" * 80)

    alpha = 1.0  # circumference
    for N in [10, 20, 50, 100, 200, 500, 1000, 2000]:
        a = alpha / N

        # Direct half-sum over the nonzero lattice frequencies.
        ms = np.arange(1, N)
        E0_half = 0.5 * np.sum(lattice_freq(ms, N, a))  # (1/2) sum omega

        # (1/2) sum omega = (1/a) cot(pi/(2N)) ≈ 2N/(pi*a) - pi/(6aN) - ...
        linear_div = 2.0 * N / (np.pi * a)

        finite_part = E0_half - linear_div
        predicted = -np.pi / (6.0 * alpha)

        print(f"{N:6d}  {E0_half:16.6f}  {linear_div:16.6f}  {finite_part:16.10f}  {predicted:18.10f}")

    predicted = -np.pi / (6.0 * alpha)
    print(f"\nContinuum prediction: -pi/(6*alpha) = {predicted:.10f}")
    print(f"Extracted intercept per chirality per D_perp: a_0 = 1/24 = {1.0/24:.10f}")
    print(f"For D = 26: D_perp = 24, total intercept per chirality = 24/24 = 1 ✓")
    print()

    # Now show the Lorentz anomaly cancellation
    print("LORENTZ ANOMALY CANCELLATION:")
    print("  For one chiral sector of the continuum closed bosonic string,")
    print("    Δ_m = m·(26-D)/12 + (1/m)·(a_L - 1).")
    print("  The lattice zero-point computation determines the intercept input")
    print("    a_L -> D_perp/24 = (D-2)/24")
    print("  in the continuum limit.")
    print("  Therefore at D = 26 one gets a_L = 1, so both anomaly coefficients vanish.")
    print("  This script checks that intercept input; it does not by itself derive")
    print("  the full discrete commutator formula for Δ_m.")
    print()


# ============================================================
# 8. Weighted Hamiltonian mismatch
# ============================================================

def test_weighted_hamiltonian_mismatch():
    """
    Compute the weighted Hamiltonian mismatch (including 1/p_r^+).

    This diagnostic is meant to show that the weighted mismatch is subtler than
    the exact unweighted join-local bond identity. The raw weighted quantity is
    not itself a pure local bond term, so the full operator-level Lorentz check
    cannot be replaced by the simple site-space locality argument alone.
    """
    print("=" * 70)
    print("8. WEIGHTED HAMILTONIAN MISMATCH")
    print("=" * 70)

    N1, N2 = 7, 5
    N3 = N1 + N2
    np.random.seed(42)
    a = 1.0
    alpha_prime = 1.0

    # Random field configuration
    X1 = np.random.randn(N1)
    X2 = np.random.randn(N2)
    X3 = np.concatenate([X1, X2])

    # Momenta satisfying the overlap: P1 = P1^T P3, P2 = P2^T P3
    P3 = np.random.randn(N3)
    P1 = P3[:N1]
    P2 = P3[N1:]

    # Bond sums
    def bond_sum(X, N):
        return sum((X[(n + 1) % N] - X[n])**2 for n in range(N))

    def mom_sum(P):
        return np.sum(P**2)

    # p_r^+ = a*N_r/(2*alpha')
    p1_plus = a * N1 / (2 * alpha_prime)
    p2_plus = a * N2 / (2 * alpha_prime)
    p3_plus = a * N3 / (2 * alpha_prime)

    # Unweighted bond mismatch
    B1, B2, B3 = bond_sum(X1, N1), bond_sum(X2, N2), bond_sum(X3, N3)
    Delta_bond_unweighted = B1 + B2 - B3

    # Weighted bond mismatch: B_r / p_r^+
    Delta_bond_weighted = B1/p1_plus + B2/p2_plus - B3/p3_plus

    # Unweighted momentum: cancels exactly
    M1, M2, M3 = mom_sum(P1), mom_sum(P2), mom_sum(P3)
    Delta_mom_unweighted = M1 + M2 - M3

    # Weighted momentum: does NOT cancel
    Delta_mom_weighted = M1/p1_plus + M2/p2_plus - M3/p3_plus

    print(f"\nN1={N1}, N2={N2}")
    print(f"Unweighted bond mismatch:    {Delta_bond_unweighted:.10f} (join-local)")
    print(f"Unweighted mom mismatch:     {Delta_mom_unweighted:.10f} (zero)")
    print(f"Weighted bond mismatch:      {Delta_bond_weighted:.10f} (NOT join-local)")
    print(f"Weighted mom mismatch:       {Delta_mom_weighted:.10f} (NOT zero)")

    # But the point is: the OSCILLATOR Hamiltonian mismatch IS join-local
    # even with weights, because:
    # H_osc,r = H_r - (p_perp^(r))^2 * pi*alpha'/(a*N_r)
    # H_osc,r / p_r^+ = H_r/p_r^+ - p_perp^2/(2*N_r * p_r^+) * stuff
    # The p_perp part is handled separately.

    # Separate zero mode and oscillator:
    p_perp_1 = np.sum(P1)
    p_perp_2 = np.sum(P2)
    p_perp_3 = np.sum(P3)  # = p_perp_1 + p_perp_2

    P1_osc = P1 - p_perp_1/N1
    P2_osc = P2 - p_perp_2/N2
    P3_osc = P3 - p_perp_3/N3

    M1_osc = mom_sum(P1_osc)
    M2_osc = mom_sum(P2_osc)
    M3_osc = mom_sum(P3_osc)

    Delta_mom_osc_unweighted = M1_osc + M2_osc - M3_osc
    Delta_mom_osc_weighted = M1_osc/p1_plus + M2_osc/p2_plus - M3_osc/p3_plus

    print(f"\nOscillator-only momentum mismatch (unweighted): {Delta_mom_osc_unweighted:.10f}")
    print(f"Oscillator-only momentum mismatch (weighted):   {Delta_mom_osc_weighted:.10f}")

    # The oscillator momentum mismatch is NOT zero (neither weighted nor unweighted)
    # This is because the oscillator momenta on different legs are related by:
    # P_n'^(1) = P_n^(1) - p_perp^(1)/N1 = P3_n - p_perp_1/N1  (for n < N1)
    # P_n'^(3) = P_n^(3) - p_perp_3/N3
    # These don't satisfy P1_osc = P1^T P3_osc in general because of the
    # different zero-mode subtractions!

    # The correct statement is: the FULL Hamiltonian including both kinetic
    # and potential, when evaluated on the overlap state, gives a join-local
    # mismatch. The proof requires using BOTH kinetic and potential together.

    # Full oscillator Hamiltonian:
    coeff_mom = np.pi * alpha_prime / a
    coeff_bond = 1.0 / (4 * np.pi * alpha_prime * a)

    H1_osc = coeff_mom * M1_osc + coeff_bond * B1
    H2_osc = coeff_mom * M2_osc + coeff_bond * B2
    H3_osc = coeff_mom * M3_osc + coeff_bond * B3

    Delta_H_osc_unweighted = H1_osc + H2_osc - H3_osc

    # This should be join-local! Let me verify by showing it only depends
    # on local join variables.
    X_Ip = X1[0]
    X_Im = X2[0]
    delta_I = X_Ip - X_Im

    # From the note: Delta_bond = 2 * delta_I * (X_{N2-1}^(2) - X_{N1-1}^(1))
    Delta_bond_pred = 2 * delta_I * (X2[N2-1] - X1[N1-1])

    print(f"\nFull oscillator H mismatch (unweighted): {Delta_H_osc_unweighted:.10f}")
    print(f"Bond-only contribution:                  {coeff_bond * Delta_bond_pred:.10f}")

    # The oscillator momentum mismatch contributes too, but it should
    # also be join-local. Let me check.
    print(f"Momentum contribution:                   {coeff_mom * Delta_mom_osc_unweighted:.10f}")
    print(f"Total = bond + mom:                      {coeff_bond * (B1+B2-B3) + coeff_mom * (M1_osc+M2_osc-M3_osc):.10f}")
    print(f"Are bond and mom contributions separately join-local? Check numerically...")

    # Actually, the oscillator momenta P'_n satisfy:
    # sum P'_n^(1) = 0, sum P'_n^(2) = 0, sum P'_n^(3) = 0
    # And on the overlap: P^(1) = P1^T P^(3), so P'_n^(1) = P^(3)_n - p_perp_1/N1
    # But P'_n^(3) = P^(3)_n - p_perp_3/N3
    # So P'_n^(1) = P'_n^(3) + p_perp_3/N3 - p_perp_1/N1
    # This offset means the oscillator momentum overlap is shifted.
    # The quadratic form sum P'^2 picks up cross terms from this shift.

    # The full unweighted mismatch H1+H2-H3 = coeff_mom*(M1_osc+M2_osc-M3_osc) + coeff_bond*(B1+B2-B3)
    # should be join-local because the FULL site-space mismatch is join-local.

    # Let me verify: the FULL site-space Hamiltonian mismatch (including zero modes):
    H1_full = coeff_mom * M1 + coeff_bond * B1
    H2_full = coeff_mom * M2 + coeff_bond * B2
    H3_full = coeff_mom * M3 + coeff_bond * B3

    Delta_H_full = H1_full + H2_full - H3_full

    # M1 + M2 = M3 on overlap, so the momentum part cancels:
    print(f"\nFull H mismatch (unweighted):             {Delta_H_full:.10f}")
    print(f"Bond-only formula:                        {coeff_bond * Delta_bond_pred:.10f}")
    print(f"Match: {np.isclose(Delta_H_full, coeff_bond * Delta_bond_pred)}")

    print("\n=> The unweighted Hamiltonian mismatch IS purely join-local (bond term).")
    print("   The weighted mismatch (H/p^+) includes non-local corrections from the")
    print("   1/p^+ weighting, but these are part of the zero-mode sector handled by")
    print("   the kinematic Lorentz generators. Only the oscillator part matters for")
    print("   the dynamical Lorentz check.")
    print()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("DISCRETE LORENTZ INVARIANCE CHECK")
    print("Lightcone Mandelstam vertex with discrete sigma")
    print("=" * 70)
    print()

    test_bond_locality()
    test_completeness()
    test_zpe_convergence()
    test_zpe_detailed()
    test_critical_dimension()
    test_neumann_convergence()
    test_forward_backward()
    test_weighted_hamiltonian_mismatch()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The current discrete Lorentz diagnostics establish:

1. BOND-LOCALITY: The Hamiltonian mismatch on the overlap state is
   purely join-local (supported at the two bonds adjacent to the cut).
   This is EXACT at finite N, not just asymptotic.

2. ZPE CONVERGENCE: After subtracting the universal weighted vacuum-energy
   divergence, the renormalized three-string zero-point mismatch converges
   to the continuum (zeta-regularized) value at rate O(1/N²) = O(a²).

3. CRITICAL DIMENSION: The renormalized intercept per chiral sector is
   a_L = D_perp/24 = (D-2)/24. Setting a_L = 1 gives D = 26. This
   is reproduced by the discrete calculation in the continuum limit.

4. LOW-MODE QUADRATIC DATA: Representative constrained quadratic entries
   stabilize with N, but this script does not yet provide a convention-matched
   continuum Neumann-coefficient comparison.

5. FORWARD/BACKWARD: Forward and backward arc differences converge
   to the same interaction-point derivative, differing by O(a).

6. FINITE-N CORRECTIONS: At finite N, the Lorentz algebra does not
   close exactly (the lattice breaks conformal invariance). The
   violations are:
   (a) Join-local (by the exact overlap)
   (b) Partly visible in the renormalized ZPE and local-stencil diagnostics
   (c) Not yet fully classified by an operator-level commutator computation

This script therefore supplies useful partial evidence for the expected
continuum D = 26 Lorentz structure, but it is not by itself a complete
operator-level proof of cubic Lorentz invariance.
""")
