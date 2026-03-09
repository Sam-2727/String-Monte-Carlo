import numpy as np
import partition_function as pf
import sympy as sp
from typing import Callable, Iterable, Sequence, Tuple

import mpmath as mp
from scipy.integrate import quad


def dedekind_eta(tau: complex) -> complex:
    tau = mp.mpc(tau)
    if mp.im(tau) <= 0:
        raise ValueError("Need Im(tau) > 0.")
    q = mp.e**(2*mp.pi*1j*tau)
    eta = mp.e**(mp.pi*1j*tau/12) * mp.qp(q)  # <-- key change
    return complex(eta)

def Z(tau: complex, prec: int = 50) -> float:
    tau = complex(tau)
    eta = dedekind_eta(tau)
    return (tau.imag ** (-13.0)) * (abs(eta) ** (-48.0))

def theta3_eta_sqrt(L: int, l1: int, l2: int, n: int = 3):
    """
    Compute sqrt(theta_n(0, tau) / eta(tau)) where tau = P2/P1
    from the periods of the Strebel differential with parameters (L, l1, l2).

    n selects the Jacobi theta function (1-4).
    Uses mpmath.jtheta and the existing dedekind_eta helper.
    """
    P1, P2, P3 = periods_improved(L, l1, l2)
    tau = P2 / P1

    tau_mp = mp.mpc(tau)
    nome = mp.exp(mp.pi * 1j * tau_mp)
    theta_n = mp.jtheta(n, 0, nome)
    eta = mp.mpc(dedekind_eta(tau))

    return abs(theta_n / eta)


def make_cyl_eqn(L: int, l1: int, l2: int, *,
                      dtype=np.complex128,
                      chop_tol: float = 1e-12,
                      use_cholesky: bool = True):
    """
    Optimized factory for the CylEqn polynomial, mirroring Mathematica CylEqnFastOptimized.

    Builds the overdetermined matrix B (m x m, with m=L/2),
    forms gram = B^H B (m x m Hermitian),
    adds +1 to gram[0,0] to impose A1=1 in normal-equation form,
    solves gram * coeffs = e1,
    returns callable f(z) = Σ coeffs[n] z^n  (n=0..m-1) with Chop-like cleanup.

    Notes:
      - This uses normal equations, which are fast but less numerically stable than lstsq.
      - If you see sensitivity, set use_cholesky=False and switch to np.linalg.solve,
        or use the slower/safer np.linalg.lstsq approach.
    """
    if L % 2 != 0:
        raise ValueError("L must be even.")
    m = L // 2
    if l1 + l2 > m:
        raise ValueError("Need l1 + l2 <= L/2.")

    # nList = 1..m
    n = np.arange(1, m + 1, dtype=np.float64)

    # anglesL = piecewise mapped angles (length m)
    # Mathematica:
    # 2π*(m + l1 + 1 - k)/L, k=1..l1
    # 2π*(m + 2l1 + l2 + 1 - k)/L, k=l1+1..l1+l2
    # 2π*(L + l1 + l2 + 1 - k)/L, k=l1+l2+1..m
    k = np.arange(1, m + 1, dtype=np.float64)
    mapped = np.empty_like(k)

    if l1 > 0:
        mapped[:l1] = (m + l1 + 1) - k[:l1]
    if l2 > 0:
        mapped[l1:l1 + l2] = (m + 2 * l1 + l2 + 1) - k[l1:l1 + l2]
    if l1 + l2 < m:
        mapped[l1 + l2:] = (L + l1 + l2 + 1) - k[l1 + l2:]

    twopi_over_L = 2.0 * np.pi / L
    anglesL = twopi_over_L * mapped          # length m
    anglesR = twopi_over_L * k               # length m

    # B = exp(i outer(anglesL, n)) + exp(i outer(anglesR, n))
    # (your Mathematica uses Take[anglesR, Length[anglesL]]; lengths are both m here)
    B = np.exp(1j * anglesL[:, None] * n[None, :]) + np.exp(1j * anglesR[:, None] * n[None, :])
    B = B.astype(dtype, copy=False)

    # gram = B^H B
    gram = (B.conj().T @ B).astype(dtype, copy=False)

    # gram[[1,1]] += 1  (Mathematica indexing) => gram[0,0] += 1
    gram[0, 0] += 1.0

    # rhs = e1
    rhs = np.zeros(m, dtype=dtype)
    rhs[0] = 1.0

    # Solve gram * coeffs = rhs
    if use_cholesky:
        # Cholesky is fastest if gram is HPD (it should be in this construction).
        Lc = np.linalg.cholesky(gram)               # gram = Lc Lc^H
        y = np.linalg.solve(Lc, rhs)
        coeffs = np.linalg.solve(Lc.conj().T, y)
    else:
        coeffs = np.linalg.solve(gram, rhs)

    # Chop coeffs once
    coeffs = coeffs.copy()
    coeffs.real[np.abs(coeffs.real) < chop_tol] = 0.0
    coeffs.imag[np.abs(coeffs.imag) < chop_tol] = 0.0
    
    #FIGURE THIS OUT LATER -- COULD BE WRONG
    coeffs[-1]=0.0

    # Fast polynomial evaluation (Horner): Σ_{n=0..m-1} coeffs[n] z^n
    def f(z):
        z = np.complex128(z)
        acc = np.complex128(0.0)
        # Horner for ascending powers: reverse coefficients
        for c in coeffs[::-1]:
            acc = acc * z + c
        # Chop output
        re = 0.0 if abs(acc.real) < chop_tol else acc.real
        im = 0.0 if abs(acc.imag) < chop_tol else acc.imag
        return np.complex128(re + 1j * im)

    # Optional metadata
    f.L = L
    f.m = m
    f.l1 = l1
    f.l2 = l2
    f.coeffs = coeffs
    return f

def make_cyl_eqn_improved(L: int, l1: int, l2: int, *,
                           dtype=np.complex128,
                           chop_tol: float = 1e-12):
    """
    Python translation of Mathematica CylEqnImproved.

    Improved version of make_cyl_eqn that factors out the cube-root
    singularities analytically via an 'improvement' pre-factor, and
    uses a half-step angle shift (-π/L).

    Returns a callable f(z) -> (singular, polynomial) where the full
    cylinder equation is singular * polynomial.
    """
    if L % 2 != 0:
        raise ValueError("L must be even.")
    m = L // 2
    l3 = m - l1 - l2
    if l3 < 0:
        raise ValueError("Need l1 + l2 <= L/2.")

    twopi_over_L = 2.0 * np.pi / L
    half_step = np.pi / L

    # Special case: all three lengths equal
    if l1 == l2 == l3:
        w1 = np.exp(2j * (-twopi_over_L * l1))
        w2 = np.exp(2j * (-twopi_over_L * (l1 + l2)))
        def f_equal(z):
            s = (1 - z**2)**(-1/3) * (1 - z**2 * w1)**(-1/3) * (1 - z**2 * w2)**(-1/3)
            return (s, 1.0)
        f_equal.L, f_equal.l1, f_equal.l2, f_equal.m = L, l1, l2, m
        f_equal.coeffs = np.array([1.0])
        return f_equal

    # Parity shift: +1 if all odd, -1 if all even, 0 otherwise
    shift = ((l1 % 2) * (l2 % 2) * (l3 % 2)
             - ((l1 + 1) % 2) * ((l2 + 1) % 2) * ((l3 + 1) % 2))
    n_coeffs = m - shift
    n_list = np.arange(1, n_coeffs + 1, dtype=np.float64)

    # Angles with half-step shift
    k = np.arange(1, m + 1, dtype=np.float64)
    mapped = np.empty_like(k)
    if l1 > 0:
        mapped[:l1] = (m + l1 + 1) - k[:l1]
    if l2 > 0:
        mapped[l1:l1 + l2] = (m + 2 * l1 + l2 + 1) - k[l1:l1 + l2]
    if l1 + l2 < m:
        mapped[l1 + l2:] = (L + l1 + l2 + 1) - k[l1 + l2:]

    angles_L = twopi_over_L * mapped - half_step
    angles_R = twopi_over_L * k - half_step

    # Improvement (singular) factor
    phase1 = twopi_over_L * l1
    phase2 = twopi_over_L * (l1 + l2)

    if (l3 == 0) or (l2==0) or (l1==0):
        def improvement(angles):
            return ((1 - np.exp(2j * angles))**(-1/2) *
                    (1 - np.exp(2j * (angles - phase1)))**(-1/2))
    else:
        def improvement(angles):
            return ((1 - np.exp(2j * angles))**(-1/3) *
                    (1 - np.exp(2j * (angles - phase1)))**(-1/3) *
                    (1 - np.exp(2j * (angles - phase2)))**(-1/3))

    imp_L = improvement(angles_L)
    imp_R = improvement(angles_R[:len(angles_L)])

    # B matrix: diag(imp) @ exp(i * outer(angles, nList))
    exp_L = np.exp(1j * np.outer(angles_L, n_list))
    exp_R = np.exp(1j * np.outer(angles_R[:len(angles_L)], n_list))
    B = (imp_L[:, None] * exp_L + imp_R[:, None] * exp_R).astype(dtype)

    # Gram matrix (plain transpose, matching Mathematica Transpose[B].B)
    gram = (B.T @ B).astype(dtype)
    gram[0, 0] += 1.0

    # Solve gram * coeffs = e1
    rhs = np.zeros(n_coeffs, dtype=dtype)
    rhs[0] = 1.0
    coeffs = np.linalg.solve(gram, rhs)

    # Chop small values
    coeffs = coeffs.copy()
    coeffs.real[np.abs(coeffs.real) < chop_tol] = 0.0
    coeffs.imag[np.abs(coeffs.imag) < chop_tol] = 0.0

    # Precompute singular factor phases
    w1 = np.exp(2j * (-phase1))
    w2 = np.exp(2j * (-phase2))

    def f(z):
        z = np.complex128(z)
        if l3 == 0:
            singular = ((1 - z**2)**(-1/2) *
                        (1 - z**2 * w1)**(-1/2))
        else:
            singular = ((1 - z**2)**(-1/3) *
                        (1 - z**2 * w1)**(-1/3) *
                        (1 - z**2 * w2)**(-1/3))
        # Horner evaluation of polynomial
        acc = np.complex128(0.0)
        for c in coeffs[::-1]:
            acc = acc * z + c
        re = 0.0 if abs(acc.real) < chop_tol else acc.real
        im = 0.0 if abs(acc.imag) < chop_tol else acc.imag
        poly = np.complex128(re + 1j * im)
        return (singular, poly)

    f.L = L
    f.m = m
    f.l1 = l1
    f.l2 = l2
    f.coeffs = coeffs
    return f

def periods_improved(L: int, l1: int, l2: int, f=None):
    """
    Python translation of Mathematica PeriodsImproved.

    Computes three period integrals by numerically integrating
    f(z) (from make_cyl_eqn_improved) along straight-line paths
    from 0 to each of three PreCorner points on the unit circle.

    Uses scipy.integrate.quad since the integrand has (1-z²)^{-1/3}
    singularities and no closed-form antiderivative.

    Returns: (P1, P2, P3) complex128.
    """
    if L % 2 != 0:
        raise ValueError("L must be even.")
    m = L // 2
    l3 = m - l1 - l2
    if l3 < 0:
        raise ValueError("Need l1 + l2 <= L/2.")
    if f is None:
        f = make_cyl_eqn_improved(L, l1, l2)

    twopi_over_L = 2.0 * np.pi / L
    theta1 = twopi_over_L * l1
    theta2 = twopi_over_L * l2

    pre_corners = [
        np.complex128(1.0),                       # singularity of (1-z²)^{-1/3}
        np.exp(1j * (theta1 + theta2)),            # singularity of 2nd factor
        np.exp(1j * (np.pi + theta1)),             # singularity of 3rd factor
    ]

    def integrate_to(p):
        """∫₀ᵖ f(z) dz  via parameterization z(t)=t·p, t∈[0,1]."""
        def integrand_re(t):
            singular, poly = f(t * p)
            return (singular * poly * p).real
        def integrand_im(t):
            singular, poly = f(t * p)
            return (singular * poly * p).imag
        re_part, _ = quad(integrand_re, 0, 1, limit=200)
        im_part, _ = quad(integrand_im, 0, 1, limit=200)
        return np.complex128(re_part + 1j * im_part)

    corners = [integrate_to(p) for p in pre_corners]

    P1 = corners[0] - corners[2]
    P2 = corners[1] - corners[2]
    P3 = corners[0] - corners[1]
    return (P1, P2, P3)

def periods(L: int,
                         l1: int,
                         l2: int,
                         m1: int | None = None,
                         m2: int | None = None,
                         m3: int | None = None):
    """
    PeriodsFastGivenF with signature (L, l1, l2, f, m1=None, m2=None, m3=None).

    Defaults if m1/m2/m3 not provided:
        m1 = Round(l1/2),  m2 = Round(l2/2),  m3 = Round(l3/2)
    where Round matches Mathematica (ties-to-even).

    f must have attribute f.coeffs of length m=L/2 for:
        f(z) = Σ_{n=1..m} c_n z^(n-1)

    Returns: (P1, P2, P3) complex128.
    """
    if L % 2 != 0:
        raise ValueError("L must be even.")
    m = L // 2
    l3 = m - l1 - l2
    if l3 < 0:
        raise ValueError("Need l1 + l2 <= L/2 so that l3 >= 0.")

    # Mathematica Round[x] uses ties-to-even; Python round() does too.
    def round_mathematica(x: float) -> int:
        return int(round(x))

    if m1 is None:
        m1 = round_mathematica(l1 / 2)
    if m2 is None:
        m2 = round_mathematica(l2 / 2)
    if m3 is None:
        m3 = round_mathematica(l3 / 2)
    f = make_cyl_eqn(L,l1,l2)
    coeffs = getattr(f, "coeffs", None)
    if coeffs is None:
        raise ValueError("Input function f must have attribute `.coeffs` (from make_cyl_eqn_fast).")
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    if coeffs.shape[0] != m:
        raise ValueError(f"Coefficient length mismatch: expected {m}, got {coeffs.shape[0]}.")

    n = np.arange(1, m + 1, dtype=np.float64)
    anti = coeffs / n  # for F(w) = Σ (c_n/n) w^n = w * Σ anti[j] w^j

    def F(w: np.complex128) -> np.complex128:
        w = np.complex128(w)
        acc = np.complex128(0.0)
        for a in anti[::-1]:
            acc = acc * w + a
        return w * acc

    k1 = int(m1)
    k2 = int(l1 + m2)
    k3 = int(l1 + l2 + m3)

    def exp_angle(s: float) -> np.complex128:
        return np.complex128(np.exp(2j * np.pi * (s / L)))

    w1L = exp_angle(m + l1 + 1 - k1)
    w1R = exp_angle(k1)

    w2L = exp_angle(m + 2 * l1 + l2 + 1 - k2)
    w2R = exp_angle(k2)

    w3L = exp_angle(L + l1 + l2 + 1 - k3)
    w3R = exp_angle(k3)

    return (np.complex128(F(w1L) - F(w1R)),
            np.complex128(F(w2L) - F(w2R)),
            np.complex128(F(w3L) - F(w3R)))

def periods_given_f(L: int,
                         l1: int,
                         l2: int,
                         f,
                         m1: int | None = None,
                         m2: int | None = None,
                         m3: int | None = None):
    """
    PeriodsFastGivenF with signature (L, l1, l2, f, m1=None, m2=None, m3=None).

    Defaults if m1/m2/m3 not provided:
        m1 = Round(l1/2),  m2 = Round(l2/2),  m3 = Round(l3/2)
    where Round matches Mathematica (ties-to-even).

    f must have attribute f.coeffs of length m=L/2 for:
        f(z) = Σ_{n=1..m} c_n z^(n-1)

    Returns: (P1, P2, P3) complex128.
    """
    if L % 2 != 0:
        raise ValueError("L must be even.")
    m = L // 2
    l3 = m - l1 - l2
    if l3 < 0:
        raise ValueError("Need l1 + l2 <= L/2 so that l3 >= 0.")

    # Mathematica Round[x] uses ties-to-even; Python round() does too.
    def round_mathematica(x: float) -> int:
        return int(round(x))

    if m1 is None:
        m1 = round_mathematica(l1 / 2)
    if m2 is None:
        m2 = round_mathematica(l2 / 2)
    if m3 is None:
        m3 = round_mathematica(l3 / 2)

    coeffs = getattr(f, "coeffs", None)
    if coeffs is None:
        raise ValueError("Input function f must have attribute `.coeffs` (from make_cyl_eqn_fast).")
    coeffs = np.asarray(coeffs, dtype=np.complex128)
    if coeffs.shape[0] != m:
        raise ValueError(f"Coefficient length mismatch: expected {m}, got {coeffs.shape[0]}.")

    n = np.arange(1, m + 1, dtype=np.float64)
    anti = coeffs / n  # for F(w) = Σ (c_n/n) w^n = w * Σ anti[j] w^j

    def F(w: np.complex128) -> np.complex128:
        w = np.complex128(w)
        acc = np.complex128(0.0)
        for a in anti[::-1]:
            acc = acc * w + a
        return w * acc

    k1 = int(m1)
    k2 = int(l1 + m2)
    k3 = int(l1 + l2 + m3)

    def exp_angle(s: float) -> np.complex128:
        return np.complex128(np.exp(2j * np.pi * (s / L)))

    w1L = exp_angle(m + l1 + 1 - k1)
    w1R = exp_angle(k1)

    w2L = exp_angle(m + 2 * l1 + l2 + 1 - k2)
    w2R = exp_angle(k2)

    w3L = exp_angle(L + l1 + l2 + 1 - k3)
    w3R = exp_angle(k3)

    return (np.complex128(F(w1L) - F(w1R)),
            np.complex128(F(w2L) - F(w2R)),
            np.complex128(F(w3L) - F(w3R)))

def make_antiderivative_from_f(f):
    """
    Return a callable F(w) = ∫ f(w) dw with integration constant = 0,
    where f(z) = Σ_{n=1..m} c_n z^(n-1) and f.coeffs stores c_n.
    """
    coeffs = getattr(f, "coeffs", None)
    if coeffs is None:
        raise ValueError("f must have attribute `.coeffs` (from make_cyl_eqn_fast).")
    coeffs = np.asarray(coeffs, dtype=np.complex128)

    m = coeffs.shape[0]
    n = np.arange(1, m + 1, dtype=np.float64)
    anti = coeffs / n  # anti[n-1] = c_n / n

    def F(w):
        w = np.complex128(w)
        # F(w) = Σ_{n=1..m} (c_n/n) w^n = w * Σ_{j=0..m-1} anti[j] w^j
        acc = np.complex128(0.0)
        for a in anti[::-1]:
            acc = acc * w + a
        return w * acc

    # attach for debugging
    F.anti_coeffs = anti
    return F

def compute_period_derivative(
    L: int,
    l1: int,
    l2: int,
    m1: int,
    m2: int,
    m3: int,
    eps: int = 1,
) -> np.complex128:
    """
    Integer-safe translation of Mathematica computePeriodDerivativePrime.

    periods_fast is a callable:
        periods_fast(l1, l2, l3, m1, m2, m3) -> (P1, P2, P3)
    with P1,P2,P3 complex.

    eps is an integer (usually 1), so l1±eps etc remain integers.

    Returns:
        np.complex128 value of
          1/(4 epsN^2) (tauL * conj(tauR) - conj(tauL) * tauR)
        where epsN = eps / (L/2) and L = 2(l1+l2+l3).
    """
    if eps <= 0:
        raise ValueError("eps should be a positive integer.")

    # L is preserved by the +/- eps updates
   
    m = L // 2  # = l1+l2+l3
    l3 = m-l1-l2
    if 2 * m != L:
        raise ValueError("L must be even (it is by construction).")

    # Compute perturbed periods
    periodsLp = periods(L,l1+eps, l2, m1, m2, m3)
    periodsLm = periods(L,l1 - eps, l2, m1, m2, m3)
    periodsRp = periods(L, l1, l2 + eps, m1, m2, m3)
    periodsRm = periods(L, l1, l2 - eps, m1, m2, m3)

    # tau = P2/P1 (Mathematica [[2]]/[[1]])
    tauLp = periodsLp[1] / periodsLp[0]
    tauLm = periodsLm[1] / periodsLm[0]
    tauRp = periodsRp[1] / periodsRp[0]
    tauRm = periodsRm[1] / periodsRm[0]

    tauL = tauLp - tauLm
    tauR = tauRp - tauRm

    # epsN = eps/(L/2) = eps/m
    epsN = eps / float(m)

    out = (1.0 / (4.0 * epsN * epsN)) * (tauL * np.conjugate(tauR) - np.conjugate(tauL) * tauR)
    return np.complex128(out)

def pole_intercept(f, k0: int, L: int,
                   alpha_min: int = -40,
                   alpha_max: int = 0,
                   fit_low: int = -4,
                   fit_high: int = -1,
                   step: int = 2) -> float:
    """
    Python analogue of the Mathematica:

      PoleIntercept[f_, k0_, L_] := Module[...; lm["BestFitParameters"][[1]]]

    It builds samples
        w(alpha) = exp(i*pi*(2*k0/L + 2*alpha/L + 1/L))
    and fits
        log|f(w)| = a + b * log|w - w0|
    on alpha in [fit_low, fit_high], returning the intercept a.

    Parameters
    ----------
    f : callable
        The CylEqn-style function; takes complex w and returns complex f(w).
    k0 : int
    L : int
    alpha_min, alpha_max, fit_low, fit_high : int
        Match the Mathematica defaults.
    step : int
        Step for alpha table (Mathematica used 2).

    Returns
    -------
    float
        Intercept a from least-squares fit y = a + b x.
    """
    # Base point w0 = exp(i*pi*(2*k0+1)/L)
    w0 = np.exp(1j * np.pi * (2.0 * k0 + 1.0) / L)

    xs = []
    ys = []

    for alpha in range(alpha_min, alpha_max + 1, step):
        w = np.exp(1j * np.pi * (2.0 * k0 / L + 2.0 * alpha / L + 1.0 / L))
        result = f(w)
        if isinstance(result, tuple):
            singular, poly = result
            fw = singular * poly
        else:
            fw = result

        if fit_low <= alpha <= fit_high:
            dx = abs(w - w0)
            dy = abs(fw)

            # avoid log(0)
            if dx > 0 and dy > 0:
                xs.append(np.log(dx))
                ys.append(np.log(dy))

    if len(xs) < 2:
        raise ValueError("Not enough points in fit window to perform linear regression.")

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)

    # Fit y = a + b x via least squares
    A = np.vstack([np.ones_like(x), x]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    return float(a)

def pole_intercept_average(f, L:int,l1: int, l2: int,
                           alpha_min: int = -40,
                           alpha_max: int = 0,
                           fit_low: int = -4,
                           fit_high: int = -1,
                           step: int = 2) -> float:
    """
    Python analogue of Mathematica PoleInterceptAverage[f,l1,l2,l3].

    For L = 2*(l1+l2+l3) and
      ells = {l1, l1+l2+l3, 2*l1+2*l2+l3},
    it computes three PoleIntercept-style log-log fits around the three
    corresponding boundary points and returns the mean intercept.

    Fit model (same as before):
      log|f(w)| = a + b * log|w - w0|
    using alpha in [fit_low, fit_high], returning mean(a) over i=1..3.

    Parameters mirror the Mathematica defaults.
    """
    m = L // 2  # = l1+l2+l3
    l3 = m-l1-l2
    if 2 * m != L:
        raise ValueError("L must be even (it is by construction).")
    ells = [l1, l1 + l2 + l3, 2 * l1 + 2 * l2 + l3]

    intercepts = []

    for ell in ells:
        # base point for this ell
        w0 = np.exp(1j * np.pi * (2.0 * ell + 1.0) / L)

        xs = []
        ys = []

        for alpha in range(alpha_min, alpha_max + 1, step):
            w = np.exp(1j * np.pi * (2.0 * ell / L + 2.0 * alpha / L + 1.0 / L))
            result = f(w)
            if isinstance(result, tuple):
                singular, poly = result
                fw = singular * poly
            else:
                fw = result

            if fit_low <= alpha <= fit_high:
                dx = abs(w - w0)
                dy = abs(fw)
                if dx > 0 and dy > 0:
                    xs.append(np.log(dx))
                    ys.append(np.log(dy))

        if len(xs) < 2:
            raise ValueError(f"Not enough fit points for ell={ell}.")
        x = np.asarray(xs, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)

        # least squares fit y = a + b x
        A = np.vstack([np.ones_like(x), x]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        intercepts.append(float(a))

    return float(np.mean(intercepts))

def boundary_intercepts(L: int, l1: int, l2: int,
                        m1: int, m2: int, m3: int,
                        *,
                        make_cyl_eqn,
                        pole_intercept_average):
    """
    Python analogue of Mathematica:

      BoundaryIntercepts[l1,l2,l3,m1,m2,m3] := {int1, p2/p1, p3/p1, (p2/p1 - p3/p1), p1}

    Inputs
    ------
    l1,l2,l3 : ints
    m1,m2,m3 : ints (these are the "m1,m2,m3" in Mathematica)
    make_cyl_eqn : function
        Should build the CylEqnFastOptimized-like function for this geometry.
        Expected signature: make_cyl_eqn(L:int, l1:int, l2:int) -> object f
        where f is callable f(z) and has attribute f.coeffs of length m=L/2
        for f(z)=Σ_{n=1..m} c_n z^(n-1) with A[_]->0 already enforced.
    pole_intercept_average : function
        The Python PoleInterceptAverage(f,l1,l2,l3) you implemented.

    Returns
    -------
    (int1, tau2, tau3, tau2_minus_tau3, period1) as complex/float.
    """
    m = L // 2  # = l1+l2+l3
    l3 = m-l1-l2
    if 2 * m != L:
        raise ValueError("L must be even (it is by construction).")

    # Map indices exactly like Mathematica
    k1 = int(m1)
    k2 = int(l1 + m2)
    k3 = int(l1 + l2 + m3)

    # Build cyl (A[_]->0 already) as a callable with coeffs
    f = make_cyl_eqn(L=L, l1=l1, l2=l2)

    coeffs = getattr(f, "coeffs", None)
    if coeffs is None or len(coeffs) != m:
        raise ValueError("make_cyl_eqn must return callable with .coeffs length L/2.")

    coeffs = np.asarray(coeffs, dtype=np.complex128)

    # Antiderivative coefficients for F(z)=∫f(z)dz = Σ (c_n/n) z^n  (n=1..m)
    n = np.arange(1, m + 1, dtype=np.float64)
    anti = coeffs / n  # length m

    def F(w: np.complex128) -> np.complex128:
        """
        Evaluate F(w)=Σ_{n=1..m} (c_n/n) w^n using Horner:
          Σ anti[n-1] w^n = w * Σ anti[n-1] w^(n-1)
        """
        w = np.complex128(w)
        acc = np.complex128(0.0)
        for a in anti[::-1]:
            acc = acc * w + a
        return w * acc

    def exp_angle(s: float) -> np.complex128:
        return np.complex128(np.exp(2j * np.pi * (s / L)))

    # Period endpoints exactly as in Mathematica:
    # period1 = F[exp(2πi*(L/2 + l1 - k1 + 1)/L)] - F[exp(2πi*k1/L)]
    w1L = exp_angle((L/2) + l1 - k1 + 1)
    w1R = exp_angle(k1)
    period1 = F(w1L) - F(w1R)

    # period2 = F[exp(2πi*(L/2 + 2l1 + l2 - k2 + 1)/L)] - F[exp(2πi*k2/L)]
    w2L = exp_angle((L/2) + 2*l1 + l2 - k2 + 1)
    w2R = exp_angle(k2)
    period2 = F(w2L) - F(w2R)

    # period3 = F[exp(2πi*(L + l1 + l2 - k3 + 1)/L)] - F[exp(2πi*k3/L)]
    w3L = exp_angle(L + l1 + l2 - k3 + 1)
    w3R = exp_angle(k3)
    period3 = F(w3L) - F(w3R)

    # int1 = PoleInterceptAverage[f, l1, l2, l3]
    int1 = pole_intercept_average(f, l1, l2, l3)

    # Chop analogue: zero tiny numerical noise
    def chop(z, tol=1e-12):
        z = complex(z)
        re = 0.0 if abs(z.real) < tol else z.real
        im = 0.0 if abs(z.imag) < tol else z.imag
        return re + 1j * im

    tau2 = chop(period2 / period1)
    tau3 = chop(period3 / period1)
    diff = chop(tau2 - tau3)

    return (float(int1), tau2, tau3, diff, chop(period1))

def compute_rhs(L: int,
                l1: int,
                l2: int,
                m1: int | None = None,
                m2: int | None = None,
                m3: int | None = None):
    """
    computeRHS with signature (L, l1, l2, m1=None, m2=None, m3=None),
    calling helper functions by their known global names.

    Uses:
      make_cyl_eqn
      periods_given_f
      pole_intercept_average
      compute_period_derivative_prime_int (and constructs periods_fast internally)
      Z_of_tau
    """
    if L % 2 != 0:
        raise ValueError("L must be even.")
    half = L // 2
    l3 = half - l1 - l2
    if l3 < 0:
        raise ValueError("Need l1 + l2 <= L/2 so that l3 >= 0.")

    # Mathematica Round ties-to-even; Python round() does too.
    def round_mathematica(x: float) -> int:
        return int(round(x))

    if m1 is None:
        m1 = round_mathematica(l1 / 2)
    if m2 is None:
        m2 = round_mathematica(l2 / 2)
    if m3 is None:
        m3 = round_mathematica(l3 / 2)

    # Build f
    f = make_cyl_eqn_improved(L=L, l1=l1, l2=l2)  # must be callable and have f.coeffs

    # Periods (PeriodsFastGivenF analogue)
    P1, P2, P3 = periods_improved(L=L, l1=l1, l2=l2, f=f)
    tau = P2 / P1

    # fZero = Abs[(f/P1) at z=0]
    coeffs = getattr(f, "coeffs", None)
    if coeffs is None:
        raise ValueError("make_cyl_eqn must return an object with attribute `.coeffs`.")
    f_at_0 = np.complex128(coeffs[0])  # f(0)
    fZero = np.abs(f_at_0 / P1)
    fZero_inv_sq = fZero ** (-2)

    # normalized f2(c) = f(c)/P1
    def f2(c):
        result = f(c)
        if isinstance(result, tuple):
            singular, poly = result
            return np.complex128(singular * poly / P1)
        else:
            return np.complex128(result / P1)

    jacobian = compute_period_derivative(
        L=L, l1=l1, l2=l2,
        m1=m1, m2=m2, m3=m3,
        eps=1,
    )
    Ztau = Z(tau)
    eta_tau = dedekind_eta(tau)
    im_tau = mp.im(mp.mpc(tau))
    b= pole_intercept_average(f,L,l1,l2)

    return (fZero_inv_sq, eta_tau,im_tau, b, jacobian, tau,P1)

def _poly_eval(coeffs: np.ndarray, z: complex) -> np.complex128:
    """Horner eval for p(z)=Σ coeffs[k] z^k (ascending powers)."""
    z = np.complex128(z)
    acc = np.complex128(0.0)
    for c in coeffs[::-1]:
        acc = acc * z + c
    return acc

def integrate_f2_times_z_between(
    f: Callable[[complex], complex],
    a: complex,
    b: complex,
) -> np.complex128:
    """
    Compute ∫_{a}^{b} z * f(z)^2 dz using polynomial coefficients f.coeffs.

    Requires: f.coeffs are ascending-power coeffs for f(z)=Σ c_k z^k.
    """
    coeffs = getattr(f, "coeffs", None)
    if coeffs is None:
        raise ValueError("f must have attribute `.coeffs` (ascending power coefficients).")
    c = np.asarray(coeffs, dtype=np.complex128)
    m = c.size
    if m == 0:
        return np.complex128(0.0)

    # d = coeffs of f(z)^2  (ascending powers), length 2m-1
    d = np.convolve(c, c)

    # Antiderivative of z*f(z)^2:
    # z*f^2 = Σ_{n=0..2m-2} d[n] z^(n+1)
    # Integral = Σ d[n]/(n+2) z^(n+2)
    n = np.arange(d.size, dtype=np.float64)  # n = 0..2m-2
    anti = np.zeros(d.size + 2, dtype=np.complex128)  # powers 0..(2m)
    anti[2:] = d / (n + 2.0)  # coefficient for z^(n+2)

    return _poly_eval(anti, b) - _poly_eval(anti, a)

def integrate_f2_times_z_over_pairs(
    f: Callable[[complex], complex],
    pairs: Sequence[Tuple[complex, complex]],
) -> np.ndarray:
    """Vector convenience: apply integrate_f2_times_z_between to many (a,b) pairs."""
    out = [integrate_f2_times_z_between(f, a, b) for a, b in pairs]
    return np.asarray(out, dtype=np.complex128)


def average_b(L,l1: int, l2: int, bs: list) -> list:
    """
    Python translation of Mathematica Averageb.

    Averages the b values using phase rotations that account for
    the cyclic symmetry of the three edges.

    Parameters
    ----------
    l1, l2, l3 : int
        Edge lengths.
    bs : list
        List of (at least 3) complex b values from calculate_b.

    Returns
    -------
    list of 2 complex values (if any l_i == 0) or 3 complex values (general case).
    """
    if L % 2 != 0:
        raise ValueError("L must be even.")
    l3=int(L/2)-l2-l1

    if l1 == 0 or l2 == 0 or l3 == 0:
        b1 = np.mean([bs[0], bs[1] * np.exp(-1j * np.pi / 2) * np.exp(2j * np.pi * l1 / L)])
        b2 = np.mean([bs[1], bs[0] * np.exp(1j * np.pi / 2) * np.exp(-2j * np.pi * l1 / L)])
        return [b1, b2]

    b1 = np.mean([bs[0],
                   bs[1] * np.exp(-1j * np.pi / 3) * np.exp(2j * np.pi * l1 / L),
                   bs[2] * np.exp(1j * np.pi / 3) * np.exp(-2j * np.pi * l3 / L)])
    b2 = np.mean([bs[1],
                   bs[2] * np.exp(-1j * np.pi / 3) * np.exp(2j * np.pi * l2 / L),
                   bs[0] * np.exp(1j * np.pi / 3) * np.exp(-2j * np.pi * l1 / L)])
    b3 = np.mean([bs[2],
                   bs[0] * np.exp(-1j * np.pi / 3) * np.exp(2j * np.pi * l3 / L),
                   bs[1] * np.exp(1j * np.pi / 3) * np.exp(-2j * np.pi * l2 / L)])
    return [b1, b2, b3]


def calculate_b(L: int, l1: int, l2: int, f=None) -> list:
    """
    Python translation of Mathematica Calculateb.

    Evaluates the improved cylinder equation polynomial at special boundary
    points with appropriate prefactors.

    For l3=0 (or l1=0 or l2=0): 4 points with (-1/2) exponent prefactors.
    Otherwise: 6 points with (-1/3) exponent prefactors.

    Parameters
    ----------
    L, l1, l2 : int
    f : callable, optional
        If provided, should be the callable from make_cyl_eqn_improved.
        If None, computes it internally.

    Returns
    -------
    list of 4 complex values (if any l_i == 0) or 6 complex values (general case).
    """
    if f is None:
        f = make_cyl_eqn_improved(L, l1, l2)

    m = L // 2
    l3 = m - l1 - l2
    theta1 = 2 * np.pi * l1 / L
    theta2 = 2 * np.pi * l2 / L

    if l1 == 0 or l2 == 0 or l3 == 0:
        # 4 special points on unit circle
        zs = np.array([
            1.0,
            np.exp(1j * theta1),
            -1.0,
            -np.exp(1j * theta1)
        ], dtype=np.complex128)

        bs = []
        for i in range(4):
            z_val = zs[i]
            if i < 2:
                # (1 + z/zs[i]) * (1 - z²/zs[j]²)  where j is the other index in {0,1}
                j = 1 - i
                factor = ((1 + z_val / zs[i]) *
                          (1 - z_val**2 / zs[j]**2))
            else:
                # (1 - z/zs[j]) * (1 - z²/zs[k]²)  where j=i-2, k is the other in {0,1}
                j = i - 2
                k = 1 - j
                factor = ((1 - z_val / zs[j]) *
                          (1 - z_val**2 / zs[k]**2))

            prefactor = factor ** (-1/2)

            result = f(z_val)
            if isinstance(result, tuple):
                singular, poly = result
                val = poly
            else:
                val = result

            bs.append(prefactor * val)

        return bs

    # General case: 6 special points on unit circle
    zs = np.array([
        1.0,
        np.exp(1j * theta1),
        np.exp(1j * (theta1 + theta2)),
        -1.0,
        -np.exp(1j * theta1),
        -np.exp(1j * (theta1 + theta2))
    ], dtype=np.complex128)

    bs = []
    for i in range(6):
        z_val = zs[i]

        if i < 3:
            # (1 + z/zs[i]) * (1 - z²/zs[j]²) * (1 - z²/zs[k]²)
            j = (i + 1) % 3
            k = (i + 2) % 3
            factor = ((1 + z_val / zs[i]) *
                      (1 - z_val**2 / zs[j]**2) *
                      (1 - z_val**2 / zs[k]**2))
        else:
            # (1 - z/zs[j]) * (1 - z²/zs[k1]²) * (1 - z²/zs[k2]²)
            j = i - 3
            k1 = (j + 1) % 3
            k2 = (j + 2) % 3
            factor = ((1 - z_val / zs[j]) *
                      (1 - z_val**2 / zs[k1]**2) *
                      (1 - z_val**2 / zs[k2]**2))

        prefactor = factor ** (-1/3)

        result = f(z_val)
        if isinstance(result, tuple):
            singular, poly = result
            val = poly
        else:
            val = result

        bs.append(prefactor * val)

    return bs