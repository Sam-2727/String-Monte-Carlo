from __future__ import annotations

"""
Analytic helpers for Riemann-surface quantities used in bc-system correlators.

This module is built on top of the holomorphic one-form and period-matrix
machinery already implemented in ell_to_tau.py. It provides:

- A-normalized surface data packaged with radial antiderivatives
- Abel-Jacobi maps in the disc frame
- Riemann theta functions with characteristics
- The prime form built from an odd characteristic

Conventions
-----------
The theta function is

    theta(y | Omega) = sum_n exp(pi i n^T Omega n + 2 pi i n^T y).

With characteristics, we use the standard half-characteristic convention

    theta[eps, delta](y | Omega)
      = sum_n exp(pi i (n+eps)^T Omega (n+eps)
                  + 2 pi i (n+eps)^T (y+delta)),

where eps_i, delta_i are 0 or 1/2.

The Abel map here is evaluated with the straight radial primitive inherited
from ell_to_tau.py. This is the natural choice in the disc frame and is enough
for the local analytic ingredients needed in the current bc-correlator work.
"""

from dataclasses import dataclass
from itertools import product
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import quad

import ell_to_tau as elt


@dataclass(frozen=True)
class RiemannSurfaceData:
    genus: int
    Omega: np.ndarray
    normalized_forms: tuple[Callable[[complex], complex], ...]
    antiderivatives: tuple[Callable[[complex], np.complex128], ...]
    A_periods: np.ndarray
    B_periods: np.ndarray
    basis_pairs: tuple[dict[str, tuple[tuple[int, int], ...]], ...]
    edge_midpoints: dict[int, tuple[np.complex128, np.complex128]]

    @property
    def tau(self) -> np.complex128 | None:
        if self.genus == 1 and self.Omega.shape == (1, 1):
            return np.complex128(self.Omega[0, 0])
        return None


def _evaluate_one_form(f, z: complex) -> np.complex128:
    value = f(np.complex128(z))
    if isinstance(value, tuple) and len(value) == 2:
        return np.complex128(value[0]) * np.complex128(value[1])
    return np.complex128(value)


def _make_radial_antiderivative(f, *, quad_limit: int = 200):
    try:
        test_val = f(np.complex128(0.0))
    except Exception:
        test_val = None

    if not (isinstance(test_val, tuple) and len(test_val) == 2) and hasattr(f, "coeffs"):
        return elt.make_antiderivative_from_f(f)

    cache: dict[tuple[float, float], np.complex128] = {}

    def F(p):
        p = np.complex128(p)
        key = (float(np.real(p)), float(np.imag(p)))
        if key in cache:
            return cache[key]

        def integrand_re(t):
            return (_evaluate_one_form(f, t * p) * p).real

        def integrand_im(t):
            return (_evaluate_one_form(f, t * p) * p).imag

        re_part, _ = quad(integrand_re, 0.0, 1.0, limit=quad_limit)
        im_part, _ = quad(integrand_im, 0.0, 1.0, limit=quad_limit)
        out = np.complex128(re_part + 1j * im_part)
        cache[key] = out
        return out

    return F


def build_surface_data(
    *,
    forms=None,
    ribbon_graph=None,
    ell_list=None,
    L: int | None = None,
    l1: int | None = None,
    l2: int | None = None,
    basis_pairs=None,
    custom_cycles=None,
) -> RiemannSurfaceData:
    """
    Build A-normalized surface data from the existing ell_to_tau.py geometry.

    Either pass:
    - `forms=...` together with optional `ribbon_graph`/`ell_list`, or
    - genus-1 theta-graph data `L`, `l1`, `l2`.
    """
    built_from_genus1_lengths = forms is None
    if built_from_genus1_lengths:
        if L is None or l1 is None or l2 is None:
            raise ValueError("Provide either forms=... or genus-1 data (L, l1, l2).")
        L = int(L)
        l1 = int(l1)
        l2 = int(l2)
        half = L // 2
        l3 = half - l1 - l2
        if L % 2 != 0:
            raise ValueError("L must be even.")
        if l3 < 0:
            raise ValueError("Need l1 + l2 <= L/2.")
        forms = [elt.make_cyl_eqn_improved(L, l1, l2)]
        ribbon_graph = (
            [(1, 2), (1, 2), (1, 2)],
            [1, 2],
            {1: [0, 1, 2], 2: [0, 1, 2]},
        )
        ell_list = [l1, l2, l3]

    norm_data = elt.normalize_holomorphic_forms(
        forms,
        ribbon_graph=ribbon_graph,
        ell_list=ell_list,
        basis_pairs=basis_pairs,
        custom_cycles=custom_cycles,
        return_data=True,
    )
    normalized_forms = tuple(norm_data["normalized_forms"])
    antiderivatives = tuple(_make_radial_antiderivative(f) for f in normalized_forms)
    basis = tuple(
        {
            "alpha": tuple((int(edge_idx), int(coeff)) for edge_idx, coeff in pair["alpha"]),
            "beta": tuple((int(edge_idx), int(coeff)) for edge_idx, coeff in pair["beta"]),
        }
        for pair in norm_data["basis_pairs"]
    )

    return RiemannSurfaceData(
        genus=len(normalized_forms),
        Omega=np.asarray(norm_data["Omega"], dtype=np.complex128),
        normalized_forms=normalized_forms,
        antiderivatives=antiderivatives,
        A_periods=np.asarray(norm_data["A_periods"], dtype=np.complex128),
        B_periods=np.asarray(norm_data["B_periods"], dtype=np.complex128),
        basis_pairs=basis,
        edge_midpoints={
            int(edge_idx): (np.complex128(z0), np.complex128(z1))
            for edge_idx, (z0, z1) in norm_data["edge_midpoints"].items()
        },
    )


def abel_map(
    z: complex,
    surface: RiemannSurfaceData,
    *,
    basepoint: complex = 0.0j,
) -> np.ndarray:
    """Return the Abel-Jacobi image zeta(z) - zeta(basepoint)."""
    z = np.complex128(z)
    base = np.complex128(basepoint)
    return np.asarray(
        [F(z) - F(base) for F in surface.antiderivatives],
        dtype=np.complex128,
    )


def abel_difference(
    z: complex,
    w: complex,
    surface: RiemannSurfaceData,
) -> np.ndarray:
    """Return zeta(z) - zeta(w) without fixing a separate basepoint."""
    z = np.complex128(z)
    w = np.complex128(w)
    return np.asarray(
        [F(z) - F(w) for F in surface.antiderivatives],
        dtype=np.complex128,
    )


def theta_characteristics(
    genus: int,
    *,
    parity: str | None = None,
    as_half_integers: bool = False,
):
    """
    Enumerate genus-g theta characteristics.

    The default output format is binary bits:
        ((a1,...,ag), (b1,...,bg)), ai,bi in {0,1}
    corresponding to eps=a/2 and delta=b/2.
    """
    if genus <= 0:
        raise ValueError("Need genus >= 1.")

    chars = []
    for a_bits in product((0, 1), repeat=genus):
        for b_bits in product((0, 1), repeat=genus):
            char = (tuple(int(x) for x in a_bits), tuple(int(x) for x in b_bits))
            if parity is None:
                chars.append(char)
                continue
            char_parity = characteristic_parity(char)
            if parity == "even" and char_parity == 0:
                chars.append(char)
            elif parity == "odd" and char_parity == 1:
                chars.append(char)
            elif parity not in {"even", "odd"}:
                raise ValueError("parity must be one of None, 'even', or 'odd'.")

    if as_half_integers:
        return tuple(
            (
                tuple(0.5 * x for x in a_bits),
                tuple(0.5 * x for x in b_bits),
            )
            for a_bits, b_bits in chars
        )
    return tuple(chars)


def _coerce_characteristic(characteristic, genus: int) -> tuple[np.ndarray, np.ndarray]:
    if characteristic is None:
        return np.zeros(genus, dtype=np.float64), np.zeros(genus, dtype=np.float64)
    if not isinstance(characteristic, (tuple, list)) or len(characteristic) != 2:
        raise ValueError("Characteristic must be a pair (epsilon, delta).")

    eps = np.asarray(characteristic[0], dtype=np.float64)
    delt = np.asarray(characteristic[1], dtype=np.float64)
    if eps.shape != (genus,) or delt.shape != (genus,):
        raise ValueError(
            f"Characteristic must have shape ({genus},) in each component; "
            f"got {eps.shape} and {delt.shape}."
        )

    def normalize_half_vector(vec):
        out = np.empty_like(vec)
        for idx, value in enumerate(vec):
            if abs(value - round(value)) < 1e-12:
                rounded = int(round(value))
                if rounded not in (0, 1):
                    raise ValueError("Binary characteristic bits must be 0 or 1.")
                out[idx] = 0.5 * rounded
            elif abs(2.0 * value - round(2.0 * value)) < 1e-12:
                rounded = 0.5 * round(2.0 * value)
                if rounded not in (0.0, 0.5):
                    raise ValueError("Half-characteristic entries must be 0 or 1/2.")
                out[idx] = rounded
            else:
                raise ValueError("Characteristic entries must be 0, 1, 0.5, or 1/2-style bits.")
        return out

    return normalize_half_vector(eps), normalize_half_vector(delt)


def characteristic_parity(characteristic) -> int:
    """
    Return the Arf parity in {0,1}; 1 means odd, 0 means even.
    """
    genus = len(characteristic[0])
    eps, delt = _coerce_characteristic(characteristic, genus)
    return int(np.rint(4.0 * float(np.dot(eps, delt)))) % 2


def theta_truncation(Omega, *, tol: float = 1e-12) -> int:
    """
    Choose an isotropic lattice cutoff from the smallest eigenvalue of Im(Omega).
    """
    Omega = np.asarray(Omega, dtype=np.complex128)
    if Omega.ndim != 2 or Omega.shape[0] != Omega.shape[1]:
        raise ValueError("Omega must be a square period matrix.")
    im_omega = np.asarray(np.imag(Omega), dtype=np.float64)
    evals = np.linalg.eigvalsh(im_omega)
    lam_min = float(np.min(evals))
    if lam_min <= 0.0:
        raise ValueError("Need Im(Omega) positive definite.")
    tol = max(float(tol), 1e-16)
    return max(4, int(np.ceil(np.sqrt(-np.log(tol) / (np.pi * lam_min)))) + 2)


def riemann_theta(
    y,
    Omega,
    *,
    characteristic=None,
    nmax: int | None = None,
    tol: float = 1e-12,
) -> np.complex128:
    theta, _ = riemann_theta_with_gradient(
        y,
        Omega,
        characteristic=characteristic,
        nmax=nmax,
        tol=tol,
        compute_gradient=False,
    )
    return theta


def riemann_theta_gradient(
    y,
    Omega,
    *,
    characteristic=None,
    nmax: int | None = None,
    tol: float = 1e-12,
) -> np.ndarray:
    _, gradient = riemann_theta_with_gradient(
        y,
        Omega,
        characteristic=characteristic,
        nmax=nmax,
        tol=tol,
        compute_gradient=True,
    )
    return gradient


def riemann_theta_with_gradient(
    y,
    Omega,
    *,
    characteristic=None,
    nmax: int | None = None,
    tol: float = 1e-12,
    compute_gradient: bool = True,
) -> tuple[np.complex128, np.ndarray | None]:
    """
    Evaluate theta[characteristic](y | Omega) and optionally its y-gradient.
    """
    Omega = np.asarray(Omega, dtype=np.complex128)
    if Omega.ndim != 2 or Omega.shape[0] != Omega.shape[1]:
        raise ValueError("Omega must be a square matrix.")
    genus = Omega.shape[0]
    y_vec = np.asarray(y, dtype=np.complex128)
    if y_vec.shape != (genus,):
        raise ValueError(f"y must have shape ({genus},), got {y_vec.shape}.")
    eps, delt = _coerce_characteristic(characteristic, genus)

    if nmax is None:
        nmax = theta_truncation(Omega, tol=tol)
    nmax = int(nmax)
    if nmax < 0:
        raise ValueError("nmax must be nonnegative.")

    theta = np.complex128(0.0)
    gradient = np.zeros(genus, dtype=np.complex128) if compute_gradient else None
    shift = y_vec + delt

    for n_tuple in product(range(-nmax, nmax + 1), repeat=genus):
        vec = np.asarray(n_tuple, dtype=np.float64) + eps
        exponent = np.complex128(
            1j
            * np.pi
            * (vec @ Omega @ vec + 2.0 * vec @ shift)
        )
        term = np.exp(exponent)
        theta += term
        if gradient is not None:
            gradient += (2j * np.pi) * vec * term

    return theta, gradient


def odd_characteristic(genus: int):
    """Return a canonical odd characteristic for the given genus."""
    chars = theta_characteristics(genus, parity="odd")
    if not chars:
        raise ValueError(f"No odd characteristics exist for genus {genus}.")
    return chars[0]


def prime_form(
    z: complex,
    w: complex,
    surface: RiemannSurfaceData,
    *,
    characteristic=None,
    nmax: int | None = None,
    tol: float = 1e-12,
) -> np.complex128:
    """
    Compute the prime form E(z,w) using an odd theta characteristic.

    The branch of the square root is the principal complex branch, so the
    result is fixed only up to the usual overall sign ambiguity of the prime
    form. Ratios and local limits are the robust quantities to compare.
    """
    if characteristic is None:
        characteristic = odd_characteristic(surface.genus)
    if characteristic_parity(characteristic) != 1:
        raise ValueError("prime_form requires an odd theta characteristic.")

    diff = abel_difference(z, w, surface)
    theta_val = riemann_theta(
        diff,
        surface.Omega,
        characteristic=characteristic,
        nmax=nmax,
        tol=tol,
    )
    grad0 = riemann_theta_gradient(
        np.zeros(surface.genus, dtype=np.complex128),
        surface.Omega,
        characteristic=characteristic,
        nmax=nmax,
        tol=tol,
    )
    omega_z = np.asarray(
        [_evaluate_one_form(f, z) for f in surface.normalized_forms],
        dtype=np.complex128,
    )
    omega_w = np.asarray(
        [_evaluate_one_form(f, w) for f in surface.normalized_forms],
        dtype=np.complex128,
    )
    spin_z = np.dot(omega_z, grad0)
    spin_w = np.dot(omega_w, grad0)
    denom = np.sqrt(spin_z) * np.sqrt(spin_w)
    if abs(denom) == 0.0:
        raise ZeroDivisionError("Prime-form denominator vanished for the chosen odd characteristic.")
    return np.complex128(theta_val / denom)


def _segment_integral(func, z0: complex, z1: complex, *, quad_limit: int = 200) -> np.complex128:
    """Numerically integrate func(z) dz along the straight segment z0 -> z1."""
    z0 = np.complex128(z0)
    z1 = np.complex128(z1)
    dz = z1 - z0

    def integrand_re(t):
        z = z0 + t * dz
        return (np.complex128(func(z)) * dz).real

    def integrand_im(t):
        z = z0 + t * dz
        return (np.complex128(func(z)) * dz).imag

    re_part, _ = quad(integrand_re, 0.0, 1.0, limit=quad_limit)
    im_part, _ = quad(integrand_im, 0.0, 1.0, limit=quad_limit)
    return np.complex128(re_part + 1j * im_part)


def riemann_constant_vector(
    surface: RiemannSurfaceData,
    *,
    quad_limit: int = 200,
) -> np.ndarray:
    r"""
    Compute the Riemann class vector Delta using the convention in Strebel.tex:

        Delta_I = 1/2 (1 - Omega_{II})
                  + sum_{J != I} int_{alpha_J} omega_J(z) zeta_I(z) dz.

    The Abel map zeta_I(z) uses the same radial-basepoint convention as the
    rest of this module.
    """
    genus = surface.genus
    Omega = np.asarray(surface.Omega, dtype=np.complex128)
    if Omega.shape != (genus, genus):
        raise ValueError(
            f"Expected Omega to have shape ({genus}, {genus}), got {Omega.shape}."
        )

    Delta = 0.5 * (1.0 - np.diag(Omega).astype(np.complex128))

    for I in range(genus):
        for J in range(genus):
            if I == J:
                continue

            F_I = surface.antiderivatives[I]
            omega_J = surface.normalized_forms[J]
            cycle = surface.basis_pairs[J]["alpha"]

            total = np.complex128(0.0)
            for edge_idx, coeff in cycle:
                z0, z1 = surface.edge_midpoints[int(edge_idx)]

                def integrand(z):
                    return F_I(z) * _evaluate_one_form(omega_J, z)

                total += np.complex128(coeff) * _segment_integral(
                    integrand,
                    z0,
                    z1,
                    quad_limit=quad_limit,
                )

            Delta[I] += total

    return np.asarray(Delta, dtype=np.complex128)


def sigma_ratio(
    z: complex,
    w: complex,
    surface: RiemannSurfaceData,
    *,
    divisor_points: Sequence[complex],
    Delta=None,
    nmax: int | None = None,
    tol: float = 1e-12,
    quad_limit: int = 200,
) -> np.complex128:
    r"""
    Solve for the normalized sigma ratio sigma(z) / sigma(w).

    The algorithm uses the lambda=1, (n,m)=(g,1) relation quoted in
    Strebel.tex. For any generic divisor points z_1,...,z_g we have

        sigma(z) / sigma(w)
        =
        theta(S - zeta(z) | Omega) / theta(S - zeta(w) | Omega)
        * prod_i E(z_i, w) / prod_i E(z_i, z),

    with

        S = sum_i zeta(z_i) - Delta.

    This determines sigma only up to an overall multiplicative constant, so
    sigma ratios are the canonical output.
    """
    divisor_points = tuple(np.complex128(point) for point in divisor_points)
    if len(divisor_points) != surface.genus:
        raise ValueError(
            f"Need exactly g={surface.genus} divisor points, got {len(divisor_points)}."
        )

    if Delta is None:
        Delta = riemann_constant_vector(surface, quad_limit=quad_limit)
    Delta = np.asarray(Delta, dtype=np.complex128)
    if Delta.shape != (surface.genus,):
        raise ValueError(
            f"Delta must have shape ({surface.genus},), got {Delta.shape}."
        )

    zeta_sum = np.sum(
        np.asarray([abel_map(point, surface) for point in divisor_points], dtype=np.complex128),
        axis=0,
    )
    arg_z = zeta_sum - abel_map(z, surface) - Delta
    arg_w = zeta_sum - abel_map(w, surface) - Delta

    theta_z = riemann_theta(arg_z, surface.Omega, nmax=nmax, tol=tol)
    theta_w = riemann_theta(arg_w, surface.Omega, nmax=nmax, tol=tol)
    if abs(theta_z) == 0.0 or abs(theta_w) == 0.0:
        raise ZeroDivisionError(
            "Theta factor vanished for the chosen divisor/reference points; "
            "pick a generic divisor."
        )

    prime_prod_z = np.complex128(1.0)
    prime_prod_w = np.complex128(1.0)
    for point in divisor_points:
        prime_prod_z *= prime_form(point, z, surface, nmax=nmax, tol=tol)
        prime_prod_w *= prime_form(point, w, surface, nmax=nmax, tol=tol)

    if abs(prime_prod_z) == 0.0 or abs(prime_prod_w) == 0.0:
        raise ZeroDivisionError(
            "Prime-form product vanished for the chosen divisor/reference points; "
            "pick generic non-coincident points."
        )

    return np.complex128((theta_z / theta_w) * (prime_prod_w / prime_prod_z))


def sigma_value(
    z: complex,
    surface: RiemannSurfaceData,
    *,
    divisor_points: Sequence[complex],
    normalization_point: complex,
    normalization_value: complex = 1.0 + 0.0j,
    Delta=None,
    nmax: int | None = None,
    tol: float = 1e-12,
    quad_limit: int = 200,
) -> np.complex128:
    r"""
    Return a normalized sigma value with sigma(normalization_point) fixed by hand.

    Since the Verlinde-Verlinde formula determines sigma only up to an overall
    multiplicative constant, this function imposes the normalization

        sigma(normalization_point) = normalization_value.
    """
    ratio = sigma_ratio(
        z,
        normalization_point,
        surface,
        divisor_points=divisor_points,
        Delta=Delta,
        nmax=nmax,
        tol=tol,
        quad_limit=quad_limit,
    )
    return np.complex128(normalization_value) * ratio
