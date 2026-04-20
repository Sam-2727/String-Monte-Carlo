from __future__ import annotations

"""
Analytic helpers for Riemann-surface quantities used in bc-system correlators.

This module is built on top of the holomorphic one-form and period-matrix
machinery already implemented in ell_to_tau.py. It provides:

- A-normalized surface data packaged with radial antiderivatives
- Abel-Jacobi maps in the disc frame
- Riemann theta functions with characteristics
- The prime form built from an odd characteristic
- Verlinde-Verlinde bc-system correlators built from these ingredients

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


@dataclass(frozen=True)
class LargeLFitResult:
    c: float
    gamma: float
    alpha: float
    r2: float
    max_abs_log_residual: float
    total_lengths: tuple[int, ...]
    log_values: tuple[float, ...]
    edge_length_sets: tuple[tuple[int, ...], ...]

    @property
    def finite_part(self) -> float:
        return float(np.exp(self.c))


@dataclass(frozen=True)
class UniversalLargeLCoefficients:
    gamma: float
    alpha: float
    r2: float
    max_abs_log_residual: float
    family_constants: tuple[float, ...]


@dataclass(frozen=True)
class RenormalizedZ1Data:
    abs_z1_sq: float
    normalization_factor: float
    renormalized_det_factor: float
    fit: LargeLFitResult
    surface: RiemannSurfaceData


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


def _ghost_number_selection_rule(
    lambda_weight: float,
    genus: int,
    *,
    n_b: int,
    n_c: int,
) -> int:
    """
    Return the expected value of n_c - n_b from the bc ghost-number anomaly.

    In the notation of Strebel.tex,

        n_c - n_b = (1 - 2 lambda) (g - 1),

    equivalently

        n_b - n_c = (2 lambda - 1) (g - 1).
    """
    expected = (1.0 - 2.0 * float(lambda_weight)) * float(genus - 1)
    rounded = int(round(expected))
    if abs(expected - rounded) > 1e-12:
        raise ValueError(
            "The bc selection rule is not integral for the supplied lambda and genus: "
            f"(1 - 2 lambda)(g - 1) = {expected}."
        )
    actual = int(n_c) - int(n_b)
    if actual != rounded:
        raise ValueError(
            "The bc correlator violates the ghost-number selection rule: "
            f"need n_c - n_b = {rounded}, got {actual}."
        )
    return rounded


def bc_correlator_geometric_factor(
    b_points: Sequence[complex],
    c_points: Sequence[complex],
    surface: RiemannSurfaceData,
    *,
    lambda_weight: float,
    divisor_points: Sequence[complex],
    normalization_point: complex,
    normalization_value: complex = 1.0 + 0.0j,
    Delta=None,
    nmax: int | None = None,
    tol: float = 1e-12,
    quad_limit: int = 200,
) -> np.complex128:
    r"""
    Compute the Verlinde-Verlinde geometric factor for the bc correlator.

    The returned value is

        theta(sum_i zeta(z_i) - sum_j zeta(w_j) - (2 lambda - 1) Delta | Omega)
        * prod_{i<i'} E(z_i, z_i')
        * prod_{j<j'} E(w_j, w_j')
        * prod_i sigma(z_i)^(2 lambda - 1)
        /
        [ prod_{i,j} E(z_i, w_j) * prod_j sigma(w_j)^(2 lambda - 1) ].

    This is the full holomorphic correlator with the chiral prefactor
    `Z_1^{-1/2}` stripped off.
    """
    b_points = tuple(np.complex128(point) for point in b_points)
    c_points = tuple(np.complex128(point) for point in c_points)
    _ghost_number_selection_rule(
        lambda_weight,
        surface.genus,
        n_b=len(b_points),
        n_c=len(c_points),
    )

    if Delta is None:
        Delta = riemann_constant_vector(surface, quad_limit=quad_limit)
    Delta = np.asarray(Delta, dtype=np.complex128)
    if Delta.shape != (surface.genus,):
        raise ValueError(
            f"Delta must have shape ({surface.genus},), got {Delta.shape}."
        )

    weight = np.complex128(2.0 * float(lambda_weight) - 1.0)
    zeta_b = (
        np.sum(
            np.asarray([abel_map(point, surface) for point in b_points], dtype=np.complex128),
            axis=0,
        )
        if b_points
        else np.zeros(surface.genus, dtype=np.complex128)
    )
    zeta_c = (
        np.sum(
            np.asarray([abel_map(point, surface) for point in c_points], dtype=np.complex128),
            axis=0,
        )
        if c_points
        else np.zeros(surface.genus, dtype=np.complex128)
    )
    theta_arg = zeta_b - zeta_c - weight * Delta
    theta_val = riemann_theta(
        theta_arg,
        surface.Omega,
        nmax=nmax,
        tol=tol,
    )

    prime_bb = np.complex128(1.0)
    for idx, zi in enumerate(b_points):
        for zj in b_points[idx + 1 :]:
            prime_bb *= prime_form(zi, zj, surface, nmax=nmax, tol=tol)

    prime_cc = np.complex128(1.0)
    for idx, wi in enumerate(c_points):
        for wj in c_points[idx + 1 :]:
            prime_cc *= prime_form(wi, wj, surface, nmax=nmax, tol=tol)

    prime_bc = np.complex128(1.0)
    for zi in b_points:
        for wj in c_points:
            prime_bc *= prime_form(zi, wj, surface, nmax=nmax, tol=tol)
    if b_points and c_points and abs(prime_bc) == 0.0:
        raise ZeroDivisionError("prod_{i,j} E(z_i, w_j) vanished for the chosen insertion points.")

    sigma_b = np.complex128(1.0)
    for zi in b_points:
        sigma_b *= sigma_value(
            zi,
            surface,
            divisor_points=divisor_points,
            normalization_point=normalization_point,
            normalization_value=normalization_value,
            Delta=Delta,
            nmax=nmax,
            tol=tol,
            quad_limit=quad_limit,
        ) ** weight

    sigma_c = np.complex128(1.0)
    for wj in c_points:
        sigma_c *= sigma_value(
            wj,
            surface,
            divisor_points=divisor_points,
            normalization_point=normalization_point,
            normalization_value=normalization_value,
            Delta=Delta,
            nmax=nmax,
            tol=tol,
            quad_limit=quad_limit,
        ) ** weight
    if c_points and abs(sigma_c) == 0.0:
        raise ZeroDivisionError("prod_j sigma(w_j)^(2 lambda - 1) vanished for the chosen insertion points.")

    return np.complex128(theta_val * prime_bb * prime_cc * sigma_b / (prime_bc * sigma_c))


def bc_correlator(
    b_points: Sequence[complex],
    c_points: Sequence[complex],
    surface: RiemannSurfaceData,
    *,
    lambda_weight: float,
    divisor_points: Sequence[complex],
    normalization_point: complex,
    normalization_value: complex = 1.0 + 0.0j,
    z1: complex | None = None,
    Delta=None,
    nmax: int | None = None,
    tol: float = 1e-12,
    quad_limit: int = 200,
) -> np.complex128:
    r"""
    Compute the holomorphic bc correlator on a Riemann surface.

    If `z1` is supplied, this returns the full Verlinde-Verlinde correlator

        Z_1^{-1/2} * geometric_factor.

    If `z1` is omitted, the function returns only the geometric factor
    `Z_1^{1/2} <prod b prod c>`, which is the part directly determined by the
    Abel map, theta function, prime form, and sigma factors.
    """
    geometric = bc_correlator_geometric_factor(
        b_points,
        c_points,
        surface,
        lambda_weight=lambda_weight,
        divisor_points=divisor_points,
        normalization_point=normalization_point,
        normalization_value=normalization_value,
        Delta=Delta,
        nmax=nmax,
        tol=tol,
        quad_limit=quad_limit,
    )
    if z1 is None:
        return geometric

    z1 = np.complex128(z1)
    if abs(z1) == 0.0:
        raise ZeroDivisionError("z1 must be nonzero to form the chiral prefactor Z_1^{-1/2}.")
    return np.complex128(geometric / np.sqrt(z1))


def _fit_large_l_behavior(
    total_lengths,
    log_values,
    *,
    fixed_gamma: float | None = None,
    fixed_alpha: float | None = None,
    edge_length_sets=None,
) -> LargeLFitResult:
    """Fit log Z(L) = c + gamma L + alpha log L to large-L data."""
    total_lengths = tuple(int(L) for L in total_lengths)
    log_values = tuple(float(value) for value in log_values)
    if len(total_lengths) != len(log_values):
        raise ValueError(
            f"Need one log value per total length; got {len(total_lengths)} and {len(log_values)}."
        )
    if any(L <= 0 for L in total_lengths):
        raise ValueError("All total lengths must be positive.")
    if (fixed_gamma is None) != (fixed_alpha is None):
        raise ValueError("Provide both fixed_gamma and fixed_alpha, or neither.")
    if fixed_gamma is None and len(total_lengths) < 3:
        raise ValueError("Need at least three large-L samples to fit c + gamma L + alpha log L.")
    if fixed_gamma is not None and len(total_lengths) < 1:
        raise ValueError("Need at least one sample when gamma and alpha are fixed.")

    order = np.argsort(np.asarray(total_lengths, dtype=np.int64))
    l_arr = np.asarray(total_lengths, dtype=np.float64)[order]
    logz_arr = np.asarray(log_values, dtype=np.float64)[order]

    if fixed_gamma is None:
        design = np.column_stack([np.ones_like(l_arr), l_arr, np.log(l_arr)])
        coef, *_ = np.linalg.lstsq(design, logz_arr, rcond=None)
        c0, gamma, alpha = coef
        predicted = design @ coef
    else:
        gamma = float(fixed_gamma)
        alpha = float(fixed_alpha)
        c0 = float(np.mean(logz_arr - gamma * l_arr - alpha * np.log(l_arr)))
        predicted = c0 + gamma * l_arr + alpha * np.log(l_arr)

    residual = logz_arr - predicted
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((logz_arr - np.mean(logz_arr)) ** 2))

    if edge_length_sets is None:
        sorted_edge_sets = tuple(() for _ in total_lengths)
    else:
        if len(edge_length_sets) != len(total_lengths):
            raise ValueError(
                "edge_length_sets must have the same length as total_lengths when provided."
            )
        sorted_edge_sets = tuple(
            tuple(int(x) for x in edge_length_sets[idx])
            for idx in order
        )

    return LargeLFitResult(
        c=float(c0),
        gamma=float(gamma),
        alpha=float(alpha),
        r2=1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0,
        max_abs_log_residual=float(np.max(np.abs(residual))),
        total_lengths=tuple(int(x) for x in l_arr.astype(np.int64)),
        log_values=tuple(float(x) for x in logz_arr),
        edge_length_sets=sorted_edge_sets,
    )


def fit_universal_large_l_coefficients(
    family_samples,
) -> UniversalLargeLCoefficients:
    r"""
    Fit moduli-independent `gamma, alpha` from multiple large-L data families.

    Each family contributes data of the form

        log Z_k(L) = c_k + gamma L + alpha log L,

    where the intercept `c_k` is family-dependent but `gamma, alpha` are
    shared. The input should be an iterable of `(total_lengths, log_values)`
    pairs, one per moduli family.
    """
    families = [
        (
            tuple(int(L) for L in total_lengths),
            tuple(float(value) for value in log_values),
        )
        for total_lengths, log_values in family_samples
    ]
    if not families:
        raise ValueError("Need at least one large-L family to fit universal coefficients.")

    n_families = len(families)
    rows: list[list[float]] = []
    targets: list[float] = []
    for family_idx, (total_lengths, log_values) in enumerate(families):
        if len(total_lengths) != len(log_values):
            raise ValueError(
                f"Family {family_idx} has mismatched lengths: "
                f"{len(total_lengths)} total lengths and {len(log_values)} log values."
            )
        for L, value in zip(total_lengths, log_values):
            if L <= 0:
                raise ValueError(f"All total lengths must be positive; got L={L}.")
            row = [0.0] * (n_families + 2)
            row[family_idx] = 1.0
            row[-2] = float(L)
            row[-1] = float(np.log(L))
            rows.append(row)
            targets.append(float(value))

    if len(rows) < n_families + 2:
        raise ValueError(
            "Need at least n_families + 2 total samples to fit shared gamma and alpha."
        )

    design = np.asarray(rows, dtype=np.float64)
    target_arr = np.asarray(targets, dtype=np.float64)
    coef, *_ = np.linalg.lstsq(design, target_arr, rcond=None)
    predicted = design @ coef
    residual = target_arr - predicted
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((target_arr - np.mean(target_arr)) ** 2))

    return UniversalLargeLCoefficients(
        gamma=float(coef[-2]),
        alpha=float(coef[-1]),
        r2=1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0,
        max_abs_log_residual=float(np.max(np.abs(residual))),
        family_constants=tuple(float(x) for x in coef[:-2]),
    )


def renormalized_aprime_factor_from_raw_det(
    total_lengths,
    aprime_determinants,
    *,
    gamma: float,
    alpha: float,
) -> float:
    r"""
    Return `exp(c(Omega))` from raw `det A'` values and fixed `gamma, alpha`.

    Given

        -1/2 log det A' = c(Omega) + gamma L + alpha log L,

    this strips the universal large-L piece and averages the remaining finite
    part over the supplied large-L samples.
    """
    log_values = []
    for det_value in aprime_determinants:
        det_value = float(det_value)
        if det_value <= 0.0:
            raise ValueError(f"Need det A' > 0, got {det_value}.")
        log_values.append(float(-0.5 * np.log(det_value)))
    fit = _fit_large_l_behavior(
        total_lengths,
        log_values,
        fixed_gamma=gamma,
        fixed_alpha=alpha,
    )
    return fit.finite_part


def renormalized_aprime_factor_from_raw_log_values(
    total_lengths,
    log_values,
    *,
    gamma: float,
    alpha: float,
) -> float:
    r"""
    Return `exp(c(Omega))` from raw `-1/2 log det A'` values and fixed coefficients.
    """
    fit = _fit_large_l_behavior(
        total_lengths,
        log_values,
        fixed_gamma=gamma,
        fixed_alpha=alpha,
    )
    return fit.finite_part


def build_surface_from_ribbon_graph(
    ribbon_graph,
    edge_lengths,
    *,
    basis_pairs=None,
    custom_cycles=None,
) -> RiemannSurfaceData:
    """
    Build a surface directly from an F=1 ribbon graph and integer edge lengths.
    """
    edge_lengths = tuple(int(x) for x in edge_lengths)
    forms = elt.make_cyl_eqn_improved_higher_genus(
        ribbon_graph,
        edge_lengths,
    )
    return build_surface_data(
        forms=forms,
        ribbon_graph=ribbon_graph,
        ell_list=edge_lengths,
        basis_pairs=basis_pairs,
        custom_cycles=custom_cycles,
    )


def _sample_aprime_large_l_data(
    ribbon_graph,
    base_edge_lengths,
    *,
    scales,
    min_edge_length: int = 200,
    kernel=None,
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[tuple[int, ...], ...]]:
    import partition_function as pf

    base_edge_lengths = tuple(int(x) for x in base_edge_lengths)
    if not base_edge_lengths:
        raise ValueError("Need at least one base edge length.")
    if any(edge <= 0 for edge in base_edge_lengths):
        raise ValueError("All base edge lengths must be positive integers.")

    scales = tuple(int(scale) for scale in scales)
    if len(scales) < 1:
        raise ValueError("Need at least one scale to sample large-L data.")
    if any(scale <= 0 for scale in scales):
        raise ValueError("All scales must be positive integers.")

    sampled_edge_lengths: list[tuple[int, ...]] = []
    total_lengths: list[int] = []
    log_values: list[float] = []
    for scale in scales:
        edge_lengths = tuple(int(scale * edge) for edge in base_edge_lengths)
        if min(edge_lengths) < int(min_edge_length):
            raise ValueError(
                "All sampled edge lengths must satisfy the large-L cutoff "
                f"ell_a >= {int(min_edge_length)}; got {edge_lengths}."
            )
        A_prime = np.asarray(
            pf.traced_matter_matrix_f1(ribbon_graph, edge_lengths, kernel=kernel),
            dtype=np.float64,
        )
        A_prime = 0.5 * (A_prime + A_prime.T)
        log_values.append(float(-0.5 * pf.logdet_cholesky(A_prime)))
        total_lengths.append(int(2 * sum(edge_lengths)))
        sampled_edge_lengths.append(edge_lengths)

    return (
        tuple(total_lengths),
        tuple(log_values),
        tuple(sampled_edge_lengths),
    )


def fit_genus_universal_aprime_coefficients(
    families,
    *,
    scales,
    min_edge_length: int = 200,
    kernel=None,
) -> UniversalLargeLCoefficients:
    r"""
    Fit universal `gamma, alpha` from raw `A'` data across same-genus families.

    Parameters
    ----------
    families:
        Iterable of `(ribbon_graph, base_edge_lengths)` pairs. Each family is
        sampled at the same `scales`, and the fit assumes

            -1/2 log det A'_k = c_k(Omega) + gamma L + alpha log L

        with shared `gamma, alpha` and family-dependent finite parts `c_k`.
    """
    family_samples = []
    for ribbon_graph, base_edge_lengths in families:
        total_lengths, log_values, _ = _sample_aprime_large_l_data(
            ribbon_graph,
            base_edge_lengths,
            scales=scales,
            min_edge_length=min_edge_length,
            kernel=kernel,
        )
        family_samples.append((total_lengths, log_values))
    return fit_universal_large_l_coefficients(family_samples)


def fit_renormalized_aprime_factor(
    ribbon_graph,
    base_edge_lengths,
    *,
    scales,
    min_edge_length: int = 200,
    kernel=None,
) -> LargeLFitResult:
    r"""
    Fit the large-L behavior of -1/2 log(det A') at fixed ribbon-graph moduli.

    This implements the same renormalization algorithm described in the
    "Renormalization of Z(l_a, R)" subsection of Strebel.tex, but applied just
    to the scheme-dependent determinant factor entering

        |Z_1|^2 = N_1 [det Im(Omega)]^{1/2} (det A')^{-1/2}.

    For edge lengths ell_a = scale * ell_a^(0) at fixed ratios, we fit

        -1/2 log det A' = c(Omega) + gamma L + alpha log L,

    where L = 2 sum_a ell_a is the total boundary lattice length. The
    renormalized determinant factor feeding into |Z_1|^2 is then exp(c).
    """
    total_lengths, log_values, sampled_edge_lengths = _sample_aprime_large_l_data(
        ribbon_graph,
        base_edge_lengths,
        scales=scales,
        min_edge_length=min_edge_length,
        kernel=kernel,
    )

    return _fit_large_l_behavior(
        total_lengths,
        log_values,
        edge_length_sets=sampled_edge_lengths,
    )


def lambda_one_geometric_z1_factor(
    b_points: Sequence[complex],
    c_point: complex,
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
    Return the geometric factor whose 2/3 power gives Z_1 in the lambda=1 case.

    For lambda=1 and (n,m)=(g,1), Strebel.tex writes

        Z_1 det(omega_I(z_i))
        =
        Z_1^{-1/2}
        theta(sum_i zeta(z_i) - zeta(w) - Delta | Omega)
        * prod_{i<i'} E(z_i, z_i') * prod_i sigma(z_i)
          / (prod_i E(z_i, w) sigma(w)).

    Therefore the geometric ratio

        A = [
            theta(...) * prod_{i<i'} E(z_i, z_i') * prod_i sigma(z_i)
            / (prod_i E(z_i, w) sigma(w))
        ] / det(omega_I(z_i))

    satisfies A = Z_1^{3/2}. This helper returns A.
    """
    b_points = tuple(np.complex128(point) for point in b_points)
    if len(b_points) != surface.genus:
        raise ValueError(
            f"Need exactly g={surface.genus} b-insertion points, got {len(b_points)}."
        )

    if Delta is None:
        Delta = riemann_constant_vector(surface, quad_limit=quad_limit)
    Delta = np.asarray(Delta, dtype=np.complex128)
    if Delta.shape != (surface.genus,):
        raise ValueError(
            f"Delta must have shape ({surface.genus},), got {Delta.shape}."
        )

    zeta_sum = np.sum(
        np.asarray([abel_map(point, surface) for point in b_points], dtype=np.complex128),
        axis=0,
    )
    theta_arg = zeta_sum - abel_map(c_point, surface) - Delta
    theta_val = riemann_theta(
        theta_arg,
        surface.Omega,
        nmax=nmax,
        tol=tol,
    )

    omega_matrix = np.asarray(
        [
            [_evaluate_one_form(form, point) for form in surface.normalized_forms]
            for point in b_points
        ],
        dtype=np.complex128,
    )
    det_omega = np.complex128(np.linalg.det(omega_matrix))
    if abs(det_omega) == 0.0:
        raise ZeroDivisionError("det(omega_I(z_i)) vanished for the chosen b-insertion points.")

    sigma_c = sigma_value(
        c_point,
        surface,
        divisor_points=divisor_points,
        normalization_point=normalization_point,
        normalization_value=normalization_value,
        Delta=Delta,
        nmax=nmax,
        tol=tol,
        quad_limit=quad_limit,
    )
    if abs(sigma_c) == 0.0:
        raise ZeroDivisionError("sigma(c_point) vanished for the chosen normalization data.")

    prime_prod_bb = np.complex128(1.0)
    for i, zi in enumerate(b_points):
        for zj in b_points[i + 1 :]:
            prime_prod_bb *= prime_form(zi, zj, surface, nmax=nmax, tol=tol)

    prime_prod_bc = np.complex128(1.0)
    sigma_prod_b = np.complex128(1.0)
    for zi in b_points:
        prime_prod_bc *= prime_form(zi, c_point, surface, nmax=nmax, tol=tol)
        sigma_prod_b *= sigma_value(
            zi,
            surface,
            divisor_points=divisor_points,
            normalization_point=normalization_point,
            normalization_value=normalization_value,
            Delta=Delta,
            nmax=nmax,
            tol=tol,
            quad_limit=quad_limit,
        )

    if abs(prime_prod_bc) == 0.0:
        raise ZeroDivisionError("prod_i E(z_i, w) vanished for the chosen insertion points.")

    return np.complex128(
        theta_val * prime_prod_bb * sigma_prod_b / (prime_prod_bc * sigma_c * det_omega)
    )


def abs_z1_sq_from_lambda_one(
    b_points: Sequence[complex],
    c_point: complex,
    surface: RiemannSurfaceData,
    *,
    divisor_points: Sequence[complex],
    normalization_point: complex,
    normalization_value: complex = 1.0 + 0.0j,
    Delta=None,
    nmax: int | None = None,
    tol: float = 1e-12,
    quad_limit: int = 200,
) -> float:
    r"""
    Compute |Z_1|^2 from the lambda=1, (n,m)=(g,1) identity in Strebel.tex.

    If A is the complex number returned by `lambda_one_geometric_z1_factor`,
    then A = Z_1^{3/2}. Taking absolute values avoids branch choices and gives

        |Z_1|^2 = |A|^{4/3}.
    """
    factor = lambda_one_geometric_z1_factor(
        b_points,
        c_point,
        surface,
        divisor_points=divisor_points,
        normalization_point=normalization_point,
        normalization_value=normalization_value,
        Delta=Delta,
        nmax=nmax,
        tol=tol,
        quad_limit=quad_limit,
    )
    return float(abs(factor) ** (4.0 / 3.0))


def abs_z1_sq_from_renormalized_det(
    surface: RiemannSurfaceData,
    *,
    normalization_factor: float = 1.0,
    renormalized_det_factor: float,
) -> float:
    r"""
    Evaluate

        |Z_1|^2 = N_1 [det Im(Omega)]^{1/2} exp(c(Omega)),

    where exp(c(Omega)) is the renormalized finite part extracted from the
    large-L fit to -1/2 log(det A').

    The default `normalization_factor=1.0` is the canonical convention where
    the renormalized determinant formula itself defines `|Z_1|^2`, i.e.

        |Z_1|^2 = [det Im(Omega)]^{1/2} exp(c(Omega)).

    In the notation of Strebel.tex this corresponds to choosing
    `mathcal{N}_1 = 1` (equivalently `log mathcal{N}_1 = 0`).
    """
    im_omega = np.asarray(np.imag(surface.Omega), dtype=np.float64)
    det_im_omega = float(np.linalg.det(im_omega))
    if det_im_omega <= 0.0:
        raise ValueError("Need det(Im Omega) > 0 to evaluate |Z_1|^2.")
    return float(normalization_factor) * np.sqrt(det_im_omega) * float(renormalized_det_factor)


def canonical_abs_z1_sq(
    surface: RiemannSurfaceData,
    *,
    renormalized_det_factor: float,
) -> float:
    r"""
    Return the canonical `|Z_1|^2` defined by the renormalized determinant.

    This is a thin wrapper for the convention

        |Z_1|^2 = [det Im(Omega)]^{1/2} exp(c(Omega)),

    with no extra moduli-independent factor.
    """
    return abs_z1_sq_from_renormalized_det(
        surface,
        normalization_factor=1.0,
        renormalized_det_factor=renormalized_det_factor,
    )


def canonical_chiral_z1(abs_z1_sq: float) -> np.complex128:
    r"""
    Return the naive chiral choice `Z_1 = +sqrt(|Z_1|^2)`.

    This is a convention choice for the gravitational-anomaly phase: we take
    the positive real square root of the canonically normalized `|Z_1|^2`.
    """
    abs_z1_sq = float(abs_z1_sq)
    if abs_z1_sq < 0.0:
        raise ValueError(f"Need |Z_1|^2 >= 0, got {abs_z1_sq}.")
    return np.complex128(np.sqrt(abs_z1_sq))


def _principal_nth_root(value: complex, n: int) -> np.complex128:
    if n <= 0:
        raise ValueError(f"Need n >= 1 for an nth root, got n={n}.")
    value = np.complex128(value)
    radius = float(abs(value))
    if radius == 0.0:
        return np.complex128(0.0)
    angle = float(np.angle(value))
    return np.complex128(radius ** (1.0 / n) * np.exp(1j * angle / n))


def sigma_scale_from_z1(
    anchor_b_points: Sequence[complex],
    anchor_c_point: complex,
    surface: RiemannSurfaceData,
    *,
    divisor_points: Sequence[complex],
    normalization_point: complex,
    normalization_value: complex = 1.0 + 0.0j,
    z1: complex | None = None,
    abs_z1_sq: float | None = None,
    renormalized_det_factor: float | None = None,
    Delta=None,
    nmax: int | None = None,
    tol: float = 1e-12,
    quad_limit: int = 200,
) -> np.complex128:
    r"""
    Fix the overall sigma normalization from a chosen chiral `Z_1`.

    Let `tilde_sigma` be the currently normalized sigma produced by
    `sigma_value(...)`. If the canonically normalized sigma is

        sigma(z) = C * tilde_sigma(z),

    then for genus `g > 1` the `lambda=1`, `(n,m)=(g,1)` Strebel equation
    determines `C` through

        C^(g-1) = Z_1^(3/2) / A_tilde,

    where `A_tilde` is the value returned by
    `lambda_one_geometric_z1_factor(...)` computed with `tilde_sigma`.

    This helper returns the principal `(g-1)`-st root `C`.

    Notes
    -----
    - For genus 1 the overall sigma constant cancels out of the special
      equation, so this procedure cannot determine sigma normalization.
    - If `z1` is omitted, the function uses the naive chiral choice
      `Z_1 = +sqrt(|Z_1|^2)`. You may supply `abs_z1_sq` directly or
      `renormalized_det_factor`, in which case the canonical convention
      `|Z_1|^2 = [det Im(Omega)]^(1/2) exp(c(Omega))` is used.
    """
    if surface.genus <= 1:
        raise ValueError(
            "sigma_scale_from_z1 only fixes an overall sigma constant for genus > 1; "
            "at genus 1 the lambda=1 special equation is insensitive to sigma -> C sigma."
        )

    if z1 is None:
        if abs_z1_sq is None:
            if renormalized_det_factor is None:
                raise ValueError(
                    "Provide either z1, abs_z1_sq, or renormalized_det_factor "
                    "to fix the sigma normalization."
                )
            abs_z1_sq = canonical_abs_z1_sq(
                surface,
                renormalized_det_factor=renormalized_det_factor,
            )
        z1 = canonical_chiral_z1(float(abs_z1_sq))
    z1 = np.complex128(z1)
    if abs(z1) == 0.0:
        raise ZeroDivisionError("z1 must be nonzero to fix the sigma normalization.")

    a_tilde = lambda_one_geometric_z1_factor(
        anchor_b_points,
        anchor_c_point,
        surface,
        divisor_points=divisor_points,
        normalization_point=normalization_point,
        normalization_value=normalization_value,
        Delta=Delta,
        nmax=nmax,
        tol=tol,
        quad_limit=quad_limit,
    )
    if abs(a_tilde) == 0.0:
        raise ZeroDivisionError("lambda_one_geometric_z1_factor vanished for the chosen anchor data.")

    z1_three_halves = z1 * np.sqrt(z1)
    scale_power = np.complex128(z1_three_halves / a_tilde)
    return _principal_nth_root(scale_power, surface.genus - 1)


def canonical_sigma_value(
    z: complex,
    surface: RiemannSurfaceData,
    *,
    anchor_b_points: Sequence[complex],
    anchor_c_point: complex,
    divisor_points: Sequence[complex],
    normalization_point: complex,
    normalization_value: complex = 1.0 + 0.0j,
    z1: complex | None = None,
    abs_z1_sq: float | None = None,
    renormalized_det_factor: float | None = None,
    Delta=None,
    nmax: int | None = None,
    tol: float = 1e-12,
    quad_limit: int = 200,
) -> np.complex128:
    r"""
    Return sigma(z) with its overall constant fixed by the chosen chiral `Z_1`.

    This multiplies the existing normalized `sigma_value(...)` by the scale
    returned from `sigma_scale_from_z1(...)`.
    """
    scale = sigma_scale_from_z1(
        anchor_b_points,
        anchor_c_point,
        surface,
        divisor_points=divisor_points,
        normalization_point=normalization_point,
        normalization_value=normalization_value,
        z1=z1,
        abs_z1_sq=abs_z1_sq,
        renormalized_det_factor=renormalized_det_factor,
        Delta=Delta,
        nmax=nmax,
        tol=tol,
        quad_limit=quad_limit,
    )
    return scale * sigma_value(
        z,
        surface,
        divisor_points=divisor_points,
        normalization_point=normalization_point,
        normalization_value=normalization_value,
        Delta=Delta,
        nmax=nmax,
        tol=tol,
        quad_limit=quad_limit,
    )


def genus2_lambda_one_sigma_kernel(
    a: complex,
    b: complex,
    w: complex,
    surface: RiemannSurfaceData,
    *,
    z1: complex,
    Delta=None,
    nmax: int | None = None,
    tol: float = 1e-12,
    quad_limit: int = 200,
) -> np.complex128:
    r"""
    Return the genus-2 kernel `H_{ab}(w)` implied by the `lambda=1` equation.

    For genus 2, the `lambda=1`, `(n,m)=(2,1)` Strebel identity gives

        sigma(w) = sigma(a) sigma(b) H_{ab}(w),

    where

        H_{ab}(w)
        =
        theta(zeta(a)+zeta(b)-zeta(w)-Delta | Omega) E(a,b)
        /
        [ Z_1^(3/2) det(omega_I(a,b)) E(a,w) E(b,w) ].
    """
    if surface.genus != 2:
        raise ValueError("genus2_lambda_one_sigma_kernel requires a genus-2 surface.")
    z1 = np.complex128(z1)
    if abs(z1) == 0.0:
        raise ZeroDivisionError("z1 must be nonzero.")

    if Delta is None:
        Delta = riemann_constant_vector(surface, quad_limit=quad_limit)
    Delta = np.asarray(Delta, dtype=np.complex128)
    if Delta.shape != (2,):
        raise ValueError(f"Delta must have shape (2,), got {Delta.shape}.")

    theta_arg = (
        abel_map(a, surface)
        + abel_map(b, surface)
        - abel_map(w, surface)
        - Delta
    )
    theta_val = riemann_theta(
        theta_arg,
        surface.Omega,
        nmax=nmax,
        tol=tol,
    )

    omega_matrix = np.asarray(
        [
            [_evaluate_one_form(form, point) for form in surface.normalized_forms]
            for point in (a, b)
        ],
        dtype=np.complex128,
    )
    det_omega = np.complex128(np.linalg.det(omega_matrix))
    if abs(det_omega) == 0.0:
        raise ZeroDivisionError("det(omega_I(a,b)) vanished for the chosen points.")

    e_ab = prime_form(a, b, surface, nmax=nmax, tol=tol)
    e_aw = prime_form(a, w, surface, nmax=nmax, tol=tol)
    e_bw = prime_form(b, w, surface, nmax=nmax, tol=tol)
    denom = z1 * np.sqrt(z1) * det_omega * e_aw * e_bw
    if abs(denom) == 0.0:
        raise ZeroDivisionError("lambda-one sigma kernel denominator vanished.")
    return np.complex128(theta_val * e_ab / denom)


def genus2_sigma_values_from_lambda_one(
    points: Sequence[complex],
    surface: RiemannSurfaceData,
    *,
    z1: complex,
    Delta=None,
    nmax: int | None = None,
    tol: float = 1e-12,
    quad_limit: int = 200,
) -> tuple[np.complex128, np.complex128, np.complex128]:
    r"""
    Solve `sigma(p_1), sigma(p_2), sigma(p_3)` directly from the genus-2
    `lambda=1` equations on the same three points.
    """
    if surface.genus != 2:
        raise ValueError("genus2_sigma_values_from_lambda_one requires a genus-2 surface.")
    points = tuple(np.complex128(point) for point in points)
    if len(points) != 3:
        raise ValueError(f"Need exactly three points, got {len(points)}.")

    p1, p2, p3 = points
    if Delta is None:
        Delta = riemann_constant_vector(surface, quad_limit=quad_limit)
    Delta = np.asarray(Delta, dtype=np.complex128)
    if Delta.shape != (2,):
        raise ValueError(f"Delta must have shape (2,), got {Delta.shape}.")

    h12 = genus2_lambda_one_sigma_kernel(
        p1,
        p2,
        p3,
        surface,
        z1=z1,
        Delta=Delta,
        nmax=nmax,
        tol=tol,
        quad_limit=quad_limit,
    )
    h13 = genus2_lambda_one_sigma_kernel(
        p1,
        p3,
        p2,
        surface,
        z1=z1,
        Delta=Delta,
        nmax=nmax,
        tol=tol,
        quad_limit=quad_limit,
    )
    h23 = genus2_lambda_one_sigma_kernel(
        p2,
        p3,
        p1,
        surface,
        z1=z1,
        Delta=Delta,
        nmax=nmax,
        tol=tol,
        quad_limit=quad_limit,
    )
    if abs(h12) == 0.0 or abs(h13) == 0.0 or abs(h23) == 0.0:
        raise ZeroDivisionError("lambda-one sigma kernel vanished for the chosen points.")

    s1_base = np.sqrt(np.complex128(1.0) / (h12 * h13))
    s2_base = np.sqrt(np.complex128(1.0) / (h12 * h23))
    s3_base = np.sqrt(np.complex128(1.0) / (h13 * h23))

    best = None
    best_residual = None
    for signs in product((1.0, -1.0), repeat=3):
        s1 = np.complex128(signs[0]) * s1_base
        s2 = np.complex128(signs[1]) * s2_base
        s3 = np.complex128(signs[2]) * s3_base
        residual = max(
            abs(s3 - s1 * s2 * h12),
            abs(s2 - s1 * s3 * h13),
            abs(s1 - s2 * s3 * h23),
        )
        if best_residual is None or residual < best_residual:
            best_residual = residual
            best = (s1, s2, s3)

    assert best is not None
    return best


def genus2_bbb_correlator_from_lambda_one(
    b_points: Sequence[complex],
    surface: RiemannSurfaceData,
    *,
    z1: complex,
    Delta=None,
    nmax: int | None = None,
    tol: float = 1e-12,
    quad_limit: int = 200,
) -> np.complex128:
    r"""
    Compute the genus-2 three-`b` correlator using direct `lambda=1` sigma data.

    This avoids the auxiliary-divisor sigma pipeline by solving the three
    needed sigma values directly from the genus-2 `lambda=1`, `(n,m)=(2,1)`
    equation on the same three insertion points.
    """
    if surface.genus != 2:
        raise ValueError("genus2_bbb_correlator_from_lambda_one requires a genus-2 surface.")
    b_points = tuple(np.complex128(point) for point in b_points)
    if len(b_points) != 3:
        raise ValueError(f"Need exactly three b-points, got {len(b_points)}.")

    if Delta is None:
        Delta = riemann_constant_vector(surface, quad_limit=quad_limit)
    Delta = np.asarray(Delta, dtype=np.complex128)
    if Delta.shape != (2,):
        raise ValueError(f"Delta must have shape (2,), got {Delta.shape}.")

    sigma_vals = genus2_sigma_values_from_lambda_one(
        b_points,
        surface,
        z1=z1,
        Delta=Delta,
        nmax=nmax,
        tol=tol,
        quad_limit=quad_limit,
    )
    sigma_prod = np.prod(np.asarray(sigma_vals, dtype=np.complex128))

    zeta_b = np.sum(
        np.asarray([abel_map(point, surface) for point in b_points], dtype=np.complex128),
        axis=0,
    )
    theta_arg = zeta_b - 3.0 * Delta
    theta_val = riemann_theta(
        theta_arg,
        surface.Omega,
        nmax=nmax,
        tol=tol,
    )

    prime_bb = np.complex128(1.0)
    for idx, zi in enumerate(b_points):
        for zj in b_points[idx + 1 :]:
            prime_bb *= prime_form(zi, zj, surface, nmax=nmax, tol=tol)

    z1 = np.complex128(z1)
    if abs(z1) == 0.0:
        raise ZeroDivisionError("z1 must be nonzero.")
    return np.complex128(theta_val * prime_bb * sigma_prod**3 / np.sqrt(z1))


def normalization_factor_from_lambda_one(
    b_points: Sequence[complex],
    c_point: complex,
    surface: RiemannSurfaceData,
    *,
    renormalized_det_factor: float,
    divisor_points: Sequence[complex],
    normalization_point: complex,
    normalization_value: complex = 1.0 + 0.0j,
    Delta=None,
    nmax: int | None = None,
    tol: float = 1e-12,
    quad_limit: int = 200,
) -> float:
    r"""
    Determine the moduli-independent normalization N_1 from the last Strebel
    equation together with the renormalized determinant finite part.

    This value is tied to the sigma normalization convention chosen through
    `normalization_point` and `normalization_value`. The same convention must
    then be used in any later bc-correlator evaluation.
    """
    abs_z1_sq = abs_z1_sq_from_lambda_one(
        b_points,
        c_point,
        surface,
        divisor_points=divisor_points,
        normalization_point=normalization_point,
        normalization_value=normalization_value,
        Delta=Delta,
        nmax=nmax,
        tol=tol,
        quad_limit=quad_limit,
    )
    im_omega = np.asarray(np.imag(surface.Omega), dtype=np.float64)
    det_im_omega = float(np.linalg.det(im_omega))
    if det_im_omega <= 0.0:
        raise ValueError("Need det(Im Omega) > 0 to determine the normalization factor.")
    if renormalized_det_factor <= 0.0:
        raise ValueError("renormalized_det_factor must be positive.")
    return float(abs_z1_sq / (np.sqrt(det_im_omega) * float(renormalized_det_factor)))


def estimate_abs_z1_sq(
    ribbon_graph,
    base_edge_lengths,
    *,
    scales,
    b_points: Sequence[complex],
    c_point: complex,
    divisor_points: Sequence[complex],
    normalization_point: complex,
    normalization_value: complex = 1.0 + 0.0j,
    surface_edge_lengths=None,
    min_edge_length: int = 200,
    basis_pairs=None,
    custom_cycles=None,
    kernel=None,
    Delta=None,
    nmax: int | None = None,
    tol: float = 1e-12,
    quad_limit: int = 200,
) -> RenormalizedZ1Data:
    r"""
    High-level helper to compute |Z_1|^2 together with the fitted N_1.

    Steps:
    1. Fit the large-L behavior of -1/2 log(det A') at fixed edge-length ratios.
    2. Build a large surface at the chosen reference edge lengths.
    3. Use the lambda=1, (n,m)=(g,1) identity to compute |Z_1|^2.
    4. Convert that into the moduli-independent normalization N_1.

    As in `normalization_factor_from_lambda_one`, the extracted N_1 is tied to
    the sigma normalization convention supplied here.
    """
    fit = fit_renormalized_aprime_factor(
        ribbon_graph,
        base_edge_lengths,
        scales=scales,
        min_edge_length=min_edge_length,
        kernel=kernel,
    )

    if surface_edge_lengths is None:
        surface_edge_lengths = fit.edge_length_sets[-1]
    surface_edge_lengths = tuple(int(x) for x in surface_edge_lengths)
    if min(surface_edge_lengths) < int(min_edge_length):
        raise ValueError(
            "surface_edge_lengths should also lie in the trusted large-L regime; "
            f"got {surface_edge_lengths} with min_edge_length={int(min_edge_length)}."
        )

    surface = build_surface_from_ribbon_graph(
        ribbon_graph,
        surface_edge_lengths,
        basis_pairs=basis_pairs,
        custom_cycles=custom_cycles,
    )
    normalization_factor = normalization_factor_from_lambda_one(
        b_points,
        c_point,
        surface,
        renormalized_det_factor=fit.finite_part,
        divisor_points=divisor_points,
        normalization_point=normalization_point,
        normalization_value=normalization_value,
        Delta=Delta,
        nmax=nmax,
        tol=tol,
        quad_limit=quad_limit,
    )
    abs_z1_sq = abs_z1_sq_from_renormalized_det(
        surface,
        normalization_factor=normalization_factor,
        renormalized_det_factor=fit.finite_part,
    )
    return RenormalizedZ1Data(
        abs_z1_sq=float(abs_z1_sq),
        normalization_factor=float(normalization_factor),
        renormalized_det_factor=float(fit.finite_part),
        fit=fit,
        surface=surface,
    )


def estimate_canonical_abs_z1_sq(
    ribbon_graph,
    base_edge_lengths,
    *,
    scales,
    surface_edge_lengths=None,
    min_edge_length: int = 200,
    basis_pairs=None,
    custom_cycles=None,
    kernel=None,
) -> RenormalizedZ1Data:
    r"""
    High-level helper for the canonical convention `mathcal{N}_1 = 1`.

    This uses the renormalized determinant formula itself as the definition of
    `|Z_1|^2`, without trying to extract a separate normalization from the
    `lambda=1`, `(n,m)=(g,1)` identity.
    """
    fit = fit_renormalized_aprime_factor(
        ribbon_graph,
        base_edge_lengths,
        scales=scales,
        min_edge_length=min_edge_length,
        kernel=kernel,
    )

    if surface_edge_lengths is None:
        surface_edge_lengths = fit.edge_length_sets[-1]
    surface_edge_lengths = tuple(int(x) for x in surface_edge_lengths)
    if min(surface_edge_lengths) < int(min_edge_length):
        raise ValueError(
            "surface_edge_lengths should also lie in the trusted large-L regime; "
            f"got {surface_edge_lengths} with min_edge_length={int(min_edge_length)}."
        )

    surface = build_surface_from_ribbon_graph(
        ribbon_graph,
        surface_edge_lengths,
        basis_pairs=basis_pairs,
        custom_cycles=custom_cycles,
    )
    abs_z1_sq = canonical_abs_z1_sq(
        surface,
        renormalized_det_factor=fit.finite_part,
    )
    return RenormalizedZ1Data(
        abs_z1_sq=float(abs_z1_sq),
        normalization_factor=1.0,
        renormalized_det_factor=float(fit.finite_part),
        fit=fit,
        surface=surface,
    )
