from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, quad

import ell_to_tau as elt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PICTURES_DIR = ROOT / "covariant formalism" / "pictures"


def build_normalized_form(L: int, l1: int, l2: int):
    """
    Return a vectorized evaluator for the A-normalized genus-1 holomorphic
    one-form coefficient f(z) appearing in u(z)=∫^z f(z') dz'.
    """
    f = elt.make_cyl_eqn_improved(L, l1, l2)
    P1, P2, _ = elt.periods_improved(L, l1, l2, f=f)

    coeffs = np.asarray(f.coeffs, dtype=np.complex128)
    half = L // 2
    l3 = half - l1 - l2
    if l3 < 0:
        raise ValueError("Need l1 + l2 <= L/2.")

    phase1 = 2.0 * np.pi * l1 / L
    phase2 = 2.0 * np.pi * (l1 + l2) / L
    w1 = np.exp(-2j * phase1)
    w2 = np.exp(-2j * phase2)

    def normalized_form(z):
        z = np.asarray(z, dtype=np.complex128)
        if l3 == 0:
            singular = (1.0 - z**2) ** (-0.5) * (1.0 - z**2 * w1) ** (-0.5)
        else:
            singular = (
                (1.0 - z**2) ** (-1.0 / 3.0)
                * (1.0 - z**2 * w1) ** (-1.0 / 3.0)
                * (1.0 - z**2 * w2) ** (-1.0 / 3.0)
            )
        poly = np.polynomial.polynomial.polyval(z, coeffs)
        return singular * poly / P1

    return normalized_form, np.complex128(P2 / P1)


def radial_u_curve(
    normalized_form,
    radius: float,
    thetas: np.ndarray,
    *,
    radial_steps: int,
) -> np.ndarray:
    """
    Evaluate u(r e^{iθ}) by integrating the normalized one-form along the
    straight radial segment from 0 to r e^{iθ}.
    """
    r_grid = np.linspace(0.0, radius, radial_steps, dtype=np.float64)
    phase = np.exp(1j * thetas)[None, :]
    z_grid = r_grid[:, None] * phase
    integrand = normalized_form(z_grid) * phase
    u_grid = cumulative_trapezoid(integrand, r_grid, axis=0, initial=0.0)
    return np.asarray(u_grid[-1], dtype=np.complex128)


def u_at_point(normalized_form, z: complex) -> np.complex128:
    """Evaluate u(z)=∫_0^z omega along the straight radial segment."""
    z = np.complex128(z)

    def integrand_re(t):
        return (normalized_form(t * z) * z).real

    def integrand_im(t):
        return (normalized_form(t * z) * z).imag

    re_part, _ = quad(integrand_re, 0.0, 1.0, limit=300)
    im_part, _ = quad(integrand_im, 0.0, 1.0, limit=300)
    return np.complex128(re_part + 1j * im_part)


def interaction_points_u(L: int, l1: int, l2: int, normalized_form) -> list[np.complex128]:
    """
    Map all six boundary prevertices to the u-plane.

    These are lattice translates of the two torus interaction points, but we
    keep all six representatives so they appear directly on the image of
    the unit circle.
    """
    geom = elt._theta_graph_prevertex_data(L, l1, l2)
    singular_points = np.asarray(geom["singular_points"], dtype=np.complex128)
    return [u_at_point(normalized_form, z) for z in singular_points]


def make_theta_segments(
    *,
    radius: float,
    n_theta: int,
    singular_angles: np.ndarray,
    gap: float,
) -> list[np.ndarray]:
    if radius < 1.0 - 1e-12:
        return [np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)]

    cuts = np.sort(np.mod(singular_angles, 2.0 * np.pi))
    segments: list[np.ndarray] = []
    for idx, start in enumerate(cuts):
        stop = cuts[(idx + 1) % len(cuts)]
        if idx == len(cuts) - 1:
            stop += 2.0 * np.pi
        left = start + gap
        right = stop - gap
        if right <= left:
            continue
        count = max(24, n_theta // len(cuts))
        theta = np.linspace(left, right, count, endpoint=True)
        theta = np.mod(theta, 2.0 * np.pi)
        segments.append(theta)
    return segments


def plot_circle_images(
    *,
    l1: int,
    l2: int,
    l3: int,
    radii: list[float],
    radial_steps: int,
    n_theta: int,
    output: Path,
):
    L = 2 * (l1 + l2 + l3)
    normalized_form, tau = build_normalized_form(L, l1, l2)
    interactions = interaction_points_u(L, l1, l2, normalized_form)

    singular_angles = np.array(
        [
            0.0,
            np.pi,
            2.0 * np.pi * l1 / L,
            np.pi + 2.0 * np.pi * l1 / L,
            2.0 * np.pi * (l1 + l2) / L,
            np.pi + 2.0 * np.pi * (l1 + l2) / L,
        ],
        dtype=np.float64,
    )

    palette = ["#440154", "#3b528b", "#21918c", "#5ec962", "#000000"]
    fig, ax = plt.subplots(figsize=(8.5, 8.5))

    for idx, radius in enumerate(radii):
        color = palette[idx % len(palette)]
        segments = make_theta_segments(
            radius=radius,
            n_theta=n_theta,
            singular_angles=singular_angles,
            gap=0.02,
        )
        for thetas in segments:
            u_vals = radial_u_curve(
                normalized_form,
                radius=radius,
                thetas=thetas,
                radial_steps=radial_steps,
            )
            ax.plot(u_vals.real, u_vals.imag, color=color, lw=1.8)

    if interactions:
        interaction_array = np.asarray(interactions, dtype=np.complex128)
        ax.scatter(
            interaction_array.real,
            interaction_array.imag,
            s=52,
            color="#d62728",
            edgecolors="black",
            linewidths=0.7,
            zorder=5,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\Re u$")
    ax.set_ylabel(r"$\Im u$")
    ax.set_title(
        rf"Images of $|z|=0.2,0.4,0.6,0.8,1$ for $(\ell_1,\ell_2,\ell_3)=({l1},{l2},{l3})$"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)

    print(f"Saved plot to {output}")
    print(f"tau = {tau}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot the image of circles |z|=const in the flat torus coordinate u."
    )
    parser.add_argument("--l1", type=int, default=200)
    parser.add_argument("--l2", type=int, default=300)
    parser.add_argument("--l3", type=int, default=600)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PICTURES_DIR / "circle_images_l200_l300_l600.png",
    )
    parser.add_argument("--radial-steps", type=int, default=600)
    parser.add_argument("--n-theta", type=int, default=360)
    args = parser.parse_args()

    plot_circle_images(
        l1=args.l1,
        l2=args.l2,
        l3=args.l3,
        radii=[0.2, 0.4, 0.6, 0.8, 1.0],
        radial_steps=args.radial_steps,
        n_theta=args.n_theta,
        output=args.output,
    )


if __name__ == "__main__":
    main()
