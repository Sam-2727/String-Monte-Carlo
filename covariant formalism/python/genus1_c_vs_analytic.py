from __future__ import annotations

"""Compare the fitted genus-1 finite term c(tau) with analytic torus data.

For each fixed theta-graph length ratio, this script sweeps

    L = L_start, L_start + L_step, ..., L_stop

at fixed radius R and fits

    log Z(L) = c(tau) + gamma * L + alpha * log L.

It then evaluates the analytic torus compact-boson partition function at the
corresponding modulus tau and reports

    exp(c(tau)) / Z_analytic(tau).

It also computes the moduli-dependent part of the Weyl anomaly in the disc
coordinate from one of several local-coefficient conventions:

    --weyl-mode nu_z:
        S_L(moduli) = (log|nu_1| + log|nu_2|) / 24

    --weyl-mode b_z36:
        S_L(moduli) = (log|b_1| + log|b_2|) / 36

    --weyl-mode b_z36_puncture:
        S_L(moduli) = (log|b_1| + log|b_2|) / 36 + log|P_1| / 12

    --weyl-mode b_u24:
        S_L(moduli) = (log|b_1| + log|b_2|) / 24

For the torus theta graph, the two cubic vertices have the same magnitude by
symmetry, so these reduce to

    S_L(moduli) = log|nu| / 12
    S_L(moduli) = log|b| / 18
    S_L(moduli) = log|b| / 12

The current genus-1 helpers expose the regularized disc-coordinate coefficient
nu directly. For the b-based conventions we therefore reconstruct |b| from
|nu| using

    |nu| = (2/3)^(1/3) |b|^(2/3),

which follows from the local relation quoted in the notes.

"""

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import compact_partition as cp
import ell_to_tau as elt


DEFAULT_RATIOS = (
    ("1:1:3", (1, 1, 3)),
    ("1:2:2", (1, 2, 2)),
    ("1:3:6", (1, 3, 6)),
    ("1:4:5", (1, 4, 5)),
    ("2:3:5", (2, 3, 5)),
    ("3:3:4", (3, 3, 4)),
)


@dataclass(frozen=True)
class CaseResult:
    ratio: str
    weyl_mode: str
    tau_raw_real: float
    tau_raw_imag: float
    tau_real: float
    tau_imag: float
    tau_drift_abs: float
    c: float
    gamma: float
    alpha: float
    exp_c: float
    z_analytic: float
    exp_c_over_z: float
    weyl_local_abs: float
    S_L_moduli: float
    exp_S_L_moduli: float
    exp_c_over_z_exp_sl: float
    r2: float
    max_abs_log_residual: float


def _parse_ratio(text: str) -> tuple[int, int, int]:
    pieces = text.split(":")
    if len(pieces) != 3:
        raise ValueError(f"Invalid ratio '{text}'. Expected form a:b:c.")
    values = tuple(int(piece) for piece in pieces)
    if any(value <= 0 for value in values):
        raise ValueError(f"Invalid ratio '{text}'. All entries must be positive.")
    return values


def _build_l_values(start: int, stop: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("step must be positive")
    values = list(range(start, stop + 1, step))
    if not values:
        raise ValueError("empty L-range")
    return values


def _scaled_lengths(total_L: int, ratio: tuple[int, int, int]) -> tuple[int, int, int]:
    denom = 2 * sum(ratio)
    if total_L % denom != 0:
        raise ValueError(
            f"L={total_L} is incompatible with ratio {ratio}; need divisibility by {denom}."
        )
    scale = total_L // denom
    return tuple(scale * value for value in ratio)


def _fit_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    l_arr = np.array([row["L"] for row in rows], dtype=float)
    logz = np.array([row["logZ"] for row in rows], dtype=float)
    design = np.column_stack([np.ones_like(l_arr), l_arr, np.log(l_arr)])
    coef, *_ = np.linalg.lstsq(design, logz, rcond=None)
    predicted = design @ coef
    residual = logz - predicted
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((logz - np.mean(logz)) ** 2))
    c0, gamma, alpha = coef
    return {
        "c": float(c0),
        "gamma": float(gamma),
        "alpha": float(alpha),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
        "max_abs_log_residual": float(np.max(np.abs(residual))),
    }


def _reduce_tau_to_fundamental_domain(tau: complex) -> complex:
    tau = complex(tau)
    if tau.imag <= 0:
        raise ValueError("Need Im(tau) > 0.")

    for _ in range(100):
        shifted = tau - round(tau.real)
        if abs(shifted.real - tau.real) > 1e-15:
            tau = shifted
            continue

        if abs(tau) < 1.0 - 1e-14:
            tau = -1.0 / tau
            continue

        if tau.real > 0.5 + 1e-14:
            tau -= 1.0
            continue
        if tau.real < -0.5 - 1e-14:
            tau += 1.0
            continue
        break
    return tau


def _compute_tau(total_L: int, ratio: tuple[int, int, int]) -> complex:
    l1, l2, _ = _scaled_lengths(total_L, ratio)
    form = elt.make_cyl_eqn_improved(total_L, l1, l2)
    p1, p2, _ = elt.periods_improved(total_L, l1, l2, form)
    return complex(p2 / p1)


def _compute_p1_abs(total_L: int, ratio: tuple[int, int, int]) -> float:
    l1, l2, _ = _scaled_lengths(total_L, ratio)
    form = elt.make_cyl_eqn_improved(total_L, l1, l2)
    p1, _, _ = elt.periods_improved(total_L, l1, l2, form)
    return float(abs(p1))


def _compute_weyl_moduli_term(
    total_L: int,
    ratio: tuple[int, int, int],
    weyl_mode: str,
) -> tuple[float, float]:
    l1, l2, _ = _scaled_lengths(total_L, ratio)
    if weyl_mode == "none":
        return 1.0, 0.0

    averaged_nu = np.asarray(
        elt.average_nu(L=total_L, l1=l1, l2=l2, normalize_A=True),
        dtype=np.complex128,
    )
    nu_abs = float(np.mean(np.abs(averaged_nu)))

    if weyl_mode == "nu_z":
        local_abs = nu_abs
        return local_abs, float(math.log(local_abs) / 12.0)

    # The current genus-1 helpers expose nu, not the u-coordinate coefficient b.
    # Reconstruct |b| from |nu| using |nu| = (2/3)^(1/3) |b|^(2/3).
    local_abs = float(math.sqrt(1.5) * (nu_abs ** 1.5))
    if weyl_mode == "b_z36":
        return local_abs, float(math.log(local_abs) / 18.0)
    if weyl_mode == "b_z36_puncture":
        p1_abs = _compute_p1_abs(total_L, ratio)
        return local_abs, float(math.log(local_abs) / 18.0 + math.log(p1_abs) / 12.0)
    if weyl_mode == "b_u24":
        return local_abs, float(math.log(local_abs) / 12.0)
    raise ValueError(f"Unsupported weyl_mode {weyl_mode!r}")


def _theta_sum_dim2(tp: np.ndarray, radius: float, cutoff: int = 20) -> float:
    side = np.arange(-cutoff, cutoff + 1, dtype=float)
    s1, s2 = np.meshgrid(side, side, indexing="ij")
    s1 = s1.ravel()
    s2 = s2.ravel()
    quad = (
        s1 * s1 * tp[0, 0]
        + s1 * s2 * tp[0, 1]
        + s2 * s1 * tp[1, 0]
        + s2 * s2 * tp[1, 1]
    )
    return float(np.sum(np.exp(-4.0 * math.pi * (radius ** 2) * quad)))


def _compute_logz(total_L: int, ratio: tuple[int, int, int], radius: float) -> float:
    l1, l2, _ = _scaled_lengths(total_L, ratio)
    mat = cp.direct_mat_n_fast(total_L)
    aprime = cp.direct_red_traced_mat(total_L, l1, l2, mat)
    w = cp.mat_w(total_L, l1, l2, mat)
    t1 = cp.mat_t_first_part(total_L, l1, l2, mat)
    t2 = cp.mat_t_second_part(total_L, l1, l2, w, aprime)
    tp = cp.mat_t_prime(cp.symm(t1 - t2))
    sign, logdet = np.linalg.slogdet(aprime)
    if sign <= 0:
        raise RuntimeError(f"Encountered non-positive determinant sign={sign} at L={total_L}.")
    theta = _theta_sum_dim2(tp, radius, cutoff=20)
    return float(math.log(radius) - 0.5 * logdet + math.log(theta))


def compute_case(
    ratio_name: str,
    ratio: tuple[int, int, int],
    radius: float,
    l_values: list[int],
    analytic_cutoff: int,
    weyl_mode: str,
) -> CaseResult:
    rows = []
    for total_L in l_values:
        rows.append({"L": total_L, "logZ": _compute_logz(total_L, ratio, radius)})

    fit = _fit_rows(rows)
    tau_start = _compute_tau(l_values[0], ratio)
    tau_end = _compute_tau(l_values[-1], ratio)
    tau_reduced = _reduce_tau_to_fundamental_domain(tau_end)
    tau_drift_abs = abs(tau_end - tau_start)
    z_analytic = float(cp.z_compact(radius, tau_reduced, N=analytic_cutoff))
    local_abs, s_l_moduli = _compute_weyl_moduli_term(l_values[-1], ratio, weyl_mode)
    exp_s_l_moduli = float(math.exp(s_l_moduli))
    exp_c = float(math.exp(fit["c"]))
    return CaseResult(
        ratio=ratio_name,
        weyl_mode=weyl_mode,
        tau_raw_real=float(tau_end.real),
        tau_raw_imag=float(tau_end.imag),
        tau_real=float(tau_reduced.real),
        tau_imag=float(tau_reduced.imag),
        tau_drift_abs=float(tau_drift_abs),
        c=fit["c"],
        gamma=fit["gamma"],
        alpha=fit["alpha"],
        exp_c=exp_c,
        z_analytic=z_analytic,
        exp_c_over_z=float(exp_c / z_analytic),
        weyl_local_abs=local_abs,
        S_L_moduli=s_l_moduli,
        exp_S_L_moduli=exp_s_l_moduli,
        exp_c_over_z_exp_sl=float(exp_c / (z_analytic * exp_s_l_moduli)),
        r2=fit["r2"],
        max_abs_log_residual=fit["max_abs_log_residual"],
    )


def _print_table(results: list[CaseResult], weyl_mode: str) -> None:
    local_label = "|nu|" if weyl_mode == "nu_z" else "|b|"
    header = (
        f"{'ratio':>8}  {'tau':>25}  {'exp(c)/(Z e^S)':>16}  "
        f"{'S_L':>10}  {local_label:>10}  {'|dtau|':>10}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        tau_text = f"{result.tau_real:+.12f} {result.tau_imag:+.12f}i"
        print(
            f"{result.ratio:>8}  {tau_text:>25}  {result.exp_c_over_z_exp_sl:16.12f}  "
            f"{result.S_L_moduli:10.6f}  {result.weyl_local_abs:10.6f}  {result.tau_drift_abs:10.3e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--R", type=float, default=1.0, help="Compactification radius.")
    parser.add_argument("--L-start", type=int, default=2000, help="First L value.")
    parser.add_argument("--L-stop", type=int, default=5000, help="Last L value.")
    parser.add_argument("--L-step", type=int, default=200, help="Step in L.")
    parser.add_argument(
        "--ratio",
        action="append",
        default=[],
        help="Length ratio a:b:c. May be passed multiple times.",
    )
    parser.add_argument(
        "--analytic-cutoff",
        type=int,
        default=40,
        help="Lattice cutoff N for the analytic torus sum.",
    )
    parser.add_argument(
        "--weyl-mode",
        choices=("none", "nu_z", "b_z36", "b_z36_puncture", "b_u24"),
        default="nu_z",
        help="Convention for the moduli-dependent Weyl anomaly term.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for machine-readable output.",
    )
    args = parser.parse_args()

    l_values = _build_l_values(args.L_start, args.L_stop, args.L_step)
    ratio_cases = list(DEFAULT_RATIOS)
    if args.ratio:
        ratio_cases.extend((text, _parse_ratio(text)) for text in args.ratio)

    results = [
        compute_case(
            ratio_name=name,
            ratio=ratio,
            radius=args.R,
            l_values=l_values,
            analytic_cutoff=args.analytic_cutoff,
            weyl_mode=args.weyl_mode,
        )
        for name, ratio in ratio_cases
    ]

    print(
        json.dumps(
            {
                "radius": args.R,
                "l_values": l_values,
                "analytic_cutoff": args.analytic_cutoff,
                "weyl_mode": args.weyl_mode,
                "results": [asdict(result) for result in results],
            },
            indent=2,
        )
    )
    print()
    _print_table(results, args.weyl_mode)

    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(
                {
                    "radius": args.R,
                    "l_values": l_values,
                    "analytic_cutoff": args.analytic_cutoff,
                    "weyl_mode": args.weyl_mode,
                    "results": [asdict(result) for result in results],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
