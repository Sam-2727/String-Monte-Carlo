#!/usr/bin/env python3
r"""
Systematic low-point numerical validation for the discrete-sigma lightcone method.

This script organizes the current low-point diagnostics into one report with
explicit continuum fits:

1. Bosonic three-tachyon (TTT):
   - exact legwise factorization residual on a finite-N grid,
   - D_perp scan around the critical bosonic value,
   - invariant tail residual after subtracting the observed 7,7,-5, pi,
     and pi^2/72 large-N structure.

2. Bosonic two-tachyon/one-massless (TTM):
   - fixed-ratio continuum fits for A_tr and B_rel,
   - ratio samples at a large common scale.

3. Superstring bosonic prefactor diagnostics:
   - fixed-ratio continuum fits for representative support-three local families,
   - ratio samples for the current best symmetric second-order candidate,
   - optional coarse family-grid scan.

The goal is not to re-prove standard continuum lightcone consistency, but to
show systematically that the discrete-sigma numerics reproduce the expected
low-point amplitude structure with smooth continuum behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bosonic_massless_check as bmc
import continuum_extrapolation as ce
import superstring_prefactor_check as spc
import tachyon_check as tc


@dataclass
class ContinuumFit:
    intercept: float
    slope: float
    rmse: float
    max_abs: float


@dataclass
class FitStability:
    primary: ContinuumFit
    window_count: int
    intercept_min: float
    intercept_max: float
    intercept_std: float
    slope_min: float
    slope_max: float
    slope_std: float
    rmse_max: float


def fit_stability(ns: list[int], values: list[float], min_points: int = 3) -> FitStability:
    """
    Fit c0 + c1/N on all suffix windows with at least `min_points` samples.

    This gives a basic continuum-extrapolation stability diagnostic without
    assuming that the smallest-N points should be trusted equally.
    """
    if len(ns) < min_points:
        raise ValueError(f"need at least {min_points} points for a stability fit")

    window_fits: list[ContinuumFit] = []
    for start in range(0, len(ns) - min_points + 1):
        window_fits.append(fit_const_plus_inv_n(ns[start:], values[start:]))

    primary = window_fits[0]
    intercepts = np.array([fit.intercept for fit in window_fits], dtype=float)
    slopes = np.array([fit.slope for fit in window_fits], dtype=float)
    rmses = np.array([fit.rmse for fit in window_fits], dtype=float)
    return FitStability(
        primary=primary,
        window_count=len(window_fits),
        intercept_min=float(np.min(intercepts)),
        intercept_max=float(np.max(intercepts)),
        intercept_std=float(np.std(intercepts)),
        slope_min=float(np.min(slopes)),
        slope_max=float(np.max(slopes)),
        slope_std=float(np.std(slopes)),
        rmse_max=float(np.max(rmses)),
    )


def fit_stability_to_dict(fit: FitStability) -> dict[str, object]:
    result = asdict(fit)
    result["primary"] = asdict(fit.primary)
    return result


def fit_const_plus_inv_n(ns: list[int], values: list[float]) -> ContinuumFit:
    """Fit values ~= c0 + c1 / N."""
    xs = 1.0 / np.array(ns, dtype=float)
    ys = np.array(values, dtype=float)
    design = np.column_stack([np.ones_like(xs), xs])
    coeffs, _, _, _ = np.linalg.lstsq(design, ys, rcond=None)
    residuals = ys - design @ coeffs
    return ContinuumFit(
        intercept=float(coeffs[0]),
        slope=float(coeffs[1]),
        rmse=float(math.sqrt(np.mean(residuals * residuals))),
        max_abs=float(np.max(np.abs(residuals))),
    )


def tachyon_rows(
    pairs: list[tuple[int, int]], alpha_prime: float, d_perp: int
) -> list[tuple[int, int, int, tc.TachyonData]]:
    rows = []
    for n1, n2 in pairs:
        data = tc.compute_tachyon_data(n1, n2, alpha_prime, d_perp)
        rows.append((n1, n2, n1 + n2, data))
    return rows


def invariant_tail_residual(
    rows: list[tuple[int, int, int, tc.TachyonData]]
) -> tuple[float, float, float]:
    """
    Residual after subtracting the observed gauge-invariant three-leg tail

        7 log N1 + 7 log N2 - 5 log N3
      + pi (1/N1 + 1/N2 - 1/N3)
      + (pi^2/72) (1/N1^2 + 1/N2^2 + 1/N3^2)

    and fitting only one overall constant.
    """
    exact_c2 = math.pi * math.pi / 72.0
    shifted = []
    for n1, n2, n3, data in rows:
        tail = (
            7.0 * math.log(n1)
            + 7.0 * math.log(n2)
            - 5.0 * math.log(n3)
            + math.pi * (1.0 / n1 + 1.0 / n2 - 1.0 / n3)
            + exact_c2 * (1.0 / (n1 * n1) + 1.0 / (n2 * n2) + 1.0 / (n3 * n3))
        )
        shifted.append(data.log_required_norm_noext - tail)
    shifted_arr = np.array(shifted)
    constant = float(np.mean(shifted_arr))
    residuals = shifted_arr - constant
    rmse = float(math.sqrt(np.mean(residuals * residuals)))
    max_abs = float(np.max(np.abs(residuals)))
    return constant, rmse, max_abs


def tachyon_summary(
    alpha_prime: float, d_perp: int, grid_min_n: int, grid_max_n: int, grid_max_n3: int
) -> dict[str, object]:
    pairs = tc.grid_pairs(grid_min_n, grid_max_n, grid_max_n3)
    rows = tachyon_rows(pairs, alpha_prime, d_perp)
    rmse, max_abs = tc.factorization_errors(pairs, alpha_prime, d_perp)
    const, tail_rmse, tail_max = invariant_tail_residual(rows)
    d_scan = []
    for d_trial in range(max(0, d_perp - 2), d_perp + 3):
        rmse_trial, max_trial = tc.factorization_errors(pairs, alpha_prime, d_trial)
        d_scan.append(
            {"d_perp": d_trial, "rmse": rmse_trial, "max_abs": max_trial}
        )
    return {
        "grid": {
            "min_n": grid_min_n,
            "max_n": grid_max_n,
            "max_n3": grid_max_n3,
            "pair_count": len(pairs),
        },
        "d_perp": d_perp,
        "factorization": {"rmse": rmse, "max_abs": max_abs},
        "d_perp_scan": d_scan,
        "invariant_tail": {
            "constant": const,
            "rmse": tail_rmse,
            "max_abs": tail_max,
        },
    }


def print_tachyon_summary(
    alpha_prime: float, d_perp: int, grid_min_n: int, grid_max_n: int, grid_max_n3: int
) -> dict[str, object]:
    summary = tachyon_summary(alpha_prime, d_perp, grid_min_n, grid_max_n, grid_max_n3)
    print("=" * 92)
    print("BOSONIC THREE-TACHYON VALIDATION")
    print("=" * 92)
    print(
        "Grid: "
        f"{summary['grid']['min_n']} <= N1,N2 <= {summary['grid']['max_n']}, "
        f"N3 <= {summary['grid']['max_n3']}, D_perp = {summary['d_perp']}"
    )
    print(
        "Exact legwise factorization residual for "
        "log C_req^(B)(N1,N2) = f_in(N1) + f_in(N2) + f_out(N3) + const:"
    )
    print(f"  rmse    = {summary['factorization']['rmse']:.12e}")
    print(f"  max_abs = {summary['factorization']['max_abs']:.12e}")
    print()
    print("D_perp scan of the same factorization residual:")
    print(f"{'D_perp':>8s} {'rmse':>18s} {'max_abs':>18s}")
    print("-" * 48)
    for row in summary["d_perp_scan"]:
        print(f"{row['d_perp']:8d} {row['rmse']:18.12e} {row['max_abs']:18.12e}")
    print()
    print("Gauge-invariant three-leg tail test:")
    print(
        "  subtract 7 log N1 + 7 log N2 - 5 log N3, "
        "pi(1/N1 + 1/N2 - 1/N3), and (pi^2/72)(1/N1^2 + 1/N2^2 + 1/N3^2),"
    )
    print("  then fit only one overall constant.")
    print(f"  fitted constant = {summary['invariant_tail']['constant']:.12f}")
    print(f"  rmse            = {summary['invariant_tail']['rmse']:.12e}")
    print(f"  max_abs         = {summary['invariant_tail']['max_abs']:.12e}")
    print()
    return summary


def massless_summary(alpha_prime: float, ratio_scale: int) -> dict[str, object]:
    fixed_ratio_pairs = [(16, 24), (32, 48), (64, 96), (128, 192), (256, 384)]
    ns = []
    a_vals = []
    b_vals = []
    for n1, n2 in fixed_ratio_pairs:
        data = bmc.compute_massless_data(n1, n2, alpha_prime)
        ns.append(n1)
        a_vals.append(data.a_trace_reduced)
        b_vals.append(data.b_rel_reduced)
    fit_a = fit_stability(ns, a_vals)
    fit_b = fit_stability(ns, b_vals)
    extrap_a = ce.summarize_extrapolation(ns, a_vals)
    extrap_b = ce.summarize_extrapolation(ns, b_vals)
    ratio_rows = []
    for a, b in bmc.RATIO_SCAN_FAMILIES:
        n1 = ratio_scale * a
        n2 = ratio_scale * b
        data = bmc.compute_massless_data(n1, n2, alpha_prime)
        ratio_rows.append(
            {
                "a": a,
                "b": b,
                "lambda": n1 / (n1 + n2),
                "N1_A_tr": n1 * data.a_trace_reduced,
                "B_rel": data.b_rel_reduced,
            }
        )
    return {
        "fixed_ratio_pairs": fixed_ratio_pairs,
        "A_tr_fit": fit_stability_to_dict(fit_a),
        "B_rel_fit": fit_stability_to_dict(fit_b),
        "A_tr_extrapolation": ce.summary_to_dict(extrap_a),
        "B_rel_extrapolation": ce.summary_to_dict(extrap_b),
        "ratio_scale": ratio_scale,
        "ratio_rows": ratio_rows,
    }


def print_massless_summary(alpha_prime: float, ratio_scale: int) -> dict[str, object]:
    summary = massless_summary(alpha_prime, ratio_scale)
    print("=" * 92)
    print("BOSONIC TWO-TACHYON / ONE-MASSLESS VALIDATION")
    print("=" * 92)
    print("Fixed ratio N1:N2 = 2:3 continuum fits:")
    a_fit = summary["A_tr_fit"]["primary"]
    b_fit = summary["B_rel_fit"]["primary"]
    print(f"  A_tr  ~= {a_fit['intercept']:.12f} + {a_fit['slope']:.12f} / N1")
    print(f"           rmse = {a_fit['rmse']:.12e}, max_abs = {a_fit['max_abs']:.12e}")
    print(
        "           intercept window = "
        f"[{summary['A_tr_fit']['intercept_min']:.12f}, {summary['A_tr_fit']['intercept_max']:.12f}], "
        f"std = {summary['A_tr_fit']['intercept_std']:.12e}"
    )
    print(f"  B_rel ~= {b_fit['intercept']:.12f} + {b_fit['slope']:.12f} / N1")
    print(f"           rmse = {b_fit['rmse']:.12e}, max_abs = {b_fit['max_abs']:.12e}")
    print(
        "           intercept window = "
        f"[{summary['B_rel_fit']['intercept_min']:.12f}, {summary['B_rel_fit']['intercept_max']:.12f}], "
        f"std = {summary['B_rel_fit']['intercept_std']:.12e}"
    )
    print(
        "  A_tr extrapolation envelope = "
        f"{summary['A_tr_extrapolation']['estimate']:.12f} +/- "
        f"{summary['A_tr_extrapolation']['uncertainty']:.12e} "
        f"(preferred model: {summary['A_tr_extrapolation']['preferred_model']})"
    )
    print(
        "  B_rel extrapolation envelope = "
        f"{summary['B_rel_extrapolation']['estimate']:.12f} +/- "
        f"{summary['B_rel_extrapolation']['uncertainty']:.12e} "
        f"(preferred model: {summary['B_rel_extrapolation']['preferred_model']})"
    )
    print()
    print(
        f"Ratio samples at scale {summary['ratio_scale']}: "
        f"N1 = {summary['ratio_scale']} * a, N2 = {summary['ratio_scale']} * b"
    )
    print(f"{'a':>3s} {'b':>3s} {'lambda':>8s} {'N1*A_tr':>14s} {'B_rel':>14s}")
    print("-" * 50)
    for row in summary["ratio_rows"]:
        print(
            f"{row['a']:3d} {row['b']:3d} {row['lambda']:8.5f} "
            f"{row['N1_A_tr']:14.9f} {row['B_rel']:14.9f}"
        )
    print()
    return summary


def family_fixed_ratio_fits(
    left_t: float, right_t: float, scales: list[int]
) -> dict[str, FitStability]:
    ns = []
    scaled_bqq = []
    scaled_eta_plus = []
    scaled_eta_minus = []
    a_delta_vals = []
    for scale in scales:
        n1 = 2 * scale
        n2 = 3 * scale
        data = spc.prefactor_data_three_point_family(n1, n2, left_t, right_t)
        ns.append(n1)
        scaled_bqq.append(n1 * data.b_qq_reduced)
        scaled_eta_plus.append(math.sqrt(n1) * data.eta_plus)
        scaled_eta_minus.append(math.sqrt(n1) * data.eta_minus)
        a_delta_vals.append(data.a_delta_reduced)
    return {
        "N1Bqq": fit_stability(ns, scaled_bqq),
        "sqrtN_eta_plus": fit_stability(ns, scaled_eta_plus),
        "sqrtN_eta_minus": fit_stability(ns, scaled_eta_minus),
        "A_delta": fit_stability(ns, a_delta_vals),
    }


def family_fixed_ratio_extrapolations(
    left_t: float, right_t: float, scales: list[int]
) -> dict[str, dict[str, object]]:
    ns = []
    scaled_bqq = []
    scaled_eta_plus = []
    scaled_eta_minus = []
    a_delta_vals = []
    for scale in scales:
        n1 = 2 * scale
        n2 = 3 * scale
        data = spc.prefactor_data_three_point_family(n1, n2, left_t, right_t)
        ns.append(n1)
        scaled_bqq.append(n1 * data.b_qq_reduced)
        scaled_eta_plus.append(math.sqrt(n1) * data.eta_plus)
        scaled_eta_minus.append(math.sqrt(n1) * data.eta_minus)
        a_delta_vals.append(data.a_delta_reduced)
    return {
        "N1Bqq": ce.summary_to_dict(ce.summarize_extrapolation(ns, scaled_bqq)),
        "sqrtN_eta_plus": ce.summary_to_dict(
            ce.summarize_extrapolation(ns, scaled_eta_plus)
        ),
        "sqrtN_eta_minus": ce.summary_to_dict(
            ce.summarize_extrapolation(ns, scaled_eta_minus)
        ),
        "A_delta": ce.summary_to_dict(ce.summarize_extrapolation(ns, a_delta_vals)),
    }


def superstring_prefactor_summary(ratio_scale: int) -> dict[str, object]:
    families = [
        ("minimal/minimal", 0.0, 0.0),
        ("minimal/second", 0.0, 0.5),
        ("second/second", -0.5, 0.5),
        ("midpoint", -0.25, 0.25),
    ]
    scales = [8, 16, 32, 64, 128, 256]
    family_rows = []
    for label, left_t, right_t in families:
        fits = family_fixed_ratio_fits(left_t, right_t, scales)
        extrapolations = family_fixed_ratio_extrapolations(left_t, right_t, scales)
        family_rows.append(
            {
                "label": label,
                "left_t": left_t,
                "right_t": right_t,
                "fits": {name: fit_stability_to_dict(fit) for name, fit in fits.items()},
                "extrapolations": extrapolations,
            }
        )
    ratio_rows = []
    for a, b in [(1, 3), (1, 2), (3, 5), (2, 3), (1, 1)]:
        n1 = ratio_scale * a
        n2 = ratio_scale * b
        data = spc.prefactor_data(
            n1, n2, left_variant="second_order", right_variant="second_order"
        )
        ratio_rows.append(
            {
                "a": a,
                "b": b,
                "lambda": n1 / (n1 + n2),
                "N1_Bqq": n1 * data.b_qq_reduced,
                "sqrtN1_eta_plus": math.sqrt(n1) * data.eta_plus,
                "sqrtN1_eta_minus": math.sqrt(n1) * data.eta_minus,
                "A_delta": data.a_delta_reduced,
            }
        )
    return {
        "scales": scales,
        "families": family_rows,
        "ratio_scale": ratio_scale,
        "ratio_rows": ratio_rows,
    }


def print_superstring_prefactor_summary(ratio_scale: int) -> dict[str, object]:
    summary = superstring_prefactor_summary(ratio_scale)
    print("=" * 92)
    print("SUPERSTRING BOSONIC PREFACTOR VALIDATION")
    print("=" * 92)
    print("Fixed ratio N1:N2 = 2:3 continuum fits for representative support-three families:")
    header = (
        f"{'family':>16s} {'lim N1*Bqq':>14s} {'lim sqrt(N1)eta_+':>18s} "
        f"{'lim sqrt(N1)eta_-':>18s} {'lim A_delta':>14s}"
    )
    print(header)
    print("-" * len(header))
    for row in summary["families"]:
        fits = row["fits"]
        extrap = row["extrapolations"]
        print(
            f"{row['label']:16s} "
            f"{extrap['N1Bqq']['estimate']:14.9f} "
            f"{extrap['sqrtN_eta_plus']['estimate']:18.9f} "
            f"{extrap['sqrtN_eta_minus']['estimate']:18.9f} "
            f"{extrap['A_delta']['estimate']:14.9f}"
        )
    print()
    print(
        f"Ratio samples for the symmetric second-order candidate at scale {summary['ratio_scale']}:"
    )
    print(
        f"{'a':>3s} {'b':>3s} {'lambda':>8s} {'N1*Bqq':>14s} "
        f"{'sqrt(N1)eta_+':>16s} {'sqrt(N1)eta_-':>16s} {'A_delta':>14s}"
    )
    print("-" * 96)
    for row in summary["ratio_rows"]:
        print(
            f"{row['a']:3d} {row['b']:3d} {row['lambda']:8.5f} "
            f"{row['N1_Bqq']:14.9f} {row['sqrtN1_eta_plus']:16.9f} "
            f"{row['sqrtN1_eta_minus']:16.9f} {row['A_delta']:14.9f}"
        )
    print()
    print("Fixed-ratio intercept stability windows for representative families:")
    print(f"{'family':>16s} {'Bqq window':>26s} {'eta_- window':>26s}")
    print("-" * 72)
    for row in summary["families"]:
        fits = row["fits"]
        print(
            f"{row['label']:16s} "
            f"[{fits['N1Bqq']['intercept_min']:.6f}, {fits['N1Bqq']['intercept_max']:.6f}] "
            f"[{fits['sqrtN_eta_minus']['intercept_min']:.6f}, {fits['sqrtN_eta_minus']['intercept_max']:.6f}]"
        )
    print()
    print("Model/window extrapolation envelopes for representative families:")
    print(f"{'family':>16s} {'N1*Bqq':>24s} {'A_delta':>24s}")
    print("-" * 72)
    for row in summary["families"]:
        extrap = row["extrapolations"]
        print(
            f"{row['label']:16s} "
            f"{extrap['N1Bqq']['estimate']:.6f} +/- {extrap['N1Bqq']['uncertainty']:.3e} "
            f"{extrap['A_delta']['estimate']:.6f} +/- {extrap['A_delta']['uncertainty']:.3e}"
        )
    print()
    return summary


def family_grid_summary(
    family_grid_max_abs_t: float, family_grid_step: float, family_grid_scale: int
) -> dict[str, object]:
    if family_grid_step <= 0.0:
        raise ValueError("family-grid-step must be positive")
    n_steps = int(round(2.0 * family_grid_max_abs_t / family_grid_step))
    ts = [
        -family_grid_max_abs_t + k * family_grid_step for k in range(n_steps + 1)
    ]
    scales = [max(4, family_grid_scale // 8), max(8, family_grid_scale // 4), max(16, family_grid_scale // 2), family_grid_scale]
    rows = []
    for left_t in ts:
        for right_t in ts:
            fits = family_fixed_ratio_fits(left_t, right_t, scales)
            rows.append(
                {
                    "left_t": left_t,
                    "right_t": right_t,
                    "fits": {name: fit_stability_to_dict(fit) for name, fit in fits.items()},
                    "extrapolations": family_fixed_ratio_extrapolations(
                        left_t, right_t, scales
                    ),
                }
            )
    return {
        "max_abs_t": family_grid_max_abs_t,
        "step": family_grid_step,
        "scale": family_grid_scale,
        "rows": rows,
    }


def print_family_grid_scan(
    family_grid_max_abs_t: float, family_grid_step: float, family_grid_scale: int
) -> dict[str, object]:
    summary = family_grid_summary(
        family_grid_max_abs_t, family_grid_step, family_grid_scale
    )
    print("=" * 92)
    print("OPTIONAL SUPPORT-THREE FAMILY GRID")
    print("=" * 92)
    print(
        "Each row reports fixed-ratio N1:N2 = 2:3 continuum fits from a small "
        "set of scales. This is raw diagnostic data, not a heuristic ranking."
    )
    print(
        f"{'t_+':>8s} {'t_-':>8s} {'lim N1*Bqq':>14s} "
        f"{'lim sqrt(N1)eta_+':>18s} {'lim sqrt(N1)eta_-':>18s}"
    )
    print("-" * 74)
    for row in summary["rows"]:
        fits = row["fits"]
        print(
            f"{row['left_t']:8.3f} {row['right_t']:8.3f} "
            f"{fits['N1Bqq']['primary']['intercept']:14.9f} "
            f"{fits['sqrtN_eta_plus']['primary']['intercept']:18.9f} "
            f"{fits['sqrtN_eta_minus']['primary']['intercept']:18.9f}"
        )
    print()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-prime", type=float, default=1.0)
    parser.add_argument("--d-perp", type=int, default=24)
    parser.add_argument("--factorized-min-n", type=int, default=4)
    parser.add_argument("--factorized-max-n", type=int, default=40)
    parser.add_argument("--factorized-max-n3", type=int, default=80)
    parser.add_argument(
        "--ratio-scale",
        type=int,
        default=128,
        help="base scale used in the large-N ratio samples",
    )
    parser.add_argument(
        "--family-grid",
        action="store_true",
        help="print an optional coarse support-three family grid scan",
    )
    parser.add_argument(
        "--family-grid-max-abs-t",
        type=float,
        default=0.75,
        help="largest |t| included in the optional family-grid scan",
    )
    parser.add_argument(
        "--family-grid-step",
        type=float,
        default=0.25,
        help="step size used in the optional family-grid scan",
    )
    parser.add_argument(
        "--family-grid-scale",
        type=int,
        default=128,
        help="largest scale used in the optional family-grid fit set",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="optional path to write the full validation report as JSON",
    )
    args = parser.parse_args()

    report: dict[str, object] = {
        "parameters": {
            "alpha_prime": args.alpha_prime,
            "d_perp": args.d_perp,
            "factorized_min_n": args.factorized_min_n,
            "factorized_max_n": args.factorized_max_n,
            "factorized_max_n3": args.factorized_max_n3,
            "ratio_scale": args.ratio_scale,
        }
    }

    report["tachyon"] = print_tachyon_summary(
        args.alpha_prime,
        args.d_perp,
        args.factorized_min_n,
        args.factorized_max_n,
        args.factorized_max_n3,
    )
    report["massless"] = print_massless_summary(args.alpha_prime, args.ratio_scale)
    report["superstring_prefactor"] = print_superstring_prefactor_summary(
        args.ratio_scale
    )
    if args.family_grid:
        report["family_grid"] = print_family_grid_scan(
            args.family_grid_max_abs_t,
            args.family_grid_step,
            args.family_grid_scale,
        )
    if args.json_out is not None:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Wrote JSON report to {out_path}")


if __name__ == "__main__":
    main()
