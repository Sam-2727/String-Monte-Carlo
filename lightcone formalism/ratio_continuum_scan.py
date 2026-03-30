#!/usr/bin/env python3
"""
Fixed-ratio continuum scans for the discrete-sigma low-point numerics.

This script complements `low_point_validation.py` by replacing single-scale
ratio snapshots with actual N -> infinity extrapolations at several fixed
ratios lambda = N1 / (N1 + N2).

Current scope:
1. Bosonic TTM sector:
   - extrapolate N1 * A_tr(lambda)
   - extrapolate B_rel(lambda)
2. Superstring bosonic prefactor sector (second/second family by default):
   - extrapolate N1 * B_qq(lambda)
   - extrapolate sqrt(N1) eta_+(lambda)
   - extrapolate sqrt(N1) eta_-(lambda)
   - extrapolate A_delta(lambda)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bosonic_massless_check as bmc
import continuum_extrapolation as ce
import superstring_prefactor_check as spc
import tachyon_check as tc


DEFAULT_RATIO_FAMILIES = [(1, 3), (1, 2), (3, 5), (2, 3), (1, 1)]
DEFAULT_SCALES = [8, 16, 32, 64, 128, 256]


def parse_family(text: str) -> tuple[int, int]:
    pieces = [piece.strip() for piece in text.split(",") if piece.strip()]
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError(f"expected a,b family, got {text!r}")
    a, b = (int(piece) for piece in pieces)
    if a <= 0 or b <= 0:
        raise argparse.ArgumentTypeError("require a,b > 0")
    return a, b


def parse_scales(text: str) -> list[int]:
    pieces = [piece.strip() for piece in text.split(",") if piece.strip()]
    scales = [int(piece) for piece in pieces]
    if not scales:
        raise argparse.ArgumentTypeError("need at least one scale")
    if any(scale <= 0 for scale in scales):
        raise argparse.ArgumentTypeError("all scales must be positive")
    return scales


def family_label(a: int, b: int) -> str:
    return f"{a}:{b}"


def tachyon_family_data(a: int, b: int, scales: list[int], alpha_prime: float) -> dict[str, object]:
    ns = []
    gamma_values = []
    rows = []
    for scale in scales:
        n1 = a * scale
        n2 = b * scale
        data = tc.compute_tachyon_data(n1, n2, alpha_prime, d_perp=24)
        lam = n1 / (n1 + n2)
        ns.append(n1)
        gamma_values.append(data.gamma_t)
        rows.append(
            {
                "scale": scale,
                "N1": n1,
                "N2": n2,
                "lambda": lam,
                "gamma_T": data.gamma_t,
            }
        )
    extrap = ce.summary_to_dict(ce.summarize_extrapolation(ns, gamma_values))
    return {"rows": rows, "gamma_T_extrapolation": extrap}


def massless_family_data(a: int, b: int, scales: list[int], alpha_prime: float) -> dict[str, object]:
    ns = []
    scaled_a = []
    b_rel = []
    rows = []
    for scale in scales:
        n1 = a * scale
        n2 = b * scale
        data = bmc.compute_massless_data(n1, n2, alpha_prime)
        lam = n1 / (n1 + n2)
        ns.append(n1)
        scaled_a.append(n1 * data.a_trace_reduced)
        b_rel.append(data.b_rel_reduced)
        rows.append(
            {
                "scale": scale,
                "N1": n1,
                "N2": n2,
                "lambda": lam,
                "N1_A_tr": n1 * data.a_trace_reduced,
                "A_tr": data.a_trace_reduced,
                "B_rel": data.b_rel_reduced,
            }
        )
    return {
        "rows": rows,
        "N1_A_tr_extrapolation": ce.summary_to_dict(ce.summarize_extrapolation(ns, scaled_a)),
        "B_rel_extrapolation": ce.summary_to_dict(ce.summarize_extrapolation(ns, b_rel)),
    }


def prefactor_family_data(
    a: int,
    b: int,
    scales: list[int],
    alpha_prime: float,
    left_variant: str,
    right_variant: str,
) -> dict[str, object]:
    ns = []
    scaled_bqq = []
    scaled_eta_plus = []
    scaled_eta_minus = []
    a_delta_vals = []
    rows = []
    for scale in scales:
        n1 = a * scale
        n2 = b * scale
        data = spc.prefactor_data(
            n1,
            n2,
            alpha_prime,
            left_variant=left_variant,
            right_variant=right_variant,
        )
        lam = n1 / (n1 + n2)
        ns.append(n1)
        scaled_bqq.append(n1 * data.b_qq_reduced)
        scaled_eta_plus.append(math.sqrt(n1) * data.eta_plus)
        scaled_eta_minus.append(math.sqrt(n1) * data.eta_minus)
        a_delta_vals.append(data.a_delta_reduced)
        rows.append(
            {
                "scale": scale,
                "N1": n1,
                "N2": n2,
                "lambda": lam,
                "N1_Bqq": n1 * data.b_qq_reduced,
                "sqrtN1_eta_plus": math.sqrt(n1) * data.eta_plus,
                "sqrtN1_eta_minus": math.sqrt(n1) * data.eta_minus,
                "A_delta": data.a_delta_reduced,
            }
        )
    return {
        "rows": rows,
        "N1_Bqq_extrapolation": ce.summary_to_dict(ce.summarize_extrapolation(ns, scaled_bqq)),
        "sqrtN1_eta_plus_extrapolation": ce.summary_to_dict(
            ce.summarize_extrapolation(ns, scaled_eta_plus)
        ),
        "sqrtN1_eta_minus_extrapolation": ce.summary_to_dict(
            ce.summarize_extrapolation(ns, scaled_eta_minus)
        ),
        "A_delta_extrapolation": ce.summary_to_dict(
            ce.summarize_extrapolation(ns, a_delta_vals)
        ),
    }


def full_report(
    families: list[tuple[int, int]],
    scales: list[int],
    alpha_prime: float,
    left_variant: str,
    right_variant: str,
) -> dict[str, object]:
    ratio_rows = []
    for a, b in families:
        ratio_rows.append(
            {
                "family": family_label(a, b),
                "a": a,
                "b": b,
                "lambda": a / (a + b),
                "tachyon": tachyon_family_data(a, b, scales, alpha_prime),
                "massless": massless_family_data(a, b, scales, alpha_prime),
                "prefactor": prefactor_family_data(
                    a, b, scales, alpha_prime, left_variant, right_variant
                ),
            }
        )
    ratio_rows.sort(key=lambda row: row["lambda"])
    return {
        "parameters": {
            "alpha_prime": alpha_prime,
            "scales": scales,
            "left_variant": left_variant,
            "right_variant": right_variant,
        },
        "families": ratio_rows,
    }


def print_report(report: dict[str, object]) -> None:
    print("=" * 108)
    print("FIXED-RATIO CONTINUUM SCAN")
    print("=" * 108)
    params = report["parameters"]
    print(
        f"Scales: {params['scales']} | prefactor stencil: "
        f"{params['left_variant']}/{params['right_variant']}"
    )
    print()

    print("Bosonic TTM continuum scan:")
    print(
        f"{'family':>8s} {'lambda':>8s} {'gamma_T(∞)':>14s} {'N1*A_tr(∞)':>16s} "
        f"{'B_rel(∞)':>14s}"
    )
    print("-" * 72)
    for row in report["families"]:
        print(
            f"{row['family']:>8s} {row['lambda']:8.5f} "
            f"{row['tachyon']['gamma_T_extrapolation']['estimate']:14.9f} "
            f"{row['massless']['N1_A_tr_extrapolation']['estimate']:16.9f} "
            f"{row['massless']['B_rel_extrapolation']['estimate']:14.9f}"
        )
    print()

    print("Superstring second-family prefactor continuum scan:")
    print(
        f"{'family':>8s} {'lambda':>8s} {'N1*Bqq(∞)':>14s} "
        f"{'sqrtN eta_-(∞)':>16s} {'A_delta(∞)':>14s}"
    )
    print("-" * 72)
    for row in report["families"]:
        pref = row["prefactor"]
        print(
            f"{row['family']:>8s} {row['lambda']:8.5f} "
            f"{pref['N1_Bqq_extrapolation']['estimate']:14.9f} "
            f"{pref['sqrtN1_eta_minus_extrapolation']['estimate']:16.9f} "
            f"{pref['A_delta_extrapolation']['estimate']:14.9f}"
        )
    print()

    print("Continuum envelopes:")
    print(
        f"{'family':>8s} {'B_rel uncertainty':>18s} {'N1*Bqq uncertainty':>22s} "
        f"{'A_delta uncertainty':>20s}"
    )
    print("-" * 76)
    for row in report["families"]:
        pref = row["prefactor"]
        massless = row["massless"]
        print(
            f"{row['family']:>8s} "
            f"{massless['B_rel_extrapolation']['uncertainty']:18.9e} "
            f"{pref['N1_Bqq_extrapolation']['uncertainty']:22.9e} "
            f"{pref['A_delta_extrapolation']['uncertainty']:20.9e}"
        )
    print()


def markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# Fixed-Ratio Continuum Scan",
        "",
        (
            "Scales: "
            f"`{report['parameters']['scales']}` | prefactor stencil: "
            f"`{report['parameters']['left_variant']}/{report['parameters']['right_variant']}`"
        ),
        "",
        "## Bosonic TTM",
        "",
        "| family | lambda | gamma_T(∞) | N1*A_tr(∞) | B_rel(∞) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["families"]:
        lines.append(
            "| "
            f"{row['family']} | {row['lambda']:.5f} | "
            f"{row['tachyon']['gamma_T_extrapolation']['estimate']:.9f} | "
            f"{row['massless']['N1_A_tr_extrapolation']['estimate']:.9f} | "
            f"{row['massless']['B_rel_extrapolation']['estimate']:.9f} |"
        )
    lines.extend(
        [
            "",
            "## Superstring Prefactor",
            "",
            "| family | lambda | N1*Bqq(∞) | sqrtN eta_-(∞) | A_delta(∞) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["families"]:
        pref = row["prefactor"]
        lines.append(
            "| "
            f"{row['family']} | {row['lambda']:.5f} | "
            f"{pref['N1_Bqq_extrapolation']['estimate']:.9f} | "
            f"{pref['sqrtN1_eta_minus_extrapolation']['estimate']:.9f} | "
            f"{pref['A_delta_extrapolation']['estimate']:.9f} |"
        )
    lines.extend(
        [
            "",
            "## Uncertainty Envelopes",
            "",
            "| family | B_rel uncertainty | N1*Bqq uncertainty | A_delta uncertainty |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["families"]:
        pref = row["prefactor"]
        massless = row["massless"]
        lines.append(
            "| "
            f"{row['family']} | "
            f"{massless['B_rel_extrapolation']['uncertainty']:.9e} | "
            f"{pref['N1_Bqq_extrapolation']['uncertainty']:.9e} | "
            f"{pref['A_delta_extrapolation']['uncertainty']:.9e} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-prime", type=float, default=1.0)
    parser.add_argument(
        "--family",
        action="append",
        type=parse_family,
        default=None,
        help="append a fixed ratio family a,b; may be passed multiple times",
    )
    parser.add_argument(
        "--scales",
        type=parse_scales,
        default=DEFAULT_SCALES,
        help="comma-separated list of common scales",
    )
    parser.add_argument(
        "--left-variant",
        type=str,
        default="second_order",
        help="left prefactor stencil variant",
    )
    parser.add_argument(
        "--right-variant",
        type=str,
        default="second_order",
        help="right prefactor stencil variant",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="optional JSON output path",
    )
    parser.add_argument(
        "--markdown-out",
        type=str,
        default=None,
        help="optional markdown output path",
    )
    args = parser.parse_args()

    families = args.family if args.family is not None else DEFAULT_RATIO_FAMILIES
    report = full_report(
        families,
        args.scales,
        args.alpha_prime,
        args.left_variant,
        args.right_variant,
    )
    print_report(report)
    if args.json_out is not None:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Wrote JSON report to {out_path}")
    if args.markdown_out is not None:
        md_path = Path(args.markdown_out)
        md_path.write_text(markdown_report(report))
        print(f"Wrote markdown report to {md_path}")


if __name__ == "__main__":
    main()
