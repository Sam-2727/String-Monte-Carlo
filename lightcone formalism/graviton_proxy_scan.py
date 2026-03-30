#!/usr/bin/env python3
"""
Symmetric-family graviton proxy scan using the Weyl vector block.

This script combines:
- the continuum-extrapolated bosonic prefactor data from the symmetric
  support-three family, and
- the Weyl-quantized vector block of v_{IJ}(Lambda),

to build a concrete vector-sector graviton proxy.

For a fixed ratio lambda = alpha_1 / alpha_3 = N1 / N3, let

  T_bos^{IJ} = A_delta delta^{IJ} + B_phys qhat^I qhat^J,

where qhat is a unit vector and

  B_phys = (N1 * B_qq) / lambda

is the inferred continuum bilinear coefficient after removing the trivial
alpha_1 factor from the finite-N proxy N1 * B_qq.

If the vector block of v_{IJ} is

  V_{IJ;KL} = A delta_{IJ} delta_{KL}
            + B delta_{IK} delta_{JL}
            + C delta_{IL} delta_{JK},

then contracting T_bos^{IJ} with V_{IJ;KL} gives

  M_{KL} = M_delta delta_{KL} + M_qq qhat_K qhat_L

with

  M_delta = A_delta (8A + B + C) + B_phys A,
  M_qq    = B_phys (B + C).

These are not yet the full three-graviton amplitudes, but they are a more
physical discriminator than bosonic-prefactor smoothness alone because they
already incorporate the explicit zero-mode tensor structure.

There are two useful modes:

- full: use the full vector block of v_{IJ},
- trace-dropped: remove the pieces proportional to delta_{IJ}, motivated by
  the flat-space on-shell three-particle matrix-element simplification.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gs_weyl_symbol_diagnostic as gw
import symmetric_prefactor_scan as sps
import weyl_vector_block_formula as wvf


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


@functools.lru_cache(maxsize=None)
def weyl_vector_coeffs(
    alpha_ratio: float,
    trace_dropped: bool = False,
) -> tuple[complex, complex, complex, float]:
    if trace_dropped:
        analytic_a, analytic_b, analytic_c = (
            wvf.analytic_trace_dropped_vector_block_invariants(alpha_ratio)
        )
        comparison = wvf.compare_trace_dropped_formula(alpha_ratio)
        return analytic_a, analytic_b, analytic_c, float(comparison["max_coeff_diff"])

    analytic_a, analytic_b, analytic_c = wvf.analytic_vector_block_invariants(alpha_ratio)
    _, max_err, _ = gw.fit_vector_block_invariants(alpha_ratio)
    return analytic_a, analytic_b, analytic_c, float(max_err)


def quadratic_rms(xs: list[float], ys: list[float]) -> float:
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    design = np.column_stack([np.ones_like(x), x, x * x])
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coeffs
    return float(math.sqrt(np.mean(residual * residual)))


def candidate_proxy_rows(
    candidate: dict[str, object],
    trace_dropped: bool = False,
) -> list[dict[str, object]]:
    out = []
    for row in candidate["rows"]:
        lam = row["lambda"]
        if lam <= 0.0:
            raise ValueError("lambda must be positive")
        A, B, C, fit_res = weyl_vector_coeffs(lam, trace_dropped=trace_dropped)
        a_delta = row["A_delta"]["estimate"]
        a_unc = row["A_delta"]["uncertainty"]
        n1_bqq = row["N1_Bqq"]["estimate"]
        n1_bqq_unc = row["N1_Bqq"]["uncertainty"]
        b_phys = n1_bqq / lam
        b_phys_unc = n1_bqq_unc / lam

        m_delta = a_delta * (8.0 * A + B + C) + b_phys * A
        m_qq = b_phys * (B + C)
        m_delta_unc = abs(8.0 * A + B + C) * a_unc + abs(A) * b_phys_unc
        m_qq_unc = abs(B + C) * b_phys_unc
        ratio = m_qq / m_delta if abs(m_delta) > 1.0e-12 else complex(np.nan, np.nan)

        out.append(
            {
                "family": row["family"],
                "lambda": lam,
                "trace_dropped": trace_dropped,
                "weyl_A": A,
                "weyl_B": B,
                "weyl_C": C,
                "weyl_fit_residual": fit_res,
                "A_delta": a_delta,
                "A_delta_uncertainty": a_unc,
                "B_phys": b_phys,
                "B_phys_uncertainty": b_phys_unc,
                "M_delta": m_delta,
                "M_delta_uncertainty": float(m_delta_unc),
                "M_qq": m_qq,
                "M_qq_uncertainty": float(m_qq_unc),
                "M_ratio": ratio,
            }
        )
    return out


def proxy_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    lambdas = [row["lambda"] for row in rows]
    m_delta_vals = [float(np.real(row["M_delta"])) for row in rows]
    m_delta_unc = [row["M_delta_uncertainty"] for row in rows]
    m_qq_vals = [float(np.real(row["M_qq"])) for row in rows]
    m_qq_unc = [row["M_qq_uncertainty"] for row in rows]
    ratio_vals = [float(np.real(row["M_ratio"])) for row in rows]

    mean_abs_delta = float(np.mean(np.abs(m_delta_vals)))
    mean_abs_qq = float(np.mean(np.abs(m_qq_vals)))
    mean_abs_ratio = float(np.mean(np.abs(ratio_vals)))

    return {
        "M_delta_smooth_rel": quadratic_rms(lambdas, m_delta_vals)
        / max(mean_abs_delta, 1.0e-12),
        "M_qq_smooth_rel": quadratic_rms(lambdas, m_qq_vals)
        / max(mean_abs_qq, 1.0e-12),
        "M_ratio_smooth_rel": quadratic_rms(lambdas, ratio_vals)
        / max(mean_abs_ratio, 1.0e-12),
        "M_delta_unc_rel": float(
            np.mean([unc / max(abs(val), 1.0e-12) for unc, val in zip(m_delta_unc, m_delta_vals)])
        ),
        "M_qq_unc_rel": float(
            np.mean([unc / max(abs(val), 1.0e-12) for unc, val in zip(m_qq_unc, m_qq_vals)])
        ),
        "M_qq_negative": all(value < 0.0 for value in m_qq_vals),
        "M_qq_monotone": all(
            m_qq_vals[index] < m_qq_vals[index + 1]
            for index in range(len(m_qq_vals) - 1)
        ),
        "mean_abs_M_delta": mean_abs_delta,
        "mean_abs_M_qq": mean_abs_qq,
        "mean_abs_M_ratio": mean_abs_ratio,
    }


def ranking_key(candidate: dict[str, object]) -> tuple[object, ...]:
    metrics = candidate["proxy_metrics"]
    return (
        not metrics["M_qq_negative"],
        not metrics["M_qq_monotone"],
        metrics["M_ratio_smooth_rel"],
        metrics["M_qq_smooth_rel"],
        metrics["M_delta_smooth_rel"],
        metrics["M_qq_unc_rel"],
        metrics["M_delta_unc_rel"],
        -metrics["mean_abs_M_qq"],
    )


def scan_proxies(
    alpha_prime: float,
    scales: list[int],
    max_t: float,
    step: float,
    min_t: float,
    trace_dropped: bool = False,
) -> list[dict[str, object]]:
    candidates = sps.scan_symmetric_family(
        scales=scales,
        families=sps.pfr.DEFAULT_RATIO_FAMILIES,
        alpha_prime=alpha_prime,
        max_t=max_t,
        step=step,
    )
    filtered = []
    for candidate in candidates:
        if candidate["t"] < min_t:
            continue
        rows = candidate_proxy_rows(candidate, trace_dropped=trace_dropped)
        filtered.append(
            {
                "t": candidate["t"],
                "trace_dropped": trace_dropped,
                "rank_prefactor": candidate["rank"],
                "prefactor_metrics": candidate["metrics"],
                "proxy_rows": rows,
                "proxy_metrics": proxy_metrics(rows),
            }
        )
    filtered.sort(key=ranking_key)
    for index, candidate in enumerate(filtered, start=1):
        candidate["rank_proxy"] = index
    return filtered


def print_report(candidates: list[dict[str, object]], top_k: int) -> None:
    print("=" * 122)
    mode = "trace-dropped" if candidates and candidates[0].get("trace_dropped") else "full"
    print(f"GRAVITON VECTOR-BLOCK PROXY SCAN ({mode})")
    print("=" * 122)
    print(
        f"{'rank':>4s} {'t':>7s} {'pref':>6s} {'Mqq<0':>7s} {'mono':>6s} "
        f"{'ratio_smooth':>12s} {'Mqq_smooth':>12s} {'Mqq_unc':>10s} {'mean|Mqq|':>12s}"
    )
    print("-" * 96)
    for candidate in candidates[:top_k]:
        m = candidate["proxy_metrics"]
        print(
            f"{candidate['rank_proxy']:4d} {candidate['t']:7.3f} "
            f"{candidate['rank_prefactor']:6d} {str(m['M_qq_negative']):>7s} "
            f"{str(m['M_qq_monotone']):>6s} {m['M_ratio_smooth_rel']:12.3e} "
            f"{m['M_qq_smooth_rel']:12.3e} {m['M_qq_unc_rel']:10.3e} "
            f"{m['mean_abs_M_qq']:12.6f}"
        )
    print()

    second_order = next((item for item in candidates if abs(item["t"] - 0.5) < 1.0e-12), None)
    if second_order is not None:
        m = second_order["proxy_metrics"]
        print("Second-order reference point:")
        print(
            f"  t = 0.500, proxy rank = {second_order['rank_proxy']}, "
            f"prefactor rank = {second_order['rank_prefactor']}"
        )
        print(
            f"  ratio_smooth = {m['M_ratio_smooth_rel']:.3e}, "
            f"Mqq_smooth = {m['M_qq_smooth_rel']:.3e}, "
            f"Mqq_unc = {m['M_qq_unc_rel']:.3e}"
        )
        print("  proxy values by lambda:")
        print(
            f"  {'family':>8s} {'lambda':>8s} {'B_phys':>12s} "
            f"{'M_delta':>14s} {'M_qq':>14s} {'M_qq/M_delta':>14s}"
        )
        for row in second_order["proxy_rows"]:
            print(
                f"  {row['family']:>8s} {row['lambda']:8.5f} "
                f"{row['B_phys']:12.9f} {np.real(row['M_delta']):14.9f} "
                f"{np.real(row['M_qq']):14.9f} {np.real(row['M_ratio']):14.9f}"
            )
        print()


def markdown_report(candidates: list[dict[str, object]], top_k: int) -> str:
    mode = "trace-dropped" if candidates and candidates[0].get("trace_dropped") else "full"
    lines = [
        f"# Graviton Vector-Block Proxy Scan ({mode})",
        "",
        "| rank | t | prefactor rank | Mqq<0 | mono | ratio_smooth | Mqq_smooth | Mqq_unc | mean|Mqq| |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in candidates[:top_k]:
        m = candidate["proxy_metrics"]
        lines.append(
            "| "
            f"{candidate['rank_proxy']} | {candidate['t']:.3f} | {candidate['rank_prefactor']} | "
            f"{m['M_qq_negative']} | {m['M_qq_monotone']} | "
            f"{m['M_ratio_smooth_rel']:.3e} | {m['M_qq_smooth_rel']:.3e} | "
            f"{m['M_qq_unc_rel']:.3e} | {m['mean_abs_M_qq']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_scales(text: str) -> list[int]:
    return sps.parse_scales(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-prime", type=float, default=1.0)
    parser.add_argument("--scales", type=parse_scales, default=sps.pfr.DEFAULT_SCALES)
    parser.add_argument("--max-t", type=float, default=0.75)
    parser.add_argument("--step", type=float, default=0.125)
    parser.add_argument("--min-t", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--trace-dropped",
        action="store_true",
        help="drop the delta_IJ pieces of v_IJ in the Weyl block, as motivated by on-shell three-particle matrix elements",
    )
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--markdown-out", type=str, default=None)
    args = parser.parse_args()

    candidates = scan_proxies(
        alpha_prime=args.alpha_prime,
        scales=args.scales,
        max_t=args.max_t,
        step=args.step,
        min_t=args.min_t,
        trace_dropped=args.trace_dropped,
    )
    print_report(candidates, args.top_k)

    report = {
        "parameters": {
            "alpha_prime": args.alpha_prime,
            "scales": args.scales,
            "max_t": args.max_t,
            "step": args.step,
            "min_t": args.min_t,
            "trace_dropped": args.trace_dropped,
        },
        "candidates": json_safe(candidates),
    }
    if args.json_out is not None:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Wrote JSON report to {out_path}")
    if args.markdown_out is not None:
        md_path = Path(args.markdown_out)
        md_path.write_text(markdown_report(candidates, args.top_k))
        print(f"Wrote markdown report to {md_path}")


if __name__ == "__main__":
    main()
