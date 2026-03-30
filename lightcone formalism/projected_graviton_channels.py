#!/usr/bin/env python3
"""
Projected vector and graviton channels from the Weyl vector-block proxy.

This module converts the reduced matrix

    M_KL = M_delta delta_KL + M_qq qhat_K qhat_L

from `graviton_proxy_scan.py` into explicit SO(8) polarization channels.

The goal is modest but useful: once the current candidate Weyl map and bosonic
prefactor are fixed, we can ask how a few concrete external channels behave
across the ratio lambda = alpha_1 / alpha_3 and across the support-three
stencil family. This is still not the full three-graviton amplitude, but it is
closer to explicit polarization data than the raw scalar proxy rankings.

It also supports the on-shell "trace-dropped" mode in which the delta_{IJ}
pieces of the Weyl block are removed before the contraction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import graviton_proxy_scan as gps


DEFAULT_D_PERP = 8


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    return value


def basis_vectors(d_perp: int = DEFAULT_D_PERP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if d_perp < 3:
        raise ValueError("need d_perp >= 3")
    e1 = np.zeros(d_perp, dtype=float)
    e2 = np.zeros(d_perp, dtype=float)
    e3 = np.zeros(d_perp, dtype=float)
    e1[0] = 1.0
    e2[1] = 1.0
    e3[2] = 1.0
    return e1, e2, e3


def polarization_tensors(d_perp: int = DEFAULT_D_PERP) -> dict[str, np.ndarray]:
    e1, e2, e3 = basis_vectors(d_perp)
    identity = np.eye(d_perp, dtype=float)
    qhat_qhat = np.outer(e1, e1)

    return {
        "dilaton": identity / math.sqrt(d_perp),
        "graviton_parallel": math.sqrt(d_perp / (d_perp - 1.0))
        * (qhat_qhat - identity / d_perp),
        "graviton_perpendicular": (np.outer(e2, e2) - np.outer(e3, e3)) / math.sqrt(2.0),
        "bfield_23": (np.outer(e2, e3) - np.outer(e3, e2)) / math.sqrt(2.0),
    }


def matrix_from_proxy_row(
    row: dict[str, object], d_perp: int = DEFAULT_D_PERP
) -> np.ndarray:
    e1, _, _ = basis_vectors(d_perp)
    identity = np.eye(d_perp, dtype=complex)
    qhat_qhat = np.outer(e1, e1)
    return complex(row["M_delta"]) * identity + complex(row["M_qq"]) * qhat_qhat


def closed_form_channel_values(
    m_delta: complex,
    m_qq: complex,
    d_perp: int = DEFAULT_D_PERP,
) -> dict[str, complex]:
    return {
        "vector_parallel": m_delta + m_qq,
        "vector_transverse": m_delta,
        "graviton_parallel": math.sqrt((d_perp - 1.0) / d_perp) * m_qq,
        "dilaton": math.sqrt(d_perp) * m_delta + m_qq / math.sqrt(d_perp),
        "graviton_perpendicular": 0.0j,
        "bfield_23": 0.0j,
    }


def closed_form_channel_uncertainties(
    m_delta_uncertainty: float,
    m_qq_uncertainty: float,
    d_perp: int = DEFAULT_D_PERP,
) -> dict[str, float]:
    return {
        "vector_parallel": float(m_delta_uncertainty + m_qq_uncertainty),
        "vector_transverse": float(m_delta_uncertainty),
        "graviton_parallel": float(
            math.sqrt((d_perp - 1.0) / d_perp) * m_qq_uncertainty
        ),
        "dilaton": float(
            math.sqrt(d_perp) * m_delta_uncertainty
            + m_qq_uncertainty / math.sqrt(d_perp)
        ),
        "graviton_perpendicular": 0.0,
        "bfield_23": 0.0,
    }


def projected_channels_from_row(
    row: dict[str, object], d_perp: int = DEFAULT_D_PERP
) -> dict[str, object]:
    matrix = matrix_from_proxy_row(row, d_perp=d_perp)
    channels = polarization_tensors(d_perp=d_perp)
    amplitudes = {}
    for name, tensor in channels.items():
        amplitudes[name] = complex(np.tensordot(tensor, matrix, axes=2))

    closed_form = closed_form_channel_values(
        complex(row["M_delta"]),
        complex(row["M_qq"]),
        d_perp=d_perp,
    )
    uncertainties = closed_form_channel_uncertainties(
        float(row["M_delta_uncertainty"]),
        float(row["M_qq_uncertainty"]),
        d_perp=d_perp,
    )
    amplitudes["vector_parallel"] = closed_form["vector_parallel"]
    amplitudes["vector_transverse"] = closed_form["vector_transverse"]

    ratio = complex(np.nan, np.nan)
    if abs(amplitudes["vector_transverse"]) > 1.0e-12:
        ratio = amplitudes["vector_parallel"] / amplitudes["vector_transverse"]

    return {
        "family": row["family"],
        "lambda": float(row["lambda"]),
        "M_delta": complex(row["M_delta"]),
        "M_qq": complex(row["M_qq"]),
        "M_delta_uncertainty": float(row["M_delta_uncertainty"]),
        "M_qq_uncertainty": float(row["M_qq_uncertainty"]),
        "channels": amplitudes,
        "channel_uncertainties": uncertainties,
        "vector_ratio": ratio,
    }


def quadratic_rms(xs: list[float], ys: list[float]) -> float:
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    design = np.column_stack([np.ones_like(x), x, x * x])
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coeffs
    return float(math.sqrt(np.mean(residual * residual)))


def channel_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    lambdas = [row["lambda"] for row in rows]
    vec_parallel = [float(np.real(row["channels"]["vector_parallel"])) for row in rows]
    vec_transverse = [float(np.real(row["channels"]["vector_transverse"])) for row in rows]
    grav_parallel = [float(np.real(row["channels"]["graviton_parallel"])) for row in rows]
    dilaton = [float(np.real(row["channels"]["dilaton"])) for row in rows]
    vector_ratio = [float(np.real(row["vector_ratio"])) for row in rows]

    vec_parallel_unc = [row["channel_uncertainties"]["vector_parallel"] for row in rows]
    grav_parallel_unc = [row["channel_uncertainties"]["graviton_parallel"] for row in rows]
    dilaton_unc = [row["channel_uncertainties"]["dilaton"] for row in rows]

    grav_perp_max = max(abs(row["channels"]["graviton_perpendicular"]) for row in rows)
    bfield_max = max(abs(row["channels"]["bfield_23"]) for row in rows)

    mean_abs_vec_parallel = float(np.mean(np.abs(vec_parallel)))
    mean_abs_grav_parallel = float(np.mean(np.abs(grav_parallel)))
    mean_abs_dilaton = float(np.mean(np.abs(dilaton)))
    mean_abs_ratio = float(np.mean(np.abs(vector_ratio)))
    graviton_blocked = mean_abs_grav_parallel < 1.0e-6

    return {
        "vector_parallel_smooth_rel": quadratic_rms(lambdas, vec_parallel)
        / max(mean_abs_vec_parallel, 1.0e-12),
        "graviton_parallel_smooth_rel": quadratic_rms(lambdas, grav_parallel)
        / max(mean_abs_grav_parallel, 1.0e-12),
        "dilaton_smooth_rel": quadratic_rms(lambdas, dilaton)
        / max(mean_abs_dilaton, 1.0e-12),
        "vector_ratio_smooth_rel": quadratic_rms(lambdas, vector_ratio)
        / max(mean_abs_ratio, 1.0e-12),
        "vector_parallel_unc_rel": float(
            np.mean(
                [
                    unc / max(abs(value), 1.0e-12)
                    for unc, value in zip(vec_parallel_unc, vec_parallel)
                ]
            )
        ),
        "graviton_parallel_unc_rel": float(
            np.mean(
                [
                    unc / max(abs(value), 1.0e-12)
                    for unc, value in zip(grav_parallel_unc, grav_parallel)
                ]
            )
        ),
        "dilaton_unc_rel": float(
            np.mean(
                [
                    unc / max(abs(value), 1.0e-12)
                    for unc, value in zip(dilaton_unc, dilaton)
                ]
            )
        ),
        "mean_abs_vector_parallel": mean_abs_vec_parallel,
        "mean_abs_graviton_parallel": mean_abs_grav_parallel,
        "mean_abs_dilaton": mean_abs_dilaton,
        "graviton_blocked": graviton_blocked,
        "vector_parallel_positive": all(value > 0.0 for value in vec_parallel),
        "graviton_parallel_positive": all(value > 0.0 for value in grav_parallel),
        "vector_ratio_monotone": all(
            vector_ratio[index] < vector_ratio[index + 1]
            for index in range(len(vector_ratio) - 1)
        ),
        "zero_graviton_perpendicular": float(grav_perp_max),
        "zero_bfield_23": float(bfield_max),
    }


def ranking_key(candidate: dict[str, object]) -> tuple[object, ...]:
    metrics = candidate["channel_metrics"]
    return (
        metrics["graviton_blocked"],
        metrics["zero_graviton_perpendicular"],
        metrics["zero_bfield_23"],
        not metrics["vector_parallel_positive"],
        not metrics["graviton_parallel_positive"],
        not metrics["vector_ratio_monotone"],
        metrics["vector_ratio_smooth_rel"],
        metrics["graviton_parallel_smooth_rel"],
        metrics["graviton_parallel_unc_rel"],
        -metrics["mean_abs_graviton_parallel"],
    )


def scan_projected_channels(
    alpha_prime: float,
    scales: list[int],
    max_t: float,
    step: float,
    min_t: float,
    d_perp: int = DEFAULT_D_PERP,
    trace_dropped: bool = False,
) -> list[dict[str, object]]:
    proxy_candidates = gps.scan_proxies(
        alpha_prime=alpha_prime,
        scales=scales,
        max_t=max_t,
        step=step,
        min_t=min_t,
        trace_dropped=trace_dropped,
    )
    candidates = []
    for candidate in proxy_candidates:
        rows = [
            projected_channels_from_row(row, d_perp=d_perp)
            for row in candidate["proxy_rows"]
        ]
        candidates.append(
            {
                "t": candidate["t"],
                "trace_dropped": trace_dropped,
                "rank_proxy": candidate["rank_proxy"],
                "rank_prefactor": candidate["rank_prefactor"],
                "prefactor_metrics": candidate["prefactor_metrics"],
                "proxy_metrics": candidate["proxy_metrics"],
                "channel_rows": rows,
                "channel_metrics": channel_metrics(rows),
            }
        )
    candidates.sort(key=ranking_key)
    for index, candidate in enumerate(candidates, start=1):
        candidate["rank_channels"] = index
    return candidates


def print_report(candidates: list[dict[str, object]], top_k: int) -> None:
    print("=" * 130)
    mode = "trace-dropped" if candidates and candidates[0].get("trace_dropped") else "full"
    print(f"PROJECTED GRAVITON CHANNEL SCAN ({mode})")
    print("=" * 130)
    print(
        f"{'rank':>4s} {'t':>7s} {'proxy':>6s} {'pref':>6s} {'blk':>5s} "
        f"{'ratio_smooth':>12s} {'grav_smooth':>12s} {'grav_unc':>10s} "
        f"{'mean|grav|':>12s} {'perp0':>10s} {'B0':>10s}"
    )
    print("-" * 112)
    for candidate in candidates[:top_k]:
        m = candidate["channel_metrics"]
        print(
            f"{candidate['rank_channels']:4d} {candidate['t']:7.3f} "
            f"{candidate['rank_proxy']:6d} {candidate['rank_prefactor']:6d} "
            f"{str(m['graviton_blocked']):>5s} "
            f"{m['vector_ratio_smooth_rel']:12.3e} "
            f"{m['graviton_parallel_smooth_rel']:12.3e} "
            f"{m['graviton_parallel_unc_rel']:10.3e} "
            f"{m['mean_abs_graviton_parallel']:12.6f} "
            f"{m['zero_graviton_perpendicular']:10.3e} "
            f"{m['zero_bfield_23']:10.3e}"
        )
    print()

    second_order = next((item for item in candidates if abs(item["t"] - 0.5) < 1.0e-12), None)
    if second_order is not None:
        m = second_order["channel_metrics"]
        print("Second-order reference point:")
        print(
            f"  t = 0.500, channel rank = {second_order['rank_channels']}, "
            f"proxy rank = {second_order['rank_proxy']}, prefactor rank = {second_order['rank_prefactor']}"
        )
        print(
            f"  graviton_blocked = {m['graviton_blocked']}, "
            f"  ratio_smooth = {m['vector_ratio_smooth_rel']:.3e}, "
            f"grav_smooth = {m['graviton_parallel_smooth_rel']:.3e}, "
            f"grav_unc = {m['graviton_parallel_unc_rel']:.3e}"
        )
        print(
            f"  zero-channel checks: perp = {m['zero_graviton_perpendicular']:.3e}, "
            f"B23 = {m['zero_bfield_23']:.3e}"
        )
        print(
            f"  mean|graviton_parallel| = {m['mean_abs_graviton_parallel']:.9f}"
        )
        print()
        print(
            f"  {'family':>8s} {'lambda':>8s} {'vec_parallel':>14s} "
            f"{'vec_ratio':>12s} {'grav_parallel':>15s} {'dilaton':>12s}"
        )
        for row in second_order["channel_rows"]:
            print(
                f"  {row['family']:>8s} {row['lambda']:8.5f} "
                f"{np.real(row['channels']['vector_parallel']):14.9f} "
                f"{np.real(row['vector_ratio']):12.9f} "
                f"{np.real(row['channels']['graviton_parallel']):15.9f} "
                f"{np.real(row['channels']['dilaton']):12.9f}"
            )
        print()


def markdown_report(candidates: list[dict[str, object]], params: dict[str, object]) -> str:
    mode = "trace-dropped" if candidates and candidates[0].get("trace_dropped") else "full"
    lines = [
        f"# Projected Graviton Channel Scan ({mode})",
        "",
        f"Range: `{params['min_t']} <= t <= {params['max_t']}`, step `{params['step']}`",
        f"Scales: `{params['scales']}`",
        "",
        "| rank | t | proxy rank | pref rank | blocked | ratio smooth | grav smooth | grav unc | mean|grav| | perp0 | B0 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in candidates:
        m = candidate["channel_metrics"]
        lines.append(
            "| "
            f"{candidate['rank_channels']} | {candidate['t']:.3f} | "
            f"{candidate['rank_proxy']} | {candidate['rank_prefactor']} | "
            f"{m['graviton_blocked']} | "
            f"{m['vector_ratio_smooth_rel']:.3e} | {m['graviton_parallel_smooth_rel']:.3e} | "
            f"{m['graviton_parallel_unc_rel']:.3e} | {m['mean_abs_graviton_parallel']:.6f} | "
            f"{m['zero_graviton_perpendicular']:.3e} | {m['zero_bfield_23']:.3e} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_scales(text: str) -> list[int]:
    return gps.sps.pfr.parse_scales(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-prime", type=float, default=1.0)
    parser.add_argument("--scales", type=parse_scales, default=gps.sps.pfr.DEFAULT_SCALES)
    parser.add_argument("--max-t", type=float, default=0.75)
    parser.add_argument("--step", type=float, default=0.125)
    parser.add_argument("--min-t", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--trace-dropped",
        action="store_true",
        help="drop the delta_IJ pieces of v_IJ before building projected channels",
    )
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--markdown-out", type=str, default=None)
    args = parser.parse_args()

    candidates = scan_projected_channels(
        alpha_prime=args.alpha_prime,
        scales=args.scales,
        max_t=args.max_t,
        step=args.step,
        min_t=args.min_t,
        trace_dropped=args.trace_dropped,
    )
    print_report(candidates, top_k=args.top_k)

    report = {
        "parameters": {
            "alpha_prime": args.alpha_prime,
            "scales": args.scales,
            "max_t": args.max_t,
            "step": args.step,
            "min_t": args.min_t,
            "trace_dropped": args.trace_dropped,
        },
        "candidates": candidates,
    }
    if args.json_out is not None:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n")
        print(f"Wrote JSON report to {out_path}")
    if args.markdown_out is not None:
        md_path = Path(args.markdown_out)
        md_path.write_text(markdown_report(candidates, report["parameters"]))
        print(f"Wrote markdown report to {md_path}")


if __name__ == "__main__":
    main()
