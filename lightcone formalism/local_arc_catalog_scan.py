#!/usr/bin/env python3
"""
Vacuum-contracted local catalog scan over simple arc-admixture candidates.

The earlier `local_arc_candidate_scan.py` established that the benchmark
three-point channels are unchanged when the canonical endpoint-difference local
fermion is supplemented by simple one-sided arc differences. This module pushes
that statement to the full sampled three-point polarization catalog.

For the family

    Lambda_trial
      = Lambda_join
      + c_{nabla,+} nabla_+ theta
      + c_{nabla,-} nabla_- theta,

the mixed zero-mode data are unchanged, and the remaining candidate dependence
enters only through the diagonal Xi two-point scalar in the three-point vacuum
reduction. We therefore contract the full local channel catalog for the
sampled arc family and compare it to the reduced degree-0 catalog.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

import fermionic_graviton_contraction as fgc
import local_arc_candidate_scan as lacs
import local_channel_response as lcr
import local_vacuum_reduction as lvr


DEFAULT_LAMBDA_GRID = (0.25, 0.4, 0.5)


def json_safe(value: Any) -> Any:
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


def _sorted_channel_keys() -> tuple[str, ...]:
    return tuple(sorted(fgc.polarization_tensors().keys()))


@lru_cache(maxsize=None)
def _channel_poly_cache(
    lambda_grid: tuple[float, ...],
    response_kind: str,
    trace_dropped: bool,
) -> tuple[dict[str, Any], ...]:
    polarizations = fgc.polarization_tensors()
    keys = _sorted_channel_keys()
    rows = []
    for key1 in keys:
        for key2 in keys:
            for key3 in keys:
                channel = (key1, key2, key3)
                polys = []
                xi_zero_values = []
                for lambda_ratio in lambda_grid:
                    poly = lcr.local_channel_response_polynomial(
                        polarizations[key1],
                        polarizations[key2],
                        polarizations[key3],
                        lambda_ratio,
                        response_kind=response_kind,
                        trace_dropped=trace_dropped,
                    )
                    polys.append(poly)
                    xi_zero_values.append(complex(lcr.xi_zero_component(poly)))
                rows.append(
                    {
                        "channel": list(channel),
                        "polys": polys,
                        "xi_zero_values": xi_zero_values,
                    }
                )
    return tuple(rows)


def contracted_catalog_for_candidate(
    n1: int,
    n2: int,
    coeff_nabla_plus: float,
    coeff_nabla_minus: float,
    *,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    response_kind: str = "qq",
    trace_dropped: bool = True,
) -> dict[str, Any]:
    two_point_scalar = lvr.local_candidate_two_point_scalar(
        n1,
        n2,
        coeff_nabla_plus=coeff_nabla_plus,
        coeff_nabla_minus=coeff_nabla_minus,
    )
    rows = []
    counts: dict[str, int] = {}
    max_abs_error = 0.0
    for row in _channel_poly_cache(lambda_grid, response_kind, trace_dropped):
        contracted_values = [
            lvr.vacuum_contract_local_polynomial(
                poly,
                two_point_scalar=two_point_scalar,
            )
            for poly in row["polys"]
        ]
        xi_zero_values = [complex(value) for value in row["xi_zero_values"]]
        max_channel_error = max(
            abs(contracted - reduced)
            for contracted, reduced in zip(contracted_values, xi_zero_values)
        )
        max_abs_value = max(abs(value) for value in contracted_values)
        category = "vanishing" if max_abs_value < 1.0e-12 else "reduced_only"
        counts[category] = counts.get(category, 0) + 1
        max_abs_error = max(max_abs_error, float(max_channel_error))
        rows.append(
            {
                "channel": row["channel"],
                "category": category,
                "contracted_values": contracted_values,
                "xi_zero_values": xi_zero_values,
                "max_abs_error": float(max_channel_error),
            }
        )

    return {
        "candidate": lacs.candidate_label(coeff_nabla_plus, coeff_nabla_minus),
        "coeff_nabla_plus": float(coeff_nabla_plus),
        "coeff_nabla_minus": float(coeff_nabla_minus),
        "response_kind": response_kind,
        "trace_dropped": bool(trace_dropped),
        "lambda_grid": list(lambda_grid),
        "two_point_scalar": float(two_point_scalar),
        "counts": counts,
        "rows": rows,
        "max_abs_error": float(max_abs_error),
    }


def sampled_arc_catalog_scan(
    n1: int = 128,
    n2: int = 192,
    *,
    arc_candidates: tuple[tuple[float, float], ...] = lacs.DEFAULT_ARC_CANDIDATES,
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
) -> dict[str, Any]:
    qq_rows = []
    delta_rows = []
    qq_max_abs_error = 0.0
    delta_max_abs_error = 0.0
    for coeff_nabla_plus, coeff_nabla_minus in arc_candidates:
        qq_report = contracted_catalog_for_candidate(
            n1,
            n2,
            coeff_nabla_plus,
            coeff_nabla_minus,
            lambda_grid=lambda_grid,
            response_kind="qq",
            trace_dropped=True,
        )
        delta_report = contracted_catalog_for_candidate(
            n1,
            n2,
            coeff_nabla_plus,
            coeff_nabla_minus,
            lambda_grid=lambda_grid,
            response_kind="delta",
            trace_dropped=True,
        )
        qq_rows.append(
            {
                "candidate": qq_report["candidate"],
                "counts": qq_report["counts"],
                "max_abs_error": qq_report["max_abs_error"],
            }
        )
        delta_rows.append(
            {
                "candidate": delta_report["candidate"],
                "counts": delta_report["counts"],
                "max_abs_error": delta_report["max_abs_error"],
            }
        )
        qq_max_abs_error = max(qq_max_abs_error, qq_report["max_abs_error"])
        delta_max_abs_error = max(delta_max_abs_error, delta_report["max_abs_error"])

    return {
        "parameters": {
            "n1": int(n1),
            "n2": int(n2),
            "lambda_grid": list(lambda_grid),
            "arc_candidates": [list(item) for item in arc_candidates],
        },
        "qq_rows": qq_rows,
        "delta_rows": delta_rows,
        "qq_max_abs_error": float(qq_max_abs_error),
        "delta_max_abs_error": float(delta_max_abs_error),
        "pass": qq_max_abs_error < 1.0e-12 and delta_max_abs_error < 1.0e-12,
    }


def print_report(report: dict[str, Any]) -> None:
    print("=" * 112)
    print("LOCAL ARC CATALOG SCAN")
    print("=" * 112)
    print(
        f"qq_max_abs_error={report['qq_max_abs_error']:.3e}, "
        f"delta_max_abs_error={report['delta_max_abs_error']:.3e}"
    )
    print("qq counts by candidate:")
    for row in report["qq_rows"]:
        print(f"  {row['candidate']:>12s}  {row['counts']}")
    print("delta counts by candidate:")
    for row in report["delta_rows"]:
        print(f"  {row['candidate']:>12s}  {row['counts']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report = sampled_arc_catalog_scan()
    print_report(report)
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(json_safe(report), indent=2, sort_keys=True) + "\n")
        print(f"Wrote JSON report to {args.json_out}")


if __name__ == "__main__":
    main()
