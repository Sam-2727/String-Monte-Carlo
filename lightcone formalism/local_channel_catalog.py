#!/usr/bin/env python3
"""
Comprehensive catalog of local Xi_loc correction sectors for cubic channels.

The earlier local-channel checks established three benchmark facts:

1. benchmark graviton qq channels collapse to the reduced Xi-degree-0 piece,
2. the benchmark dilaton qq channel is pure quartic in Xi_loc,
3. the sampled trace-dropped delta benchmark channel vanishes.

This helper extends that from a few benchmark channels to the full small
polarization basis used in the code:

    {parallel, perp23, perp24, dilaton, b23}^3.

For each channel and sampled lambda-ratio we record the Xi-degree profile of the
exact local polynomial. This identifies where genuine local corrections survive
before any nonzero-mode Xi contraction is performed.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

import fermionic_graviton_contraction as fgc
import local_channel_response as lcr


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


def _channel_label(channel: tuple[str, str, str]) -> str:
    return f"({channel[0]}, {channel[1]}, {channel[2]})"


def classify_profile_signature(
    profiles: tuple[tuple[tuple[int, int], ...], ...],
) -> str:
    if all(not profile for profile in profiles):
        return "vanishing"
    if all(profile == ((0, 1),) for profile in profiles):
        return "reduced_only"
    if all(profile == ((2, 4),) for profile in profiles):
        return "pure_quadratic_local"
    if all(profile == ((4, 14),) for profile in profiles):
        return "pure_quartic_local"
    if all(profile == ((0, 1), (4, 14)) for profile in profiles):
        return "reduced_plus_quartic"
    return "other"


@lru_cache(maxsize=None)
def channel_catalog(
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    *,
    response_kind: str = "qq",
    trace_dropped: bool = True,
) -> dict[str, Any]:
    polarizations = fgc.polarization_tensors()
    keys = _sorted_channel_keys()

    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    channels_by_category: dict[str, list[tuple[str, str, str]]] = {}
    profile_signatures: dict[str, int] = {}

    for key1 in keys:
        for key2 in keys:
            for key3 in keys:
                channel = (key1, key2, key3)
                profiles = []
                xi_zero_values = []
                max_abs_coeff = 0.0
                for lambda_ratio in lambda_grid:
                    poly = lcr.local_channel_response_polynomial(
                        polarizations[key1],
                        polarizations[key2],
                        polarizations[key3],
                        lambda_ratio,
                        response_kind=response_kind,
                        trace_dropped=trace_dropped,
                    )
                    profile = tuple(sorted(lcr.xi_degree_profile(poly).items()))
                    profiles.append(profile)
                    xi_zero_values.append(complex(lcr.xi_zero_component(poly)))
                    if poly:
                        max_abs_coeff = max(
                            max_abs_coeff,
                            max(float(abs(value)) for value in poly.values()),
                        )

                profile_signature = tuple(profiles)
                category = classify_profile_signature(profile_signature)
                counts[category] = counts.get(category, 0) + 1
                channels_by_category.setdefault(category, []).append(channel)
                profile_key = repr(profile_signature)
                profile_signatures[profile_key] = profile_signatures.get(profile_key, 0) + 1
                rows.append(
                    {
                        "channel": list(channel),
                        "label": _channel_label(channel),
                        "category": category,
                        "profiles": [[list(item) for item in profile] for profile in profile_signature],
                        "xi_zero_values": xi_zero_values,
                        "max_abs_xi_zero": float(max(abs(value) for value in xi_zero_values)),
                        "max_abs_coeff": max_abs_coeff,
                    }
                )

    return {
        "lambda_grid": list(lambda_grid),
        "response_kind": response_kind,
        "trace_dropped": trace_dropped,
        "counts": counts,
        "profile_signatures": profile_signatures,
        "channels_by_category": {
            category: [list(channel) for channel in channels]
            for category, channels in channels_by_category.items()
        },
        "rows": rows,
    }


def summarize_catalog(
    lambda_grid: tuple[float, ...] = DEFAULT_LAMBDA_GRID,
    *,
    response_kind: str = "qq",
    trace_dropped: bool = True,
) -> dict[str, Any]:
    catalog = channel_catalog(
        lambda_grid,
        response_kind=response_kind,
        trace_dropped=trace_dropped,
    )
    counts = dict(sorted(catalog["counts"].items()))
    dominant = sorted(
        catalog["profile_signatures"].items(),
        key=lambda item: (-item[1], item[0]),
    )
    examples: dict[str, list[list[str]]] = {}
    for category, channels in catalog["channels_by_category"].items():
        examples[category] = channels[:5]
    return {
        "lambda_grid": catalog["lambda_grid"],
        "response_kind": response_kind,
        "trace_dropped": trace_dropped,
        "counts": counts,
        "dominant_profile_signatures": dominant[:10],
        "example_channels": examples,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("=" * 96)
    print("LOCAL CHANNEL CATALOG")
    print("=" * 96)
    print(
        f"response_kind={summary['response_kind']}, "
        f"trace_dropped={summary['trace_dropped']}, "
        f"lambda_grid={summary['lambda_grid']}"
    )
    print("Category counts:")
    for category, count in summary["counts"].items():
        print(f"  {category:24s} {count:3d}")
    print()
    print("Representative channels:")
    for category, channels in summary["example_channels"].items():
        print(f"  {category:24s} {channels}")
    print()
    print("Dominant profile signatures:")
    for signature, count in summary["dominant_profile_signatures"]:
        print(f"  {count:3d}  {signature}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--response-kind",
        choices=("qq", "delta"),
        default="qq",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="optional path for a JSON dump",
    )
    args = parser.parse_args()

    summary = summarize_catalog(response_kind=args.response_kind)
    print_summary(summary)

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(json_safe(summary), indent=2))
        print()
        print(f"Wrote JSON summary to {args.json_out}")


if __name__ == "__main__":
    main()
