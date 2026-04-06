from __future__ import annotations

"""Reproduce the large-L compact-boson scaling fits for genera 1 through 4.

This script runs the same lattice partition-function sweeps used in the recent
notes, fitting

    log Z(L) = c + gamma * L + alpha * log L

over a fixed L-window and a fixed set of genus-dependent moduli/topology
families.

Defaults are chosen to reproduce the latest R=2.0 scan, but the radius can be
changed with ``--R``.

Examples
--------
Reproduce the latest sweep at R=2.0:

    ./.venv/bin/python "covariant formalism/python/genus1_to_4_scaling_sweep.py"

Repeat the same scan at R=1.4:

    ./.venv/bin/python "covariant formalism/python/genus1_to_4_scaling_sweep.py" --R 1.4

Also save a machine-readable summary:

    ./.venv/bin/python "covariant formalism/python/genus1_to_4_scaling_sweep.py" \
        --R 2.0 --json-out /tmp/radius_2_scaling.json
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import compact_partition as cp
import genus3_t_duality as g3
import genus4_single_topology_t_duality as g4


GENUS1_THETA_CUTOFF = 20
GENUS2_THETA_CUTOFF = 6
GENUS3_THETA_CUTOFF = 4
GENUS4_THETA_CUTOFF = 2

GENUS1_CASES = (
    ("3:5:2", (3, 5, 2)),
    ("1:2:2", (1, 2, 2)),
    ("1:1:3", (1, 1, 3)),
    ("2:3:5", (2, 3, 5)),
)

GENUS2_CASES = (
    ("top1_A", 1, (1, 1, 2, 2, 2, 2, 3, 3, 4)),
    ("top1_B", 1, (1, 1, 1, 2, 2, 3, 3, 3, 4)),
    ("top5_A", 5, (1, 1, 2, 2, 2, 2, 3, 3, 4)),
    ("top9_A", 9, (1, 1, 2, 2, 2, 2, 3, 3, 4)),
)

GENUS3_TOPOLOGIES = (1, 541, 1000, 1726)
GENUS3_BASE = (1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 8)

GENUS4_TOPOLOGY = 1
GENUS4_BASE = (1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 5, 5)


@dataclass(frozen=True)
class FitResult:
    genus: int
    case: str
    c: float
    gamma: float
    alpha: float
    A_prefactor: float
    mu: float
    r2: float
    max_abs_log_residual: float
    l2000_logz: float
    l5000_logz: float
    extra: dict[str, float | int | str]


def _build_l_values(start: int, stop: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("step must be positive")
    values = list(range(start, stop + 1, step))
    if not values:
        raise ValueError("empty L-range")
    return values


def _points_cache(dim: int, cutoff: int) -> np.ndarray:
    side = np.arange(-cutoff, cutoff + 1, dtype=float)
    grids = np.meshgrid(*([side] * dim), indexing="ij")
    return np.stack(grids, axis=-1).reshape(-1, dim)


def _theta_sum_dim2(Tp: np.ndarray, radius: float, cutoff: int = GENUS1_THETA_CUTOFF) -> float:
    side = np.arange(-cutoff, cutoff + 1, dtype=float)
    s1, s2 = np.meshgrid(side, side)
    s1 = s1.ravel()
    s2 = s2.ravel()
    quad = (
        s1 ** 2 * Tp[0, 0]
        + s1 * s2 * Tp[0, 1]
        + s2 * s1 * Tp[1, 0]
        + s2 ** 2 * Tp[1, 1]
    )
    return float(np.sum(np.exp(-4.0 * math.pi * (radius ** 2) * quad)))


def _theta_sum_points(T_reduced: np.ndarray, radius: float, points: np.ndarray) -> float:
    quad = np.einsum("ni,ij,nj->n", points, T_reduced, points, optimize=True)
    return float(np.sum(np.exp(-4.0 * math.pi * (radius ** 2) * quad)))


def _fit_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    Larr = np.array([row["L"] for row in rows], dtype=float)
    logz = np.array([row["logZ"] for row in rows], dtype=float)
    design = np.column_stack([np.ones_like(Larr), Larr, np.log(Larr)])
    coef, *_ = np.linalg.lstsq(design, logz, rcond=None)
    predicted = design @ coef
    ss_res = float(np.sum((logz - predicted) ** 2))
    ss_tot = float(np.sum((logz - np.mean(logz)) ** 2))
    c0, gamma, alpha = coef
    return {
        "c": float(c0),
        "gamma": float(gamma),
        "alpha": float(alpha),
        "A_prefactor": float(math.exp(c0)),
        "mu": float(math.exp(gamma)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
        "max_abs_log_residual": float(np.max(np.abs(logz - predicted))),
        "L2000_logZ": float(rows[0]["logZ"]),
        "L5000_logZ": float(rows[-1]["logZ"]),
    }


def _fit_alpha_linear(alpha_means: dict[int, float]) -> dict[str, float | list[float]]:
    genus = np.array(sorted(alpha_means), dtype=float)
    alpha = np.array([alpha_means[g] for g in sorted(alpha_means)], dtype=float)
    design = np.column_stack([np.ones_like(genus), genus])
    coef, *_ = np.linalg.lstsq(design, alpha, rcond=None)
    predicted = design @ coef
    residuals = alpha - predicted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((alpha - np.mean(alpha)) ** 2))
    return {
        "A": float(coef[0]),
        "B": float(coef[1]),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
        "residuals": residuals.tolist(),
    }


def _print_fit(result: FitResult) -> None:
    payload = {
        "genus": result.genus,
        "case": result.case,
        "c": result.c,
        "gamma": result.gamma,
        "alpha": result.alpha,
        "A_prefactor": result.A_prefactor,
        "mu": result.mu,
        "r2": result.r2,
        "max_abs_log_residual": result.max_abs_log_residual,
        "L2000_logZ": result.l2000_logz,
        "L5000_logZ": result.l5000_logz,
        **result.extra,
    }
    print(payload)


def _scale_edges(total_L: int, base: Iterable[int]) -> list[int]:
    base = list(int(x) for x in base)
    factor = total_L // (2 * sum(base))
    out = [factor * x for x in base]
    if 2 * sum(out) != total_L:
        raise ValueError(f"Scaled edge lengths do not reproduce L={total_L}")
    return out


def run_sweep(radius: float, l_values: list[int]) -> dict[str, object]:
    summary: list[FitResult] = []
    alpha_by_genus: dict[int, list[float]] = {}

    genus2_points = _points_cache(4, GENUS2_THETA_CUTOFF)
    genus3_points = _points_cache(6, GENUS3_THETA_CUTOFF)
    genus4_points = _points_cache(8, GENUS4_THETA_CUTOFF)

    print("BEGIN genus1")
    g1_rows = {name: [] for name, _ in GENUS1_CASES}
    for L in l_values:
        Mat = cp.direct_mat_n_fast(L)
        for name, (a, b, c) in GENUS1_CASES:
            denom = 2 * (a + b + c)
            l1 = a * L // denom
            l2 = b * L // denom
            Aprime = cp.direct_red_traced_mat(L, l1, l2, Mat)
            W = cp.mat_w(L, l1, l2, Mat)
            T1 = cp.mat_t_first_part(L, l1, l2, Mat)
            T2 = cp.mat_t_second_part(L, l1, l2, W, Aprime)
            Tp = cp.mat_t_prime(cp.symm(T1 - T2))
            sign, logdet = np.linalg.slogdet(Aprime)
            if sign <= 0:
                raise RuntimeError(f"genus 1 case {name} at L={L} has sign={sign}")
            theta = _theta_sum_dim2(Tp, radius)
            logZ = math.log(radius) - 0.5 * float(logdet) + math.log(theta)
            g1_rows[name].append({"L": L, "logZ": logZ})

    alpha_by_genus[1] = []
    for name, _ in GENUS1_CASES:
        fit = _fit_rows(g1_rows[name])
        alpha_by_genus[1].append(fit["alpha"])
        result = FitResult(
            genus=1,
            case=name,
            c=fit["c"],
            gamma=fit["gamma"],
            alpha=fit["alpha"],
            A_prefactor=fit["A_prefactor"],
            mu=fit["mu"],
            r2=fit["r2"],
            max_abs_log_residual=fit["max_abs_log_residual"],
            l2000_logz=fit["L2000_logZ"],
            l5000_logz=fit["L5000_logZ"],
            extra={},
        )
        summary.append(result)
        _print_fit(result)

    print("BEGIN genus2")
    genus2_graphs = {topology: cp.get_stored_genus2_graph(topology) for _, topology, _ in GENUS2_CASES}
    g2_rows = {name: [] for name, _, _ in GENUS2_CASES}
    for L in l_values:
        Mat = cp.direct_mat_n_fast(L)
        for name, topology, base in GENUS2_CASES:
            edge_lengths = _scale_edges(L, base)
            geom = cp.compact_boson_graph_geometry(edge_lengths, genus2_graphs[topology], Mat=Mat)
            sign, logdet = np.linalg.slogdet(geom["A_prime"])
            if sign <= 0:
                raise RuntimeError(f"genus 2 case {name} at L={L} has sign={sign}")
            theta = _theta_sum_points(geom["T_reduced"], radius, genus2_points)
            logZ = math.log(radius) - 0.5 * float(logdet) + math.log(theta)
            g2_rows[name].append({"L": L, "logZ": logZ})

    alpha_by_genus[2] = []
    for name, topology, _ in GENUS2_CASES:
        fit = _fit_rows(g2_rows[name])
        alpha_by_genus[2].append(fit["alpha"])
        result = FitResult(
            genus=2,
            case=name,
            c=fit["c"],
            gamma=fit["gamma"],
            alpha=fit["alpha"],
            A_prefactor=fit["A_prefactor"],
            mu=fit["mu"],
            r2=fit["r2"],
            max_abs_log_residual=fit["max_abs_log_residual"],
            l2000_logz=fit["L2000_logZ"],
            l5000_logz=fit["L5000_logZ"],
            extra={"topology": topology},
        )
        summary.append(result)
        _print_fit(result)

    print("BEGIN genus3")
    genus3_graphs = {topology: g3.get_stored_genus3_graph(topology) for topology in GENUS3_TOPOLOGIES}
    g3_rows = {topology: [] for topology in GENUS3_TOPOLOGIES}
    for L in l_values:
        edge_lengths = _scale_edges(L, GENUS3_BASE)
        Mat = cp.direct_mat_n_fast(L)
        for topology in GENUS3_TOPOLOGIES:
            geom = cp.compact_boson_graph_geometry(edge_lengths, genus3_graphs[topology], Mat=Mat)
            sign, logdet = np.linalg.slogdet(geom["A_prime"])
            if sign <= 0:
                raise RuntimeError(f"genus 3 topology {topology} at L={L} has sign={sign}")
            theta = _theta_sum_points(geom["T_reduced"], radius, genus3_points)
            logZ = math.log(radius) - 0.5 * float(logdet) + math.log(theta)
            g3_rows[topology].append({"L": L, "logZ": logZ})

    alpha_by_genus[3] = []
    for topology in GENUS3_TOPOLOGIES:
        fit = _fit_rows(g3_rows[topology])
        alpha_by_genus[3].append(fit["alpha"])
        result = FitResult(
            genus=3,
            case=f"topology_{topology}",
            c=fit["c"],
            gamma=fit["gamma"],
            alpha=fit["alpha"],
            A_prefactor=fit["A_prefactor"],
            mu=fit["mu"],
            r2=fit["r2"],
            max_abs_log_residual=fit["max_abs_log_residual"],
            l2000_logz=fit["L2000_logZ"],
            l5000_logz=fit["L5000_logZ"],
            extra={"topology": topology},
        )
        summary.append(result)
        _print_fit(result)

    print("BEGIN genus4")
    genus4_graph, topology_count = g4.current_genus4_graph_data(GENUS4_TOPOLOGY)
    g4_rows = []
    for L in l_values:
        edge_lengths = _scale_edges(L, GENUS4_BASE)
        Mat = cp.direct_mat_n_fast(L)
        geom = cp.compact_boson_graph_geometry(edge_lengths, genus4_graph, Mat=Mat)
        sign, logdet = np.linalg.slogdet(geom["A_prime"])
        if sign <= 0:
            raise RuntimeError(f"genus 4 topology {GENUS4_TOPOLOGY} at L={L} has sign={sign}")
        theta = _theta_sum_points(geom["T_reduced"], radius, genus4_points)
        logZ = math.log(radius) - 0.5 * float(logdet) + math.log(theta)
        g4_rows.append({"L": L, "logZ": logZ})

    fit = _fit_rows(g4_rows)
    alpha_by_genus[4] = [fit["alpha"]]
    result = FitResult(
        genus=4,
        case=f"topology_{GENUS4_TOPOLOGY}",
        c=fit["c"],
        gamma=fit["gamma"],
        alpha=fit["alpha"],
        A_prefactor=fit["A_prefactor"],
        mu=fit["mu"],
        r2=fit["r2"],
        max_abs_log_residual=fit["max_abs_log_residual"],
        l2000_logz=fit["L2000_logZ"],
        l5000_logz=fit["L5000_logZ"],
        extra={"topology": GENUS4_TOPOLOGY, "generator_topology_count": topology_count},
    )
    summary.append(result)
    _print_fit(result)

    print("ALPHA_GENUS_MEANS")
    alpha_means = {}
    alpha_stats = []
    for genus in sorted(alpha_by_genus):
        arr = np.array(alpha_by_genus[genus], dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=0))
        alpha_means[genus] = mean
        alpha_stats.append({"genus": genus, "alpha_mean": mean, "alpha_std": std})
        print(alpha_stats[-1])

    linear_fit = _fit_alpha_linear(alpha_means)
    print("ALPHA_LINEAR_FIT")
    print({"R": radius, **linear_fit})

    return {
        "radius": radius,
        "L_values": l_values,
        "theta_cutoffs": {
            "genus1": GENUS1_THETA_CUTOFF,
            "genus2": GENUS2_THETA_CUTOFF,
            "genus3": GENUS3_THETA_CUTOFF,
            "genus4": GENUS4_THETA_CUTOFF,
        },
        "results": [
            {
                "genus": row.genus,
                "case": row.case,
                "c": row.c,
                "gamma": row.gamma,
                "alpha": row.alpha,
                "A_prefactor": row.A_prefactor,
                "mu": row.mu,
                "r2": row.r2,
                "max_abs_log_residual": row.max_abs_log_residual,
                "L2000_logZ": row.l2000_logz,
                "L5000_logZ": row.l5000_logz,
                **row.extra,
            }
            for row in summary
        ],
        "alpha_genus_means": alpha_stats,
        "alpha_linear_fit": linear_fit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep the large-L compact-boson scaling fits for genera 1 through 4."
    )
    parser.add_argument(
        "--R",
        type=float,
        default=2.0,
        help="Compactification radius used in the lattice theta sum. Default: 2.0",
    )
    parser.add_argument(
        "--L-start",
        type=int,
        default=2000,
        help="First L value in the sweep. Default: 2000",
    )
    parser.add_argument(
        "--L-stop",
        type=int,
        default=5000,
        help="Last L value in the sweep. Default: 5000",
    )
    parser.add_argument(
        "--L-step",
        type=int,
        default=200,
        help="Step size in L. Default: 200",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path where the full summary is written as JSON.",
    )
    args = parser.parse_args()

    l_values = _build_l_values(args.L_start, args.L_stop, args.L_step)
    summary = run_sweep(radius=float(args.R), l_values=l_values)

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote JSON summary to {args.json_out}")


if __name__ == "__main__":
    main()
