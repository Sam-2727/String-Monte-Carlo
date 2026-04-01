from __future__ import annotations

"""Single-topology genus-4 compact-boson T-duality check.

Usage reminders:
  Reproduce the default run:
    ./.venv/bin/python "covariant formalism/python/genus4_single_topology_t_duality.py"

  Pick a different topology from the current generator output:
    ./.venv/bin/python "covariant formalism/python/genus4_single_topology_t_duality.py" --topology 5

  Pick custom random seed:
    ./.venv/bin/python "covariant formalism/python/genus4_single_topology_t_duality.py" --seed 123

  Supply custom edge lengths:
    ./.venv/bin/python "covariant formalism/python/genus4_single_topology_t_duality.py" --lengths 73 289 154 279 51 183 74 225 203 75 136 85 152 297 76 56 276 218 221 266 173

The default random lengths are drawn uniformly from the integers in (50, 300),
namely 51 through 299 inclusive.

The check performed is:
  R^4 Theta(4 i R^2 T') - R^(-4) Theta(4 i R^(-2) T').

Note:
  This script uses the current genus-4 output of Fast_Ribbon_Generator.py and
  selects one topology from that list. It is intended as a quick diagnostic,
  not as an exhaustive genus-4 scan.
"""

import argparse
import os
import random
import sys
from typing import Sequence

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import Fast_Ribbon_Generator as frg
import compact_partition as cp


GENUS4_DEFAULT_TOPOLOGY = 1
GENUS4_DEFAULT_SEED = 404
GENUS4_DEFAULT_EDGE_COUNT = 21
GENUS4_DEFAULT_EDGE_LENGTH_BOUNDS = (51, 299)
GENUS4_DEFAULT_R_VALUES = tuple(round(1.0 + 0.1 * i, 1) for i in range(1, 11))
GENUS4_DEFAULT_THETA_CUTOFFS = (2, 3)


def random_genus4_edge_lengths(
    seed: int = GENUS4_DEFAULT_SEED,
    count: int = GENUS4_DEFAULT_EDGE_COUNT,
    bounds: tuple[int, int] = GENUS4_DEFAULT_EDGE_LENGTH_BOUNDS,
) -> tuple[int, ...]:
    """Draw reproducible random edge lengths from the inclusive integer interval."""
    lo, hi = bounds
    rng = random.Random(seed)
    return tuple(rng.randint(lo, hi) for _ in range(count))


def current_genus4_graph_data(topology: int = GENUS4_DEFAULT_TOPOLOGY) -> tuple[dict, int]:
    """Return one genus-4 topology from the current generator output."""
    rgs = frg.generate_ribbon_graphs_fixed_genus(4, n_faces=1, verbose=False, workers=1)
    count = len(rgs)
    if not 1 <= topology <= count:
        raise ValueError(f"topology must be in 1..{count}, got {topology}")

    rg = rgs[topology - 1]
    boundary_data = frg.get_disc_boundary(rg)
    graph_data = {
        "edges": tuple((i + 1, a, b) for i, (a, b) in enumerate(rg[0])),
        "boundary": tuple((frm, to, e + 1) for frm, to, e in boundary_data["boundary"]),
        "sewing_pairs": tuple(
            (e, p1, p2) for e, (p1, p2) in sorted(boundary_data["sewing"].items())
        ),
    }
    return graph_data, count


def genus4_t_duality_row(T_reduced, R: float, N: int) -> dict:
    """One row of the genus-4 theta-only duality check."""
    inv_R = 1.0 / R
    theta_R = cp.theta_sum_reduced(T_reduced, R, N=N)
    theta_inv = cp.theta_sum_reduced(T_reduced, inv_R, N=N)
    lhs = (R ** 4) * theta_R
    rhs = (inv_R ** 4) * theta_inv
    residual = lhs - rhs
    relative = abs(residual) / max(abs(lhs), abs(rhs))
    return {
        "R": float(R),
        "lhs": float(lhs),
        "rhs": float(rhs),
        "residual": float(residual),
        "relative_residual": float(relative),
    }


def check_single_genus4_topology(
    topology: int = GENUS4_DEFAULT_TOPOLOGY,
    edge_lengths: Sequence[int] | None = None,
    seed: int = GENUS4_DEFAULT_SEED,
    r_values: Sequence[float] = GENUS4_DEFAULT_R_VALUES,
    theta_cutoffs: Sequence[int] = GENUS4_DEFAULT_THETA_CUTOFFS,
) -> dict:
    """Run the genus-4 check for one selected topology."""
    if edge_lengths is None:
        edge_lengths = random_genus4_edge_lengths(seed=seed)

    edge_lengths = tuple(int(x) for x in edge_lengths)
    if len(edge_lengths) != GENUS4_DEFAULT_EDGE_COUNT:
        raise ValueError(
            f"Expected {GENUS4_DEFAULT_EDGE_COUNT} edge lengths, got {len(edge_lengths)}"
        )

    graph_data, topology_count = current_genus4_graph_data(topology)
    geom = cp.compact_boson_graph_geometry(edge_lengths, graph_data)
    T_reduced = geom["T_reduced"]

    cutoff_results = []
    for cutoff in theta_cutoffs:
        rows = []
        worst = None
        for r in r_values:
            row = genus4_t_duality_row(T_reduced, float(r), int(cutoff))
            rows.append(row)
            if worst is None or row["relative_residual"] > worst["relative_residual"]:
                worst = row
        cutoff_results.append(
            {
                "theta_cutoff": int(cutoff),
                "rows": tuple(rows),
                "worst": worst,
            }
        )

    return {
        "topology": int(topology),
        "topology_count": topology_count,
        "seed": int(seed),
        "edge_lengths": edge_lengths,
        "total_L": int(geom["L"]),
        "T_shape": tuple(T_reduced.shape),
        "cutoff_results": tuple(cutoff_results),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a single-topology genus-4 compact-boson T-duality check."
    )
    parser.add_argument(
        "--topology",
        type=int,
        default=GENUS4_DEFAULT_TOPOLOGY,
        help="Topology index in the current genus-4 generator output.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=GENUS4_DEFAULT_SEED,
        help="Seed for the default random edge-length draw.",
    )
    parser.add_argument(
        "--lengths",
        nargs=GENUS4_DEFAULT_EDGE_COUNT,
        type=int,
        metavar="L",
        help="Override the 21 edge lengths used in the check.",
    )
    args = parser.parse_args()

    edge_lengths = tuple(args.lengths) if args.lengths is not None else None
    summary = check_single_genus4_topology(
        topology=args.topology,
        edge_lengths=edge_lengths,
        seed=args.seed,
    )

    print(f"generator topology count = {summary['topology_count']}")
    print(f"topology index           = {summary['topology']}")
    print(f"seed                     = {summary['seed']}")
    print(f"edge lengths             = {list(summary['edge_lengths'])}")
    print(f"total L                  = {summary['total_L']}")
    print(f"T_reduced shape          = {summary['T_shape']}")
    print("check                    = R^4 Theta(4 i R^2 T') - R^(-4) Theta(4 i R^(-2) T')")

    for cutoff_data in summary["cutoff_results"]:
        print(f"\ntheta cutoff = {cutoff_data['theta_cutoff']}")
        for row in cutoff_data["rows"]:
            print(
                f"  R={row['R']:.1f}"
                f"  lhs={row['lhs']:.12f}"
                f"  rhs={row['rhs']:.12f}"
                f"  resid={row['residual']:>12.6e}"
                f"  rel={row['relative_residual']:>10.6e}"
            )

        worst = cutoff_data["worst"]
        print("  worst = "
              f"R={worst['R']:.1f}, "
              f"resid={worst['residual']:.6e}, "
              f"rel={worst['relative_residual']:.6e}")
