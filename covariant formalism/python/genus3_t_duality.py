from __future__ import annotations

"""Load genus-3 ribbon graphs from a text file and scan T-duality.

Usage reminders:
  Full default scan over all stored topologies:
    ./.venv/bin/python "covariant formalism/python/genus3_t_duality.py"

  Quick check of one stored topology:
    ./.venv/bin/python "covariant formalism/python/genus3_t_duality.py" --topology 541

  Full scan with custom edge lengths:
    ./.venv/bin/python "covariant formalism/python/genus3_t_duality.py" --lengths 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134

The check performed is:
  R^3 Theta(4 i R^2 T') - R^(-3) Theta(4 i R^(-2) T').
"""

import argparse
import ast
import math
import os
import sys
import time
from functools import lru_cache
from typing import Sequence

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import compact_partition as cp

GENUS3_GRAPH_DATA_PATH = os.path.join(_here, "genus3_ribbon_graph_data.txt")
GENUS3_DEFAULT_EDGE_LENGTHS = tuple(100 + i for i in range(15))
GENUS3_DEFAULT_R_VALUES = tuple(round(1.0 + 0.1 * i, 1) for i in range(1, 11))
GENUS3_DEFAULT_THETA_CUTOFF = 4


def _normalize_triples(value: object, *, name: str) -> tuple[tuple[int, int, int], ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a tuple of triples")
    triples = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != 3:
            raise ValueError(f"{name} contains a non-triple entry: {item!r}")
        triples.append(tuple(int(x) for x in item))
    return tuple(triples)


@lru_cache(maxsize=1)
def _load_stored_genus3_graphs() -> tuple[dict, ...]:
    if not os.path.exists(GENUS3_GRAPH_DATA_PATH):
        raise FileNotFoundError(
            f"Missing genus-3 graph data file: {GENUS3_GRAPH_DATA_PATH}"
        )

    topology_count: int | None = None
    graphs = []
    current_topology: int | None = None
    current_data: dict[str, tuple[tuple[int, int, int], ...]] | None = None

    def finalize_current() -> None:
        nonlocal current_topology, current_data
        if current_topology is None:
            return
        if current_data is None:
            raise ValueError(f"Topology {current_topology} is missing all data")
        required = {"edges", "boundary", "sewing_pairs"}
        missing = required.difference(current_data)
        if missing:
            raise ValueError(
                f"Topology {current_topology} is missing fields: {sorted(missing)}"
            )
        expected_topology = len(graphs) + 1
        if current_topology != expected_topology:
            raise ValueError(
                f"Expected Topology {expected_topology}, found Topology {current_topology}"
            )
        graphs.append(
            {
                "edges": current_data["edges"],
                "boundary": current_data["boundary"],
                "sewing_pairs": current_data["sewing_pairs"],
            }
        )
        current_topology = None
        current_data = None

    with open(GENUS3_GRAPH_DATA_PATH, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("topology_count"):
                topology_count = int(line.split("=", 1)[1].strip())
                continue
            if line.startswith("Topology "):
                finalize_current()
                current_topology = int(line.split()[1])
                current_data = {}
                continue
            if "=" not in line:
                raise ValueError(f"Unrecognized line in genus-3 data file: {line!r}")

            if current_data is None or current_topology is None:
                raise ValueError(
                    f"Encountered data line outside a topology block: {line!r}"
                )
            key, rhs = (part.strip() for part in line.split("=", 1))
            if key not in {"edges", "boundary", "sewing_pairs"}:
                raise ValueError(f"Unexpected key {key!r} in genus-3 data file")
            current_data[key] = _normalize_triples(
                ast.literal_eval(rhs),
                name=key,
            )

    finalize_current()

    if topology_count is None:
        raise ValueError("genus3_ribbon_graph_data.txt does not declare topology_count")
    if len(graphs) != topology_count:
        raise ValueError(
            f"Declared topology_count = {topology_count}, but parsed {len(graphs)} topologies"
        )

    return tuple(graphs)


GENUS3_GRAPH_COUNT = len(_load_stored_genus3_graphs())


def get_stored_genus3_graph(topology: int) -> dict:
    if not 1 <= topology <= GENUS3_GRAPH_COUNT:
        raise ValueError(
            f"topology must be in 1..{GENUS3_GRAPH_COUNT}, got {topology}"
        )
    entry = _load_stored_genus3_graphs()[topology - 1]
    return {
        "edges": entry["edges"],
        "boundary": entry["boundary"],
        "sewing_pairs": entry["sewing_pairs"],
    }


def iter_stored_genus3_graphs() -> tuple[dict, ...]:
    return tuple(get_stored_genus3_graph(t) for t in range(1, GENUS3_GRAPH_COUNT + 1))


@lru_cache(maxsize=None)
def theta_lattice_points(dim: int, N: int) -> np.ndarray:
    side = np.arange(-N, N + 1, dtype=float)
    grids = np.meshgrid(*([side] * dim), indexing="ij")
    return np.stack(grids, axis=-1).reshape(-1, dim)


def theta_quadratic_form(
    T_reduced: np.ndarray,
    N: int = GENUS3_DEFAULT_THETA_CUTOFF,
) -> np.ndarray:
    points = theta_lattice_points(T_reduced.shape[0], N)
    return np.einsum("ni,ij,nj->n", points, T_reduced, points, optimize=True)


def genus3_t_duality_row(quad: np.ndarray, R: float) -> dict:
    inv_R = 1.0 / R
    theta_R = float(np.sum(np.exp(-4.0 * math.pi * (R ** 2) * quad)))
    theta_inv = float(np.sum(np.exp(-4.0 * math.pi * (inv_R ** 2) * quad)))
    lhs = (R ** 3) * theta_R
    rhs = (inv_R ** 3) * theta_inv
    residual = lhs - rhs
    relative = abs(residual) / max(abs(lhs), abs(rhs))
    return {
        "R": float(R),
        "lhs": lhs,
        "rhs": rhs,
        "residual": residual,
        "relative_residual": relative,
    }


def scan_genus3_t_duality(
    edge_lengths: Sequence[int] = GENUS3_DEFAULT_EDGE_LENGTHS,
    r_values: Sequence[float] = GENUS3_DEFAULT_R_VALUES,
    N: int = GENUS3_DEFAULT_THETA_CUTOFF,
    topologies: Sequence[int] | None = None,
    progress: bool = False,
) -> dict:
    edge_lengths = tuple(int(x) for x in edge_lengths)
    if len(edge_lengths) != 15:
        raise ValueError(f"Expected 15 edge lengths, got {len(edge_lengths)}")

    if topologies is None:
        topology_list = list(range(1, GENUS3_GRAPH_COUNT + 1))
    else:
        topology_list = [int(t) for t in topologies]
        for topology in topology_list:
            if not 1 <= topology <= GENUS3_GRAPH_COUNT:
                raise ValueError(
                    f"topology must be in 1..{GENUS3_GRAPH_COUNT}, got {topology}"
                )

    r_values = tuple(float(r) for r in r_values)
    total_L = 2 * sum(edge_lengths)
    Mat = cp.direct_mat_n_fast(total_L)

    start = time.time()
    results = []
    worst_by_R = {r: None for r in r_values}
    overall_worst = None

    for count, topology in enumerate(topology_list, start=1):
        geom = cp.compact_boson_graph_geometry(
            edge_lengths,
            get_stored_genus3_graph(topology),
            Mat=Mat,
        )
        if geom["T_reduced"].shape != (6, 6):
            raise ValueError(
                f"Topology {topology} produced T_reduced shape {geom['T_reduced'].shape}, expected (6, 6)"
            )
        quad = theta_quadratic_form(geom["T_reduced"], N=N)
        rows = []
        for r in r_values:
            row = genus3_t_duality_row(quad, r)
            row["topology"] = topology
            rows.append(row)
            current = worst_by_R[r]
            if current is None or row["relative_residual"] > current["relative_residual"]:
                worst_by_R[r] = row
            if overall_worst is None or row["relative_residual"] > overall_worst["relative_residual"]:
                overall_worst = row
        results.append({"topology": topology, "rows": tuple(rows)})

        if progress and (count % 100 == 0 or count == len(topology_list)):
            elapsed = time.time() - start
            print(
                f"Processed {count:>4d} / {len(topology_list)} genus-3 topologies "
                f"in {elapsed:>7.2f} s"
            )

    return {
        "edge_lengths": edge_lengths,
        "total_L": total_L,
        "theta_cutoff": int(N),
        "r_values": r_values,
        "topology_count": len(topology_list),
        "results": tuple(results),
        "worst_by_R": tuple(worst_by_R[r] for r in r_values),
        "overall_worst": overall_worst,
        "elapsed_seconds": time.time() - start,
    }


def quick_check_genus3_topology(
    topology: int,
    edge_lengths: Sequence[int] = GENUS3_DEFAULT_EDGE_LENGTHS,
    r_values: Sequence[float] = GENUS3_DEFAULT_R_VALUES,
    N: int = GENUS3_DEFAULT_THETA_CUTOFF,
) -> dict:
    """Run the genus-3 duality scan for a single stored topology."""
    summary = scan_genus3_t_duality(
        edge_lengths=edge_lengths,
        r_values=r_values,
        N=N,
        topologies=[topology],
        progress=False,
    )
    return {
        "topology": int(topology),
        "rows": summary["results"][0]["rows"],
        "overall_worst": summary["overall_worst"],
        "edge_lengths": summary["edge_lengths"],
        "total_L": summary["total_L"],
        "theta_cutoff": summary["theta_cutoff"],
        "elapsed_seconds": summary["elapsed_seconds"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check the genus-3 compact-boson T-duality relation."
    )
    parser.add_argument(
        "--topology",
        type=int,
        help="Run only one stored genus-3 topology for a quick check.",
    )
    parser.add_argument(
        "--lengths",
        nargs=15,
        type=int,
        metavar="L",
        help="Override the 15 edge lengths used in the genus-3 check.",
    )
    args = parser.parse_args()

    edge_lengths = (
        tuple(args.lengths)
        if args.lengths is not None
        else GENUS3_DEFAULT_EDGE_LENGTHS
    )

    print(f"Stored genus-3 one-face ribbon graphs: {GENUS3_GRAPH_COUNT}")
    print(f"graph data    = {GENUS3_GRAPH_DATA_PATH}")
    print(f"edge lengths = {list(edge_lengths)}")
    print(f"total L      = {2 * sum(edge_lengths)}")
    print(f"theta cutoff = {GENUS3_DEFAULT_THETA_CUTOFF}")
    print(
        "R scan       = "
        f"{GENUS3_DEFAULT_R_VALUES[0]:.1f} to {GENUS3_DEFAULT_R_VALUES[-1]:.1f}"
        " in steps of 0.1"
    )
    print("check        = R^3 Theta(4 i R^2 T') - R^(-3) Theta(4 i R^(-2) T')")

    if args.topology is not None:
        print(f"topology     = {args.topology}")
        quick = quick_check_genus3_topology(
            args.topology,
            edge_lengths=edge_lengths,
        )
        print("\nRows for selected topology:")
        for row in quick["rows"]:
            print(
                f"  R={row['R']:.1f}"
                f"  lhs={row['lhs']:>14.10f}"
                f"  rhs={row['rhs']:>14.10f}"
                f"  resid={row['residual']:>12.6e}"
                f"  rel={row['relative_residual']:>10.3e}"
            )

        worst = quick["overall_worst"]
        print("\nWorst case for selected topology:")
        print(
            f"  R={worst['R']:.1f}  topology={worst['topology']:>4d}"
            f"  resid={worst['residual']:>12.6e}"
            f"  rel={worst['relative_residual']:>10.3e}"
        )
        print(f"\nElapsed = {quick['elapsed_seconds']:.2f} s")
    else:
        summary = scan_genus3_t_duality(
            edge_lengths=edge_lengths,
            progress=True,
        )
        print("\nWorst relative residual by R:")
        for row in summary["worst_by_R"]:
            print(
                f"  R={row['R']:.1f}  topology={row['topology']:>4d}"
                f"  resid={row['residual']:>12.6e}"
                f"  rel={row['relative_residual']:>10.3e}"
            )

        worst = summary["overall_worst"]
        print("\nOverall worst case:")
        print(
            f"  R={worst['R']:.1f}  topology={worst['topology']:>4d}"
            f"  resid={worst['residual']:>12.6e}"
            f"  rel={worst['relative_residual']:>10.3e}"
        )
        print(f"\nElapsed = {summary['elapsed_seconds']:.2f} s")
