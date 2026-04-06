from __future__ import annotations

"""Python translation of the fixed-L pairwise checks from WeylTest.wlnb.

The notebook compares genus-1 theta-graph data at the same lattice size L but
different moduli. This script reproduces that workflow in Python using the
existing lattice, period, and analytic torus helpers in this repo.

For a pair of moduli (1, 2), it computes

    log_det_piece = (1/2) log(det(A'_2) / det(A'_1))

which is the log of the ratio of the determinant factors

    det(A'_1)^(-1/2) / det(A'_2)^(-1/2).

It also computes the analytic torus ratio

    log_z_ratio = log(Z(tau_1) / Z(tau_2)),

and the notebook-style residual

    log_residual = log_z_ratio - log_det_piece.

To mirror the old WeylTest logic, the local Weyl term is extracted from the
raw disc-coordinate singular coefficient without the puncture log|P1| term.
The raw local coefficient is measured by the same boundary-point linear fit
used in the notebook, so

    delta_S_vertex = (log|nu_2| - log|nu_1|) / 12.

In the original notebook, Z[tau] is the noncompact genus-1 free-boson partition
function, so the default analytic choice here is

    Z(tau) = Z_boson(tau) = 1 / (sqrt(Im tau) |eta(tau)|^2).

Numerically, WeylTest is checking whether

    log_residual ~= delta_S_vertex.
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


DEFAULT_L = 903 * 2
DEFAULT_REFERENCE = (301, 301)
DEFAULT_TARGETS = (
    (129, 129),
    (215, 301),
    (129, 387),
    (43, 301),
)


@dataclass(frozen=True)
class BoundaryData:
    l1: int
    l2: int
    l3: int
    log_nu_fit: float
    tau_raw_real: float
    tau_raw_imag: float
    tau_reduced_real: float
    tau_reduced_imag: float
    period1_real: float
    period1_imag: float
    period1_abs: float


@dataclass(frozen=True)
class PairResult:
    L: int
    reference: str
    target: str
    log_det_ratio_pf: float
    det_ratio_pf: float
    log_z_ratio: float
    z_ratio: float
    log_residual: float
    residual_ratio: float
    delta_S_vertex: float
    delta_log_p1: float
    delta_S_full: float
    exp_vertex_ratio: float
    exp_full_ratio: float
    log_residual_minus_delta_S_vertex: float
    log_residual_minus_delta_S_full: float
    reference_tau_real: float
    reference_tau_imag: float
    target_tau_real: float
    target_tau_imag: float
    reference_period1_abs: float
    target_period1_abs: float
    analytic_kind: str


def _format_modulus(l1: int, l2: int, l3: int) -> str:
    return f"({l1},{l2},{l3})"


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


def _log_det_cholesky(matrix: np.ndarray) -> float:
    chol = np.linalg.cholesky(matrix)
    return float(2.0 * np.sum(np.log(np.diag(chol))))


def _pole_intercept(f, k0: float, L: int) -> float:
    alpha_values = range(-40, 1, 2)
    fit_low, fit_high = -4, -1
    z0 = np.exp(1j * np.pi * (2.0 * k0 + 1.0) / L)

    xs: list[float] = []
    ys: list[float] = []
    for alpha in alpha_values:
        if not (fit_low <= alpha <= fit_high):
            continue
        z = np.exp(1j * np.pi * (2.0 * k0 / L + 2.0 * alpha / L + 1.0 / L))
        xs.append(float(np.log(abs(z - z0))))
        ys.append(float(np.log(abs(f(z)))))

    design = np.column_stack([np.ones(len(xs), dtype=float), np.asarray(xs, dtype=float)])
    coeff, *_ = np.linalg.lstsq(design, np.asarray(ys, dtype=float), rcond=None)
    return float(coeff[0])


def _pole_intercept_average(f, l1: int, l2: int, l3: int) -> float:
    L = 2 * (l1 + l2 + l3)
    ells = (l1, l1 + l2 + l3, 2 * l1 + 2 * l2 + l3)
    values = [_pole_intercept(f, float(k0), L) for k0 in ells]
    return float(np.mean(values))


def _boundary_data(L: int, l1: int, l2: int) -> BoundaryData:
    l3 = L // 2 - l1 - l2
    if l3 <= 0:
        raise ValueError(f"Need positive l3, got {l3} for L={L}, l1={l1}, l2={l2}")

    form = elt.make_cyl_eqn_improved(L, l1, l2)
    p1, p2, _ = elt.periods_improved(L, l1, l2, form)
    tau_raw = complex(p2 / p1)
    tau_reduced = _reduce_tau_to_fundamental_domain(tau_raw)

    def f(c: complex) -> complex:
        result = form(c)
        if isinstance(result, tuple):
            singular, poly = result
            return complex(singular * poly)
        return complex(result)

    log_nu_fit = _pole_intercept_average(f, l1, l2, l3)
    return BoundaryData(
        l1=l1,
        l2=l2,
        l3=l3,
        log_nu_fit=log_nu_fit,
        tau_raw_real=float(tau_raw.real),
        tau_raw_imag=float(tau_raw.imag),
        tau_reduced_real=float(tau_reduced.real),
        tau_reduced_imag=float(tau_reduced.imag),
        period1_real=float(complex(p1).real),
        period1_imag=float(complex(p1).imag),
        period1_abs=float(abs(p1)),
    )


def _det_ratio_piece(L: int, l1_ref: int, l2_ref: int, l1_tgt: int, l2_tgt: int) -> tuple[float, float]:
    base = cp.direct_mat_n_fast(L)
    aprime_ref = cp.direct_red_traced_mat(L, l1_ref, l2_ref, base)
    aprime_tgt = cp.direct_red_traced_mat(L, l1_tgt, l2_tgt, base)

    logdet_ref = _log_det_cholesky(aprime_ref) - math.log(L / 2.0)
    logdet_tgt = _log_det_cholesky(aprime_tgt) - math.log(L / 2.0)
    log_ratio_pf = 0.5 * (logdet_tgt - logdet_ref)
    return float(log_ratio_pf), float(math.exp(log_ratio_pf))


def _pair_result(
    L: int,
    reference: tuple[int, int],
    target: tuple[int, int],
    analytic_kind: str,
    analytic_cutoff: int,
) -> PairResult:
    ref = _boundary_data(L, reference[0], reference[1])
    tgt = _boundary_data(L, target[0], target[1])

    log_det_ratio_pf, det_ratio_pf = _det_ratio_piece(L, ref.l1, ref.l2, tgt.l1, tgt.l2)

    tau_ref = complex(ref.tau_reduced_real, ref.tau_reduced_imag)
    tau_tgt = complex(tgt.tau_reduced_real, tgt.tau_reduced_imag)
    if analytic_kind == "boson":
        z_ref = float(cp.z_boson(tau_ref))
        z_tgt = float(cp.z_boson(tau_tgt))
    elif analytic_kind == "compact":
        z_ref = float(cp.z_compact(1.0, tau_ref, N=analytic_cutoff))
        z_tgt = float(cp.z_compact(1.0, tau_tgt, N=analytic_cutoff))
    else:
        raise ValueError(f"Unsupported analytic_kind {analytic_kind!r}")
    log_z_ratio = float(math.log(z_ref / z_tgt))
    z_ratio = float(z_ref / z_tgt)

    log_residual = float(log_z_ratio - log_det_ratio_pf)
    residual_ratio = float(math.exp(log_residual))

    delta_s_vertex = float((tgt.log_nu_fit - ref.log_nu_fit) / 12.0)
    delta_log_p1 = float(math.log(tgt.period1_abs / ref.period1_abs) / 12.0)
    delta_s_full = float(delta_s_vertex + delta_log_p1)
    exp_vertex_ratio = float(math.exp(delta_s_vertex))
    exp_full_ratio = float(math.exp(delta_s_full))

    return PairResult(
        L=L,
        reference=_format_modulus(ref.l1, ref.l2, ref.l3),
        target=_format_modulus(tgt.l1, tgt.l2, tgt.l3),
        log_det_ratio_pf=log_det_ratio_pf,
        det_ratio_pf=det_ratio_pf,
        log_z_ratio=log_z_ratio,
        z_ratio=z_ratio,
        log_residual=log_residual,
        residual_ratio=residual_ratio,
        delta_S_vertex=delta_s_vertex,
        delta_log_p1=delta_log_p1,
        delta_S_full=delta_s_full,
        exp_vertex_ratio=exp_vertex_ratio,
        exp_full_ratio=exp_full_ratio,
        log_residual_minus_delta_S_vertex=float(log_residual - delta_s_vertex),
        log_residual_minus_delta_S_full=float(log_residual - delta_s_full),
        reference_tau_real=ref.tau_reduced_real,
        reference_tau_imag=ref.tau_reduced_imag,
        target_tau_real=tgt.tau_reduced_real,
        target_tau_imag=tgt.tau_reduced_imag,
        reference_period1_abs=ref.period1_abs,
        target_period1_abs=tgt.period1_abs,
        analytic_kind=analytic_kind,
    )


def _parse_pair(text: str) -> tuple[int, int]:
    parts = [piece.strip() for piece in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected l1,l2 pair, got {text!r}")
    l1, l2 = (int(parts[0]), int(parts[1]))
    if l1 <= 0 or l2 <= 0:
        raise ValueError("l1 and l2 must be positive")
    return (l1, l2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=int, default=DEFAULT_L, help="Total lattice size L.")
    parser.add_argument(
        "--reference",
        type=str,
        default=f"{DEFAULT_REFERENCE[0]},{DEFAULT_REFERENCE[1]}",
        help="Reference modulus as l1,l2. l3 is inferred from L/2-l1-l2.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        help="Target modulus as l1,l2. Can be passed multiple times.",
    )
    parser.add_argument(
        "--analytic-kind",
        choices=("boson", "compact"),
        default="boson",
        help="Analytic torus factor used in the comparison. WeylTest uses boson.",
    )
    parser.add_argument(
        "--analytic-cutoff",
        type=int,
        default=60,
        help="Theta cutoff used only when --analytic-kind=compact.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to save the full result as JSON.",
    )
    args = parser.parse_args()

    reference = _parse_pair(args.reference)
    targets = [_parse_pair(text) for text in args.target] if args.target else list(DEFAULT_TARGETS)

    results = [
        _pair_result(args.L, reference, target, args.analytic_kind, args.analytic_cutoff)
        for target in targets
    ]

    print(f"Fixed-L Weyl test at L={args.L}")
    print(f"analytic factor = Z_{args.analytic_kind}")
    print(f"reference modulus = {results[0].reference if results else _format_modulus(reference[0], reference[1], args.L // 2 - sum(reference))}")
    print(
        "| target | tau_target | log residual | delta S_vertex | delta S_full | ratio vs vertex | ratio vs full |"
    )
    print("|---|---|---:|---:|---:|---:|---:|")
    for row in results:
        tau_text = f"{row.target_tau_real:.10f} + {row.target_tau_imag:.10f} i"
        vertex_ratio = row.residual_ratio / row.exp_vertex_ratio
        full_ratio = row.residual_ratio / row.exp_full_ratio
        print(
            f"| {row.target} | {tau_text} | "
            f"{row.log_residual:.12f} | {row.delta_S_vertex:.12f} | "
            f"{row.delta_S_full:.12f} | {vertex_ratio:.12f} | {full_ratio:.12f} |"
        )

    print("\nDetailed pairwise data:")
    for row in results:
        print(
            f"{row.reference} -> {row.target}: "
            f"log_det_ratio_pf={row.log_det_ratio_pf:.12f}, "
            f"log_z_ratio={row.log_z_ratio:.12f}, "
            f"log_residual={row.log_residual:.12f}, "
            f"delta_S_vertex={row.delta_S_vertex:.12f}, "
            f"delta_log_p1={row.delta_log_p1:.12f}, "
            f"delta_S_full={row.delta_S_full:.12f}, "
            f"diff_vertex={row.log_residual_minus_delta_S_vertex:.12e}, "
            f"diff_full={row.log_residual_minus_delta_S_full:.12e}"
        )

    if args.json_out is not None:
        payload = {
            "L": args.L,
            "reference": reference,
            "targets": targets,
            "analytic_kind": args.analytic_kind,
            "analytic_cutoff": args.analytic_cutoff,
            "results": [asdict(row) for row in results],
        }
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved JSON to {args.json_out}")


if __name__ == "__main__":
    main()
