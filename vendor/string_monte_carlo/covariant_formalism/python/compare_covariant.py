#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import ell_to_tau as et
import partition_function as pf


REPO_ROOT = Path(__file__).resolve().parents[4]
LEAN_EXE = REPO_ROOT / ".lake" / "build" / "bin" / "covariant-bench"
LEAN_IMPROVED_EXE = REPO_ROOT / ".lake" / "build" / "bin" / "covariant-improved-bench"
LEAN_DET_EXE = REPO_ROOT / ".lake" / "build" / "bin" / "covariant-det-bench"
LEAN_PERIOD_DET_EXE = REPO_ROOT / ".lake" / "build" / "bin" / "covariant-period-det-bench"


@dataclass(frozen=True)
class Case:
    L: int
    l1: int
    l2: int

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.L, self.l1, self.l2)

    @property
    def label(self) -> str:
        return f"(L={self.L}, l1={self.l1}, l2={self.l2})"


DEFAULT_CASES = [
    Case(12, 2, 2),
    Case(16, 4, 4),
    Case(20, 4, 4),
    Case(40, 10, 8),
]

IMPROVED_ONLY_CASES = [
    Case(160, 40, 30),
    Case(320, 80, 60),
    Case(640, 160, 120),
]

DET_ONLY_CASES = [
    Case(160, 40, 30),
    Case(320, 80, 60),
    Case(640, 160, 120),
    Case(960, 240, 180),
]

PERIOD_DET_CASES = [
    Case(160, 40, 30),
    Case(320, 80, 60),
    Case(640, 160, 120),
]

TOLERANCES = {
    "make_cyl_eqn": 1e-6,
    "make_cyl_eqn_improved": 1e-6,
    "periods": 1e-6,
    "periods_improved(total)": 1e-6,
    "periods_improved(f prebuilt)": 1e-6,
    "bdet_log": 1e-6,
    "prime_det_log": 1e-6,
}


def parse_case(raw: str) -> Case:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Invalid case '{raw}'. Expected format L,l1,l2.")
    try:
        L, l1, l2 = map(int, parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid case '{raw}'. Values must be integers.") from exc
    if L <= 0 or L % 2 != 0 or l1 < 0 or l2 < 0 or l1 + l2 > L // 2:
        raise argparse.ArgumentTypeError(f"Invalid case '{raw}'.")
    return Case(L, l1, l2)


def progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def measure(
    fn: Callable[[], Any],
    repeats: int,
    number: int,
    warmup: int,
    target_ms: float,
) -> dict[str, float]:
    calls_per_sample = max(1, number)
    target_s = max(0.0, target_ms) / 1000.0
    while True:
        start = time.perf_counter()
        last = None
        for _ in range(calls_per_sample):
            last = fn()
        elapsed_s = time.perf_counter() - start
        if elapsed_s >= target_s or calls_per_sample >= 1_048_576:
            break
        calls_per_sample *= 2
    for _ in range(warmup):
        for _ in range(calls_per_sample):
            last = fn()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(calls_per_sample):
            last = fn()
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / calls_per_sample
        samples.append(elapsed_ms)
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "stdev_ms": statistics.pstdev(samples),
    }


def complex_json(z: complex) -> dict[str, float]:
    return {"re": float(z.real), "im": float(z.imag)}


def function_names(mode: str) -> list[str]:
    if mode == "period_det":
        return [
            "periods",
            "periods_improved(total)",
            "periods_improved(f prebuilt)",
            "bdet_log",
            "prime_det_log",
        ]
    if mode == "improved":
        return [
            "make_cyl_eqn_improved",
            "periods_improved(total)",
            "periods_improved(f prebuilt)",
        ]
    if mode == "det":
        return [
            "bdet_log",
            "prime_det_log",
        ]
    return [
        "make_cyl_eqn",
        "make_cyl_eqn_improved",
        "periods",
        "periods_improved(total)",
        "periods_improved(f prebuilt)",
        "bdet_log",
        "prime_det_log",
    ]


def python_case_report(
    case: Case,
    repeats: int,
    number: int,
    warmup: int,
    target_ms: float,
    mode: str = "full",
) -> dict[str, Any]:
    L, l1, l2 = case.L, case.l1, case.l2
    progress(f"[python] preparing {case.label}")
    improved = None if mode == "det" else et.make_cyl_eqn_improved(L=L, l1=l1, l2=l2)

    def value_make_cyl() -> dict[str, Any]:
        f = et.make_cyl_eqn(L=L, l1=l1, l2=l2)
        return {"coeffs": [complex_json(complex(c)) for c in f.coeffs]}

    def value_make_cyl_improved() -> dict[str, Any]:
        f = et.make_cyl_eqn_improved(L=L, l1=l1, l2=l2)
        return {"coeffs": [complex_json(complex(c)) for c in f.coeffs]}

    def value_periods() -> dict[str, Any]:
        p1, p2, p3 = et.periods(L=L, l1=l1, l2=l2)
        return {"p1": complex_json(p1), "p2": complex_json(p2), "p3": complex_json(p3)}

    def value_periods_improved_total() -> dict[str, Any]:
        p1, p2, p3 = et.periods_improved(L=L, l1=l1, l2=l2)
        return {"p1": complex_json(p1), "p2": complex_json(p2), "p3": complex_json(p3)}

    def value_periods_improved_prebuilt() -> dict[str, Any]:
        p1, p2, p3 = et.periods_improved(L=L, l1=l1, l2=l2, f=improved)
        return {"p1": complex_json(p1), "p2": complex_json(p2), "p3": complex_json(p3)}

    def value_bdet() -> float:
        return float(pf.bdet_log(L=L, l1=l1, l2=l2))

    def value_prime_det() -> float:
        return float(pf.prime_det_log(L=L, l1=l1, l2=l2))

    funcs_by_name: dict[str, Callable[[], Any]] = {
        "make_cyl_eqn": value_make_cyl,
        "make_cyl_eqn_improved": value_make_cyl_improved,
        "periods": value_periods,
        "periods_improved(total)": value_periods_improved_total,
        "periods_improved(f prebuilt)": value_periods_improved_prebuilt,
        "bdet_log": value_bdet,
        "prime_det_log": value_prime_det,
    }

    out: dict[str, Any] = {"case": {"L": L, "l1": l1, "l2": l2}}
    for name in function_names(mode):
        fn = funcs_by_name[name]
        progress(f"[python] timing {case.label} {name}")
        timing = measure(fn, repeats=repeats, number=number, warmup=warmup, target_ms=target_ms)
        progress(f"[python] value  {case.label} {name}")
        value = fn()
        out[name] = {
            "name": name,
            "ok": True,
            "timing": timing,
            "value": value,
            "error": None,
        }
    return out


def ensure_lean_exe(mode: str = "full") -> None:
    target = {
        "full": "covariant-bench",
        "period_det": "covariant-period-det-bench",
        "improved": "covariant-improved-bench",
        "det": "covariant-det-bench",
    }[mode]
    progress(f"[lean] building {target}")
    subprocess.run(
        ["lake", "build", target],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    progress("[lean] build complete")


def run_lean(
    cases: list[Case],
    repeats: int,
    number: int,
    warmup: int,
    target_ms: float,
    mode: str = "full",
) -> dict[tuple[int, int, int], dict[str, Any]]:
    ensure_lean_exe(mode=mode)
    exe = {
        "full": LEAN_EXE,
        "period_det": LEAN_PERIOD_DET_EXE,
        "improved": LEAN_IMPROVED_EXE,
        "det": LEAN_DET_EXE,
    }[mode]
    cmd = [
        str(exe),
        "--json",
        "--repeat",
        str(repeats),
        "--number",
        str(number),
        "--warmup",
        str(warmup),
        "--target-ms",
        str(int(target_ms)),
    ]
    for case in cases:
        cmd += ["--case", f"{case.L},{case.l1},{case.l2}"]
    progress(f"[lean] running {len(cases)} case(s)")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    progress("[lean] run complete")
    payload = json.loads(proc.stdout)
    return {
        (item["case"]["L"], item["case"]["l1"], item["case"]["l2"]): item
        for item in payload["cases"]
    }


def abs_complex_delta(a: dict[str, float], b: dict[str, float]) -> float:
    return abs(complex(a["re"], a["im"]) - complex(b["re"], b["im"]))


def max_delta(a: Any, b: Any) -> float:
    if a is None or b is None:
        return math.inf
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b))
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) == {"re", "im"} and set(b) == {"re", "im"}:
            return abs_complex_delta(a, b)
        return max((max_delta(a[k], b[k]) for k in a.keys() & b.keys()), default=0.0)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return math.inf
        return max((max_delta(x, y) for x, y in zip(a, b)), default=0.0)
    return 0.0 if a == b else math.inf


def format_ms(value: float) -> str:
    return f"{value:.6f}"


def compare_case(
    case: Case,
    lean: dict[str, Any],
    py: dict[str, Any],
    mode: str,
) -> list[tuple[str, float, float]]:
    failures: list[tuple[str, float, float]] = []
    print()
    print(f"Case {case.label}")
    print("function                           py_ms    lean_ms   py/lean   max_abs_delta")
    print("--------------------------------------------------------------------------------")
    for name in function_names(mode):
        py_entry = py[name]
        lean_entry = lean[name]
        py_ms = py_entry["timing"]["median_ms"]
        lean_ms = lean_entry["timing"]["median_ms"] if lean_entry["timing"] else math.inf
        ratio = py_ms / lean_ms if lean_ms and math.isfinite(lean_ms) else 0.0
        delta = max_delta(py_entry["value"], lean_entry["value"])
        print(
            f"{name:32}  {format_ms(py_ms):>8}  {format_ms(lean_ms):>8}  {ratio:>7.2f}  {delta:.6g}"
        )
        tol = TOLERANCES[name]
        if not math.isfinite(delta) or delta > tol:
            failures.append((name, delta, tol))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Lean and Python covariant benchmark kernels.")
    parser.add_argument("--case", action="append", type=parse_case, dest="cases")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--number", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--target-ms", type=float, default=50.0)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--full", action="store_true")
    mode_group.add_argument("--improved-only", action="store_true")
    mode_group.add_argument("--det-only", action="store_true")
    args = parser.parse_args()

    if args.repeat <= 0 or args.number <= 0 or args.warmup < 0 or args.target_ms < 0:
        raise SystemExit("repeat and number must be positive; warmup and target-ms must be >= 0.")

    if args.det_only:
        mode = "det"
    elif args.improved_only:
        mode = "improved"
    elif args.full:
        mode = "full"
    else:
        mode = "period_det"
    if args.cases:
        cases = args.cases
    elif mode == "improved":
        cases = IMPROVED_ONLY_CASES
    elif mode == "det":
        cases = DET_ONLY_CASES
    elif mode == "period_det":
        cases = PERIOD_DET_CASES
    else:
        cases = DEFAULT_CASES
    progress(
        f"[compare] cases={len(cases)} repeat={args.repeat} number={args.number} "
        f"warmup={args.warmup} target_ms={args.target_ms:g} mode={mode}"
    )
    lean_reports = run_lean(
        cases,
        repeats=args.repeat,
        number=args.number,
        warmup=args.warmup,
        target_ms=args.target_ms,
        mode=mode,
    )
    py_reports = {
        case.key: python_case_report(
            case,
            args.repeat,
            args.number,
            args.warmup,
            args.target_ms,
            mode=mode,
        )
        for case in cases
    }

    failures: list[tuple[Case, str, float, float]] = []
    for case in cases:
        case_failures = compare_case(
            case,
            lean_reports[case.key],
            py_reports[case.key],
            mode=mode,
        )
        failures.extend((case, name, delta, tol) for name, delta, tol in case_failures)

    if failures:
        print()
        print("Tolerance failures")
        print("------------------")
        for case, name, delta, tol in failures:
          print(f"{case.label} {name}: delta={delta:.6g} tol={tol:.6g}")
        return 1

    print()
    print("All outputs matched within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
