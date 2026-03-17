#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import ell_to_tau as et
import partition_function as pf


@dataclass(frozen=True)
class Case:
    L: int
    l1: int
    l2: int

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


@dataclass
class BenchResult:
    name: str
    ok: bool
    mean_s: float | None = None
    median_s: float | None = None
    min_s: float | None = None
    max_s: float | None = None
    stdev_s: float | None = None
    error: str | None = None


def parse_case(raw: str) -> Case:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Invalid case '{raw}'. Expected format: L,l1,l2"
        )
    try:
        L, l1, l2 = (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid case '{raw}'. Values must be integers."
        ) from exc
    if L <= 0 or L % 2 != 0:
        raise argparse.ArgumentTypeError("L must be a positive even integer.")
    if l1 < 0 or l2 < 0 or l1 + l2 > (L // 2):
        raise argparse.ArgumentTypeError("Need 0 <= l1, l2 and l1 + l2 <= L/2.")
    return Case(L=L, l1=l1, l2=l2)


def measure(
    fn: Callable[[], object],
    repeat: int,
    number: int,
    warmup: int,
    target_ms: float,
) -> tuple[int, list[float]]:
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
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(calls_per_sample):
            last = fn()
        elapsed = (time.perf_counter() - start) / calls_per_sample
        samples.append(elapsed)
    return calls_per_sample, samples


def bench(
    name: str,
    fn: Callable[[], object],
    repeat: int,
    number: int,
    warmup: int,
    target_ms: float,
) -> BenchResult:
    try:
        _, samples = measure(
            fn=fn,
            repeat=repeat,
            number=number,
            warmup=warmup,
            target_ms=target_ms,
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        return BenchResult(name=name, ok=False, error=f"{type(exc).__name__}: {exc}")
    return BenchResult(
        name=name,
        ok=True,
        mean_s=statistics.mean(samples),
        median_s=statistics.median(samples),
        min_s=min(samples),
        max_s=max(samples),
        stdev_s=statistics.pstdev(samples),
    )


def ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 1e3:.3f}"


def print_results(case: Case, results: Sequence[BenchResult]) -> None:
    print()
    print(f"Case {case.label}")
    print("function                         ok   median_ms   mean_ms   min_ms   max_ms   stdev_ms")
    print("-------------------------------------------------------------------------------------------")
    for r in results:
        if not r.ok:
            print(f"{r.name:32}  no   -          -        -        -        -")
            print(f"  error: {r.error}")
            continue
        print(
            f"{r.name:32}  yes  {ms(r.median_s):>9}  {ms(r.mean_s):>7}  "
            f"{ms(r.min_s):>7}  {ms(r.max_s):>7}  {ms(r.stdev_s):>8}"
        )

    by_name = {r.name: r for r in results if r.ok and r.median_s}
    if "make_cyl_eqn" in by_name and "make_cyl_eqn_improved" in by_name:
        base = by_name["make_cyl_eqn"].median_s or 0.0
        new = by_name["make_cyl_eqn_improved"].median_s or 0.0
        if new > 0:
            print(f"speedup make_cyl_eqn_improved vs make_cyl_eqn: {base / new:.2f}x")
    if "periods" in by_name and "periods_improved(total)" in by_name:
        base = by_name["periods"].median_s or 0.0
        new = by_name["periods_improved(total)"].median_s or 0.0
        if new > 0:
            print(f"speedup periods_improved(total) vs periods: {base / new:.2f}x")


def run_case(
    case: Case,
    repeat: int,
    number: int,
    warmup: int,
    target_ms: float,
    mode: str,
) -> list[BenchResult]:
    L, l1, l2 = case.L, case.l1, case.l2
    improved_make_results: list[BenchResult] = []
    period_results: list[BenchResult] = []
    if mode != "det":
        f_improved = et.make_cyl_eqn_improved(L=L, l1=l1, l2=l2)
        improved_make_results = [
            bench(
                "make_cyl_eqn_improved",
                lambda: et.make_cyl_eqn_improved(L=L, l1=l1, l2=l2),
                repeat,
                number,
                warmup,
                target_ms,
            ),
        ]
        period_results = [
            bench("periods", lambda: et.periods(L=L, l1=l1, l2=l2), repeat, number, warmup, target_ms),
            bench(
                "periods_improved(total)",
                lambda: et.periods_improved(L=L, l1=l1, l2=l2),
                repeat,
                number,
                warmup,
                target_ms,
            ),
            bench(
                "periods_improved(f prebuilt)",
                lambda: et.periods_improved(L=L, l1=l1, l2=l2, f=f_improved),
                repeat,
                number,
                warmup,
                target_ms,
            ),
        ]

    det_results = [
        bench("bdet_log", lambda: pf.bdet_log(L=L, l1=l1, l2=l2), repeat, number, warmup, target_ms),
        bench("prime_det_log", lambda: pf.prime_det_log(L=L, l1=l1, l2=l2), repeat, number, warmup, target_ms),
    ]

    if mode == "improved":
        return improved_make_results + period_results[1:]
    if mode == "det":
        return det_results
    if mode == "period_det":
        return period_results + det_results
    return [
        bench("make_cyl_eqn", lambda: et.make_cyl_eqn(L=L, l1=l1, l2=l2), repeat, number, warmup, target_ms),
        *improved_make_results,
        *period_results,
        *det_results,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark covariant-formalism kernels for f/periods and determinant paths."
        )
    )
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        dest="cases",
        help="Case in format L,l1,l2. Repeat flag for multiple cases.",
    )
    parser.add_argument("--repeat", type=int, default=5, help="Number of timing samples per function.")
    parser.add_argument("--number", type=int, default=1, help="Minimum calls per timing sample.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup samples before timing.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--full",
        action="store_true",
        help="Benchmark the full suite, including make_cyl_eqn and make_cyl_eqn_improved.",
    )
    mode.add_argument(
        "--improved-only",
        action="store_true",
        help="Benchmark only make_cyl_eqn_improved and periods_improved paths.",
    )
    mode.add_argument(
        "--det-only",
        action="store_true",
        help="Benchmark only bdet_log and prime_det_log.",
    )
    parser.add_argument(
        "--target-ms",
        type=float,
        default=50.0,
        help="Minimum wall-clock time per timing sample after call-count calibration.",
    )
    args = parser.parse_args()

    if args.det_only:
        bench_mode = "det"
    elif args.improved_only:
        bench_mode = "improved"
    elif args.full:
        bench_mode = "full"
    else:
        bench_mode = "period_det"
    if args.cases:
        cases = args.cases
    elif bench_mode == "improved":
        cases = IMPROVED_ONLY_CASES
    elif bench_mode == "det":
        cases = DET_ONLY_CASES
    elif bench_mode == "period_det":
        cases = PERIOD_DET_CASES
    else:
        cases = DEFAULT_CASES

    if args.repeat <= 0 or args.number <= 0 or args.warmup < 0 or args.target_ms < 0:
        raise SystemExit("repeat and number must be positive; warmup and target-ms must be >= 0.")

    for case in cases:
        results = run_case(
            case=case,
            repeat=args.repeat,
            number=args.number,
            warmup=args.warmup,
            target_ms=args.target_ms,
            mode=bench_mode,
        )
        print_results(case=case, results=results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
