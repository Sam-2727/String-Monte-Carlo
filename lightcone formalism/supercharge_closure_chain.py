#!/usr/bin/env python3
r"""
Minimal algebraic scaffold for the block-triangular cubic supercharge chain.

This script does not know the SO(8) component coefficients by itself. Instead,
it encodes the sparse projected closure system described in the note:

    2 (a0, a2, a4, a6, a8)^T = M(lambda) (b1, b3, b5, b7)^T

with

    [ lambda0   0         0         0      ]
    [ lambda21  lambda23  0         0      ]
M = [ 0         lambda43  lambda45  0      ]
    [ 0         0         lambda65  lambda67]
    [ 0         0         0         lambda8 ]

The point is not to solve the superstring prefactor outright, but to make the
recursive solve order explicit and reproducible once the projected free
supercharge matrix elements have been computed in a definite SO(8) basis.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ChainCoefficients:
    lambda0: float
    lambda21: float
    lambda23: float
    lambda43: float
    lambda45: float
    lambda65: float
    lambda67: float
    lambda8: float

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> "ChainCoefficients":
        vals = list(values)
        if len(vals) != 8:
            raise ValueError("expected 8 chain coefficients")
        return cls(*vals)


@dataclass(frozen=True)
class OddCoefficients:
    b1: float
    b3: float
    b5: float
    b7: float

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> "OddCoefficients":
        vals = list(values)
        if len(vals) != 4:
            raise ValueError("expected 4 odd coefficients")
        return cls(*vals)

    def as_array(self) -> np.ndarray:
        return np.array([self.b1, self.b3, self.b5, self.b7], dtype=float)


@dataclass(frozen=True)
class EvenCoefficients:
    a0: float
    a2: float
    a4: float
    a6: float
    a8: float

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> "EvenCoefficients":
        vals = list(values)
        if len(vals) != 5:
            raise ValueError("expected 5 even coefficients")
        return cls(*vals)

    def as_array(self) -> np.ndarray:
        return np.array([self.a0, self.a2, self.a4, self.a6, self.a8], dtype=float)


def parse_float_list(text: str, expected: int) -> list[float]:
    values = [float(piece) for piece in text.split(",") if piece.strip()]
    if len(values) != expected:
        raise argparse.ArgumentTypeError(
            f"expected {expected} comma-separated floats, got {len(values)} entries"
        )
    return values


def chain_matrix(chain: ChainCoefficients) -> np.ndarray:
    return np.array(
        [
            [chain.lambda0, 0.0, 0.0, 0.0],
            [chain.lambda21, chain.lambda23, 0.0, 0.0],
            [0.0, chain.lambda43, chain.lambda45, 0.0],
            [0.0, 0.0, chain.lambda65, chain.lambda67],
            [0.0, 0.0, 0.0, chain.lambda8],
        ],
        dtype=float,
    )


def even_from_odd(chain: ChainCoefficients, odd: OddCoefficients) -> EvenCoefficients:
    even = 0.5 * (chain_matrix(chain) @ odd.as_array())
    return EvenCoefficients(*even.tolist())


def odd_from_even_recursive(
    chain: ChainCoefficients, even: EvenCoefficients
) -> tuple[OddCoefficients, float]:
    if chain.lambda0 == 0.0:
        raise ValueError("cannot solve recursively when lambda0 = 0")
    if chain.lambda23 == 0.0:
        raise ValueError("cannot solve recursively when lambda23 = 0")
    if chain.lambda45 == 0.0:
        raise ValueError("cannot solve recursively when lambda45 = 0")
    if chain.lambda67 == 0.0:
        raise ValueError("cannot solve recursively when lambda67 = 0")
    if chain.lambda8 == 0.0:
        raise ValueError("cannot solve recursively when lambda8 = 0")

    b1 = 2.0 * even.a0 / chain.lambda0
    b3 = (2.0 * even.a2 - chain.lambda21 * b1) / chain.lambda23
    b5 = (2.0 * even.a4 - chain.lambda43 * b3) / chain.lambda45
    b7 = (2.0 * even.a6 - chain.lambda65 * b5) / chain.lambda67
    top_residual = 2.0 * even.a8 - chain.lambda8 * b7
    return OddCoefficients(b1, b3, b5, b7), top_residual


def print_template() -> None:
    print("=" * 78)
    print("PROJECTED CUBIC SUPERCHARGE CHAIN")
    print("=" * 78)
    print("Left-moving projected closure template:")
    print("  2 a0 = lambda0  b1")
    print("  2 a2 = lambda21 b1 + lambda23 b3")
    print("  2 a4 = lambda43 b3 + lambda45 b5")
    print("  2 a6 = lambda65 b5 + lambda67 b7")
    print("  2 a8 = lambda8  b7")
    print()
    print("Recursive solve order if the pivot coefficients are nonzero:")
    print("  1. b1 from the degree-0 block")
    print("  2. b3 from the degree-2 block")
    print("  3. b5 from the degree-4 block")
    print("  4. b7 from the degree-6 block")
    print("  5. check consistency in the degree-8 block")
    print()
    print(
        "The right-moving chain has the same form with tilded coefficients."
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chain",
        type=lambda text: parse_float_list(text, 8),
        default=None,
        help="comma-separated values for lambda0,lambda21,lambda23,lambda43,lambda45,lambda65,lambda67,lambda8",
    )
    parser.add_argument(
        "--odd",
        type=lambda text: parse_float_list(text, 4),
        default=None,
        help="comma-separated values for b1,b3,b5,b7",
    )
    parser.add_argument(
        "--even",
        type=lambda text: parse_float_list(text, 5),
        default=None,
        help="comma-separated values for a0,a2,a4,a6,a8",
    )
    parser.add_argument(
        "--print-template",
        action="store_true",
        help="print the generic sparse closure chain and solve order",
    )
    args = parser.parse_args()

    if args.print_template or (args.chain is None and args.odd is None and args.even is None):
        print_template()

    if args.chain is None:
        return

    chain = ChainCoefficients.from_iterable(args.chain)
    matrix = chain_matrix(chain)
    print("Numerical chain matrix:")
    print(matrix)
    print()

    if args.odd is not None:
        odd = OddCoefficients.from_iterable(args.odd)
        even = even_from_odd(chain, odd)
        print("Input odd coefficients:")
        print(odd)
        print()
        print("Derived even coefficients:")
        print(even)
        print()

    if args.even is not None:
        even = EvenCoefficients.from_iterable(args.even)
        odd, residual = odd_from_even_recursive(chain, even)
        print("Input even coefficients:")
        print(even)
        print()
        print("Recursive odd solution from the lower blocks:")
        print(odd)
        print()
        print(f"Top-block consistency residual 2 a8 - lambda8 b7 = {residual:.12g}")
        print()


if __name__ == "__main__":
    main()
