#!/usr/bin/env python3
r"""
SO(8) triality-friendly gamma matrices for local Green-Schwarz diagnostics.

The construction follows the recursive Euclidean gamma-matrix convention stated
in `SYM/string notes.tex` around the `trialitycliff` equations:

  Gamma^1 = sigma_1, Gamma^2 = sigma_2,
  Gamma^mu_(d) = Gamma^mu_(d-2) \otimes (-sigma_3),
  Gamma^(d-1)_(d) = I \otimes sigma_1,
  Gamma^d_(d)     = I \otimes sigma_2.

For d = 8 we then:
  1. split into positive/negative chirality blocks,
  2. change basis so that the charge-conjugation matrix becomes the identity,
  3. expose the 8_s <-> 8_c gamma blocks used in the note.

The output matrices are generally complex. That is fine: a fixed complex
convention is enough for finite-dimensional Clifford-algebra checks, and the
Clifford / triality identities are verified numerically by the module itself.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


def _sigma_1() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


def _sigma_2() -> np.ndarray:
    return np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)


def _sigma_3() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


@lru_cache(maxsize=None)
def euclidean_gamma_matrices(dimension: int = 8) -> tuple[np.ndarray, ...]:
    if dimension % 2 != 0 or dimension < 2:
        raise ValueError("dimension must be an even integer >= 2")
    if dimension == 2:
        return (_sigma_1(), _sigma_2())

    previous = euclidean_gamma_matrices(dimension - 2)
    lifted = tuple(np.kron(gamma, -_sigma_3()) for gamma in previous)
    size = 2 ** (dimension // 2 - 1)
    identity = np.eye(size, dtype=complex)
    return lifted + (np.kron(identity, _sigma_1()), np.kron(identity, _sigma_2()))


def chirality_matrix(gammas: tuple[np.ndarray, ...]) -> np.ndarray:
    product = np.eye(gammas[0].shape[0], dtype=complex)
    for gamma in gammas:
        product = product @ gamma
    return (1.0j) ** (-len(gammas) // 2) * product


def charge_conjugation_matrix(gammas: tuple[np.ndarray, ...]) -> np.ndarray:
    product = np.eye(gammas[0].shape[0], dtype=complex)
    for index in range(0, len(gammas), 2):
        product = product @ gammas[index]
    return product


def _chirality_split_matrix(gammas: tuple[np.ndarray, ...]) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(chirality_matrix(gammas))
    positive = np.where(eigenvalues > 0)[0]
    negative = np.where(eigenvalues < 0)[0]
    if len(positive) != len(negative):
        raise ValueError("unexpected chirality multiplicities")
    return np.hstack([eigenvectors[:, positive], eigenvectors[:, negative]])


def _charge_identity_change(split_charge_conjugation: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(split_charge_conjugation)
    sqrt_eigenvalues = np.sqrt(eigenvalues + 0.0j)
    return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.conj().T


@dataclass(frozen=True)
class SO8GammaData:
    gamma_s_to_c: tuple[np.ndarray, ...]
    gamma_c_to_s: tuple[np.ndarray, ...]
    gamma_ss: dict[tuple[int, int], np.ndarray]
    gamma_cc: dict[tuple[int, int], np.ndarray]
    max_clifford_error: float
    max_same_side_triality_error: float
    max_cross_triality_error: float
    max_transpose_error: float


def _antisymmetrized_two_index(
    left: tuple[np.ndarray, ...], right: tuple[np.ndarray, ...]
) -> dict[tuple[int, int], np.ndarray]:
    tensors: dict[tuple[int, int], np.ndarray] = {}
    for i in range(8):
        for j in range(i + 1, 8):
            tensors[(i + 1, j + 1)] = 0.5 * (left[i] @ right[j] - left[j] @ right[i])
    return tensors


@lru_cache(maxsize=1)
def so8_gamma_data() -> SO8GammaData:
    gammas = euclidean_gamma_matrices(8)
    split = _chirality_split_matrix(gammas)
    split_gammas = tuple(split.conj().T @ gamma @ split for gamma in gammas)
    split_charge = split.conj().T @ charge_conjugation_matrix(gammas) @ split
    change = _charge_identity_change(split_charge)
    change_inv = np.linalg.inv(change)
    charge_identity_gammas = tuple(change @ gamma @ change_inv for gamma in split_gammas)

    gamma_s_to_c = tuple(gamma[:8, 8:] for gamma in charge_identity_gammas)
    gamma_c_to_s = tuple(gamma[8:, :8] for gamma in charge_identity_gammas)

    max_clifford_error = 0.0
    max_same_side_triality_error = 0.0
    max_cross_triality_error = 0.0
    max_transpose_error = 0.0
    eye8 = np.eye(8, dtype=complex)

    for i in range(8):
        max_transpose_error = max(
            max_transpose_error,
            float(np.max(np.abs(gamma_s_to_c[i] + gamma_c_to_s[i].T))),
        )
        for j in range(8):
            cliff_s = gamma_s_to_c[i] @ gamma_c_to_s[j] + gamma_s_to_c[j] @ gamma_c_to_s[i]
            cliff_c = gamma_c_to_s[i] @ gamma_s_to_c[j] + gamma_c_to_s[j] @ gamma_s_to_c[i]
            same_side = gamma_s_to_c[i].T @ gamma_s_to_c[j] + gamma_s_to_c[j].T @ gamma_s_to_c[i]
            max_clifford_error = max(
                max_clifford_error,
                float(np.max(np.abs(cliff_s - 2.0 * (i == j) * eye8))),
                float(np.max(np.abs(cliff_c - 2.0 * (i == j) * eye8))),
            )
            max_same_side_triality_error = max(
                max_same_side_triality_error,
                float(np.max(np.abs(same_side + 2.0 * (i == j) * eye8))),
            )

    for a in range(8):
        for b in range(8):
            for dot_c in range(8):
                for dot_d in range(8):
                    cross = sum(
                        gamma_s_to_c[i][a, dot_c] * gamma_c_to_s[i][dot_d, b]
                        + gamma_s_to_c[i][b, dot_c] * gamma_c_to_s[i][dot_d, a]
                        for i in range(8)
                    )
                    max_cross_triality_error = max(
                        max_cross_triality_error,
                        float(abs(cross - 2.0 * (a == b) * (dot_c == dot_d))),
                    )

    gamma_ss = _antisymmetrized_two_index(gamma_s_to_c, gamma_c_to_s)
    gamma_cc = _antisymmetrized_two_index(gamma_c_to_s, gamma_s_to_c)

    return SO8GammaData(
        gamma_s_to_c=gamma_s_to_c,
        gamma_c_to_s=gamma_c_to_s,
        gamma_ss=gamma_ss,
        gamma_cc=gamma_cc,
        max_clifford_error=max_clifford_error,
        max_same_side_triality_error=max_same_side_triality_error,
        max_cross_triality_error=max_cross_triality_error,
        max_transpose_error=max_transpose_error,
    )


def print_summary(show_sample: bool) -> None:
    data = so8_gamma_data()
    print("=" * 78)
    print("SO(8) GAMMA / TRIALITY CONVENTION")
    print("=" * 78)
    print(f"max Clifford error   : {data.max_clifford_error:.3e}")
    print(f"max same-side error  : {data.max_same_side_triality_error:.3e}")
    print(f"max cross triality   : {data.max_cross_triality_error:.3e}")
    print(f"max transpose error  : {data.max_transpose_error:.3e}")
    print()
    print(
        "Convention: rows of gamma_s_to_c[i] carry the chiral 8_s index a, "
        "columns carry the antichiral 8_c index dot a."
    )
    print(
        "The two-index chiral generators gamma_ss[(I,J)] are 1/2 [gamma^I gamma^J - gamma^J gamma^I] on 8_s."
    )
    print(
        "Here gamma_c_to_s[i] = -gamma_s_to_c[i]^T, so the same-side spinor "
        "contraction carries the expected minus sign."
    )
    if show_sample:
        print()
        sample = data.gamma_s_to_c[0]
        print("Sample gamma_s_to_c[1] matrix:")
        with np.printoptions(precision=3, suppress=True):
            print(sample)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-sample",
        action="store_true",
        help="also print one sample 8x8 gamma block",
    )
    args = parser.parse_args()
    print_summary(args.show_sample)


if __name__ == "__main__":
    main()
