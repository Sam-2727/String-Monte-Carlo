#!/usr/bin/env python3
"""
Reusable continuum extrapolation utilities for the discrete-sigma numerics.

The goal is modest but important: give low-point sequences a structured
continuum estimate that is less brittle than a single c0 + c1 / N fit.

We use a small family of polynomial models in x = 1 / N:

    c0 + c1 x
    c0 + c2 x^2
    c0 + c1 x + c2 x^2

and combine:

- model selection on the largest available window,
- suffix-window stability for the preferred model,
- and model-spread diagnostics across the full data set.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    name: str
    powers: tuple[int, ...]
    min_points: int


@dataclass
class PolynomialFit:
    model_name: str
    powers: tuple[int, ...]
    start_index: int
    n_points: int
    coeffs: list[float]
    intercept: float
    intercept_stderr: float | None
    rmse: float
    max_abs: float
    rss: float
    aic: float
    bic: float


@dataclass
class ExtrapolationSummary:
    preferred_model: str
    estimate: float
    uncertainty: float
    combined_lower: float
    combined_upper: float
    preferred_full_fit: PolynomialFit
    preferred_window_lower: float
    preferred_window_upper: float
    preferred_window_std: float
    model_lower: float
    model_upper: float
    full_window_fits: dict[str, PolynomialFit]
    preferred_window_fits: list[PolynomialFit]


DEFAULT_MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("c0+c1/N", (0, 1), 3),
    ModelSpec("c0+c1/N+c2/N^2", (0, 1, 2), 6),
)


def _safe_information_criterion(rss: float, n_points: int, n_params: int) -> tuple[float, float]:
    rss_per_point = max(rss / max(n_points, 1), 1.0e-300)
    aic = n_points * math.log(rss_per_point) + 2.0 * n_params
    bic = n_points * math.log(rss_per_point) + math.log(max(n_points, 1)) * n_params
    return aic, bic


def fit_model(
    ns: list[int],
    values: list[float],
    model: ModelSpec,
    start_index: int = 0,
) -> PolynomialFit:
    window_ns = np.array(ns[start_index:], dtype=float)
    window_values = np.array(values[start_index:], dtype=float)
    if len(window_ns) < model.min_points:
        raise ValueError(
            f"need at least {model.min_points} points for model {model.name}"
        )

    x = 1.0 / window_ns
    design = np.column_stack([x ** power for power in model.powers])
    coeffs, _, _, _ = np.linalg.lstsq(design, window_values, rcond=None)
    fitted = design @ coeffs
    residuals = window_values - fitted
    rss = float(np.sum(residuals * residuals))
    rmse = float(math.sqrt(np.mean(residuals * residuals)))
    max_abs = float(np.max(np.abs(residuals)))
    aic, bic = _safe_information_criterion(rss, len(window_ns), len(model.powers))

    intercept_index = model.powers.index(0)
    intercept = float(coeffs[intercept_index])
    intercept_stderr = None
    dof = len(window_ns) - len(model.powers)
    if dof > 0:
        gram = design.T @ design
        try:
            inv_gram = np.linalg.inv(gram)
            sigma2 = rss / dof
            intercept_stderr = float(math.sqrt(max(sigma2 * inv_gram[intercept_index, intercept_index], 0.0)))
        except np.linalg.LinAlgError:
            intercept_stderr = None

    return PolynomialFit(
        model_name=model.name,
        powers=model.powers,
        start_index=start_index,
        n_points=len(window_ns),
        coeffs=[float(value) for value in coeffs],
        intercept=intercept,
        intercept_stderr=intercept_stderr,
        rmse=rmse,
        max_abs=max_abs,
        rss=rss,
        aic=aic,
        bic=bic,
    )


def summarize_extrapolation(
    ns: list[int],
    values: list[float],
    models: tuple[ModelSpec, ...] = DEFAULT_MODELS,
) -> ExtrapolationSummary:
    if len(ns) != len(values):
        raise ValueError("ns and values must have the same length")
    if len(ns) < 3:
        raise ValueError("need at least three data points for extrapolation")

    full_window_fits: dict[str, PolynomialFit] = {}
    for model in models:
        if len(ns) >= model.min_points:
            full_window_fits[model.name] = fit_model(ns, values, model, start_index=0)
    if not full_window_fits:
        raise ValueError("no extrapolation models are available for this data")

    preferred_full_fit = min(
        full_window_fits.values(),
        key=lambda fit: (fit.bic, fit.rmse, fit.model_name),
    )

    preferred_model = next(model for model in models if model.name == preferred_full_fit.model_name)
    preferred_window_fits = []
    max_start = len(ns) - preferred_model.min_points
    for start_index in range(max_start + 1):
        preferred_window_fits.append(
            fit_model(ns, values, preferred_model, start_index=start_index)
        )

    window_intercepts = np.array(
        [fit.intercept for fit in preferred_window_fits], dtype=float
    )
    model_intercepts = np.array(
        [fit.intercept for fit in full_window_fits.values()], dtype=float
    )
    combined = np.concatenate([window_intercepts, model_intercepts])

    combined_lower = float(np.min(combined))
    combined_upper = float(np.max(combined))
    estimate = preferred_full_fit.intercept
    uncertainty = float(
        max(abs(estimate - combined_lower), abs(combined_upper - estimate))
    )

    return ExtrapolationSummary(
        preferred_model=preferred_full_fit.model_name,
        estimate=estimate,
        uncertainty=uncertainty,
        combined_lower=combined_lower,
        combined_upper=combined_upper,
        preferred_full_fit=preferred_full_fit,
        preferred_window_lower=float(np.min(window_intercepts)),
        preferred_window_upper=float(np.max(window_intercepts)),
        preferred_window_std=float(np.std(window_intercepts)),
        model_lower=float(np.min(model_intercepts)),
        model_upper=float(np.max(model_intercepts)),
        full_window_fits=full_window_fits,
        preferred_window_fits=preferred_window_fits,
    )


def fit_to_dict(fit: PolynomialFit) -> dict[str, object]:
    return asdict(fit)


def summary_to_dict(summary: ExtrapolationSummary) -> dict[str, object]:
    result = asdict(summary)
    result["preferred_full_fit"] = fit_to_dict(summary.preferred_full_fit)
    result["full_window_fits"] = {
        name: fit_to_dict(fit) for name, fit in summary.full_window_fits.items()
    }
    result["preferred_window_fits"] = [
        fit_to_dict(fit) for fit in summary.preferred_window_fits
    ]
    return result
