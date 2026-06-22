# SPDX-FileCopyrightText: 2025 - 2026 Rahil Miten Doshi, Friedrich-Alexander-Universität Erlangen-Nürnberg
# SPDX-License-Identifier: BSD-3-Clause

"""Fit-quality metrics for a built :class:`~materforge.core.materials.Material`.

A regression- or interpolation-backed property is an *approximation* of the data
points it was built from. These helpers quantify how well that approximation
reproduces its source data:

- :func:`r_squared`, :func:`rmse`, :func:`mae`, :func:`max_abs_error` are plain
  array metrics (``y_true`` vs ``y_pred``) - usable on their own.
- :func:`fit_quality`, :func:`residuals`, :func:`fit_report` work directly on a
  material, using the source points it kept in ``material.sample_data`` and the
  compiled property callable from :meth:`Material.compile`.

**What "fit quality" means here:** the stored property expression evaluated at
the source sample points, compared with the sample values. For a property with
``regression: {simplify: pre, ...}`` this is the regression error, the number you
usually care about. For plain interpolation (or ``simplify: post``) the stored
curve passes through every point, so the error is ~0 by construction - correct,
if not very informative.

Everything routes through :class:`~materforge.core.evaluator.MaterialEvaluator`,
which currently supports a single dependency symbol; pass ``symbol=`` to pick one
explicitly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import sympy as sp

if TYPE_CHECKING:
    from materforge.core.evaluator import MaterialEvaluator
    from materforge.core.materials import Material

logger = logging.getLogger(__name__)


# ====================================================================
# PURE ARRAY METRICS
# ====================================================================

def _as_pair(y_true, y_pred) -> Tuple[np.ndarray, np.ndarray]:
    """Coerce two array-likes to equal-length 1-D float arrays."""
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length, got {yt.size} and {yp.size}")
    if yt.size == 0:
        raise ValueError("need at least one point to compute a metric")
    return yt, yp


def r_squared(y_true, y_pred) -> float:
    """Coefficient of determination, ``1 - SS_res / SS_tot``.

    ``SS_tot`` is taken about the mean of ``y_true``. When the observations are
    constant (``SS_tot == 0``) there is no variance to explain: returns ``1.0``
    if they are reproduced exactly (``SS_res == 0``), otherwise ``0.0``.
    """
    yt, yp = _as_pair(y_true, y_pred)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def rmse(y_true, y_pred) -> float:
    """Root-mean-square error between observations and predictions."""
    yt, yp = _as_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error between observations and predictions."""
    yt, yp = _as_pair(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def max_abs_error(y_true, y_pred) -> float:
    """Largest single absolute error between observations and predictions."""
    yt, yp = _as_pair(y_true, y_pred)
    return float(np.max(np.abs(yt - yp)))


@dataclass(frozen=True)
class FitQuality:
    """Goodness-of-fit summary for one property against its source data.

    Attributes:
        property:      Property name.
        r_squared:     Coefficient of determination (``1.0`` is a perfect fit).
        rmse:          Root-mean-square error, in the property's units.
        mae:           Mean absolute error.
        max_abs_error: Largest single absolute error.
        n_points:      Number of source data points compared.
    """
    property: str
    r_squared: float
    rmse: float
    mae: float
    max_abs_error: float
    n_points: int

    def __str__(self) -> str:
        return (f"{self.property}: R²={self.r_squared:.6f}  RMSE={self.rmse:.4g}  "
                f"MAE={self.mae:.4g}  max|err|={self.max_abs_error:.4g}  (n={self.n_points})")


# ====================================================================
# MATERIAL-AWARE HELPERS
# ====================================================================

def _require_material(material: "Material") -> None:
    from materforge.core.materials import Material  # local import avoids a cycle
    if not isinstance(material, Material):
        raise TypeError(f"Expected Material instance, got {type(material).__name__}")


def _predict_at_samples(material: "Material", prop_name: str,
                        evaluator: "MaterialEvaluator") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x, y_observed, y_predicted)`` at a property's source points.

    Raises:
        KeyError: The property has no retained source data, or could not be
            compiled to a numeric callable.
    """
    samples = material.sample_data.get(prop_name)
    if samples is None:
        available = sorted(material.sample_data)
        raise KeyError(
            f"No source data for property '{prop_name}'. Fit quality is only "
            f"available for data-backed properties; this material has data for: "
            f"{available or '(none)'}.")
    func = evaluator.function(prop_name)  # KeyError if the property did not compile
    y_pred = np.asarray(func(samples.x), dtype=float).reshape(-1)
    if y_pred.shape != samples.x.shape:
        y_pred = np.broadcast_to(y_pred, samples.x.shape).astype(float, copy=True)
    return samples.x, samples.y, y_pred


def fit_quality(material: "Material", prop_name: str, *,
                symbol: Optional[sp.Symbol] = None) -> FitQuality:
    """Goodness-of-fit of a property against the data it was built from.

    Args:
        material:  A material built by :func:`materforge.create_material`.
        prop_name: Name of a data-backed property (file-import, tabular, or
                   computed, with or without regression).
        symbol:    Dependency symbol to evaluate against; inferred when omitted.
    Returns:
        A :class:`FitQuality` summary.
    Raises:
        TypeError: ``material`` is not a Material.
        KeyError:  The property has no retained source data.
    Example:
        >>> fq = fit_quality(mat, 'heat_capacity')
        >>> print(fq.rmse, fq.r_squared)
    """
    _require_material(material)
    evaluator = material.compile(symbol)
    _, y_obs, y_pred = _predict_at_samples(material, prop_name, evaluator)
    return FitQuality(
        property=prop_name,
        r_squared=r_squared(y_obs, y_pred),
        rmse=rmse(y_obs, y_pred),
        mae=mae(y_obs, y_pred),
        max_abs_error=max_abs_error(y_obs, y_pred),
        n_points=int(y_obs.size),
    )


def residuals(material: "Material", prop_name: str, *,
              symbol: Optional[sp.Symbol] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Per-point fit residuals of a property.

    Args:
        material:  A built material.
        prop_name: Name of a data-backed property.
        symbol:    Dependency symbol to evaluate against; inferred when omitted.
    Returns:
        ``(x, predicted - observed)`` at the property's source data points.
    Raises:
        TypeError: ``material`` is not a Material.
        KeyError:  The property has no retained source data.
    """
    _require_material(material)
    evaluator = material.compile(symbol)
    x, y_obs, y_pred = _predict_at_samples(material, prop_name, evaluator)
    return x, y_pred - y_obs


def fit_report(material: "Material", *,
               symbol: Optional[sp.Symbol] = None) -> Dict[str, FitQuality]:
    """Fit quality for every data-backed property of a material.

    Compiles the material once and evaluates each property with retained source
    data. Properties that cannot be assessed are skipped with a warning.

    Args:
        material: A built material.
        symbol:   Dependency symbol to evaluate against; inferred when omitted.
    Returns:
        Mapping of property name to :class:`FitQuality`.
    Raises:
        TypeError: ``material`` is not a Material.
    """
    _require_material(material)
    evaluator = material.compile(symbol)
    report: Dict[str, FitQuality] = {}
    for name in sorted(material.sample_data):
        try:
            _, y_obs, y_pred = _predict_at_samples(material, name, evaluator)
        except KeyError as error:
            logger.warning("Skipping fit quality for '%s': %s", name, error)
            continue
        report[name] = FitQuality(
            property=name,
            r_squared=r_squared(y_obs, y_pred),
            rmse=rmse(y_obs, y_pred),
            mae=mae(y_obs, y_pred),
            max_abs_error=max_abs_error(y_obs, y_pred),
            n_points=int(y_obs.size),
        )
    return report


__all__ = [
    "r_squared",
    "rmse",
    "mae",
    "max_abs_error",
    "FitQuality",
    "fit_quality",
    "residuals",
    "fit_report",
]
