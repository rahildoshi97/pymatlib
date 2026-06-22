# SPDX-FileCopyrightText: 2025 - 2026 Rahil Miten Doshi, Friedrich-Alexander-Universität Erlangen-Nürnberg
# SPDX-License-Identifier: BSD-3-Clause

"""Post-build plotting helpers that return a Matplotlib ``Axes``.

These complement the parse-time :class:`~materforge.visualization.plotters.PropertyVisualizer`
(which auto-saves a composite PNG during ``create_material(enable_plotting=True)``).
Here, each function works on an *already built* material, draws onto an ``Axes``
you pass (or a fresh one), and **returns it without saving or closing** - so a
Jupyter cell renders the figure inline and you stay in control of styling and
output.

- :func:`plot_property`     - the fitted curve, optionally over its source data.
- :func:`plot_residuals`    - per-point fit residuals with a zero reference line.
- :func:`compare_materials` - the same property from several materials on one axis.

The curve comes from :meth:`Material.compile`; the data points and the plot range
come from ``material.sample_data``. Pass ``symbol=`` to pick the dependency for a
material whose properties could be read against more than one symbol.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.axes import Axes

from materforge.core.materials import Material


def _require_material(material: Material) -> None:
    if not isinstance(material, Material):
        raise TypeError(f"Expected Material instance, got {type(material).__name__}")


def _dep_label(material: Material, prop_name: str, symbol: Optional[sp.Symbol]) -> str:
    """A readable x-axis label for the dependency, without compiling."""
    if symbol is not None:
        return str(symbol)
    expr = material.properties.get(prop_name)
    free: set = getattr(expr, "free_symbols", set())
    if len(free) == 1:
        return str(next(iter(free)))
    return "dependency"


def _curve(func, x: np.ndarray) -> np.ndarray:
    """Evaluate a compiled property callable over ``x``, broadcasting constants."""
    y = np.asarray(func(x), dtype=float)
    if y.shape != x.shape:
        y = np.broadcast_to(y, x.shape).astype(float, copy=True)
    return y


def _resolve_range(prop_name: str, dep_range, samples) -> tuple[float, float]:
    if dep_range is not None:
        lo, hi = float(dep_range[0]), float(dep_range[1])
    elif samples is not None and samples.x.size > 0:
        lo, hi = float(np.min(samples.x)), float(np.max(samples.x))
    else:
        raise ValueError(
            f"No source data for '{prop_name}' to infer a plot range; "
            f"pass dep_range=(lower, upper).")
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def plot_property(material: Material, prop_name: str, *,
                  ax: Optional[Axes] = None,
                  symbol: Optional[sp.Symbol] = None,
                  num: int = 300,
                  show_data: bool = True,
                  dep_range: Optional[Sequence[float]] = None,
                  label: Optional[str] = None,
                  **plot_kw) -> Axes:
    """Plot a property's fitted curve, optionally over its source data points.

    Args:
        material:  A material built by :func:`materforge.create_material`.
        prop_name: Property to plot.
        ax:        Axes to draw on; a new figure/axes is created when omitted.
        symbol:    Dependency symbol to evaluate against; inferred when omitted.
        num:       Number of points sampled along the curve.
        show_data: Overlay the retained source data points (when available).
        dep_range: ``(lower, upper)`` plot range; defaults to the source data
                   range. Required for a property with no retained data.
        label:     Legend label for the curve (defaults to ``"<prop> (fit)"``).
        **plot_kw: Forwarded to ``Axes.plot`` for the curve.
    Returns:
        The Axes drawn on (not saved or shown).
    Raises:
        TypeError:  ``material`` is not a Material.
        KeyError:   The property could not be compiled to a numeric callable.
        ValueError: No range is available and ``dep_range`` was not given.
    """
    _require_material(material)
    func = material.compile(symbol).function(prop_name)
    samples = material.sample_data.get(prop_name)
    lo, hi = _resolve_range(prop_name, dep_range, samples)
    x = np.linspace(lo, hi, num)
    y = _curve(func, x)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(x, y, label=label if label is not None else f"{prop_name} (fit)", **plot_kw)
    if show_data and samples is not None and samples.x.size > 0:
        ax.scatter(samples.x, samples.y, s=20, alpha=0.8, zorder=3,
                   color="black", label="data")
    ax.set_xlabel(_dep_label(material, prop_name, symbol))
    ax.set_ylabel(prop_name)
    ax.set_title(f"{material.name}: {prop_name}")
    ax.legend()
    return ax


def plot_residuals(material: Material, prop_name: str, *,
                   ax: Optional[Axes] = None,
                   symbol: Optional[sp.Symbol] = None,
                   **scatter_kw) -> Axes:
    """Plot per-point fit residuals (fit − data) with a zero reference line.

    Args:
        material:    A built material.
        prop_name:   A data-backed property.
        ax:          Axes to draw on; a new one is created when omitted.
        symbol:      Dependency symbol to evaluate against; inferred when omitted.
        **scatter_kw: Forwarded to ``Axes.scatter``.
    Returns:
        The Axes drawn on.
    Raises:
        TypeError: ``material`` is not a Material.
        KeyError:  The property has no retained source data.
    """
    from materforge.analysis import residuals  # local import avoids a cycle

    _require_material(material)
    x, res = residuals(material, prop_name, symbol=symbol)
    if ax is None:
        _, ax = plt.subplots()
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--", zorder=1)
    scatter_kw.setdefault("s", 20)
    scatter_kw.setdefault("alpha", 0.8)
    ax.scatter(x, res, zorder=3, **scatter_kw)
    ax.set_xlabel(_dep_label(material, prop_name, symbol))
    ax.set_ylabel(f"{prop_name} residual (fit − data)")
    ax.set_title(f"{material.name}: {prop_name} residuals")
    return ax


def compare_materials(materials: Iterable[Material], prop_name: str, *,
                      labels: Optional[Sequence[str]] = None,
                      ax: Optional[Axes] = None,
                      symbol: Optional[sp.Symbol] = None,
                      num: int = 300,
                      show_data: bool = False,
                      dep_range: Optional[Sequence[float]] = None) -> Axes:
    """Overlay one property from several materials on a single axis.

    Useful for comparing two alloys, or the same material built with different
    regression settings.

    Args:
        materials: Materials to overlay.
        prop_name: Property to compare (must exist in each material).
        labels:    Legend labels, one per material (defaults to each ``name``).
        ax:        Axes to draw on; a new one is created when omitted.
        symbol:    Dependency symbol to evaluate against; inferred when omitted.
        num:       Number of points sampled along each curve.
        show_data: Overlay each material's source data points.
        dep_range: Shared ``(lower, upper)`` plot range; per-material data range
                   is used when omitted.
    Returns:
        The Axes drawn on.
    Raises:
        ValueError: No materials given, or ``labels`` length mismatch.
    """
    mats: List[Material] = list(materials)
    if not mats:
        raise ValueError("compare_materials needs at least one material")
    if labels is not None and len(labels) != len(mats):
        raise ValueError(
            f"labels has {len(labels)} entries but there are {len(mats)} materials")
    if ax is None:
        _, ax = plt.subplots()
    for i, mat in enumerate(mats):
        plot_property(mat, prop_name, ax=ax, symbol=symbol, num=num,
                      show_data=show_data, dep_range=dep_range,
                      label=labels[i] if labels is not None else mat.name)
    ax.set_title(prop_name)
    ax.legend()
    return ax


__all__ = ["plot_property", "plot_residuals", "compare_materials"]
