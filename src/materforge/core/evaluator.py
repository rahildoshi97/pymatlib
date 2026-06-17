# SPDX-FileCopyrightText: 2025 - 2026 Rahil Miten Doshi, Friedrich-Alexander-Universität Erlangen-Nürnberg
# SPDX-License-Identifier: BSD-3-Clause

"""Fast numeric evaluation of a :class:`~materforge.core.materials.Material`.

:class:`Material.evaluate` substitutes the dependency symbolically on every call,
which is fine for a one-off scalar but slow when sweeping many values. A
:class:`MaterialEvaluator` compiles each property to a NumPy callable once
(via :func:`sympy.lambdify`) and reuses it, so evaluating over an array of
dependency values is a single vectorised call per property.

Build one with :meth:`Material.compile`::

    ev = material.compile()
    ev(500.0)                       # {'density': 2634.5, ...}
    ev(np.linspace(300, 900, 200))  # {'density': array([...]), ...}
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Union

import numpy as np
import sympy as sp

if TYPE_CHECKING:
    from materforge.core.materials import Material

logger = logging.getLogger(__name__)

# A scalar in, or anything array-like (list, tuple, ndarray).
Number = Union[int, float]
ArrayLike = Union[Number, "np.ndarray", list, tuple]


class MaterialEvaluator:
    """Compiled, reusable numeric evaluator for a material's properties.

    Each property is lambdified once against the material's dependency symbol and
    cached. Calling the evaluator with a scalar returns a ``dict`` of floats;
    calling it with an array returns a ``dict`` of NumPy arrays shaped like the
    input. Constant properties are broadcast to match.

    The evaluator is a snapshot taken at construction time: if the material's
    properties change afterwards, build a fresh evaluator with
    :meth:`Material.compile`.

    Args:
        material: A fully processed Material instance.
        symbol:   Dependency symbol to evaluate against. If omitted, it is
                  inferred from the free symbols of the material's properties
                  (the common case, where every property is a function of a
                  single dependency). Pass it explicitly to disambiguate or to
                  pin the variable for an all-constant material.
    Raises:
        TypeError:  symbol is given but is not a sympy Symbol.
        ValueError: The properties depend on more than one symbol (multivariate
                    materials are not yet supported), or on a symbol other than
                    the one supplied.
    """

    def __init__(self, material: "Material", symbol: Optional[sp.Symbol] = None) -> None:
        self._material_name = material.name
        self._symbol = self._resolve_symbol(material.properties, symbol)
        self._functions: Dict[str, Callable] = {}
        self._build(material.properties)

    # --- Construction helpers ---
    @staticmethod
    def _free_symbols(expr: Any) -> Set[sp.Symbol]:
        """Free symbols of a property expression, or empty for a plain number."""
        if isinstance(expr, sp.Basic):
            return expr.free_symbols
        return set()

    def _resolve_symbol(self, properties: Dict[str, Any],
                        symbol: Optional[sp.Symbol]) -> Optional[sp.Symbol]:
        all_symbols: Set[sp.Symbol] = set()
        for expr in properties.values():
            all_symbols |= self._free_symbols(expr)

        if symbol is not None:
            if not isinstance(symbol, sp.Symbol):
                raise TypeError(
                    f"symbol must be a sympy Symbol, got {type(symbol).__name__}")
            extra = all_symbols - {symbol}
            if extra:
                raise ValueError(
                    f"Properties depend on {sorted(map(str, extra))}, which is not "
                    f"the supplied symbol '{symbol}'")
            return symbol

        if len(all_symbols) > 1:
            raise ValueError(
                f"Material '{self._material_name}' depends on multiple symbols "
                f"{sorted(map(str, all_symbols))}; pass symbol= to pick one. "
                f"Multivariate evaluation is not yet supported.")
        return next(iter(all_symbols)) if all_symbols else None

    def _build(self, properties: Dict[str, Any]) -> None:
        for name, expr in properties.items():
            if expr is None:
                logger.warning("Skipping property '%s' with no expression", name)
                continue
            free = self._free_symbols(expr)
            if not free:
                try:
                    self._functions[name] = _Constant(float(expr))
                    continue
                except (TypeError, ValueError):
                    logger.warning(
                        "Skipping constant property '%s' that is not a real number", name)
                    continue
            if self._symbol is None:
                # Cannot happen given _resolve_symbol, but guards against misuse.
                logger.warning("Skipping property '%s'; no dependency symbol", name)
                continue
            try:
                self._functions[name] = sp.lambdify(self._symbol, expr, "numpy")
            except (TypeError, ValueError, NameError) as exc:
                logger.warning("Could not compile property '%s': %s", name, exc)

    # --- Public API ---
    @property
    def symbol(self) -> Optional[sp.Symbol]:
        """The dependency symbol this evaluator substitutes (None if all-constant)."""
        return self._symbol

    def property_names(self) -> Set[str]:
        """Names of the properties this evaluator can compute."""
        return set(self._functions.keys())

    def function(self, name: str) -> Callable:
        """Returns the cached numeric callable for a single property.

        Useful when you only need one property, or want to hand the raw callable
        to another numerical routine.

        Raises:
            KeyError: No compiled property with that name.
        """
        if name not in self._functions:
            raise KeyError(
                f"No compiled property '{name}'. Available: {sorted(self._functions)}")
        return self._functions[name]

    def evaluate(self, value: ArrayLike) -> Dict[str, object]:
        """Alias for calling the evaluator; see :meth:`__call__`."""
        return self(value)

    def __call__(self, value: ArrayLike) -> Dict[str, object]:
        """Evaluates every property at a scalar or array of dependency values.

        Args:
            value: A scalar, or an array-like of dependency values.
        Returns:
            Mapping of property name to result: a float for a scalar input, or a
            NumPy array shaped like the input for an array. Constant properties
            are broadcast to match.
        Raises:
            ValueError: value is None or cannot be read as a real number/array.
        """
        if value is None:
            raise ValueError("value must not be None")
        if _is_scalar(value):
            return self._evaluate_scalar(float(value))  # type: ignore[arg-type]
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"value must be a real number or array-like: {exc}") from exc
        return self._evaluate_array(array)

    def _evaluate_scalar(self, value: float) -> Dict[str, object]:
        results: Dict[str, object] = {}
        for name, func in self._functions.items():
            results[name] = float(np.asarray(func(value)).reshape(-1)[0])
        return results

    def _evaluate_array(self, array: "np.ndarray") -> Dict[str, object]:
        results: Dict[str, object] = {}
        for name, func in self._functions.items():
            raw = np.asarray(func(array), dtype=float)
            if raw.shape != array.shape:
                raw = np.broadcast_to(raw, array.shape).astype(float, copy=True)
            results[name] = raw
        return results

    def __len__(self) -> int:
        return len(self._functions)

    def __repr__(self) -> str:
        symbol = self._symbol if self._symbol is not None else "const"
        return (f"MaterialEvaluator('{self._material_name}', symbol={symbol}, "
                f"properties={len(self._functions)})")


class _Constant:
    """Callable wrapper for a constant property, so every property is a function."""

    __slots__ = ("value",)

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, _: ArrayLike) -> float:
        return self.value


def _is_scalar(value: object) -> bool:
    """True for a single number, including a 0-d NumPy array."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float, np.number)):
        return True
    return isinstance(value, np.ndarray) and value.ndim == 0


__all__: List[str] = ["MaterialEvaluator"]
