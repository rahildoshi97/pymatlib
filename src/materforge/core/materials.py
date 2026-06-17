# SPDX-FileCopyrightText: 2025 - 2026 Rahil Miten Doshi, Friedrich-Alexander-Universität Erlangen-Nürnberg
# SPDX-FileCopyrightText: 2026 Matthias Markl, Friedrich-Alexander-Universität Erlangen-Nürnberg
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import sympy as sp

if TYPE_CHECKING:
    from materforge.core.evaluator import MaterialEvaluator

logger = logging.getLogger(__name__)


@dataclass
class Material:
    """Generic material container with fully dynamic property tracking.
    All material properties are assigned dynamically
    via setattr and tracked automatically.

    Attributes:
        name: Human-readable material identifier.
        properties: Dictionary with all properties.
    """
    name: str
    properties: Dict[str, sp.Basic] = field(default_factory=dict)

    # --- Dynamic property tracking ---
    def __setattr__(self, name: str, value) -> None:
        if name in {"name", "properties"}:
            super().__setattr__(name, value)
        else:
            self.properties[name] = value

    def __getattr__(self, name: str) -> Any:
        # Only fires when normal lookup in properties fails
        if name in self.properties:
            return self.properties[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def property_names(self) -> set:
        """Returns all dynamically assigned property names."""
        return set(self.properties.keys())

    # --- Property evaluation ---
    def evaluate(self, symbol: sp.Symbol, value) -> Material:
        """Evaluates all properties by substituting symbol=value.

        Args:
            symbol: SymPy symbol to substitute (e.g. sp.Symbol('T')).
            value:  Value to substitute.
        Returns:
            New Material instance with name '{name}@{symbol}={value}' and
            all properties substituted at the given value.
            Properties that fail evaluation are silently excluded.
        Raises:
            ValueError: If symbol is not sp.Symbol, or value is non-numeric.
        """
        if not isinstance(symbol, sp.Symbol):
            raise ValueError(f"symbol must be sp.Symbol, got {type(symbol).__name__}")

        if value is None:
            raise ValueError("value must not be None")
        try:
            value = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"value must be convertible to float, got {type(value).__name__}") from e

        logger.info("Evaluating '%s' at %s=%.2f", self.name, symbol, value)

        evaluated_properties: Dict[str, sp.Basic] = {}

        for prop_name, expr in self.properties.items():
            if expr is None:
                logger.warning("Dropped None expression for '%s'", prop_name)
                continue

            try:
                substituted = expr.subs(symbol, value).evalf(chop=True)
            except (TypeError, ValueError, AttributeError) as e:
                logger.error("Failed to evaluate '%s': %s", prop_name, e, exc_info=True)
                continue
            if substituted.free_symbols:
                logger.error("Property '%s' still has free symbols %s after substituting %s; expression requires %s",
                    prop_name, substituted.free_symbols, symbol, expr.free_symbols, exc_info=True)
                continue
            evaluated_properties[prop_name] = substituted

        return Material(
            name=f"{self.name}@{symbol}={value}",
            properties=evaluated_properties,
        )

    # --- Fast numeric evaluation ---
    def compile(self, symbol: Optional[sp.Symbol] = None) -> "MaterialEvaluator":
        """Builds a reusable evaluator with cached numeric callables.

        Each property is lambdified once and reused, so sweeping many dependency
        values - or evaluating over a NumPy array in one call - is far faster
        than repeated symbolic :meth:`evaluate`.

        Args:
            symbol: Dependency symbol to evaluate against. Inferred from the
                    properties' free symbols when omitted.
        Returns:
            A :class:`~materforge.core.evaluator.MaterialEvaluator` snapshot of
            this material.
        Example:
            >>> ev = material.compile()
            >>> ev(500.0)                       # {'density': 2634.5, ...}
            >>> ev(np.linspace(300, 900, 200))  # arrays, one call per property
        """
        from materforge.core.evaluator import MaterialEvaluator
        return MaterialEvaluator(self, symbol)

    def __str__(self) -> str:
        return f"Material: {self.name} ({len(self.properties)} properties)"

    def __repr__(self) -> str:
        return f"Material(name='{self.name}', properties={sorted(self.property_names())})"
