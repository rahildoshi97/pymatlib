"""
Core data structures and material definitions.

This module contains the fundamental classes and interfaces that define
materials, chemical elements, and the core abstractions used throughout
the PyMatLib library.
"""

from .materials import Material
from .elements import ChemicalElement
from .symbol_registry import SymbolRegistry
from .interfaces import PropertyProcessor, TemperatureResolver, DataHandler, Visualizer
from .exceptions import MaterialError, MaterialCompositionError, MaterialTemperatureError

__all__ = [
    "Material",
    "ChemicalElement",
    "SymbolRegistry",
    "PropertyProcessor",
    "TemperatureResolver",
    "DataHandler",
    "Visualizer",
    "MaterialError",
    "MaterialCompositionError",
    "MaterialTemperatureError"
]
