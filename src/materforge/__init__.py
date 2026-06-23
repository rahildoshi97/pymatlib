# SPDX-FileCopyrightText: 2025 - 2026 Rahil Miten Doshi, Friedrich-Alexander-Universität Erlangen-Nürnberg
# SPDX-License-Identifier: BSD-3-Clause

"""
MaterForge - Materials Formulation Engine with Python

A high-performance Python library for material property modeling
and analysis. MaterForge provides comprehensive tools for defining, processing, and
evaluating material properties as functions of temperature and other dependencies.

Key Features:
- Temperature-dependent material property modeling
- Multiple property definition formats (YAML-based)
- Symbolic mathematics integration with SymPy
- Piecewise function creation and evaluation
- Material property visualization
- Regression and data analysis capabilities
- Integration with numerical simulation frameworks

Main Components:
- Core: Material definitions and fundamental data structures
- Parsing: YAML configuration parsing and property processing
- Algorithms: Mathematical operations and property computations
- Visualization: Property plotting and analysis tools
- Data: Material databases and physical constants
"""

# Version handling - Python >=3.10 guarantees importlib.metadata is available
try:
    from ._version import version as __version__
except ImportError:
    # Underscore aliases so these helpers don't leak into the public namespace.
    from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PackageNotFoundError
    try:
        __version__ = _pkg_version("materforge")
    except _PackageNotFoundError:
        # Last resort: neither the generated _version.py nor installed package
        # metadata is available. Stay version-neutral so this can't go stale.
        __version__ = "0+unknown"

# Core material definitions
from .core.materials import Material, PropertySamples
from .core.symbol_registry import SymbolRegistry
from .core.evaluator import MaterialEvaluator

# Main API functions
from .parsing.api import (
    create_material,
    validate_yaml_file,
    get_material_info,
    get_material_property_names,
    evaluate_material_properties,
    clear_cache,
)

# Bundled example materials
from .catalog import list_materials, load_material, get_material_path

# Property processing
from .parsing.processors.property_processor import PropertyProcessor
from .parsing.validation.property_type_detector import PropertyType

# Algorithms
from .algorithms.interpolation import interpolate_value, ensure_ascending_order
from .algorithms.piecewise_builder import PiecewiseBuilder
from .algorithms.piecewise_inverter import PiecewiseInverter

# Visualization
from .visualization.plotters import PropertyVisualizer
from .visualization.plots import plot_property, plot_residuals, compare_materials

# Fit-quality analysis
from .analysis import (
    r_squared,
    rmse,
    mae,
    max_abs_error,
    FitQuality,
    fit_quality,
    residuals,
    fit_report,
)

__all__ = [
    # Version
    "__version__",
    # Core classes
    "Material",
    "PropertySamples",
    "SymbolRegistry",
    "MaterialEvaluator",
    # Main API
    "create_material",
    "validate_yaml_file",
    "get_material_info",
    "get_material_property_names",
    "evaluate_material_properties",
    "clear_cache",
    # Bundled example materials
    "list_materials",
    "load_material",
    "get_material_path",
    # Processing
    "PropertyProcessor",
    "PropertyType",
    # Algorithms
    "interpolate_value",
    "ensure_ascending_order",
    "PiecewiseBuilder",
    "PiecewiseInverter",
    # Visualization
    "PropertyVisualizer",
    "plot_property",
    "plot_residuals",
    "compare_materials",
    # Fit-quality analysis
    "r_squared",
    "rmse",
    "mae",
    "max_abs_error",
    "FitQuality",
    "fit_quality",
    "residuals",
    "fit_report",
]

# Package metadata
__author__ = "Rahil Doshi"
__email__ = "rahil.doshi@fau.de"
__description__ = "Materials Formulation Engine with Python"
