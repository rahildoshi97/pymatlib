"""
PyMatLib - Python Material Properties Library
=============================================

A high-performance Python library for material simulation and analysis with a focus on
temperature-dependent properties. PyMatLib enables efficient modeling of pure metals and
alloys through YAML configuration files, providing symbolic and numerical property
evaluation for various material properties.

Main Features:
- YAML-driven material configuration
- Temperature-dependent property modeling
- Symbolic mathematics with SymPy
- Piecewise function support with regression
- Property inversion capabilities
- Comprehensive visualization tools
"""

# Enhanced version handling with multiple fallbacks
try:
    from ._version import version as __version__
except ImportError:
    try:
        from importlib.metadata import version
        __version__ = version("pymatlib")
    except ImportError:
        try:
            from importlib_metadata import version
            __version__ = version("pymatlib")
        except ImportError:
            __version__ = "0.3.0+unknown"  # Fallback version

# Core API exports
from pymatlib.parsing.api import (
    create_material,
    get_supported_properties,
    validate_yaml_file,
)

# Core classes
from pymatlib.core.material import Material
from pymatlib.parsing.validation.type_detection import PropertyType

# Piecewise utilities
from pymatlib.algorithms.inversion import (
    PiecewiseInverter,
    create_energy_density_inverse,
)

# Constants
from pymatlib.data.constants import (
    ProcessingConstants,
    ErrorMessages,
    FileConstants,
)

__all__ = [
    # Version
    "__version__",
    # Main API
    "create_material",
    "get_supported_properties",
    "validate_yaml_file",
    # Core classes
    "Material",
    "PropertyType",
    # Piecewise utilities
    "PiecewiseInverter",
    "create_energy_density_inverse",
    # Constants
    "ProcessingConstants",
    "ErrorMessages",
    "FileConstants",
]

# Package metadata
__author__ = "Rahil Doshi"
__email__ = "rahil.doshi@fau.de"
__license__ = "GPL-3.0"
__description__ = "A high-performance Python library for material simulation and analysis"
