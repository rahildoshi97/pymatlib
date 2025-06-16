"""Abstract base classes for PyMatLib components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union
import numpy as np
import sympy as sp
from pymatlib.core.material import Material


class PropertyProcessor(ABC):
    """Abstract base class for material property processors.

    Property processors handle the conversion of raw property data
    (from YAML configurations, data files, or symbolic expressions)
    into temperature-dependent functions that can be evaluated
    by the materials modeling system.

    Examples
    --------
    >>> class CustomProcessor(PropertyProcessor):
    ...     def process_property(self, material, prop_name, config, T):
    ...         # Custom processing logic
    ...         pass
    """

    @abstractmethod
    def process_property(self, material: Material, prop_name: str,
                         config: Dict[str, Any], T: Union[float, sp.Symbol]) -> None:
        """Process a single material property."""
        pass


class TemperatureResolver(ABC):
    """Abstract base class for temperature resolution."""

    @abstractmethod
    def resolve_temperature(self, temp_def: Any, material: Material) -> Union[float, np.ndarray]:
        """Resolve temperature definition to numeric values."""
        pass


class DataHandler(ABC):
    """Abstract base class for data file handling."""

    @abstractmethod
    def read_data(self, file_config: Dict[str, Any]) -> tuple:
        """Read data from file configuration."""
        pass


class Visualizer(ABC):
    """Abstract base class for property visualization."""

    @abstractmethod
    def visualize_property(self, material: Material, prop_name: str, **kwargs) -> None:
        """Visualize a material property."""
        pass

    @abstractmethod
    def save_plots(self) -> None:
        """Save generated plots."""
        pass
