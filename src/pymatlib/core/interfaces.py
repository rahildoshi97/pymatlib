"""Abstract base classes for PyMatLib components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union, Tuple
import numpy as np
import sympy as sp
from pymatlib.core.materials import Material

class PropertyProcessor(ABC):
    """Abstract base class for material property processors.

    Property processors handle the conversion of raw property data
    (from YAML configurations, data files, or symbolic expressions)
    into temperature-dependent functions that can be evaluated
    by the materials modeling system.
    """

    @abstractmethod
    def process_property(self, material: Material, prop_name: str,
                         config: Dict[str, Any], T: Union[float, sp.Symbol]) -> None:
        """Process a single material property.
        Args:
            material: Material instance to modify
            prop_name: Name of the property being processed
            config: Property configuration from YAML
            T: Temperature symbol or value
        """
        pass

class TemperatureResolver(ABC):
    """Abstract base class for temperature resolution."""

    @abstractmethod
    def resolve_temperature(self, temp_def: Any, material: Material) -> Union[float, np.ndarray]:
        """Resolve temperature definition to numeric values.
        Args:
            temp_def: Temperature definition (various formats)
            material: Material for reference resolution
        Returns:
            Resolved temperature value(s)
        """
        pass

    @abstractmethod
    def process_temperature_definition(self, temp_def: Any, n_values: int = None,
                                       material: Material = None) -> np.ndarray:
        """Process temperature definition into array.
        Args:
            temp_def: Temperature definition
            n_values: Number of expected values
            material: Material for reference resolution
        Returns:
            Temperature array
        """
        pass

class DataHandler(ABC):
    """Abstract base class for data file handling."""

    @abstractmethod
    def read_data(self, file_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Read data from file configuration.
        Args:
            file_config: File configuration dictionary
        Returns:
            Tuple of (temperature_array, property_array)
        """
        pass

class Visualizer(ABC):
    """Abstract base class for property visualization."""

    @abstractmethod
    def visualize_property(self, material: Material, prop_name: str,
                           T: Union[float, sp.Symbol], prop_type: str,
                           **kwargs) -> None:
        """Visualize a material property.
        Args:
            material: Material instance
            prop_name: Property name
            T: Temperature symbol or value
            prop_type: Type of property visualization
            **kwargs: Additional visualization parameters
        """
        pass

    @abstractmethod
    def save_plots(self) -> None:
        """Save generated plots to files."""
        pass

    @abstractmethod
    def initialize_plots(self) -> None:
        """Initialize plotting system."""
        pass

    @abstractmethod
    def reset_visualization_tracking(self) -> None:
        """Reset visualization state."""
        pass
