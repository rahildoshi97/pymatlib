import numpy as np
import sympy as sp
from dataclasses import dataclass, field
from pymatlib.core.alloy import Alloy
from pymatlib.core.typedefs import MaterialProperty


@dataclass
class PropertyArrayExtractor:
    """
    Extracts arrays of property values from an Alloy at specified temperatures.
    Attributes:
        alloy (Alloy): The alloy object containing material properties.
        temperature_array (np.ndarray): Array of temperature values to evaluate properties at.
        symbol (sp.Symbol): Symbol to use for property evaluation (e.g., u.center()).
        specific_enthalpy_array (np.ndarray): Extracted specific enthalpy values.
        energy_density_array (np.ndarray): Extracted energy density values.
    """
    alloy: Alloy
    temperature_array: np.ndarray
    symbol: sp.Symbol

    # Arrays will be populated during extraction
    specific_enthalpy_array: np.ndarray = field(default_factory=lambda: np.array([]))
    energy_density_array: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        """Initialize arrays after instance creation."""
        # Extract property arrays after initialization if temperature array is provided.
        if len(self.temperature_array) >= 2:
            self.extract_all_arrays()

    def extract_property_array(self, property_name: str) -> np.ndarray:
        """
        Extract array of property values at each temperature point using the provided symbol.
        Args:
            property_name (str): Name of the property to extract.
        Returns:
            np.ndarray: Array of property values corresponding to temperature_array.
        Raises:
            ValueError: If property doesn't exist or isn't a MaterialProperty.
        """
        # Get the property from the alloy
        property_obj = getattr(self.alloy, property_name, None)
        # Check if property exists
        if property_obj is None:
            raise ValueError(f"Property '{property_name}' not found in alloy")
        # Check if it's a MaterialProperty
        if isinstance(property_obj, MaterialProperty):
            # Use the symbolic temperature variable from the MaterialProperty
            return property_obj.evalf(self.symbol, self.temperature_array)
        else:
            raise ValueError(f"Property '{property_name}' is not a MaterialProperty")

    def extract_all_arrays(self) -> None:
        """Extract arrays for all supported properties."""
        # Extract specific enthalpy array if available
        if hasattr(self.alloy, 'specific_enthalpy') and self.alloy.specific_enthalpy is not None:
            self.specific_enthalpy_array = self.extract_property_array('specific_enthalpy')
        # Extract energy density array if available
        if hasattr(self.alloy, 'energy_density') and self.alloy.energy_density is not None:
            self.energy_density_array = self.extract_property_array('energy_density')
