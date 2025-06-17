import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import sympy as sp

from pymatlib.core.elements import (ChemicalElement,
                                    interpolate_atomic_mass,
                                    interpolate_atomic_number,
                                    interpolate_boiling_temperature,
                                    interpolate_melting_temperature)

logger = logging.getLogger(__name__)

class MaterialCompositionError(ValueError):
    """Exception raised when material composition validation fails."""
    pass

class MaterialTemperatureError(ValueError):
    """Exception raised when material temperature validation fails."""
    pass

@dataclass
class Material:
    """Represents a material (alloy or pure metal) composed of various elements with given fractions."""
    name: str
    material_type: str  # 'alloy' or 'pure_metal'
    elements: List[ChemicalElement]
    composition: Union[np.ndarray, List, Tuple]  # List of fractions summing to 1.0

    # Temperature properties - different requirements based on material_type
    # For pure metals
    melting_temperature: sp.Float = None
    boiling_temperature: sp.Float = None

    # For alloys
    # interval for melting
    solidus_temperature: sp.Float = None
    liquidus_temperature: sp.Float = None
    # interval for boiling
    initial_boiling_temperature: sp.Float = None
    final_boiling_temperature: sp.Float = None

    # Properties to be calculated post-init based on composition
    atomic_mass: float = field(init=False)
    atomic_number: float = field(init=False)

    # Optional properties with default values
    density: sp.Expr = None
    dynamic_viscosity: sp.Expr = None
    energy_density: Optional[Union[sp.Piecewise, sp.Expr]] = None
    energy_density_solidus: sp.Float = None
    energy_density_liquidus: sp.Float = None
    heat_capacity: sp.Expr = None
    heat_conductivity: sp.Expr = None
    kinematic_viscosity: sp.Expr = None
    latent_heat_of_fusion: sp.Expr = None
    latent_heat_of_vaporization: sp.Expr = None
    specific_enthalpy: sp.Expr = None
    surface_tension: sp.Expr = None
    thermal_diffusivity: sp.Expr = None
    thermal_expansion_coefficient: sp.Expr = None

    def __post_init__(self) -> None:
        """
        Initialize and validate the material properties.
        Called automatically after the dataclass initialization.
        Validates composition and temperatures, then calculates derived properties.
        """
        self._validate_composition()
        self._validate_temperatures()
        self._calculate_properties()

    def solidification_interval(self) -> Tuple[sp.Float, sp.Float]:
        """
        Calculate the solidification interval based on solidus and liquidus temperature of the material.
        Returns:
            Tuple[float, float]: A tuple containing the solidus and liquidus temperatures.
        """
        return self.solidus_temperature, self.liquidus_temperature

    def _validate_composition(self) -> None:
        """Validate the material composition."""
        if not self.elements:
            raise ValueError("Elements list cannot be empty")
        if len(self.elements) != len(self.composition):
            raise ValueError(f"Number of elements ({len(self.elements)}) must match composition length ({len(self.composition)})")
        if not np.isclose(sum(self.composition), 1.0, atol=1e-10):
            raise MaterialCompositionError(f"The sum of the composition array must be 1.0, got {sum(self.composition)}")
        if self.material_type == 'pure_metal' and len(self.elements) != 1:
            raise MaterialCompositionError(f"Pure metals must have exactly 1 element, got {len(self.elements)}")
        if self.material_type == 'alloy' and len(self.elements) < 2:
            raise MaterialCompositionError(f"Alloys must have at least 2 elements, got {len(self.elements)}")

    def _validate_temperatures(self) -> None:
        """Validate the temperature properties based on material type using centralized validator."""
        from pymatlib.parsing.validation.temperature_validator import TemperatureValidator
        TemperatureValidator.validate_material_temperatures(self)

    def _calculate_properties(self) -> None:
        """Calculate derived properties based on composition."""
        # Always calculate these fundamental properties
        self.atomic_number = interpolate_atomic_number(self.elements, self.composition)
        self.atomic_mass = interpolate_atomic_mass(self.elements, self.composition)
        # For pure metals, use interpolated values only if not provided in config
        if self.material_type == 'pure_metal':
            if self.melting_temperature is None:
                logger.warning("Melting temperature not provided, using interpolated value as fallback")
                # Calculate interpolated temperature values but store them separately
                self._calculated_melting_temperature = interpolate_melting_temperature(self.elements, self.composition)
                self.melting_temperature = sp.Float(self._calculated_melting_temperature)
            if self.boiling_temperature is None:
                logger.warning("Boiling temperature not provided, using interpolated value as fallback")
                # Calculate interpolated temperature values but store them separately
                self._calculated_boiling_temperature = interpolate_boiling_temperature(self.elements, self.composition)
                self.boiling_temperature = sp.Float(self._calculated_boiling_temperature)
