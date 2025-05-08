import numpy as np
import sympy as sp
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
from pymatlib.core.elements import (ChemicalElement,
                                    interpolate_atomic_mass,
                                    interpolate_atomic_number,
                                    interpolate_boiling_temperature,
                                    interpolate_melting_temperature)
from pymatlib.core.typedefs import ArrayTypes #, PropertyTypes

import logging
logger = logging.getLogger(__name__)

class AlloyCompositionError(ValueError):
    """Exception raised when alloy composition validation fails."""
    pass

class AlloyTemperatureError(ValueError):
    """Exception raised when alloy temperature validation fails."""
    pass

@dataclass
class Alloy:
    """Represents an alloy or pure metal composed of various elements with given fractions."""
    material_type: str  # 'alloy' or 'pure_metal'
    elements: List[ChemicalElement]
    composition: ArrayTypes  # List of fractions summing to 1.0

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
    atomic_number: float = field(init=False)
    atomic_mass: float = field(init=False)

    # Optional properties with default values
    density: sp.Expr = None
    dynamic_viscosity: sp.Expr = None
    energy_density: sp.Expr = None
    energy_density_solidus: float = None
    energy_density_liquidus: float = None
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
        Initialize and validate the alloy properties.

        Called automatically after the dataclass initialization.
        Validates composition and temperatures, then calculates derived properties.
        """
        self._validate_composition()
        self._validate_temperatures()
        self._calculate_properties()

    def solidification_interval(self) -> Tuple[sp.Float, sp.Float]:
        """
        Calculate the solidification interval based on solidus and liquidus temperature of the alloy.

        Returns:
            Tuple[float, float]: A tuple containing the solidus and liquidus temperatures.
        """
        return self.solidus_temperature, self.liquidus_temperature

    def _validate_composition(self) -> None:
        """Validate the alloy composition."""
        if not self.elements:
            raise ValueError("Elements list cannot be empty")
        if len(self.elements) != len(self.composition):
            raise ValueError(f"Number of elements ({len(self.elements)}) must match composition length ({len(self.composition)})")
        if not np.isclose(sum(self.composition), 1.0, atol=1e-10):
            raise AlloyCompositionError(f"The sum of the composition array must be 1.0, got {sum(self.composition)}")
        if self.material_type == 'alloy' and len(self.elements) < 2:
            raise AlloyCompositionError(f"Alloys must have at least 2 elements, got {len(self.elements)}")

    def _validate_temperatures(self) -> None:
        """Validate the temperature properties based on material type."""
        if self.material_type == 'pure_metal':
            self._validate_pure_metal_temperatures()
        else:  # alloy
            self._validate_alloy_temperatures()

    def _validate_pure_metal_temperatures(self) -> None:
        """Validate temperatures for pure metal materials."""
        if self.melting_temperature is None:
            raise AlloyTemperatureError("Pure metals must specify melting_temperature")
        if not isinstance(self.melting_temperature, sp.Float):
            raise AlloyTemperatureError(f"Melting temperature must be a Float, got {type(self.melting_temperature).__name__}.")
        if self.boiling_temperature is None:
            raise AlloyTemperatureError("Pure metals must specify boiling_temperature")
        if not isinstance(self.boiling_temperature, sp.Float):
            raise AlloyTemperatureError(f"Boiling temperature must be a Float, got {type(self.boiling_temperature).__name__}.")

    def _validate_alloy_temperatures(self) -> None:
        """Validate temperatures for alloy materials."""
        if self.solidus_temperature is None or self.liquidus_temperature is None:
            raise AlloyTemperatureError("Alloys must specify both solidus_temperature and liquidus_temperature")
        if not isinstance(self.solidus_temperature, sp.Float):
            raise AlloyTemperatureError(f"Solidus temperature must be a Float, got {type(self.solidus_temperature).__name__}.")
        if not isinstance(self.liquidus_temperature, sp.Float):
            raise AlloyTemperatureError(f"Liquidus temperature must be a Float, got {type(self.liquidus_temperature).__name__}.")
        if self.solidus_temperature >= self.liquidus_temperature:
            raise AlloyTemperatureError("The solidus temperature must be less than or equal to the liquidus temperature.")
        if self.initial_boiling_temperature is not None and self.final_boiling_temperature is not None:
            if self.initial_boiling_temperature >= self.final_boiling_temperature:
                raise AlloyTemperatureError("The liquidus boiling temperature must be less than or equal to the vapor boiling temperature.")
        if not (450 <= self.solidus_temperature <= 1900):
            raise AlloyTemperatureError(f"Solidus temperature {self.solidus_temperature} K is out of range (450 K - 1900 K).")
        if not (600 <= self.liquidus_temperature <= 2000):
            raise AlloyTemperatureError(f"Liquidus temperature {self.liquidus_temperature} K is out of range (600 K - 2000 K).")

    def _calculate_properties(self) -> None:
        """Calculate derived properties based on composition."""
        # Always calculate these fundamental properties
        self.atomic_number = interpolate_atomic_number(self.elements, self.composition)
        self.atomic_mass = interpolate_atomic_mass(self.elements, self.composition)

        # Calculate interpolated temperature values but store them separately
        self._calculated_melting_temperature = interpolate_melting_temperature(self.elements, self.composition)
        self._calculated_boiling_temperature = interpolate_boiling_temperature(self.elements, self.composition)

        # For pure metals, use interpolated values only if not provided in config
        if self.material_type == 'pure_metal':
            if self.melting_temperature is None:
                logger.warning("Melting temperature not provided, using interpolated value as fallback")
                self.melting_temperature = sp.Float(self._calculated_melting_temperature)

            if self.boiling_temperature is None:
                logger.warning("Boiling temperature not provided, using interpolated value as fallback")
                self.boiling_temperature = sp.Float(self._calculated_boiling_temperature)
