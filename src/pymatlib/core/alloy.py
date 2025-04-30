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


class AlloyCompositionError(ValueError):
    """Exception raised when alloy composition validation fails."""
    pass

class AlloyTemperatureError(ValueError):
    """Exception raised when alloy temperature validation fails."""
    pass


@dataclass
class Alloy:
    """
    Represents an alloy composed of various elements with given fractions.

    Attributes:
        elements (List[ChemicalElement]): List of `ChemicalElement` objects that make up the alloy.
        composition (ArrayTypes): List of fractions of each element in the alloy. Fractions should sum to 1.0.
        solidus_temperature (float): Solidus temperature of the alloy in Kelvin.
        liquidus_temperature (float): Liquidus temperature of the alloy in Kelvin.
        atomic_number (float): Average atomic number of the alloy, calculated post-init.
        atomic_mass (float): Average atomic mass of the alloy, calculated post-init.
        boiling_temperature (float): Boiling temperature of the alloy, calculated post-init.
        melting_temperature (float): Melting temperature of the alloy, calculated post-init.

        Optional properties:
            density (sp.Expr): Density of the alloy.
            dynamic_viscosity (sp.Expr): Dynamic viscosity of the alloy.
            heat_capacity (sp.Expr): Heat capacity of the alloy.
            heat_conductivity (sp.Expr): Heat conductivity of the alloy.
            kinematic_viscosity (sp.Expr): Kinematic viscosity of the alloy.
            latent_heat_of_fusion (sp.Expr): Latent heat of fusion of the alloy.
            latent_heat_of_vaporization (sp.Expr): Latent heat of vaporization of the alloy.
            surface_tension (sp.Expr): Surface tension of the alloy.
            thermal_diffusivity (sp.Expr): Thermal diffusivity of the alloy.
            thermal_expansion_coefficient (sp.Expr): Thermal expansion coefficient of the alloy.

    Raises:
        AlloyCompositionError: If the composition fractions don't sum to 1.0.
        AlloyTemperatureError: If solidus temperature is greater than liquidus temperature
            or temperatures are out of the general valid range for alloys.
        ValueError: If elements list is empty or composition length doesn't match elements length.
    """
    elements: List[ChemicalElement]
    composition: ArrayTypes  # List of fractions summing to 1.0
    solidus_temperature: sp.Float
    liquidus_temperature: sp.Float

    # Properties to be calculated post-init based on composition
    atomic_number: float = field(init=False)
    atomic_mass: float = field(init=False)
    boiling_temperature: float = field(init=False)
    melting_temperature: float = field(init=False)

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

    def solidification_interval(self) -> Tuple[sp.Float, sp.Float]:
        """
        Calculate the solidification interval based on solidus and liquidus temperature of the alloy.

        Returns:
            Tuple[float, float]: A tuple containing the solidus and liquidus temperatures.
        """
        return self.solidus_temperature, self.liquidus_temperature

    def _validate_composition(self) -> None:
        """
        Validate the alloy composition.

        Raises:
            ValueError: If elements list is empty or composition length doesn't match elements length.
            AlloyCompositionError: If composition fractions don't sum to 1.0.
        """
        if not self.elements:
            raise ValueError("Elements list cannot be empty")
        if len(self.elements) != len(self.composition):
            raise ValueError(f"Number of elements ({len(self.elements)}) must match composition length ({len(self.composition)})")
        if not np.isclose(sum(self.composition), 1.0, atol=1e-10):
            raise AlloyCompositionError(f"The sum of the composition array must be 1.0, got {sum(self.composition)}")

    def _validate_temperatures(self) -> None:
        """
        Validate the alloy temperatures.

        Raises:
            AlloyTemperatureError:
                - If solidus temperature is greater than liquidus temperature.
                - If temperatures are outside the general range for alloys (450 K to 2000 K).
        """
        if not isinstance(self.solidus_temperature, sp.Float):
            raise AlloyTemperatureError(f"Solidus temperature must be a Float, "
                                        f"got {type(self.solidus_temperature).__name__}.")
        if not isinstance(self.liquidus_temperature, sp.Float):
            raise AlloyTemperatureError(f"Liquidus temperature must be a Float, "
                                        f"got {type(self.liquidus_temperature).__name__}.")
        if self.solidus_temperature > self.liquidus_temperature:
            raise AlloyTemperatureError("The solidus temperature must be less than or equal to the liquidus temperature.")
        if not (450 <= self.solidus_temperature <= 1900):
            raise AlloyTemperatureError(f"Solidus temperature {self.solidus_temperature} K is out of range (450 K - 1900 K).")
        if not (600 <= self.liquidus_temperature <= 2000):
            raise AlloyTemperatureError(f"Liquidus temperature {self.liquidus_temperature} K is out of range (600 K - 2000 K).")

    def _calculate_properties(self) -> None:
        """Calculate derived properties based on composition."""
        self.atomic_number = interpolate_atomic_number(self.elements, self.composition)
        self.atomic_mass = interpolate_atomic_mass(self.elements, self.composition)
        self.boiling_temperature = interpolate_boiling_temperature(self.elements, self.composition)
        self.melting_temperature = interpolate_melting_temperature(self.elements, self.composition)

    def __post_init__(self) -> None:
        """
        Initialize and validate the alloy properties.

        Called automatically after the dataclass initialization.
        Validates composition and temperatures, then calculates derived properties.
        """
        self._validate_composition()
        self._validate_temperatures()
        self._calculate_properties()