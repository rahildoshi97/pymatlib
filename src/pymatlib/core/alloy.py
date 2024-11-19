import numpy as np
from typing import List, Tuple
from dataclasses import dataclass, field
from pymatlib.core.elements import (ChemicalElement,
                                    interpolate_atomic_mass,
                                    interpolate_atomic_number,
                                    interpolate_temperature_boil)
from pymatlib.data.element_data import Ti, Al, V
from pymatlib.core.typedefs import ArrayTypes, PropertyTypes


@dataclass
class Alloy:
    """
    Represents an alloy composed of various elements with given fractions.

    Attributes:
        elements (List[ChemicalElement]): List of `ChemicalElement` objects that make up the alloy.
        composition (ArrayTypes): List of fractions of each element in the alloy. Fractions should sum to 1.0.
        temperature_solidus (float): Solidus temperature of the alloy in Kelvin.
        temperature_liquidus (float): Liquidus temperature of the alloy in Kelvin.
        atomic_number (float): Average atomic number of the alloy, calculated post-init.
        atomic_mass (float): Average atomic mass of the alloy, calculated post-init.
        temperature_boil (float): Boiling temperature of the alloy, calculated post-init.

        Optional properties:
            density (PropertyTypes): Density of the alloy.
            dynamic_viscosity (PropertyTypes): Dynamic viscosity of the alloy.
            heat_capacity (PropertyTypes): Heat capacity of the alloy.
            heat_conductivity (PropertyTypes): Heat conductivity of the alloy.
            kinematic_viscosity (PropertyTypes): Kinematic viscosity of the alloy.
            latent_heat_of_fusion (PropertyTypes): Latent heat of fusion of the alloy.
            latent_heat_of_vaporization (PropertyTypes): Latent heat of vaporization of the alloy.
            surface_tension (PropertyTypes): Surface tension of the alloy.
            thermal_diffusivity (PropertyTypes): Thermal diffusivity of the alloy.
            thermal_expansion_coefficient (PropertyTypes): Thermal expansion coefficient of the alloy.
    """
    elements: List[ChemicalElement]
    composition: ArrayTypes  # List of fractions summing to 1.0
    temperature_solidus: float
    temperature_liquidus: float

    # Properties to be calculated post-init based on composition
    atomic_number: float = field(init=False)
    atomic_mass: float = field(init=False)
    temperature_boil: float = field(init=False)

    # Optional properties with default values
    density: PropertyTypes = None
    dynamic_viscosity: PropertyTypes = None
    heat_capacity: PropertyTypes = None
    heat_conductivity: PropertyTypes = None
    kinematic_viscosity: PropertyTypes = None
    latent_heat_of_fusion: PropertyTypes = None
    latent_heat_of_vaporization: PropertyTypes = None
    surface_tension: PropertyTypes = None
    thermal_diffusivity: PropertyTypes = None
    thermal_expansion_coefficient: PropertyTypes = None


    def solidification_interval(self) -> Tuple[float, float]:
        """
        Calculate the solidification interval based on solidus and liquidus temperature of the alloy.

        Returns:
            Tuple[float, float]: A tuple containing the solidus and liquidus temperatures.
        """
        return self.temperature_solidus, self.temperature_liquidus

    def __post_init__(self) -> None:
        """
        Initializes properties based on elemental composition and validates the composition and phase transition temperatures.

        Raises:
            ValueError: If the sum of the composition array does not equal 1 or if the solidus temperature is greater than the liquidus temperature.
        """
        if not np.isclose(sum(self.composition), 1.0, atol=1e-12):
            raise ValueError(f"The sum of the composition array must be 1.0, got {sum(self.composition)}")

        if self.temperature_solidus > self.temperature_liquidus:
            raise ValueError("The solidus temperature must be less than or equal to the liquidus temperature.")

        # utils = ElementUtils()
        self.atomic_number = interpolate_atomic_number(self.elements, self.composition)
        self.atomic_mass = interpolate_atomic_mass(self.elements, self.composition)
        self.temperature_boil = interpolate_temperature_boil(self.elements, self.composition)


if __name__ == '__main__':
    try:
        # Valid case
        Ti64 = Alloy([Ti, Al, V], [0.90, 0.06, 0.04], 1878, 1928)
        print(f"Calculated Atomic Number: {0.90 * Ti.atomic_number + 0.06 * Al.atomic_number + 0.04 * V.atomic_number}")
        print(f"Alloy Atomic Number: {Ti64.atomic_number}")
        print(f"Initial Heat Conductivity: {Ti64.heat_conductivity}")
        Ti64.heat_conductivity = 34
        print(f"Updated Heat Conductivity: {Ti64.heat_conductivity}")
        print(f"Boiling Temperature (Before Change): {Ti64.temperature_boil}")
        Ti64.temperature_boil = 1000
        print(f"Boiling Temperature (After Change): {Ti64.temperature_boil}")

        # Invalid Composition
        try:
            invalid_alloy = Alloy([Ti, Al], [0.5, 0.5], 1878, 1928)
        except ValueError as e:
            print(f"Invalid Composition Test Passed: {e}")

        # Empty Composition
        try:
            empty_composition_alloy = Alloy([Ti], [], 1878, 1928)
        except ValueError as e:
            print(f"Empty Composition Test Passed: {e}")

        # Single Element Alloy
        single_element_alloy = Alloy([Ti], [1.0], 1878, 1928)
        print(f"Single Element Alloy Atomic Number: {single_element_alloy.atomic_number}")

        # Invalid Property Assignment
        try:
            Ti64.heat_conductivity = "invalid_value"  # type: ignore
        except TypeError as e:
            print(f"Invalid Property Assignment Test Passed: {e}")

        # Boundary Values for Temperatures
        boundary_alloy = Alloy([Ti, Al, V], [0.33, 0.33, 0.34], -273.15, 10000)
        print(f"Boundary Temperatures: Solidus={boundary_alloy.temperature_solidus}, Liquidus={boundary_alloy.temperature_liquidus}")

        # Properties Initialization
        default_alloy = Alloy([Ti], [1.0], 1878, 1928)
        print(f"Default Density: {default_alloy.density}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
