import numpy as np
from typing import List, Tuple
from dataclasses import dataclass, field
from pymatlib.core.elements import (ChemicalElement,
                                    interpolate_atomic_mass,
                                    interpolate_atomic_number,
                                    interpolate_temperature_boil)
from pymatlib.core.typedefs import ArrayTypes, PropertyTypes


class AlloyCompositionError(ValueError):
    """Exception raised when alloy composition validation fails."""
    pass

class AlloyTemperatureError(ValueError):
    """Exception raised when alloy temperature validation fails."""
    pass

'''class PropertyTypeChecker:
    """
    A descriptor class for handling property type checking.
    
    Args:
        name (str): Name of the property to check.
    
    Raises:
        TypeError: If the value being set is not None and not of type PropertyTypes.
    """
    def __init__(self, name: str):
        self.private_name = f'_{name}'

    def __get__(self, obj, objtype=None):
        """Get the property value."""
        if obj is None:
            return self
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        """
        Set the property value with type checking.
        
        Args:
            obj: The object instance.
            value: The value to set.
            
        Raises:
            TypeError: If value is not None and not of type PropertyTypes.
        """
        if value is not None and not isinstance(value, get_args(PropertyTypes)):
            raise TypeError(f"{self.private_name[1:]} must be of type PropertyTypes (float or MaterialProperty)")
        setattr(obj, self.private_name, value)'''

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

    Raises:
        AlloyCompositionError: If the composition fractions don't sum to 1.0.
        AlloyTemperatureError: If solidus temperature is greater than liquidus temperature
            or temperatures are out of the general valid range for alloys.
        ValueError: If elements list is empty or composition length doesn't match elements length.
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
    base_temperature: float = None
    base_density: float = None
    density: PropertyTypes = None
    dynamic_viscosity: PropertyTypes = None
    energy_density: PropertyTypes = None
    energy_density_solidus: float = None
    energy_density_liquidus: float = None
    energy_density_temperature_array: np.ndarray = field(default_factory=lambda: np.array([]))
    energy_density_array: np.ndarray = field(default_factory=lambda: np.array([]))
    heat_capacity: PropertyTypes = None
    heat_conductivity: PropertyTypes = None
    kinematic_viscosity: PropertyTypes = None
    latent_heat_of_fusion: PropertyTypes = None
    latent_heat_of_vaporization: PropertyTypes = None
    surface_tension: PropertyTypes = None
    temperature_array: np.ndarray = field(default_factory=lambda: np.array([]))
    thermal_diffusivity: PropertyTypes = None
    thermal_expansion_coefficient: PropertyTypes = None

    '''# Private fields for properties
    _density: PropertyTypes = field(default=None, init=False, repr=False)
    _dynamic_viscosity: PropertyTypes = field(default=None, init=False, repr=False)
    _heat_capacity: PropertyTypes = field(default=None, init=False, repr=False)
    _heat_conductivity: PropertyTypes = field(default=None, init=False, repr=False)
    _kinematic_viscosity: PropertyTypes = field(default=None, init=False, repr=False)
    _latent_heat_of_fusion: PropertyTypes = field(default=None, init=False, repr=False)
    _latent_heat_of_vaporization: PropertyTypes = field(default=None, init=False, repr=False)
    _surface_tension: PropertyTypes = field(default=None, init=False, repr=False)
    _thermal_diffusivity: PropertyTypes = field(default=None, init=False, repr=False)
    _thermal_expansion_coefficient: PropertyTypes = field(default=None, init=False, repr=False)
    
    # Property descriptors
    density = PropertyTypeChecker("density")
    dynamic_viscosity = PropertyTypeChecker("dynamic_viscosity")
    heat_capacity = PropertyTypeChecker("heat_capacity")
    heat_conductivity = PropertyTypeChecker("heat_conductivity")
    kinematic_viscosity = PropertyTypeChecker("kinematic_viscosity")
    latent_heat_of_fusion = PropertyTypeChecker("latent_heat_of_fusion")
    latent_heat_of_vaporization = PropertyTypeChecker("latent_heat_of_vaporization")
    surface_tension = PropertyTypeChecker("surface_tension")
    thermal_diffusivity = PropertyTypeChecker("thermal_diffusivity")
    thermal_expansion_coefficient = PropertyTypeChecker("thermal_expansion_coefficient")'''

    def solidification_interval(self) -> Tuple[float, float]:
        """
        Calculate the solidification interval based on solidus and liquidus temperature of the alloy.

        Returns:
            Tuple[float, float]: A tuple containing the solidus and liquidus temperatures.
        """
        return self.temperature_solidus, self.temperature_liquidus

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
        if self.temperature_solidus > self.temperature_liquidus:
            raise AlloyTemperatureError("The solidus temperature must be less than or equal to the liquidus temperature.")
        if not (450 <= self.temperature_solidus <= 1900):
            raise AlloyTemperatureError(f"Solidus temperature {self.temperature_solidus} K is out of range (450 K - 1900 K).")
        if not (600 <= self.temperature_liquidus <= 2000):
            raise AlloyTemperatureError(f"Liquidus temperature {self.temperature_liquidus} K is out of range (600 K - 2000 K).")

    def _calculate_properties(self) -> None:
        """Calculate derived properties based on composition."""
        self.atomic_number = interpolate_atomic_number(self.elements, self.composition)
        self.atomic_mass = interpolate_atomic_mass(self.elements, self.composition)
        self.temperature_boil = interpolate_temperature_boil(self.elements, self.composition)

    def __post_init__(self) -> None:
        """
        Initialize and validate the alloy properties.

        Called automatically after the dataclass initialization.
        Validates composition and temperatures, then calculates derived properties.
        """
        self._validate_composition()
        self._validate_temperatures()
        self._calculate_properties()