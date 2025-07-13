import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import sympy as sp

from pymatlib.core.elements import (ChemicalElement,
                                    interpolate_atomic_mass,
                                    interpolate_atomic_number,
                                    interpolate_boiling_temperature)
from pymatlib.core.exceptions import MaterialCompositionError, MaterialTemperatureError

logger = logging.getLogger(__name__)


@dataclass
class Material:
    """
    Represents a material with its composition and properties.

    This class supports both pure metals and alloys with comprehensive
    temperature validation and property calculation.
    """
    # Basic material information
    name: str
    material_type: str  # 'pure_metal' or 'alloy'
    elements: List[ChemicalElement]
    composition: Union[np.ndarray, List[float], Tuple]  # List of fractions summing to 1.0
    # Temperature properties for pure metals
    melting_temperature: Optional[sp.Float] = None
    boiling_temperature: Optional[sp.Float] = None
    # Temperature properties for alloys
    solidus_temperature: Optional[sp.Float] = None
    liquidus_temperature: Optional[sp.Float] = None
    initial_boiling_temperature: Optional[sp.Float] = None
    final_boiling_temperature: Optional[sp.Float] = None
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
    # Calculated properties (set during initialization)
    atomic_number: Optional[float] = field(default=None, init=False)
    atomic_mass: Optional[float] = field(default=None, init=False)
    # Pure metal ranges (based on periodic table data)
    MIN_MELTING_TEMP = 302.0  # Cesium (lowest solid metal at room conditions)
    MAX_MELTING_TEMP = 3695.0  # Tungsten (highest melting point)
    MIN_BOILING_TEMP = 630.0  # Mercury (lowest, but practical metals ~1000K)
    MAX_BOILING_TEMP = 6203.0  # Tungsten (highest boiling point)
    # Alloy ranges (engineering alloys have wider practical ranges)
    MIN_SOLIDUS_TEMP = 250.0  # Low-temperature solders (Bi-based alloys)
    MAX_SOLIDUS_TEMP = 2000.0  # High-temperature superalloys
    MIN_LIQUIDUS_TEMP = 300.0  # Low-melting alloys
    MAX_LIQUIDUS_TEMP = 2200.0  # Refractory alloys (Mo-Re, W-Re systems)

    def __post_init__(self) -> None:
        """
        Initialize and validate the material properties.
        Called automatically after the dataclass initialization.
        Validates composition and temperatures, then calculates derived properties.
        """
        logger.info("Initializing material: %s (type: %s)", self.name, self.material_type)
        self._validate_composition()
        self._validate_temperatures()
        self._calculate_properties()

    def _validate_composition(self) -> None:
        """Validate the material composition."""
        logger.debug("Validating composition for material: %s", self.name)
        if not self.elements:
            raise ValueError("Elements list cannot be empty")
        if len(self.elements) != len(self.composition):
            raise ValueError(
                f"Number of elements ({len(self.elements)}) must match composition length ({len(self.composition)})")
        if not np.isclose(sum(self.composition), 1.0, atol=1e-10):
            raise MaterialCompositionError(f"The sum of the composition array must be 1.0, got {sum(self.composition)}")
        if self.material_type == 'pure_metal' and len(self.elements) != 1:
            raise MaterialCompositionError(f"Pure metals must have exactly 1 element, got {len(self.elements)}")
        if self.material_type == 'alloy' and len(self.elements) < 2:
            raise MaterialCompositionError(f"Alloys must have at least 2 elements, got {len(self.elements)}")

    def _validate_temperatures(self) -> None:
        """Validate temperature properties based on material type."""
        logger.debug(f"Starting temperature validation for {self.material_type}: {self.name}")
        try:
            if self.material_type == 'pure_metal':
                self._validate_pure_metal_temperatures()
            elif self.material_type == 'alloy':
                self._validate_alloy_temperatures()
            else:
                raise MaterialTemperatureError(
                    f"Unknown material type: {self.material_type}. "
                    f"Must be 'pure_metal' or 'alloy'"
                )
            logger.debug(f"Temperature validation passed for {self.material_type}: {self.name}")
        except MaterialTemperatureError as e:
            logger.error(f"Temperature validation failed for {self.name}: {e}")
            raise

    def _validate_pure_metal_temperatures(self) -> None:
        """Validate temperature properties for pure metals."""
        # Check required temperatures are present
        if self.melting_temperature is None:
            raise MaterialTemperatureError("Pure metals must specify melting_temperature")
        if self.boiling_temperature is None:
            raise MaterialTemperatureError("Pure metals must specify boiling_temperature")
        # Validate temperature types
        if not isinstance(self.melting_temperature, sp.Float):
            raise MaterialTemperatureError(
                f"melting_temperature must be a SymPy Float, got {type(self.melting_temperature).__name__}"
            )
        if not isinstance(self.boiling_temperature, sp.Float):
            raise MaterialTemperatureError(
                f"boiling_temperature must be a SymPy Float, got {type(self.boiling_temperature).__name__}"
            )
        # Validate temperature ranges
        self._validate_temperature_value(
            self.melting_temperature, "melting_temperature",
            self.MIN_MELTING_TEMP, self.MAX_MELTING_TEMP
        )
        self._validate_temperature_value(
            self.boiling_temperature, "boiling_temperature",
            self.MIN_BOILING_TEMP, self.MAX_BOILING_TEMP
        )
        # Validate temperature relationships
        if float(self.melting_temperature) >= float(self.boiling_temperature):
            raise MaterialTemperatureError(
                f"melting_temperature ({float(self.melting_temperature)}K) must be less than "
                f"boiling_temperature ({float(self.boiling_temperature)}K)"
            )

    def _validate_alloy_temperatures(self) -> None:
        """Validate temperature properties for alloys."""
        # Check required temperatures are present
        required_temps = [
            ('solidus_temperature', self.solidus_temperature),
            ('liquidus_temperature', self.liquidus_temperature),
            ('initial_boiling_temperature', self.initial_boiling_temperature),
            ('final_boiling_temperature', self.final_boiling_temperature)
        ]
        missing_temps = [name for name, temp in required_temps if temp is None]
        if missing_temps:
            raise MaterialTemperatureError(
                f"Alloys must specify all temperature properties. Missing: {', '.join(missing_temps)}"
            )
        # Validate temperature types
        temp_checks = [
            ('solidus_temperature', self.solidus_temperature),
            ('liquidus_temperature', self.liquidus_temperature),
            ('initial_boiling_temperature', self.initial_boiling_temperature),
            ('final_boiling_temperature', self.final_boiling_temperature)
        ]
        for temp_name, temp_val in temp_checks:
            if temp_val is not None and not isinstance(temp_val, sp.Float):
                raise MaterialTemperatureError(
                    f"{temp_name} must be a SymPy Float, got {type(temp_val).__name__}"
                )
        # Validate temperature ranges
        self._validate_temperature_value(
            self.solidus_temperature, "solidus_temperature",
            self.MIN_SOLIDUS_TEMP, self.MAX_SOLIDUS_TEMP
        )
        self._validate_temperature_value(
            self.liquidus_temperature, "liquidus_temperature",
            self.MIN_LIQUIDUS_TEMP, self.MAX_LIQUIDUS_TEMP
        )
        self._validate_temperature_value(
            self.initial_boiling_temperature, "initial_boiling_temperature",
            self.MIN_BOILING_TEMP, self.MAX_BOILING_TEMP
        )
        self._validate_temperature_value(
            self.final_boiling_temperature, "final_boiling_temperature",
            self.MIN_BOILING_TEMP, self.MAX_BOILING_TEMP
        )
        # Validate temperature relationships
        # Solidus <= Liquidus
        if float(self.solidus_temperature) > float(self.liquidus_temperature):
            raise MaterialTemperatureError(
                f"solidus_temperature ({float(self.solidus_temperature)}K) must be less than or equal to "
                f"liquidus_temperature ({float(self.liquidus_temperature)}K)"
            )
        # Initial boiling <= Final boiling
        if float(self.initial_boiling_temperature) > float(self.final_boiling_temperature):
            raise MaterialTemperatureError(
                f"initial_boiling_temperature ({float(self.initial_boiling_temperature)}K) must be "
                f"less than or equal to final_boiling_temperature ({float(self.final_boiling_temperature)}K)"
            )
        # Liquidus < Initial boiling
        if float(self.liquidus_temperature) >= float(self.initial_boiling_temperature):
            raise MaterialTemperatureError(
                f"liquidus_temperature ({float(self.liquidus_temperature)}K) must be less than "
                f"initial_boiling_temperature ({float(self.initial_boiling_temperature)}K)"
            )

    @staticmethod
    def _validate_temperature_value(temperature: Union[float, sp.Float], temp_name: str,
                                    min_temp: Optional[float] = None, max_temp: Optional[float] = None) -> None:
        """Validate a single temperature value."""
        if temperature is None:
            raise MaterialTemperatureError(f"{temp_name} cannot be None")
        # Convert to float for validation
        temp_val = float(temperature)
        from pymatlib.data.constants import PhysicalConstants
        # Check absolute zero
        if temp_val <= PhysicalConstants.ABSOLUTE_ZERO:
            raise MaterialTemperatureError(
                f"{temp_name} must be above absolute zero, got {temp_val}K"
            )
        # Check range if specified
        if min_temp is not None and temp_val < min_temp:
            raise MaterialTemperatureError(
                f"{temp_name} {temp_val}K is below minimum allowed value ({min_temp}K)"
            )
        if max_temp is not None and temp_val > max_temp:
            raise MaterialTemperatureError(
                f"{temp_name} {temp_val}K is above maximum allowed value ({max_temp}K)"
            )

    def _calculate_properties(self) -> None:
        """Calculate interpolated atomic properties based on composition."""
        logger.debug("Calculating interpolated properties for %s", self.name)
        if self.material_type == 'pure_metal':
            # For pure metals, use the single element's properties
            element = self.elements[0]
            self.atomic_number = float(element.atomic_number)
            self.atomic_mass = float(element.atomic_mass)
            logger.debug("Pure metal properties - atomic_number: %.1f, atomic_mass: %.3f",
                         self.atomic_number, self.atomic_mass)
        elif self.material_type == 'alloy':
            # For alloys, calculate weighted averages
            self.atomic_number = interpolate_atomic_number(self.elements, self.composition)
            self.atomic_mass = interpolate_atomic_mass(self.elements, self.composition)
            logger.debug("Alloy properties - atomic_number: %.3f, atomic_mass: %.3f",
                         self.atomic_number, self.atomic_mass)

    def solidification_interval(self) -> Tuple[sp.Float, sp.Float]:
        """
        Get the solidification interval for alloys.
        Returns:
            Tuple of (solidus_temperature, liquidus_temperature)
        Raises:
            ValueError: If called on a pure metal
        """
        if self.material_type != 'alloy':
            raise ValueError("Solidification interval is only applicable to alloys")
        return self.solidus_temperature, self.liquidus_temperature

    def melting_point(self) -> sp.Float:
        """
        Get the melting point.

        For pure metals, returns melting_temperature.
        For alloys, returns solidus_temperature.
        Returns:
            Melting point temperature
        """
        if self.material_type == 'pure_metal':
            return self.melting_temperature
        else:
            return self.solidus_temperature

    def boiling_point(self) -> sp.Float:
        """
        Get the boiling point.

        For pure metals, returns boiling_temperature.
        For alloys, returns initial_boiling_temperature.
        Returns:
            Boiling point temperature
        """
        if self.material_type == 'pure_metal':
            return self.boiling_temperature
        else:
            return self.initial_boiling_temperature

    def __str__(self) -> str:
        """String representation of the material."""
        element_names = [element.name for element in self.elements]
        if self.material_type == 'pure_metal':
            return f"Pure Metal: {self.name} ({element_names[0]})"
        else:
            composition_str = ", ".join([f"{elem}: {comp:.3f}"
                                         for elem, comp in zip(element_names, self.composition)])
            return f"Alloy: {self.name} ({composition_str})"

    def __repr__(self) -> str:
        """Detailed representation of the material."""
        return (f"Material(name='{self.name}', material_type='{self.material_type}', "
                f"elements={len(self.elements)}, composition={self.composition})")
