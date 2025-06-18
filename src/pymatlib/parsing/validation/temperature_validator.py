import logging
from typing import Optional, Union
import sympy as sp
from pymatlib.core.materials import Material, MaterialTemperatureError

logger = logging.getLogger(__name__)

class TemperatureValidator:
    """Centralized temperature validation for materials."""
    # Temperature range constants (in Kelvin)
    ABSOLUTE_ZERO = 0.0
    # Pure metal ranges (based on periodic table data)
    MIN_MELTING_TEMP = 302.0    # Cesium (lowest solid metal at room conditions)
    MAX_MELTING_TEMP = 3695.0   # Tungsten (highest melting point)
    MIN_BOILING_TEMP = 630.0    # Mercury (lowest, but practical metals ~1000K)
    MAX_BOILING_TEMP = 6203.0   # Tungsten (highest boiling point)
    # Alloy ranges (engineering alloys have wider practical ranges)
    MIN_SOLIDUS_TEMP = 250.0    # Low-temperature solders (Bi-based alloys)
    MAX_SOLIDUS_TEMP = 2000.0   # High-temperature superalloys
    MIN_LIQUIDUS_TEMP = 300.0   # Low-melting alloys
    MAX_LIQUIDUS_TEMP = 2200.0  # Refractory alloys (Mo-Re, W-Re systems)

    @staticmethod
    def validate_material_temperatures(material: Material) -> None:
        """Main entry point for temperature validation."""
        logger.debug(f"Starting temperature validation for {material.material_type}: {material.name}")
        try:
            if material.material_type == 'pure_metal':
                TemperatureValidator.validate_pure_metal_temperatures(material)
            elif material.material_type == 'alloy':
                TemperatureValidator.validate_alloy_temperatures(material)
            else:
                raise MaterialTemperatureError(
                    f"Unknown material type: {material.material_type}. "
                    f"Must be 'pure_metal' or 'alloy'"
                )
            logger.debug(f"Temperature validation passed for {material.material_type}: {material.name}")
        except MaterialTemperatureError as e:
            logger.error(f"Temperature validation failed for {material.name}: {e}")
            raise

    @staticmethod
    def validate_pure_metal_temperatures(material: Material) -> None:
        """
        Consolidated validation logic for pure metal temperature properties.
        Args:
            material: Material instance to validate
        Raises:
            MaterialTemperatureError: If any temperature validation fails
        """
        # Check required temperatures are present
        TemperatureValidator._validate_required_pure_metal_temperatures(material)
        # Validate temperature types
        TemperatureValidator._validate_pure_metal_temperature_types(material)
        # Validate temperature relationships
        TemperatureValidator._validate_pure_metal_temperature_relationships(material)
        # Validate temperature ranges
        TemperatureValidator._validate_pure_metal_temperature_ranges(material)

    @staticmethod
    def validate_alloy_temperatures(material: Material) -> None:
        """
        Consolidated validation logic for alloy temperature properties.
        Args:
            material: Material instance to validate
        Raises:
            MaterialTemperatureError: If any temperature validation fails
        """
        # Check required temperatures are present
        TemperatureValidator._validate_required_alloy_temperatures(material)
        # Validate temperature types
        TemperatureValidator._validate_alloy_temperature_types(material)
        # Validate temperature relationships
        TemperatureValidator._validate_alloy_temperature_relationships(material)
        # Validate temperature ranges
        TemperatureValidator._validate_alloy_temperature_ranges(material)

    @staticmethod
    def validate_temperature_value(
            temperature: Union[float, sp.Float],
            temp_name: str,
            min_temp: Optional[float] = None,
            max_temp: Optional[float] = None) -> None:
        """
        Validate a single temperature value.
        Args:
            temperature: Temperature value to validate
            temp_name: Name of the temperature for error messages
            min_temp: Minimum allowed temperature (optional)
            max_temp: Maximum allowed temperature (optional)
        Raises:
            MaterialTemperatureError: If temperature validation fails
        """
        if temperature is None:
            raise MaterialTemperatureError(f"{temp_name} cannot be None")
        # Convert to float for validation
        temp_val = float(temperature)
        # Check absolute zero
        if temp_val <= TemperatureValidator.ABSOLUTE_ZERO:
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

    @staticmethod
    def validate_temperature_range(
            temp1: Union[float, sp.Float],
            temp2: Union[float, sp.Float],
            temp1_name: str,
            temp2_name: str,
            allow_equal: bool = True) -> None:
        """
        Validate that two temperatures have the correct relationship.
        Args:
            temp1: First temperature (should be lower)
            temp2: Second temperature (should be higher)
            temp1_name: Name of first temperature
            temp2_name: Name of second temperature
            allow_equal: Whether temperatures can be equal
        Raises:
            MaterialTemperatureError: If temperature relationship is invalid
        """
        val1 = float(temp1)
        val2 = float(temp2)
        if allow_equal:
            if val1 > val2:
                raise MaterialTemperatureError(
                    f"{temp1_name} ({val1}K) must be less than or equal to "
                    f"{temp2_name} ({val2}K)"
                )
        else:
            if val1 >= val2:
                raise MaterialTemperatureError(
                    f"{temp1_name} ({val1}K) must be less than "
                    f"{temp2_name} ({val2}K)"
                )

    # Private helper methods for pure metal validation
    @staticmethod
    def _validate_required_pure_metal_temperatures(material: Material) -> None:
        """Check that all required pure metal temperatures are present."""
        if material.melting_temperature is None:
            raise MaterialTemperatureError("Pure metals must specify melting_temperature")
        if material.boiling_temperature is None:
            raise MaterialTemperatureError("Pure metals must specify boiling_temperature")

    @staticmethod
    def _validate_pure_metal_temperature_types(material: Material) -> None:
        """Validate that pure metal temperatures are correct types."""
        if (material.melting_temperature is not None and
                not isinstance(material.melting_temperature, sp.Float)):
            raise MaterialTemperatureError(
                f"melting_temperature must be a SymPy Float, got {type(material.melting_temperature).__name__}"
            )
        if (material.boiling_temperature is not None and
                not isinstance(material.boiling_temperature, sp.Float)):
            raise MaterialTemperatureError(
                f"boiling_temperature must be a SymPy Float, got {type(material.boiling_temperature).__name__}"
            )

    @staticmethod
    def _validate_pure_metal_temperature_relationships(material: Material) -> None:
        """Validate temperature relationships for pure metals."""
        if (material.melting_temperature is not None and
                material.boiling_temperature is not None):
            TemperatureValidator.validate_temperature_range(
                material.melting_temperature,
                material.boiling_temperature,
                "melting_temperature",
                "boiling_temperature",
                allow_equal=False
            )

    @staticmethod
    def _validate_pure_metal_temperature_ranges(material: Material) -> None:
        """Validate that pure metal temperatures are within reasonable ranges."""
        if material.melting_temperature is not None:
            TemperatureValidator.validate_temperature_value(
                material.melting_temperature,
                "melting_temperature",
                TemperatureValidator.MIN_MELTING_TEMP,
                TemperatureValidator.MAX_MELTING_TEMP
            )
        if material.boiling_temperature is not None:
            TemperatureValidator.validate_temperature_value(
                material.boiling_temperature,
                "boiling_temperature",
                TemperatureValidator.MIN_BOILING_TEMP,
                TemperatureValidator.MAX_BOILING_TEMP
            )

    # Private helper methods for alloy validation
    @staticmethod
    def _validate_required_alloy_temperatures(material: Material) -> None:
        """Check that all required alloy temperatures are present."""
        required_temps = [
            ('solidus_temperature', material.solidus_temperature),
            ('liquidus_temperature', material.liquidus_temperature),
            ('initial_boiling_temperature', material.initial_boiling_temperature),
            ('final_boiling_temperature', material.final_boiling_temperature)
        ]
        missing_temps = [name for name, temp in required_temps if temp is None]
        if missing_temps:
            raise MaterialTemperatureError(
                f"Alloys must specify all temperature properties. Missing: {', '.join(missing_temps)}"
            )

    @staticmethod
    def _validate_alloy_temperature_types(material: Material) -> None:
        """Validate that alloy temperatures are correct types."""
        temp_checks = [
            ('solidus_temperature', material.solidus_temperature),
            ('liquidus_temperature', material.liquidus_temperature),
            ('initial_boiling_temperature', material.initial_boiling_temperature),
            ('final_boiling_temperature', material.final_boiling_temperature)
        ]
        for temp_name, temp_val in temp_checks:
            if temp_val is not None and not isinstance(temp_val, sp.Float):
                raise MaterialTemperatureError(
                    f"{temp_name} must be a SymPy Float, got {type(temp_val).__name__}"
                )

    @staticmethod
    def _validate_alloy_temperature_relationships(material: Material) -> None:
        """Validate temperature relationships for alloys."""
        # Solidus <= Liquidus
        TemperatureValidator.validate_temperature_range(
            material.solidus_temperature,
            material.liquidus_temperature,
            "solidus_temperature",
            "liquidus_temperature",
            allow_equal=True
        )
        # Initial boiling <= Final boiling (if both present)
        if (material.initial_boiling_temperature is not None and
                material.final_boiling_temperature is not None):
            TemperatureValidator.validate_temperature_range(
                material.initial_boiling_temperature,
                material.final_boiling_temperature,
                "initial_boiling_temperature",
                "final_boiling_temperature",
                allow_equal=True
            )
        # Liquidus < Initial boiling (if present)
        if material.initial_boiling_temperature is not None:
            TemperatureValidator.validate_temperature_range(
                material.liquidus_temperature,
                material.initial_boiling_temperature,
                "liquidus_temperature",
                "initial_boiling_temperature",
                allow_equal=False
            )

    @staticmethod
    def _validate_alloy_temperature_ranges(material: Material) -> None:
        """Validate that alloy temperatures are within reasonable ranges."""
        TemperatureValidator.validate_temperature_value(
            material.solidus_temperature,
            "solidus_temperature",
            TemperatureValidator.MIN_SOLIDUS_TEMP,
            TemperatureValidator.MAX_SOLIDUS_TEMP
        )
        TemperatureValidator.validate_temperature_value(
            material.liquidus_temperature,
            "liquidus_temperature",
            TemperatureValidator.MIN_LIQUIDUS_TEMP,
            TemperatureValidator.MAX_LIQUIDUS_TEMP
        )
        if material.initial_boiling_temperature is not None:
            TemperatureValidator.validate_temperature_value(
                material.initial_boiling_temperature,
                "initial_boiling_temperature",
                TemperatureValidator.MIN_BOILING_TEMP,
                TemperatureValidator.MAX_BOILING_TEMP
            )
        if material.final_boiling_temperature is not None:
            TemperatureValidator.validate_temperature_value(
                material.final_boiling_temperature,
                "final_boiling_temperature",
                TemperatureValidator.MIN_BOILING_TEMP,
                TemperatureValidator.MAX_BOILING_TEMP
            )
