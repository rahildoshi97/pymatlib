from dataclasses import dataclass
from typing import Final

@dataclass
class Constants:
    """
    A dataclass to store fundamental constants.
    Attributes:
        ROOM_TEMPERATURE (float): Room temperature in Kelvin.
        N_A (float): Avogadro's number, the number of constituent particles (usually atoms or molecules) in one mole of a given substance.
        AMU (float): Atomic mass unit in kilograms.
        E (float): Elementary charge, the electric charge carried by a single proton or the magnitude of the electric charge carried by a single electron.
        SPEED_OF_LIGHT (float): Speed of light in vacuum in meters per second.
    """
    ROOM_TEMPERATURE: float = 298.15  # Room temperature in Kelvin
    N_A: float = 6.022141e23  # Avogadro's number, /mol
    AMU: float = 1.660538e-27  # Atomic mass unit, kg
    E: float = 1.60217657e-19  # Elementary charge, C
    SPEED_OF_LIGHT: float = 0.299792458e9  # Speed of light, m/s

@dataclass(frozen=True)
class ProcessingConstants:
    """Processing constants used throughout the YAML parser."""
    # Tolerance and precision
    DEFAULT_TOLERANCE: Final[float] = 1e-8
    TEMPERATURE_EPSILON: Final[float] = 1e-8
    MONOTONICITY_THRESHOLD: Final[float] = 1e-8
    # Temperature limits
    ABSOLUTE_ZERO: Final[float] = 0.0
    MIN_TEMPERATURE: Final[float] = 0.0
    # Data validation
    MIN_DATA_POINTS: Final[int] = 2
    MIN_TEMPERATURE_POINTS: Final[int] = 2
    # Regression limits
    MAX_REGRESSION_SEGMENTS: Final[int] = 8
    WARNING_REGRESSION_SEGMENTS: Final[int] = 6
    DEFAULT_REGRESSION_SEED: Final[int] = 13579
    # Visualization
    DEFAULT_VISUALIZATION_POINTS: Final[int] = 1000
    TEMPERATURE_PADDING_FACTOR: Final[float] = 0.1
    STEP_FUNCTION_OFFSET: Final[float] = 100.0
    # File processing
    MAX_MISSING_VALUE_PERCENTAGE: Final[float] = 50.0
    # Default ranges
    DEFAULT_TEMP_LOWER: Final[float] = 273.15
    DEFAULT_TEMP_UPPER: Final[float] = 3000.0
    # Shared regex for temperature arithmetic
    TEMP_ARITHMETIC_REGEX: Final[str] = r'^(\w+)\s*([+-])\s*(\d+(?:\.\d+)?)$'

@dataclass(frozen=True)
class ErrorMessages:
    """Standardized error message templates."""
    TEMPERATURE_BELOW_ZERO: Final[str] = "Temperature must be above absolute zero ({min_temp}K), got {temp}K"
    INSUFFICIENT_DATA_POINTS: Final[str] = "Insufficient data points ({count}), minimum required: {min_points}"
    INVALID_PROPERTY_TYPE: Final[str] = "Invalid property type for '{prop_name}': {prop_type}"
    MISSING_DEPENDENCY: Final[str] = "Missing dependency '{dep}' for property '{prop_name}'"
    CIRCULAR_DEPENDENCY: Final[str] = "Circular dependency detected: {cycle_path}"

@dataclass(frozen=True)
class FileConstants:
    """File processing related constants."""
    SUPPORTED_EXTENSIONS: Final[tuple] = ('.csv', '.xlsx', '.txt')
    MAX_FILE_SIZE_MB: Final[int] = 100
    DEFAULT_ENCODING: Final[str] = 'utf-8'
    # Missing value indicators
    NA_VALUES: Final[tuple] = ('', ' ', '  ', '   ', 'nan', 'NaN', 'NULL', 'null', 'N/A', 'n/a', 'NA')
