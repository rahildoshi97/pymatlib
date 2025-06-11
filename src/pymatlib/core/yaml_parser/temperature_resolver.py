import logging
import numpy as np
import re
from typing import List, Union, Optional

from pymatlib.core.material import Material
from pymatlib.core.yaml_parser.data_handler import read_data_from_file
from pymatlib.core.yaml_parser.yaml_keys import MELTING_TEMPERATURE_KEY, BOILING_TEMPERATURE_KEY, \
    SOLIDUS_TEMPERATURE_KEY, LIQUIDUS_TEMPERATURE_KEY, INITIAL_BOILING_TEMPERATURE_KEY, FINAL_BOILING_TEMPERATURE_KEY, \
    FILE_PATH_KEY, TEMPERATURE_KEY, VALUE_KEY
from pymatlib.data.constants import ProcessingConstants

logger = logging.getLogger(__name__)

class TemperatureResolver:
    """Handles processing of different temperature definition formats in YAML configurations."""

    # --- Class Constants ---
    ABSOLUTE_ZERO = ProcessingConstants.ABSOLUTE_ZERO
    EPSILON = ProcessingConstants.TEMPERATURE_EPSILON
    MIN_POINTS = ProcessingConstants.MIN_TEMPERATURE_POINTS

    # Temperature reference mapping
    TEMPERATURE_REFERENCE_MAP = {
        MELTING_TEMPERATURE_KEY: 'melting_temperature',
        BOILING_TEMPERATURE_KEY: 'boiling_temperature',
        SOLIDUS_TEMPERATURE_KEY: 'solidus_temperature',
        LIQUIDUS_TEMPERATURE_KEY: 'liquidus_temperature',
        INITIAL_BOILING_TEMPERATURE_KEY: 'initial_boiling_temperature',
        FINAL_BOILING_TEMPERATURE_KEY: 'final_boiling_temperature'
    }

    # --- Main Public API ---
    @staticmethod
    def process_temperature_definition(temp_def: Union[List, str, int, float],
                                       n_values: Optional[int] = None,
                                       material: Optional[Material] = None) -> np.ndarray:
        """
        Process different temperature definition formats with optional material reference support.
        Args:
            temp_def: Temperature definition (list, string, numeric, or reference)
            n_values: Number of values (required for equidistant format)
            material: Material object for temperature reference resolution (optional)
        Returns:
            np.ndarray: Processed temperature array
        Examples:
            # Direct numeric value
            process_temperature_definition(500.0)  # Returns [500.0]

            # List of temperatures
            process_temperature_definition([300, 400, 500])  # Returns [300, 400, 500]

            # Equidistant format
            process_temperature_definition("(300, 50)", n_values=5)  # Returns [300, 350, 400, 450, 500]

            # Range format
            process_temperature_definition("(300, 500, 50)")  # Returns [300, 350, 400, 450, 500]

            # Temperature reference (requires material)
            process_temperature_definition("melting_temperature", material=material)
        """
        if isinstance(temp_def, list):
            return TemperatureResolver._process_temperature_list(temp_def, material)
        elif isinstance(temp_def, str):
            return TemperatureResolver._process_temperature_string(temp_def, n_values, material)
        elif isinstance(temp_def, (int, float)):
            temp_val = float(temp_def)
            if temp_val <= TemperatureResolver.ABSOLUTE_ZERO:
                raise ValueError(f"Temperature must be above absolute zero ({TemperatureResolver.ABSOLUTE_ZERO}K), got {temp_val}K")
            return np.array([temp_val], dtype=float)
        else:
            raise ValueError(f"Unsupported temperature definition format: {type(temp_def)}")

    @staticmethod
    def extract_from_config(prop_config: dict, material: Material) -> np.ndarray:
        """
        Extract temperature array from property configuration.
        Args:
            prop_config: Property configuration dictionary
            material: Material object for reference resolution
        Returns:
            np.ndarray: Temperature array extracted from configuration
        """
        # Handle FILE properties (need to re-read temperature data from file)
        if FILE_PATH_KEY in prop_config:
            try:
                temp_array, _ = read_data_from_file(prop_config)
                return temp_array
            except Exception as e:
                raise ValueError(f"Failed to extract temperature array from file: {str(e)}") from e
        # Handle properties with explicit temperature definitions
        if TEMPERATURE_KEY in prop_config:
            temp_def = prop_config[TEMPERATURE_KEY]
            n_values = len(prop_config[VALUE_KEY]) if VALUE_KEY in prop_config else None
            return TemperatureResolver.process_temperature_definition(temp_def, n_values, material)
        raise ValueError(f"Cannot extract temperature array: no temperature information in config")

    # --- Temperature Reference Resolution ---
    @staticmethod
    def resolve_temperature_reference(temp_ref: Union[str, float, int], material: Material) -> float:
        """
        Consolidated temperature reference resolution.
        Handles numeric values, simple references, and arithmetic expressions.
        Args:
            temp_ref: Temperature reference (string, float, or int)
            material: Material object for reference resolution
        Returns:
            float: Resolved temperature value
        """
        # Handle direct numeric values (int/float)
        if isinstance(temp_ref, (int, float)):
            result = float(temp_ref)
            if result <= TemperatureResolver.ABSOLUTE_ZERO:
                raise ValueError(f"Temperature must be above absolute zero, got {result}K")
            return result
        # Handle string-based definitions
        if isinstance(temp_ref, str):
            # Try direct numeric conversion first
            try:
                result = float(temp_ref)
                if result <= TemperatureResolver.ABSOLUTE_ZERO:
                    raise ValueError(f"Temperature must be above absolute zero, got {result}K")
                return result
            except ValueError:
                pass  # Not a numeric string, continue with reference resolution
            # Handle arithmetic expressions with temperature references
            if '+' in temp_ref or '-' in temp_ref:
                # match = re.match(r'(\w+)\s*([+-])\s*(\d+(?:\.\d+)?)', temp_ref.strip())
                match = re.match(ProcessingConstants.TEMP_ARITHMETIC_REGEX, temp_ref.strip())
                if match:
                    base_temp_name, operator, offset = match.groups()
                    base_temp = TemperatureResolver.get_temperature_value(base_temp_name, material)
                    offset_val = float(offset)
                    result = base_temp + offset_val if operator == '+' else base_temp - offset_val
                    return result
            # Direct temperature reference
            result = TemperatureResolver.get_temperature_value(temp_ref, material)
            return result
        raise ValueError(f"Unsupported temperature reference type: {type(temp_ref)} for value {temp_ref}")

    @staticmethod
    def get_temperature_value(temp_ref: Union[str, float, int], material: Material) -> float:
        """
        Enhanced helper function to get temperature value from material or direct numeric input.
        Replaces both get_transition_temperature and _get_temperature_value from common_utils.py
        Args:
            temp_ref: Temperature reference (string, float, or int)
            material: Material object for reference resolution
        Returns:
            float: Temperature value
        """
        # Handle direct numeric values
        if isinstance(temp_ref, (int, float)):
            result = float(temp_ref)
            if result <= TemperatureResolver.ABSOLUTE_ZERO:
                raise ValueError(f"Temperature must be above absolute zero, got {result}K")
            return result
        # Handle string references
        if isinstance(temp_ref, str):
            # Try numeric conversion first
            try:
                result = float(temp_ref)
                if result <= TemperatureResolver.ABSOLUTE_ZERO:
                    raise ValueError(f"Temperature must be above absolute zero, got {result}K")
                return result
            except ValueError:
                pass # Not numeric, try material reference
            # Material reference lookup
            if temp_ref in TemperatureResolver.TEMPERATURE_REFERENCE_MAP:
                attr_name = TemperatureResolver.TEMPERATURE_REFERENCE_MAP[temp_ref]
                if hasattr(material, attr_name):
                    return float(getattr(material, attr_name))
                else:
                    raise ValueError(f"Material does not have attribute '{attr_name}' for reference '{temp_ref}'")
            else:
                raise ValueError(f"Unknown temperature reference: '{temp_ref}'")
        raise ValueError(f"Unsupported temperature value type: {type(temp_ref)}")

    # --- Private Processing Methods ---
    @staticmethod
    def _process_temperature_list(temp_list: List[Union[int, float, str]],
                                  material: Optional[Material] = None) -> np.ndarray:
        """
        Process explicit temperature list with optional material reference support.
        Args:
            temp_list: List of temperature values or references
            material: Material object for reference resolution (optional)
        Returns:
            np.ndarray: Processed temperature array
        """
        try:
            temp_array = []
            for temp_item in temp_list:
                if isinstance(temp_item, str) and material is not None:
                    # Handle temperature references like "solidus_temperature", "melting_temperature + 50"
                    temp_array.append(TemperatureResolver.resolve_temperature_reference(temp_item, material))
                else:
                    temp_array.append(float(temp_item))
            temp_array = np.array(temp_array)
            # Validate all temperatures are above absolute zero
            if np.any(temp_array <= TemperatureResolver.ABSOLUTE_ZERO):
                invalid_temps = temp_array[temp_array <= TemperatureResolver.ABSOLUTE_ZERO]
                raise ValueError(f"Temperature must be above absolute zero ({TemperatureResolver.ABSOLUTE_ZERO}K), "
                                 f"got {invalid_temps}")
            return temp_array
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid temperature list: {temp_list} \n -> {str(e)}") from e

    @staticmethod
    def _process_temperature_string(temp_str: str,
                                    n_values: Optional[int] = None,
                                    material: Optional[Material] = None) -> np.ndarray:
        """
        Process string-based temperature definitions.
        Args:
            temp_str: String temperature definition
            n_values: Number of values (required for equidistant format)
            material: Material object for reference resolution (optional)
        Returns:
            np.ndarray: Processed temperature array
        """
        # Handle single temperature references
        if not (temp_str.startswith('(') and temp_str.endswith(')')):
            if material is not None:
                # Single temperature reference like "melting_temperature"
                return np.array([TemperatureResolver.resolve_temperature_reference(temp_str, material)])
            else:
                raise ValueError(f"String temperature definition must be enclosed in parentheses or require material for reference: {temp_str}")
        try:
            content = temp_str.strip('()')
            values = [x.strip() for x in content.split(',')]
            if len(values) == 2:
                # Format: (start, increment/decrement) - requires n_values
                return TemperatureResolver._process_equidistant_temperature(values, n_values)
            elif len(values) == 3:
                # Format: (start, stop, difference/points)
                return TemperatureResolver._process_range_temperature(values)
            else:
                raise ValueError(f"Temperature string must have 2 or 3 comma-separated values, got {len(values)}")
        except Exception as e:
            raise ValueError(f"Invalid temperature string format: {temp_str} \n -> {str(e)}") from e

    @staticmethod
    def _process_equidistant_temperature(values: List[str], n_values: Optional[int]) -> np.ndarray:
        """
        Process equidistant temperature format: (start, increment).
        Args:
            values: List containing [start, increment] as strings
            n_values: Number of values to generate
        Returns:
            np.ndarray: Generated temperature array
        """
        if n_values is None:
            raise ValueError("Number of values required for equidistant temperature format (start, increment/decrement)")
        if n_values < TemperatureResolver.MIN_POINTS:
            raise ValueError(f"Number of values must be at least {TemperatureResolver.MIN_POINTS}, got {n_values}")
        try:
            start, increment = float(values[0]), float(values[1])
            if abs(increment) <= TemperatureResolver.EPSILON:
                raise ValueError("Temperature increment/decrement cannot be zero")
            if start <= TemperatureResolver.ABSOLUTE_ZERO:
                raise ValueError(f"Start temperature must be above absolute zero ({TemperatureResolver.ABSOLUTE_ZERO}K), got {start}K")
            # Generate temperature array
            temp_array = np.array([start + i * increment for i in range(n_values)])
            # Validate all temperatures are above absolute zero
            if np.any(temp_array <= TemperatureResolver.ABSOLUTE_ZERO):
                invalid_temps = temp_array[temp_array <= TemperatureResolver.ABSOLUTE_ZERO]
                raise ValueError(f"Generated temperatures must be above absolute zero, got {invalid_temps}")
            return temp_array
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid equidistant temperature format: ({values[0]}, {values[1]}) \n -> {str(e)}") from e

    @staticmethod
    def _process_range_temperature(values: List[str]) -> np.ndarray:
        """
        Process range temperature format: (start, stop, difference/points).
        Args:
            values: List containing [start, stop, step_or_points] as strings
        Returns:
            np.ndarray: Generated temperature array
        """
        try:
            start, stop = float(values[0]), float(values[1])
            # Validate temperatures
            if start <= TemperatureResolver.ABSOLUTE_ZERO or stop <= TemperatureResolver.ABSOLUTE_ZERO:
                raise ValueError(f"Temperatures must be above absolute zero ({TemperatureResolver.ABSOLUTE_ZERO}K), got start={start}K, stop={stop}K")
            # Parse third parameter (could be step size or number of points)
            third_param_str = values[2].strip()
            third_param = float(third_param_str)
            is_integer_format = (
                # Check if string represents an integer (no decimal point)
                    '.' not in third_param_str and
                    'e' not in third_param_str.lower() and
                    third_param == int(third_param) and
                    0 < third_param <= 1000
            )
            # Determine if it's number of points or step size
            if is_integer_format:
                # Likely number of points (reasonable upper limit)
                n_points = int(third_param)
                if n_points < TemperatureResolver.MIN_POINTS:
                    raise ValueError(f"Number of points must be at least {TemperatureResolver.MIN_POINTS}, got {n_points}")
                temp_array = np.linspace(start, stop, n_points)
            else:
                # Step size
                if abs(third_param) <= TemperatureResolver.EPSILON:
                    raise ValueError("Temperature step cannot be zero")
                if (start < stop and third_param <= 0) or (start > stop and third_param >= 0):
                    raise ValueError("Step sign must match range direction")
                if abs(third_param) > abs(stop - start):
                    raise ValueError(f"Absolute value of step ({abs(third_param)}) is too large for the range. It should be <= {abs(stop - start)}")
                temp_array = np.arange(start, stop + third_param/2, third_param)
            return temp_array
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid range temperature format: ({', '.join(values)}) \n -> {str(e)}") from e

    # --- Validation Methods ---
    @staticmethod
    def validate_temperature_array(temp_array: np.ndarray, context: str = "") -> None:
        """
        Validate a temperature array for common issues.
        Args:
            temp_array: Temperature array to validate
            context: Context string for error messages
        """
        if len(temp_array) == 0:
            raise ValueError(f"Temperature array is empty{' for ' + context if context else ''}")
        if len(temp_array) < TemperatureResolver.MIN_POINTS:
            raise ValueError(f"Temperature array must have at least {TemperatureResolver.MIN_POINTS} points, got {len(temp_array)}{' for ' + context if context else ''}")
        if np.any(temp_array <= TemperatureResolver.ABSOLUTE_ZERO):
            invalid_temps = temp_array[temp_array <= TemperatureResolver.ABSOLUTE_ZERO]
            raise ValueError(f"All temperatures must be above absolute zero ({TemperatureResolver.ABSOLUTE_ZERO}K), got {invalid_temps}{' for ' + context if context else ''}")
        if not np.all(np.isfinite(temp_array)):
            raise ValueError(f"Temperature array contains non-finite values{' for ' + context if context else ''}")
