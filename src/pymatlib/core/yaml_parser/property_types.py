import logging
from enum import auto, Enum
from typing import Any, Dict, Set, Union

import sympy as sp

from pymatlib.core.yaml_parser.yaml_keys import FILE_PATH_KEY, TEMPERATURE_HEADER_KEY, VALUE_HEADER_KEY, BOUNDS_KEY, \
    REGRESSION_KEY, TEMPERATURE_KEY, VALUE_KEY, EQUATION_KEY, CONSTANT_KEY, EXTRAPOLATE_KEY, SIMPLIFY_KEY, DEGREE_KEY, \
    SEGMENTS_KEY, PRE_KEY, POST_KEY

logger = logging.getLogger(__name__)

# --- Enum ---
class PropertyType(Enum):
    CONSTANT = auto()
    FILE = auto()
    KEY_VAL = auto()
    PIECEWISE_EQUATION = auto()
    COMPUTE = auto()
    INVALID = auto()

# --- Main Class ---
class PropertyTypeDetector:
    """Utility class for detecting property types from configuration values."""

    # --- Main API ---'
    @staticmethod
    def determine_property_type(prop_name: str, config: Any) -> PropertyType:
        """Determine the property type from its configuration."""
        logger.debug("""PropertyTypeDetector: determine_property_type:
            prop_name: %r
            config: %r""", prop_name, config)
        try:
            if isinstance(config, int):
                raise ValueError(f"Property '{prop_name}' must be defined as a float, got {config} of type {type(config).__name__}")
            if PropertyTypeDetector._check_constant_property(config):
                return PropertyType.CONSTANT
            # For dictionary configurations, try to identify the type and collect validation errors
            if isinstance(config, dict):
                validation_errors = {}
                if FILE_PATH_KEY in config:
                    try:
                        PropertyTypeDetector._check_file_property(config)
                        return PropertyType.FILE
                    except ValueError as e:
                        validation_errors['FILE'] = str(e)
                if TEMPERATURE_KEY in config and VALUE_KEY in config:
                    try:
                        PropertyTypeDetector._check_key_val_property(config)
                        return PropertyType.KEY_VAL
                    except ValueError as e:
                        validation_errors['KEY_VAL'] = str(e)
                if TEMPERATURE_KEY in config and EQUATION_KEY in config:
                    try:
                        PropertyTypeDetector._check_piecewise_equation_property(config)
                        return PropertyType.PIECEWISE_EQUATION
                    except ValueError as e:
                        validation_errors['PIECEWISE_EQUATION'] = str(e)
                if EQUATION_KEY in config:
                    try:
                        PropertyTypeDetector._check_compute_property(config)
                        return PropertyType.COMPUTE
                    except ValueError as e:
                        validation_errors['COMPUTE'] = str(e)
                if validation_errors:
                    error_message = next(iter(validation_errors.values()))
                    raise ValueError(f"Invalid property configuration \n -> {error_message}")
                # No specific errors found, but still invalid
                raise ValueError(f"Property has an  invalid format: {config}")
            if isinstance(config, str):
                try:
                    if PropertyTypeDetector._check_compute_property(config):
                        return PropertyType.COMPUTE
                except ValueError as e:
                    raise ValueError(f"Invalid COMPUTE property: {str(e)}") from e
            # If we get here, we couldn't determine the type
            return PropertyType.INVALID
        except Exception as e:
            logger.error(f"Failed to determine property type for {prop_name}: {e}")
            raise ValueError(f"Failed to determine property type for '{prop_name}' \n -> {str(e)}") from e

    # --- Type Checker Methods ---
    @staticmethod
    def _check_constant_property(value: Any) -> bool:
        """Check if the value is a numeric constant."""
        logger.debug("""PropertyTypeDetector: is_numeric:
            value: %r""", value)
        if isinstance(value, float):
            return True
        if isinstance(value, str):
            try:
                float(value)
                return '.' in value or 'e' in value.lower()
            except ValueError:
                return False
        return False

    @staticmethod
    def _check_file_property(value: Any) -> bool:
        """Check if the value is a file-based property definition."""
        logger.debug("""PropertyTypeDetector: _check_file_property:
            value: %r""", value)
        if isinstance(value, dict) and FILE_PATH_KEY in value:
            required_keys = {FILE_PATH_KEY, TEMPERATURE_HEADER_KEY, VALUE_HEADER_KEY, BOUNDS_KEY}
            optional_keys = {REGRESSION_KEY}
            PropertyTypeDetector._check_keys(value, required_keys, optional_keys, "FILE configuration")
            PropertyTypeDetector._check_bounds(value[BOUNDS_KEY], "FILE bounds")
            if REGRESSION_KEY in value:
                PropertyTypeDetector._check_regression(value[REGRESSION_KEY])
            return True
        return False

    @staticmethod
    def _check_key_val_property(value: Any) -> bool:
        """Check if the value is a key-value property definition."""
        logger.debug("""PropertyTypeDetector: _check_key_val_property:
            value: %r""", value)
        if isinstance(value, dict) and TEMPERATURE_KEY in value and VALUE_KEY in value:
            required_keys = {TEMPERATURE_KEY, VALUE_KEY, BOUNDS_KEY}
            optional_keys = {REGRESSION_KEY}
            PropertyTypeDetector._check_keys(value, required_keys, optional_keys, "KEY_VAL configuration")
            PropertyTypeDetector._check_bounds(value[BOUNDS_KEY], "KEY_VAL bounds")
            if REGRESSION_KEY in value:
                PropertyTypeDetector._check_regression(value[REGRESSION_KEY])
            return True
        return False

    @staticmethod
    def _check_piecewise_equation_property(value: Any) -> bool:
        """Check if the value is a piecewise equation property definition."""
        logger.debug("""PropertyTypeDetector: _check_piecewise_equation:
            value: %r""", value)
        if isinstance(value, dict) and TEMPERATURE_KEY in value and EQUATION_KEY in value:
            required_keys = {TEMPERATURE_KEY, EQUATION_KEY, BOUNDS_KEY}
            optional_keys = {REGRESSION_KEY}
            PropertyTypeDetector._check_keys(value, required_keys, optional_keys, "PIECEWISE_EQUATION configuration")
            PropertyTypeDetector._check_bounds(value[BOUNDS_KEY], "PIECEWISE_EQUATION bounds")
            if REGRESSION_KEY in value:
                PropertyTypeDetector._check_regression(value[REGRESSION_KEY])
            return True
        return False

    @staticmethod
    def _check_compute_property(value: Any) -> bool:
        """Check if the value is a computed (formula) property definition."""
        logger.debug("""PropertyTypeDetector: _check_compute_property:
            value: %r""", value)
        if isinstance(value, str):
            try:
                expr = sp.sympify(value)
                return len(expr.free_symbols) > 0 or not expr.is_number
            except (sp.SympifyError, ValueError, TypeError):
                math_operators = ['+', '-', '*', '/', '**', '(', ')', ' ']
                return any(op in value for op in math_operators)
        elif isinstance(value, dict) and EQUATION_KEY in value:
            required_keys = {EQUATION_KEY, BOUNDS_KEY}
            optional_keys = {REGRESSION_KEY}
            PropertyTypeDetector._check_keys(value, required_keys, optional_keys, "COMPUTE configuration")
            PropertyTypeDetector._check_bounds(value[BOUNDS_KEY], "COMPUTE bounds")
            if REGRESSION_KEY in value:
                PropertyTypeDetector._check_regression(value[REGRESSION_KEY])
            return True
        return False

    # --- Validation Helpers ---
    @staticmethod
    def _check_keys(value: Dict[str, Any], required_keys: Set[str], optional_keys: Set[str], context: str) -> None:
        """Validate dictionary keys with clear error messages."""
        logger.debug("""PropertyTypeDetector: _check_keys:
            value: %r
            required_keys: %r
            optional_keys: %r
            context: %r""", value, required_keys, optional_keys, context)
        if not isinstance(value, dict):
            raise ValueError(f"{context} must be a dictionary, got {type(value).__name__}")
        value_keys = set(value.keys())
        missing_keys = required_keys - value_keys
        if missing_keys:
            raise ValueError(f"Missing required keys in {context}: {missing_keys}")
        extra_keys = value_keys - required_keys - optional_keys
        if extra_keys:
            raise ValueError(f"Unexpected keys in {context}: {extra_keys}. Allowed keys are: {required_keys | optional_keys}")

    @staticmethod
    def _check_bounds(bounds: Any, context: str = "bound") -> None:
        """Validate bounds with clear error messages."""
        logger.debug("""PropertyTypeDetector: _check_bounds:
            bounds: %r
            context: %r""", bounds, context)
        if not isinstance(bounds, list):
            raise ValueError(f"{context} must be a list, got {type(bounds).__name__}")
        if len(bounds) != 2:
            raise ValueError(f"{context} must have exactly two elements, got {len(bounds)}")
        valid_bound_types = {CONSTANT_KEY, EXTRAPOLATE_KEY}
        if bounds[0] not in valid_bound_types:
            raise ValueError(f"Lower {context} type must be one of: {valid_bound_types}, got '{bounds[0]}'")
        if bounds[1] not in valid_bound_types:
            raise ValueError(f"Upper {context} type must be one of: {valid_bound_types}, got '{bounds[1]}'")

    @staticmethod
    def _check_regression(regression: Union[Dict[str, Any], Any]) -> None:
        """Validate the regression configuration with clear error messages."""
        logger.debug("""PropertyTypeDetector: _check_regression:
            regression: %r""", regression)
        if not isinstance(regression, dict):
            raise ValueError(f"Regression must be a dictionary, got {type(regression).__name__}")
        required_keys = {SIMPLIFY_KEY, DEGREE_KEY, SEGMENTS_KEY}
        optional_keys = set()
        missing_keys = required_keys - set(regression.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys in regression configuration: {missing_keys}")
        extra_keys = set(regression.keys()) - required_keys - optional_keys
        if extra_keys:
            raise ValueError(f"Unexpected keys in regression configuration: {extra_keys}")
        if not isinstance(regression[SIMPLIFY_KEY], str) or regression[SIMPLIFY_KEY] not in {PRE_KEY, POST_KEY}:
            raise ValueError(f"Invalid regression simplify type '{regression[SIMPLIFY_KEY]}'. Must be '{PRE_KEY}' or '{POST_KEY}'")
        if not isinstance(regression[DEGREE_KEY], int) or regression[DEGREE_KEY] < 1:
            raise ValueError(f"Regression degree must be a positive integer, got {regression[DEGREE_KEY]}")
        if not isinstance(regression[SEGMENTS_KEY], int) or regression[SEGMENTS_KEY] < 1:
            raise ValueError(f"Regression segments must be a positive integer >= 1, got {regression[SEGMENTS_KEY]}")
