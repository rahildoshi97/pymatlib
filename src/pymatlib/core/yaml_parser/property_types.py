import logging
from enum import auto, Enum
from typing import Any, Dict, Set, Union

import sympy as sp

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

    # --- Main API ---
    @staticmethod
    def determine_property_type(prop_name: str, config: Any) -> PropertyType:
        """Determine the property type from its configuration."""
        logger.debug("""PropertyTypeDetector: determine_property_type:
            prop_name: %r
            config: %r""", prop_name, config)
        try:
            if isinstance(config, int):
                raise ValueError(f"Property '{prop_name}' must be defined as a float, got {config} of type {type(config).__name__}")
            if PropertyTypeDetector.is_numeric(config):
                return PropertyType.CONSTANT
            elif PropertyTypeDetector.is_data_file(config):
                return PropertyType.FILE
            elif PropertyTypeDetector.is_key_val_property(config):
                return PropertyType.KEY_VAL
            elif PropertyTypeDetector.is_piecewise_equation(config):
                return PropertyType.PIECEWISE_EQUATION
            elif PropertyTypeDetector.is_compute_property(config):
                return PropertyType.COMPUTE
            else:
                return PropertyType.INVALID
        except Exception as e:
            logger.error(f"Failed to determine property type for {prop_name}: {e}")
            raise ValueError(f"Failed to determine property type \n -> {e}")

    # --- Type Checker Methods ---
    @staticmethod
    def is_numeric(value: Any) -> bool:
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
    def is_data_file(value: Union[str, Dict[str, Any]]) -> bool:
        """Check if the value is a file-based property definition."""
        logger.debug("""PropertyTypeDetector: is_data_file:
            value: %r""", value)
        if isinstance(value, str):
            return value.endswith(('.txt', '.csv', '.xlsx'))
        if isinstance(value, dict) and 'file' in value:
            required_keys = {'file', 'temp_col', 'prop_col', 'bounds'}
            optional_keys = {'regression'}
            PropertyTypeDetector.validate_keys(value, required_keys, optional_keys, "'FILE' config")
            PropertyTypeDetector.validate_bounds(value['bounds'], "'FILE' config bound")
            if 'regression' in value:
                PropertyTypeDetector.validate_regression(value['regression'])
            return True
        return False

    @staticmethod
    def is_key_val_property(value: Any) -> bool:
        """Check if the value is a key-value property definition."""
        logger.debug("""PropertyTypeDetector: is_key_val_property:
            value: %r""", value)
        required_keys = {'key', 'val', 'bounds'}
        optional_keys = {'regression'}
        try:
            PropertyTypeDetector.validate_keys(value, required_keys, optional_keys, "'KEY_VAL' config")
            PropertyTypeDetector.validate_bounds(value['bounds'], "'KEY_VAL' config bound")
            if 'regression' in value:
                PropertyTypeDetector.validate_regression(value['regression'])
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def is_piecewise_equation(value: Any) -> bool:
        """Check if the value is a piecewise equation property definition."""
        logger.debug("""PropertyTypeDetector: is_piecewise_equation:
            value: %r""", value)
        required_keys = {'temperature', 'equation', 'bounds'}
        optional_keys = {'regression'}
        try:
            PropertyTypeDetector.validate_keys(value, required_keys, optional_keys, "'PIECEWISE_EQUATION' config bound")
            PropertyTypeDetector.validate_bounds(value['bounds'], "'PIECEWISE_EQUATION' config bound")
            if 'regression' in value:
                PropertyTypeDetector.validate_regression(value['regression'])
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def is_compute_property(value: Any) -> bool:
        """Check if the value is a computed (formula) property definition."""
        logger.debug("""PropertyTypeDetector: is_compute_property:
            value: %r""", value)
        if isinstance(value, str):
            try:
                expr = sp.sympify(value)
                return len(expr.free_symbols) > 0 or not expr.is_number
            except (sp.SympifyError, ValueError, TypeError):
                math_operators = ['+', '-', '*', '/', '**', '(', ')', ' ']
                return any(op in value for op in math_operators)
        elif isinstance(value, dict) and 'equation' in value:
            required_keys = {'equation', 'bounds'}
            optional_keys = {'regression'}
            PropertyTypeDetector.validate_keys(value, required_keys, optional_keys, "'COMPUTE' Config")
            PropertyTypeDetector.validate_bounds(value['bounds'], "'COMPUTE' config bound")
            if 'regression' in value:
                PropertyTypeDetector.validate_regression(value['regression'])
            return True
        return False

    # --- Validation Helpers ---
    @staticmethod
    def validate_keys(value: Dict[str, Any], required_keys: Set[str], optional_keys: Set[str], context: str) -> None:
        """Validate that a dictionary has the required and only allowed keys."""
        logger.debug("""PropertyTypeDetector: validate_keys:
            value: %r
            required_keys: %r
            optional_keys: %r
            context: %r""", value, required_keys, optional_keys, context)
        if not isinstance(value, dict):
            raise ValueError(f"{context} must be a dictionary, got {type(value).__name__}")
        value_keys = set(value.keys())
        missing_keys = required_keys - value_keys
        if missing_keys:
            raise ValueError(f"Missing required keys for {context}: {missing_keys}")
        extra_keys = value_keys - required_keys - optional_keys
        if extra_keys:
            raise ValueError(f"Extra keys found in {context}: {extra_keys}. Allowed keys are: {required_keys | optional_keys}")

    @staticmethod
    def validate_bounds(bounds: Any, context: str = "bound") -> None:
        """Validate that bounds are a list of two valid types."""
        logger.debug("""PropertyTypeDetector: validate_bounds:
            bounds: %r
            context: %r""", bounds, context)
        if not isinstance(bounds, list):
            raise ValueError(f"{context}s must be a list, got {type(bounds).__name__}")
        if len(bounds) != 2:
            raise ValueError(f"{context} must have exactly two elements")
        valid_bound_types = {'constant', 'extrapolate'}
        if bounds[0] not in valid_bound_types:
            raise ValueError(f"Lower {context} type must be one of: {valid_bound_types}, got '{bounds[0]}'")
        if bounds[1] not in valid_bound_types:
            raise ValueError(f"Upper {context} type must be one of: {valid_bound_types}, got '{bounds[1]}'")

    @staticmethod
    def validate_regression(regression: Union[Dict[str, Any], Any]) -> None:
        """Validate the regression configuration."""
        logger.debug("""PropertyTypeDetector: validate_regression:
            regression: %r""", regression)
        if not isinstance(regression, dict):
            raise ValueError(f"Regression must be a dictionary, got {type(regression).__name__}")
        required_keys = {'simplify', 'degree', 'segments'}
        optional_keys = set()
        PropertyTypeDetector.validate_keys(regression, required_keys, optional_keys, "regression")
        if not isinstance(regression['simplify'], str) or regression['simplify'] not in {'pre', 'post'}:
            raise ValueError(f"Invalid regression simplify type '{regression['simplify']}'. Must be 'pre', or 'post'")
        if not isinstance(regression['degree'], int) or regression['degree'] < 1:
            raise ValueError("Regression degree must be a positive integer")
        if not isinstance(regression['segments'], int) or regression['segments'] < 1:
            raise ValueError("Regression segments must be an integer >= 1")
