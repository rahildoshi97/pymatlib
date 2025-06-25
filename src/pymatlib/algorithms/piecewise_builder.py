import logging
from typing import Dict, List, Union
import numpy as np
import sympy as sp

from pymatlib.algorithms.interpolation import ensure_ascending_order
from pymatlib.algorithms.regression_processor import RegressionProcessor
from pymatlib.data.constants import ProcessingConstants
from pymatlib.parsing.config.yaml_keys import CONSTANT_KEY, EXTRAPOLATE_KEY, BOUNDS_KEY, PRE_KEY

logger = logging.getLogger(__name__)


class PiecewiseBuilder:
    """Centralized piecewise function creation with different strategies."""

    @staticmethod
    def build_from_data(temp_array: np.ndarray, prop_array: np.ndarray,
                        T: sp.Symbol, config: Dict, prop_name: str) -> sp.Piecewise:
        """
        Main entry point for data-based piecewise creation.
        Args:
            temp_array: Temperature data points
            prop_array: Property values corresponding to temperatures
            T: Temperature symbol for the piecewise function
            config: Configuration dictionary containing bounds and regression settings
            prop_name: Name of the property (for logging and error messages)
        Returns:
            sp.Piecewise: Symbolic piecewise function
        """
        # Validate input arrays
        if temp_array is None or prop_array is None:
            raise ValueError(f"Temperature and property arrays cannot be None for '{prop_name}'")
        if len(temp_array) != len(prop_array):
            raise ValueError(f"Temperature and property arrays must have same length for '{prop_name}'")
        if len(temp_array) == 0:
            raise ValueError(f"Empty data arrays provided for '{prop_name}'")
        try:
            # Ensure ascending order (handles both ascending and descending input)
            temp_array, prop_array = ensure_ascending_order(temp_array, prop_array)
            # Extract configuration
            lower_bound_type, upper_bound_type = config[BOUNDS_KEY]
            T_standard = sp.Symbol('T')
            # Check for regression configuration
            has_regression, simplify_type, degree, segments = RegressionProcessor.process_regression_params(
                config, prop_name, len(temp_array))
            # Create piecewise function based on regression settings
            if has_regression and simplify_type == PRE_KEY:
                pw_result = PiecewiseBuilder._build_with_regression(
                    temp_array, prop_array, T_standard, lower_bound_type, upper_bound_type,
                    degree, segments)
            else:
                pw_result = PiecewiseBuilder._build_without_regression(
                    temp_array, prop_array, T_standard, lower_bound_type, upper_bound_type)
            # Handle symbol substitution if needed
            if isinstance(T, sp.Symbol) and str(T) != 'T':
                pw_result = pw_result.subs(T_standard, T)
            return pw_result
        except Exception as e:
            logger.error(f"Failed to build piecewise from data for '{prop_name}': {e}", exc_info=True)
            raise ValueError(f"Failed building piecewise from data for '{prop_name}': {str(e)}") from e

    @staticmethod
    def build_from_formulas(temp_points: np.ndarray, equations: List[Union[str, sp.Expr]],
                            T: sp.Symbol, lower_bound_type: str = CONSTANT_KEY,
                            upper_bound_type: str = CONSTANT_KEY) -> sp.Piecewise:
        """
        Create piecewise from symbolic equations.
        Args:
            temp_points: Temperature breakpoints
            equations: List of symbolic expressions (strings or SymPy expressions)
            T: Temperature symbol
            lower_bound_type: Boundary behavior below first breakpoint
            upper_bound_type: Boundary behavior above last breakpoint
        Returns:
            sp.Piecewise: Symbolic piecewise function
        """
        logger.debug(f"Building piecewise function from {len(equations)} formulas and {len(temp_points)} breakpoints")
        if len(equations) != len(temp_points) - 1:
            raise ValueError(
                f"Number of equations ({len(equations)}) must be one less than temperature/break points ({len(temp_points)})")
        if len(temp_points) < 2:
            raise ValueError("At least 2 temperature points required for piecewise equations")
        try:
            temp_points = np.asarray(temp_points, dtype=float)
            # Parse equations into SymPy expressions using the provided symbol T
            parsed_equations = []
            for i, eqn_str in enumerate(equations):
                try:
                    expr = sp.sympify(eqn_str)
                    # Validate that only T symbol is used
                    for symbol in expr.free_symbols:
                        if str(symbol) != str(T):
                            raise ValueError(
                                f"Invalid symbol '{symbol}' in equation '{eqn_str}'. Only '{T}' is allowed.")
                    parsed_equations.append(expr)
                except Exception as e:
                    raise ValueError(f"Failed to parse equation {i + 1}: '{eqn_str}': {e}")
            # Special case: single expression with extrapolation at both ends
            if (len(parsed_equations) == 1 and
                    lower_bound_type == EXTRAPOLATE_KEY and
                    upper_bound_type == EXTRAPOLATE_KEY):
                logger.warning(
                    "Using a single expression with extrapolation at both ends. "
                    "Consider simplifying your YAML definition to use a direct equation."
                )
                # Return as Piecewise for consistency
                # Force Piecewise by creating a trivial condition
                return sp.Piecewise((parsed_equations[0], T >= -sp.oo))  # Always true, but explicit
            # Build piecewise conditions for multiple equations or different boundary types
            conditions = []
            # Handle lower bound
            if lower_bound_type == CONSTANT_KEY:
                conditions.append((parsed_equations[0].subs(T, temp_points[0]), T < temp_points[0]))
            # Handle intervals (including special cases for first and last)
            for i, expr in enumerate(parsed_equations):
                if i == 0 and lower_bound_type == EXTRAPOLATE_KEY:
                    # First segment with extrapolation
                    conditions.append((expr, T < temp_points[i + 1]))
                elif i == len(parsed_equations) - 1 and upper_bound_type == EXTRAPOLATE_KEY:
                    # Last segment with extrapolation
                    conditions.append((expr, T >= temp_points[i]))
                else:  # Regular interval
                    conditions.append((expr, sp.And(T >= temp_points[i], T < temp_points[i + 1])))
            # Handle upper bound
            if upper_bound_type == CONSTANT_KEY:
                conditions.append((parsed_equations[-1].subs(T, temp_points[-1]), T >= temp_points[-1]))
            return sp.Piecewise(*conditions)
        except Exception as e:
            logger.error(f"Failed to build piecewise from formulas: {e}", exc_info=True)
            raise ValueError(f"Failed building piecewise from formulas: {str(e)}") from e

    @staticmethod
    def _build_without_regression(temp_array: np.ndarray, prop_array: np.ndarray,
                                  T: sp.Symbol, lower: str, upper: str) -> sp.Piecewise:
        """
        Create basic linear interpolation piecewise function.
        Args:
            temp_array: Temperature data points (must be sorted)
            prop_array: Property values
            T: Temperature symbol
            lower: Lower boundary behavior ('constant' or 'extrapolate')
            upper: Upper boundary behavior ('constant' or 'extrapolate')
        Returns:
            sp.Piecewise: Linear interpolation piecewise function
        """
        conditions = []
        # Handle lower bound (T < temp_array[0])
        if lower == CONSTANT_KEY:
            lower_expr = prop_array[0]
        else:  # extrapolate
            if len(temp_array) > 1:
                slope = (prop_array[1] - prop_array[0]) / (temp_array[1] - temp_array[0])
                lower_expr = prop_array[0] + slope * (T - temp_array[0])
            else:
                lower_expr = prop_array[0]
        conditions.append((lower_expr, T < temp_array[0]))
        # Handle main interpolation segments
        for i in range(len(temp_array) - 1):
            slope = (prop_array[i + 1] - prop_array[i]) / (temp_array[i + 1] - temp_array[i])
            expr = prop_array[i] + slope * (T - temp_array[i])
            condition = sp.And(T >= temp_array[i], T < temp_array[i + 1])
            conditions.append((expr, condition))
        # Handle upper bound (T >= temp_array[-1])
        if upper == CONSTANT_KEY:
            upper_expr = prop_array[-1]
        else:  # extrapolate
            if len(temp_array) > 1:
                slope = (prop_array[-1] - prop_array[-2]) / (temp_array[-1] - temp_array[-2])
                upper_expr = prop_array[-1] + slope * (T - temp_array[-1])
            else:
                upper_expr = prop_array[-1]
        conditions.append((upper_expr, T >= temp_array[-1]))
        return sp.Piecewise(*conditions)

    @staticmethod
    def _build_with_regression(temp_array: np.ndarray, prop_array: np.ndarray,
                               T: sp.Symbol, lower: str, upper: str,
                               degree: int, segments: int) -> sp.Piecewise:
        """
        Create piecewise with regression.
        This delegates to RegressionProcessor but provides a unified interface.
        Args:
            temp_array: Temperature data points
            prop_array: Property values
            T: Temperature symbol
            lower: Lower boundary behavior
            upper: Upper boundary behavior
            degree: Polynomial degree for regression
            segments: Number of segments for regression
        Returns:
            sp.Piecewise: Regression-based piecewise function
        """
        return RegressionProcessor.process_regression(temp_array, prop_array, T, lower, upper, degree, segments,
                                                      seed=ProcessingConstants.DEFAULT_REGRESSION_SEED)
