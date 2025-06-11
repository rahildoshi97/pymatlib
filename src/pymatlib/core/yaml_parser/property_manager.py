import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import sympy as sp

from pymatlib.core.material import Material
from pymatlib.core.yaml_parser.data_handler import read_data_from_file
from pymatlib.core.symbol_registry import SymbolRegistry
from pymatlib.core.yaml_parser.common_utils import (
    ensure_ascending_order,
    validate_energy_density_monotonicity,
    generate_step_plot_data,
    evaluate_numeric_temperature
)
from pymatlib.core.yaml_parser.custom_error import DependencyError, CircularDependencyError
from pymatlib.core.yaml_parser.piecewise_builder import PiecewiseBuilder
from pymatlib.core.yaml_parser.property_type_detector import PropertyType
from pymatlib.core.yaml_parser.regression_processor import RegressionManager
from pymatlib.core.yaml_parser.temperature_resolver import TemperatureResolver
from pymatlib.core.yaml_parser.yaml_keys import MELTING_TEMPERATURE_KEY, BOILING_TEMPERATURE_KEY, \
    SOLIDUS_TEMPERATURE_KEY, LIQUIDUS_TEMPERATURE_KEY, INITIAL_BOILING_TEMPERATURE_KEY, FINAL_BOILING_TEMPERATURE_KEY, \
    TEMPERATURE_KEY, VALUE_KEY, BOUNDS_KEY, CONSTANT_KEY, REGRESSION_KEY, SIMPLIFY_KEY, \
    PRE_KEY, FILE_PATH_KEY, EQUATION_KEY, POST_KEY
from pymatlib.data.constants import ProcessingConstants

logger = logging.getLogger(__name__)

seed = ProcessingConstants.DEFAULT_REGRESSION_SEED

class PropertyManager:
    """Handles processing of different property types for material objects."""

    # --- Constructor ---
    def __init__(self) -> None:
        """Initialize processor state."""
        self.properties: Optional[Dict[str, Any]] = None
        self.categorized_properties: Optional[Dict[PropertyType, List[Tuple[str, Any]]]] = None
        self.base_dir: Optional[Path] = None
        self.visualizer = None
        self.processed_properties: set = set()

    def _finalize_property_processing(self, material: Material, prop_name: str,
                                      temp_array: np.ndarray, prop_array: np.ndarray,
                                      T: Union[float, sp.Symbol], config: Dict,
                                      prop_type: str, skip_numeric_check: bool = False) -> bool:
        """
        Common finalization logic for property processing.
        Args:
            skip_numeric_check: If True, skips the numeric temperature handling
                               (useful for piecewise and computed properties that handle this separately)
        """
        lower_bound_type, upper_bound_type = config[BOUNDS_KEY]
        if not skip_numeric_check:
            if evaluate_numeric_temperature(material, prop_name, T, self,
                                            interpolation_data=(temp_array, prop_array,
                                                                   lower_bound_type, upper_bound_type)):
                return True
        piecewise_func = PiecewiseBuilder.create_from_data(temp_array, prop_array, T, config, prop_name)
        setattr(material, prop_name, piecewise_func)
        self._visualize_if_enabled(material, prop_name, T, prop_type, temp_array, prop_array,
                                   config, (np.min(temp_array), np.max(temp_array)))
        self.processed_properties.add(prop_name)
        return False

    # --- Public API ---
    def process_properties(self, material: Material, T: Union[float, sp.Symbol],
                           properties: Dict[str, Any],
                           categorized_properties: Dict[PropertyType, List[Tuple[str, Any]]],
                           base_dir: Path, visualizer) -> None:
        """Process all properties for the material."""
        self.properties = properties
        self.categorized_properties = categorized_properties
        self.base_dir = base_dir
        self.visualizer = visualizer
        self.processed_properties = set()
        try:
            for prop_type, prop_list in self.categorized_properties.items():
                sorted_props = sorted(prop_list,
                                      key=lambda x: 0 if x[0] in [MELTING_TEMPERATURE_KEY, BOILING_TEMPERATURE_KEY,
                                                                  SOLIDUS_TEMPERATURE_KEY, LIQUIDUS_TEMPERATURE_KEY,
                                                                  INITIAL_BOILING_TEMPERATURE_KEY,
                                                                  FINAL_BOILING_TEMPERATURE_KEY] else 1)
                for prop_name, config in sorted_props:
                    if prop_type == PropertyType.CONSTANT:
                        self._process_constant_property(material, prop_name, config, T)
                    elif prop_type == PropertyType.STEP_FUNCTION:
                        self._process_step_function_property(material, prop_name, config, T)
                    elif prop_type == PropertyType.FILE:
                        self._process_file_property(material, prop_name, config, T)
                    elif prop_type == PropertyType.KEY_VAL:
                        self._process_key_val_property(material, prop_name, config, T)
                    elif prop_type == PropertyType.PIECEWISE_EQUATION:
                        self._process_piecewise_equation_property(material, prop_name, config, T)
                    elif prop_type == PropertyType.COMPUTE:
                        self._process_computed_property(material, prop_name, T)
            self._post_process_properties(material, T)
        except Exception as e:
            raise ValueError(f"Failed to process properties \n -> {str(e)}") from e

    # --- Property-Type Processing Methods ---
    def _process_constant_property(self, material: Material, prop_name: str, prop_config: Union[float, str],
                                   T: Union[float, sp.Symbol]) -> None:
        """Process constant float property."""
        try:
            value = float(prop_config)
            setattr(material, prop_name, sp.Float(value))
            self._visualize_if_enabled(material=material, prop_name=prop_name, T=T, prop_type='CONSTANT')
            self.processed_properties.add(prop_name)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to process constant property \n -> {str(e)}") from e

    def _process_step_function_property(self, material: Material, prop_name: str,
                                        prop_config: Dict[str, Any], T: Union[float, sp.Symbol]) -> None:
        """Process step function with unified symbol handling."""
        try:
            temp_key = prop_config[TEMPERATURE_KEY]
            val_array = prop_config[VALUE_KEY]
            transition_temp = TemperatureResolver.resolve_temperature_reference(temp_key, material)
            T_standard = sp.Symbol('T')
            step_function = sp.Piecewise((val_array[0], T_standard < transition_temp), (val_array[1], True))
            if evaluate_numeric_temperature(material, prop_name, T, self, piecewise_expr=step_function):
                return
            if str(T) != 'T':
                step_function = step_function.subs(T_standard, T)
            setattr(material, prop_name, step_function)
            offset = ProcessingConstants.STEP_FUNCTION_OFFSET
            val1 = max(transition_temp - offset, ProcessingConstants.ABSOLUTE_ZERO) # Ensure non-negative temperature
            val2 = transition_temp + offset
            step_temp_array = np.array([val1, transition_temp, val2])
            x_data, y_data = generate_step_plot_data(transition_temp, val_array, step_temp_array)
            self._visualize_if_enabled(
                material=material, prop_name=prop_name, T=T, prop_type='STEP_FUNCTION',
                x_data=x_data, y_data=y_data, config=prop_config,
                bounds=(np.min(step_temp_array), np.max(step_temp_array))
            )
            self.processed_properties.add(prop_name)
        except Exception as e:
            raise ValueError(f"Failed to process step function property '{prop_name}'\n -> {str(e)}") from e

    def _process_file_property(self, material: Material, prop_name: str, file_config: Dict[str, Any], T: Union[float, sp.Symbol]) -> None:
        """Process property data from a file configuration."""
        try:
            yaml_dir = self.base_dir
            file_config[FILE_PATH_KEY] = str(yaml_dir / file_config[FILE_PATH_KEY])
            temp_array, prop_array = read_data_from_file(file_config)
            validate_energy_density_monotonicity(prop_name, temp_array, prop_array)
            self._finalize_property_processing(material, prop_name, temp_array, prop_array,
                                               T, file_config, 'FILE')
        except Exception as e:
            raise ValueError(f"Failed to process file property {prop_name} \n -> {str(e)}") from e

    def _process_key_val_property(self, material: Material, prop_name: str, prop_config: Dict[str, Any],
                                  T: Union[float, sp.Symbol]) -> None:
        """Process property defined with key-val pairs."""
        try:
            temp_def = prop_config[TEMPERATURE_KEY]
            val_array = prop_config[VALUE_KEY]
            key_array = TemperatureResolver.process_temperature_definition(temp_def, len(val_array), material)
            if len(key_array) != len(val_array):
                raise ValueError(f"Length mismatch in {prop_name}: key and val arrays must have same length")
            key_array, val_array = ensure_ascending_order(key_array, val_array)
            validate_energy_density_monotonicity(prop_name, key_array, val_array)
            self._finalize_property_processing(material, prop_name, key_array, val_array,
                                               T, prop_config, 'KEY_VAL')
        except Exception as e:
            raise ValueError(f"Failed to process key-val property '{prop_name}' \n -> {str(e)}") from e

    def _process_piecewise_equation_property(self, material: Material, prop_name: str, prop_config: Dict[str, Any],
                                             T: Union[float, sp.Symbol]) -> None:
        """Process piecewise equation property."""
        try:
            eqn_strings = prop_config[EQUATION_KEY]
            temp_def = prop_config[TEMPERATURE_KEY]
            temp_points = TemperatureResolver.process_temperature_definition(temp_def, len(eqn_strings) + 1)
            for eqn in eqn_strings:
                expr = sp.sympify(eqn)
                # invalid_symbols = [str(symbol) for symbol in expr.free_symbols if str(symbol) != 'T']
                for symbol in expr.free_symbols:
                    if str(symbol) != 'T':
                        raise ValueError(f"Unsupported symbol '{symbol}' in equation '{eqn}' for property '{prop_name}'."
                                         f"Only 'T' is allowed.")
            lower_bound_type, upper_bound_type = prop_config[BOUNDS_KEY]
            temp_points, eqn_strings = ensure_ascending_order(temp_points, eqn_strings)
            diff = max(np.min(np.diff(np.sort(temp_points))) / 10.0, 1.0)
            temp_dense = np.arange(temp_points[0], temp_points[-1] + diff / 2, diff)
            T_standard = sp.Symbol('T')
            piecewise_standard = PiecewiseBuilder.create_from_formulas(temp_points, list(eqn_strings), T_standard, lower_bound_type, upper_bound_type)
            if evaluate_numeric_temperature(material, prop_name, T, self, piecewise_expr=piecewise_standard):
                return
            f_pw = sp.lambdify(T_standard, piecewise_standard, 'numpy')
            y_dense = f_pw(temp_dense)
            validate_energy_density_monotonicity(prop_name, temp_dense, y_dense)
            self._finalize_property_processing(material, prop_name, temp_dense, y_dense,
                                               T, prop_config, 'PIECEWISE_EQUATION', skip_numeric_check=True)
        except Exception as e:
            raise ValueError(f"Failed to process piecewise equation property '{prop_name}' \n -> {str(e)}") from e

    def _process_computed_property(self, material: Material, prop_name: str, T: Union[float, sp.Symbol]) -> None:
        """Process computed properties using predefined models with dependency checking."""
        if prop_name in self.processed_properties:
            return
        try: # Get property configuration
            prop_config = self.properties[prop_name]
            if not isinstance(prop_config, dict) or EQUATION_KEY not in prop_config:
                raise ValueError(f"Invalid COMPUTE property configuration for {prop_name}")
            temp_def = prop_config[TEMPERATURE_KEY]
            temp_array = TemperatureResolver.process_temperature_definition(temp_def, material=material)
            expression = prop_config[EQUATION_KEY]
            try:
                material_property = self._parse_and_process_expression(expression, material, T, prop_name)
            except CircularDependencyError:
                raise # Re-raise CircularDependencyError without wrapping it
            except Exception as e: # Wrap other exceptions with more context
                raise ValueError(f"Failed to process computed property '{prop_name}' \n -> {str(e)}") from e
            if evaluate_numeric_temperature(material, prop_name, T, self, piecewise_expr=material_property):
                return
            T_standard = sp.Symbol('T')
            if isinstance(T, sp.Symbol) and str(T) != 'T':
                standard_expr = material_property.subs(T, T_standard)
                f_pw = sp.lambdify(T_standard, standard_expr, 'numpy')
            else:
                f_pw = sp.lambdify(T, material_property, 'numpy')
            try:
                y_dense = f_pw(temp_array)
                if not np.all(np.isfinite(y_dense)):
                    invalid_count = np.sum(~np.isfinite(y_dense))
                    logger.warning(
                        f"Property '{prop_name}' has {invalid_count} non-finite values. "
                        f"This may indicate issues with the expression: {expression}")
                validate_energy_density_monotonicity(prop_name, temp_array, y_dense)
                self._finalize_property_processing(material, prop_name, temp_array, y_dense,
                                                   T, prop_config, 'COMPUTE', skip_numeric_check=True)
            except Exception as e:
                raise ValueError(f"Error evaluating expression for '{prop_name}' \n -> {str(e)}") from e
        except CircularDependencyError:
            raise # Re-raise CircularDependencyError without wrapping it
        except Exception as e:
            raise ValueError(f"Failed to process computed property '{prop_name}' \n -> {str(e)}") from e

    # --- Expression/Dependency Helpers ---
    def _parse_and_process_expression(self, expression: str, material: Material, T: Union[float, sp.Symbol], prop_name: str) -> sp.Expr:
        """Parse and process a mathematical expression string into a SymPy expression."""
        try:
            # Create a standard symbol 'T' for parsing the expression
            T_standard = sp.Symbol('T')
            # Parse the expression with the standard symbol
            sympy_expr = sp.sympify(expression, evaluate=False)
            # Extract dependencies (always excluding 'T' as it's handled separately)
            dependencies = [str(symbol) for symbol in sympy_expr.free_symbols if str(symbol) != 'T']
            # Check for missing dependencies before processing
            missing_deps = []
            for dep in dependencies:
                if not hasattr(material, dep) and dep not in self.properties:
                    missing_deps.append(dep)
            if missing_deps:
                available_props = sorted(list(self.properties.keys()))
                raise DependencyError(expression=expression, missing_deps=missing_deps, available_props=available_props)
            # Check for circular dependencies
            self._check_circular_dependencies(prop_name=prop_name, current_deps=dependencies, visited=set())
            # Process dependencies first
            for dep in dependencies:
                if not hasattr(material, dep) or getattr(material, dep) is None:
                    if dep in self.properties:
                        self._process_computed_property(material, dep, T)
                    else: # This should not happen due to the check above, but just in case
                        available_props = sorted(list(self.properties.keys()))
                        raise DependencyError(expression=expression, missing_deps=[dep], available_props=available_props)
            # Verify all dependencies are now available
            missing_deps = [dep for dep in dependencies if not hasattr(material, dep) or getattr(material, dep) is None]
            if missing_deps:
                raise ValueError(f"Cannot compute expression. Missing dependencies: {missing_deps}")
            # Create substitution dictionary
            substitutions = {}
            for dep in dependencies:
                dep_value = getattr(material, dep, None)
                if dep_value is None:
                    # This is a safety check - if we get here, something went wrong during dependency processing
                    raise ValueError(f"Dependency '{dep}' was processed but is still not available on the material")
                dep_symbol = SymbolRegistry.get(dep)
                if dep_symbol is None:
                    raise ValueError(f"Symbol '{dep}' not found in symbol registry")
                substitutions[dep_symbol] = dep_value
            # Handle temperature substitution based on type
            if isinstance(T, sp.Symbol): # If T is a symbolic variable, substitute the standard 'T' with it
                substitutions[T_standard] = T
            else: # For numeric T, substitute with the value directly
                substitutions[T_standard] = T
            # Perform substitution and evaluate integrals if present
            result_expr = sympy_expr.subs(substitutions)
            if isinstance(result_expr, sp.Integral):
                result_expr = result_expr.doit()
            return result_expr
        except CircularDependencyError: # Re-raise the circular dependency error directly without wrapping it
            raise
        except DependencyError as e: # Re-raise with the original exception as the cause
            raise e
        except Exception as e: # Wrap other exceptions with more context
            raise ValueError(f"Failed to parse and process expression: {expression}") from e

    def _check_circular_dependencies(self, prop_name, current_deps, visited, path=None):
        """Check for circular dependencies in property definitions."""
        if path is None:
            path = []
        # Always filter out 'T' from dependencies to check
        current_deps = [dep for dep in current_deps if dep != 'T']
        if prop_name is not None:
            if prop_name in visited:
                cycle_path = path + [prop_name]
                raise CircularDependencyError(dependency_path=cycle_path)
            visited.add(prop_name)
            path = path + [prop_name]
        for dep in current_deps:
            if dep in self.properties:
                dep_config = self.properties[dep]
                if isinstance(dep_config, str):
                    expr = sp.sympify(dep_config)
                    # Always exclude 'T' from dependencies
                    dep_deps = [str(symbol) for symbol in expr.free_symbols if str(symbol) != 'T']
                elif isinstance(dep_config, dict) and EQUATION_KEY in dep_config:
                    eqn = dep_config[EQUATION_KEY]
                    if isinstance(eqn, list):
                        symbols = set()
                        for eq in eqn:
                            expr = sp.sympify(eq)
                            symbols.update(expr.free_symbols)
                        # Always exclude 'T' from dependencies
                        dep_deps = [str(symbol) for symbol in symbols if str(symbol) != 'T']
                    else:
                        expr = sp.sympify(eqn)
                        # Always exclude 'T' from dependencies
                        dep_deps = [str(symbol) for symbol in expr.free_symbols if str(symbol) != 'T']
                else:
                    dep_deps = []
                if dep_deps:
                    self._check_circular_dependencies(dep, dep_deps, visited.copy(), path)

    def _visualize_if_enabled(self, material: Material, prop_name: str, T: sp.Symbol, prop_type: str,
                              x_data: Optional[np.ndarray] = None,
                              y_data: Optional[np.ndarray] = None,
                              config: Optional[Dict] = None,
                              bounds: Optional[Tuple[float, float]] = None) -> None:
        """Visualize property only if visualizer is available and T is symbolic."""
        if self.visualizer is None or not isinstance(T, sp.Symbol):
            return
        lower_bound_type, upper_bound_type = CONSTANT_KEY, CONSTANT_KEY
        has_regression, simplify_type, degree, segments = False, None, None, None
        if config is not None and isinstance(config, dict):
            if BOUNDS_KEY in config:
                bounds_config = config[BOUNDS_KEY]
                if isinstance(bounds_config, list) and len(bounds_config) == 2:
                    lower_bound_type, upper_bound_type = bounds_config
            data_length = len(x_data) if x_data is not None else 100 # fallback
            has_regression, simplify_type, degree, segments = RegressionManager.process_regression_params(
                config, prop_name, data_length
            )
        if bounds is None:
            if x_data is not None:
                bounds = (np.min(x_data), np.max(x_data))
            else:
                bounds = (ProcessingConstants.DEFAULT_TEMP_LOWER, ProcessingConstants.DEFAULT_TEMP_UPPER)
        self.visualizer.visualize_property(
            material=material, prop_name=prop_name, T=T, prop_type=prop_type,
            x_data=x_data, y_data=y_data,
            has_regression=has_regression, simplify_type=simplify_type,
            degree=degree, segments=segments,
            lower_bound=bounds[0], upper_bound=bounds[1],
            lower_bound_type=lower_bound_type, upper_bound_type=upper_bound_type
        )

    # --- Post-Processing ---
    def _post_process_properties(self, material: Material, T: Union[float, sp.Symbol]) -> None:
        """Perform post-processing regression on properties after all have been initially processed."""
        if not isinstance(T, sp.Symbol):
            logger.debug("Skipping post-processing for numeric temperature")
            return
        errors = []
        for prop_name, prop_config in self.properties.items():
            try:
                if not isinstance(prop_config, dict) or REGRESSION_KEY not in prop_config:
                    continue
                temp_array = TemperatureResolver.extract_from_config(prop_config, material)
                has_regression, simplify_type, degree, segments = RegressionManager.process_regression_params(
                    prop_config, prop_name, len(temp_array)
                )
                if not has_regression or simplify_type != POST_KEY:
                    continue
                if not hasattr(material, prop_name):
                    logger.warning(f"Property '{prop_name}' not found on material during post-processing")
                    continue
                prop_value = getattr(material, prop_name)
                if isinstance(prop_value, sp.Integral):
                    logger.warning(f"Property '{prop_name}' is an integral and cannot be post-processed")
                    continue
                if not isinstance(prop_value, sp.Expr):
                    logger.debug(f"Skipping non-symbolic property: {prop_name}")
                    continue
                self._apply_post_regression(material, prop_name, prop_config, T)
            except Exception as e:
                error_msg = f"Failed to post-process {prop_name}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        if errors:
            error_summary = "\n".join(errors)
            raise ValueError(f"Post-processing errors occurred:\n{error_summary}")

    @staticmethod
    def _apply_post_regression(material: Material, prop_name: str, prop_config: Dict, T: sp.Symbol) -> None:
        """Apply post-processing regression with proper type validation."""
        prop_value = getattr(material, prop_name)
        try:
            temp_array = TemperatureResolver.extract_from_config(prop_config, material)
        except Exception as e:
            raise ValueError(f"Failed to extract temperature array for {prop_name}: {str(e)}") from e
        # Validate temp_array type and convert if necessary
        if isinstance(temp_array, str):
            raise ValueError(f"Temperature array for {prop_name} is a string: '{temp_array}'. Expected numpy array.")
        if not isinstance(temp_array, np.ndarray):
            try:
                temp_array = np.array(temp_array, dtype=np.float64)
            except Exception as e:
                raise ValueError(f"Cannot convert temperature data to numpy array for {prop_name}: {str(e)}") from e
        # Validate dtype
        if temp_array.dtype.kind not in ['f', 'i']:  # not float or int
            logger.warning(f"Temperature array for {prop_name} has non-numeric dtype {temp_array.dtype}. Converting to float64.")
            try:
                temp_array = np.asarray(temp_array, dtype=np.float64)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Cannot convert temperature array to numeric format for {prop_name}: {str(e)}") from e
        f_prop = sp.lambdify(T, prop_value, 'numpy')
        prop_array = f_prop(temp_array)
        # Validate prop_array type
        if hasattr(prop_array, 'dtype') and prop_array.dtype.kind not in ['f', 'i']:
            try:
                prop_array = np.asarray(prop_array, dtype=np.float64)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Cannot convert property array to numeric format for {prop_name}: {str(e)}") from e
        temp_array, prop_array = ensure_ascending_order(temp_array, prop_array)
        validate_energy_density_monotonicity(prop_name, temp_array, prop_array)
        # Temporarily modify config to force PRE_KEY regression
        original_simplify_type = prop_config[REGRESSION_KEY][SIMPLIFY_KEY]
        prop_config[REGRESSION_KEY][SIMPLIFY_KEY] = PRE_KEY
        try:
            piecewise_func = PiecewiseBuilder.create_from_data(temp_array, prop_array, T, prop_config, prop_name)
        finally: # Restore original config
            prop_config[REGRESSION_KEY][SIMPLIFY_KEY] = original_simplify_type
        setattr(material, prop_name, piecewise_func)
