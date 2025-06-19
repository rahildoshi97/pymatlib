import logging
from typing import Any, Dict, Union
from pathlib import Path

import numpy as np
import sympy as sp

from pymatlib.core.materials import Material
from pymatlib.parsing.processors.property_processor_base import PropertyProcessorBase
from pymatlib.parsing.processors.temperature_resolver import TemperatureResolver
from pymatlib.parsing.io.data_handler import load_property_data
from pymatlib.parsing.utils.utilities import handle_numeric_temperature, create_step_visualization_data
from pymatlib.parsing.validation.property_validator import validate_monotonic_energy_density
from pymatlib.algorithms.interpolation import ensure_ascending_order
from pymatlib.algorithms.piecewise_builder import PiecewiseBuilder
from pymatlib.parsing.config.yaml_keys import (
    TEMPERATURE_KEY, VALUE_KEY, BOUNDS_KEY, FILE_PATH_KEY, EQUATION_KEY
)
from pymatlib.data.constants import ProcessingConstants

logger = logging.getLogger(__name__)

class BasePropertyHandler:
    """Base class for property handlers with common functionality.

    This class provides shared functionality for processing material properties,
    including piecewise function creation, visualization, and property assignment.
    It serves as the foundation for MaterialPropertyProcessor and other specialized
    processors in the PyMatLib library.
    Attributes:
        processed_properties (set): Set of property names that have been processed
        visualizer: Optional visualizer instance for property plotting
    """
    def __init__(self):
        self.base_dir: Path = None
        self.visualizer = None
        self.processed_properties = None

    def set_processing_context(self, base_dir: Path, visualizer, processed_properties: set):
        """Set processing context shared across handlers."""
        self.base_dir = base_dir
        self.visualizer = visualizer
        self.processed_properties = processed_properties

    def _finalize_property_processing(self, material: Material, prop_name: str,
                                      temp_array: np.ndarray, prop_array: np.ndarray,
                                      T: Union[float, sp.Symbol], config: Dict,
                                      prop_type: str, skip_numeric_check: bool = False) -> bool:
        """
        Common finalization logic for property processing.

        This method handles the final steps of property processing including:
        - Numeric temperature evaluation (if applicable)
        - Piecewise function creation
        - Property assignment to material
        - Visualization (if enabled)
        - Processing tracking
        Args:
            material: Material object to modify
            prop_name: Name of the property being processed
            temp_array: Temperature data points
            prop_array: Property values corresponding to temperatures
            T: Temperature symbol or numeric value
            config: Configuration dictionary containing bounds and settings
            prop_type: Type of property for logging and visualization
            skip_numeric_check: If True, skips numeric temperature handling
                              (useful for piecewise and computed properties)
        Returns:
            bool: True if numeric evaluation was performed and processing is complete,
                  False if symbolic processing should continue
        Raises:
            ValueError: If piecewise function creation fails
        """
        logger.debug(f"Finalizing processing for property '{prop_name}' of type {prop_type}")
        # Validate input arrays
        if temp_array is None or prop_array is None:
            raise ValueError(f"Temperature and property arrays cannot be None for '{prop_name}'")
        if len(temp_array) != len(prop_array):
            raise ValueError(f"Temperature and property arrays must have same length for '{prop_name}'")
        # Extract boundary configuration
        lower_bound_type, upper_bound_type = config.get('bounds', ['constant', 'constant'])
        # Handle numeric temperature case
        if not skip_numeric_check:
            if handle_numeric_temperature(
                    material, prop_name, T, self,
                    interpolation_data=(temp_array, prop_array, lower_bound_type, upper_bound_type)
            ):
                logger.debug(f"Numeric temperature evaluation completed for '{prop_name}'")
                return True
        # Create symbolic piecewise function
        try:
            piecewise_func = PiecewiseBuilder.build_from_data(
                temp_array, prop_array, T, config, prop_name
            )
            setattr(material, prop_name, piecewise_func)
            logger.debug(f"Created piecewise function for property '{prop_name}'")
        except Exception as e:
            logger.error(f"Failed to create piecewise function for '{prop_name}': {e}")
            raise ValueError(f"Failed to finalize property '{prop_name}': {str(e)}") from e
        # Generate visualization if enabled
        self._visualize_if_enabled(
            material=material, prop_name=prop_name, T=T, prop_type=prop_type,
            x_data=temp_array, y_data=prop_array, config=config,
            bounds=(np.min(temp_array), np.max(temp_array))
        )
        # Track processed property
        self.processed_properties.add(prop_name)
        logger.debug(f"Successfully finalized property '{prop_name}'")
        return False

    def _visualize_if_enabled(self, material: Material, prop_name: str,
                              T: Union[float, sp.Symbol], prop_type: str,
                              x_data: np.ndarray = None, y_data: np.ndarray = None,
                              config: Dict = None, bounds: tuple = None) -> None:
        """
        Generate visualization if visualizer is available and enabled.

        This method handles the visualization of processed properties by extracting
        visualization parameters from the configuration and calling the appropriate
        visualizer methods. It gracefully handles cases where visualization is
        disabled or unavailable.
        Args:
            material: Material object containing the processed property
            prop_name: Name of the property to visualize
            T: Temperature symbol or numeric value
            prop_type: Type of property for visualization context
            x_data: Temperature data points (optional)
            y_data: Property values corresponding to temperatures (optional)
            config: Configuration dictionary containing visualization settings (optional)
            bounds: Temperature bounds tuple (min_temp, max_temp) (optional)
        Note:
            This method logs warnings for visualization failures but does not raise
            exceptions, ensuring that visualization issues don't interrupt property
            processing.
        """
        if self.visualizer is None or not hasattr(self.visualizer, 'is_visualization_enabled'):
            return
        if not self.visualizer.is_visualization_enabled():
            return
        try:
            # Extract visualization parameters
            has_regression = False
            simplify_type = None
            degree = 1
            segments = 1
            lower_bound_type = 'constant'
            upper_bound_type = 'constant'
            if config:
                bounds_config = config.get('bounds', ['constant', 'constant'])
                lower_bound_type, upper_bound_type = bounds_config
                if 'regression' in config:
                    has_regression = True
                    regression_config = config['regression']
                    simplify_type = regression_config.get('simplify', 'pre')
                    degree = regression_config.get('degree', 1)
                    segments = regression_config.get('segments', 1)
            # Set bounds
            lower_bound = bounds[0] if bounds else None
            upper_bound = bounds[1] if bounds else None
            # Call visualizer
            self.visualizer.visualize_property(
                material=material, prop_name=prop_name, T=T, prop_type=prop_type,
                x_data=x_data, y_data=y_data, has_regression=has_regression,
                simplify_type=simplify_type, degree=degree, segments=segments,
                lower_bound=lower_bound, upper_bound=upper_bound,
                lower_bound_type=lower_bound_type, upper_bound_type=upper_bound_type
            )
            logger.debug(f"Generated visualization for property '{prop_name}'")
        except Exception as e:
            logger.warning(f"Failed to generate visualization for '{prop_name}': {e}")

class ConstantPropertyHandler(BasePropertyHandler):
    """Handler for constant properties."""
    def process_property(self, material: Material, prop_name: str,
                         prop_config: Union[float, str], T: Union[float, sp.Symbol]) -> None:
        """Process constant float property."""
        try:
            value = float(prop_config)
            setattr(material, prop_name, sp.Float(value))
            logger.debug(f"Set constant property {prop_name} = {value}")
            self._visualize_if_enabled(material=material, prop_name=prop_name, T=T, prop_type='CONSTANT',
                                       x_data=None, y_data=None)
            self.processed_properties.add(prop_name)
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to process constant property '{prop_name}': {e}", exc_info=True)
            raise ValueError(f"Failed to process constant property \n -> {str(e)}") from e

class StepFunctionPropertyHandler(BasePropertyHandler):
    """Handler for step function properties."""
    def process_property(self, material: Material, prop_name: str,
                         prop_config: Dict[str, Any], T: Union[float, sp.Symbol]) -> None:
        """Process step function with unified symbol handling."""
        try:
            temp_key = prop_config[TEMPERATURE_KEY]
            val_array = prop_config[VALUE_KEY]
            transition_temp = TemperatureResolver.resolve_temperature_reference(temp_key, material)
            T_standard = sp.Symbol('T')
            step_function = sp.Piecewise((val_array[0], T_standard < transition_temp), (val_array[1], True))
            if handle_numeric_temperature(material, prop_name, T, self, piecewise_expr=step_function):
                return
            if str(T) != 'T':
                step_function = step_function.subs(T_standard, T)
            setattr(material, prop_name, step_function)
            # Create visualization data
            offset = ProcessingConstants.STEP_FUNCTION_OFFSET
            val1 = max(transition_temp - offset, ProcessingConstants.ABSOLUTE_ZERO)
            val2 = transition_temp + offset
            step_temp_array = np.array([val1, transition_temp, val2])
            x_data, y_data = create_step_visualization_data(transition_temp, val_array, step_temp_array)
            self._visualize_if_enabled(
                material=material, prop_name=prop_name, T=T, prop_type='STEP_FUNCTION',
                x_data=x_data, y_data=y_data, config=prop_config,
                bounds=(np.min(step_temp_array), np.max(step_temp_array))
            )
            self.processed_properties.add(prop_name)
        except Exception as e:
            raise ValueError(f"Failed to process step function property '{prop_name}'\n -> {str(e)}") from e

class FilePropertyHandler(BasePropertyHandler):
    """Handler for file-based properties."""
    def process_property(self, material: Material, prop_name: str,
                         file_config: Dict[str, Any], T: Union[float, sp.Symbol]) -> None:
        """Process property data from a file configuration."""
        try:
            file_path = self.base_dir / file_config[FILE_PATH_KEY]
            file_config[FILE_PATH_KEY] = str(file_path)
            logger.debug(f"Loading property '{prop_name}' from file: {file_path}")
            temp_array, prop_array = load_property_data(file_config)
            logger.debug(f"Loaded {len(temp_array)} data points for property '{prop_name}' "
                         f"(T range: {np.min(temp_array):.1f}K - {np.max(temp_array):.1f}K)")
            validate_monotonic_energy_density(prop_name, temp_array, prop_array)
            self._finalize_property_processing(material, prop_name, temp_array, prop_array,
                                               T, file_config, 'FILE')
        except FileNotFoundError as e:
            logger.error(f"File not found for property '{prop_name}': {file_path}", exc_info=True)
            raise ValueError(f"File not found for property '{prop_name}': {file_path}") from e
        except Exception as e:
            logger.error(f"Failed to process file property '{prop_name}': {e}", exc_info=True)
            raise ValueError(f"Failed to process file property {prop_name} \n -> {str(e)}") from e

class KeyValPropertyHandler(BasePropertyHandler):
    """Handler for key-value properties."""
    def process_property(self, material: Material, prop_name: str,
                         prop_config: Dict[str, Any], T: Union[float, sp.Symbol]) -> None:
        """Process property defined with key-val pairs."""
        try:
            temp_def = prop_config[TEMPERATURE_KEY]
            val_array = prop_config[VALUE_KEY]
            key_array = TemperatureResolver.resolve_temperature_definition(temp_def, len(val_array), material)
            if len(key_array) != len(val_array):
                raise ValueError(f"Length mismatch in {prop_name}: key and val arrays must have same length")
            key_array, val_array = ensure_ascending_order(key_array, val_array)
            validate_monotonic_energy_density(prop_name, key_array, val_array)
            self._finalize_property_processing(material, prop_name, key_array, val_array,
                                               T, prop_config, 'KEY_VAL')
        except Exception as e:
            raise ValueError(f"Failed to process key-val property '{prop_name}' \n -> {str(e)}") from e

class PiecewiseEquationPropertyHandler(BasePropertyHandler):
    """Handler for piecewise equation properties."""
    def process_property(self, material: Material, prop_name: str,
                         prop_config: Dict[str, Any], T: Union[float, sp.Symbol]) -> None:
        """Process piecewise equation property."""
        try:
            eqn_strings = prop_config[EQUATION_KEY]
            temp_def = prop_config[TEMPERATURE_KEY]
            temp_points = TemperatureResolver.resolve_temperature_definition(temp_def, len(eqn_strings) + 1)
            # Validate equations
            for eqn in eqn_strings:
                expr = sp.sympify(eqn)
                for symbol in expr.free_symbols:
                    if str(symbol) != 'T':
                        raise ValueError(f"Unsupported symbol '{symbol}' in equation '{eqn}' for property '{prop_name}'. "
                                         f"Only 'T' is allowed.")
            lower_bound_type, upper_bound_type = prop_config[BOUNDS_KEY]
            temp_points, eqn_strings = ensure_ascending_order(temp_points, eqn_strings)
            # Create dense temperature array for validation
            diff = max(np.min(np.diff(np.sort(temp_points))) / 10.0, 1.0)
            temp_dense = np.arange(temp_points[0], temp_points[-1] + diff / 2, diff)
            T_standard = sp.Symbol('T')
            piecewise_standard = PiecewiseBuilder.build_from_formulas(temp_points, list(eqn_strings), T_standard,
                                                                      lower_bound_type, upper_bound_type)
            if handle_numeric_temperature(material, prop_name, T, self, piecewise_expr=piecewise_standard):
                return
            # Evaluate for validation
            f_pw = sp.lambdify(T_standard, piecewise_standard, 'numpy')
            y_dense = f_pw(temp_dense)
            validate_monotonic_energy_density(prop_name, temp_dense, y_dense)
            self._finalize_property_processing(material, prop_name, temp_dense, y_dense,
                                               T, prop_config, 'PIECEWISE_EQUATION', skip_numeric_check=True)
        except Exception as e:
            raise ValueError(f"Failed to process piecewise equation property '{prop_name}' \n -> {str(e)}") from e

'''class ComputedPropertyHandler(BasePropertyHandler):
    """Handler for computed properties."""
    def process_property(self, material: Material, prop_name: str,
                         T: Union[float, sp.Symbol]) -> None:
        """Process computed properties using predefined models with dependency checking."""
        # This will be implemented by the dependency processor
        pass'''

class ComputedPropertyHandler(BasePropertyHandler):
    """Handler for computed properties."""
    def __init__(self):
        super().__init__()
        self.dependency_processor = None
    def set_processing_context(self, base_dir, visualizer, processed_properties):
        """Set processing context and initialize dependency processor."""
        super().set_processing_context(base_dir, visualizer, processed_properties)
        # dependency_processor will be set by the main processor

    def set_dependency_processor(self, properties: Dict[str, Any]):
        """Set the dependency processor with access to all properties."""
        from pymatlib.parsing.processors.dependency_processor import DependencyProcessor
        self.dependency_processor = DependencyProcessor(properties, self.processed_properties)

    def process_property(self, material: Material, prop_name: str,
                         config: Dict[str, Any], T: Union[float, sp.Symbol]) -> None:
        """Process computed properties using dependency processor."""
        if self.dependency_processor is None:
            raise ValueError("Dependency processor not initialized")
        # Pass the config to the dependency processor if needed, or ignore it
        self.dependency_processor.process_computed_property(material, prop_name, T)
