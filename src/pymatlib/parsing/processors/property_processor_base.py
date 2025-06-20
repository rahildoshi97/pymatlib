"""
Base processor class for material property processing with common finalization logic.

This module provides the PropertyProcessorBase class that serves as a foundation
for all property processors in the PyMatLib library, ensuring consistent
processing patterns and reducing code duplication.
"""

import logging
import numpy as np
import sympy as sp
from typing import Dict, Union, Tuple, Optional
from pathlib import Path

from pymatlib.algorithms.piecewise_builder import PiecewiseBuilder
from pymatlib.parsing.utils.utilities import handle_numeric_temperature
from pymatlib.core.materials import Material

logger = logging.getLogger(__name__)

class PropertyProcessorBase:
    """
    Base class for material property processors with common finalization logic.

    This class provides shared functionality for processing material properties,
    including piecewise function creation, visualization, and property assignment.
    It serves as the foundation for all property handlers in the PyMatLib library.
    Attributes:
        processed_properties (set): Set of property names that have been processed
        visualizer: Optional visualizer instance for property plotting
        base_dir (Path): Base directory for file operations
    """

    def __init__(self):
        """Initialize the base processor with empty state."""
        self.processed_properties = set()
        self.visualizer = None
        self.base_dir: Path = None
        logger.debug("PropertyProcessorBase initialized")

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
        # Handle numeric temperature case (if not skipped)
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
            # Assign property to material
            setattr(material, prop_name, piecewise_func)
            logger.debug(f"Created piecewise function for property '{prop_name}'")
        except Exception as e:
            logger.error(f"Failed to create piecewise function for '{prop_name}': {e}")
            raise ValueError(f"Failed to finalize property '{prop_name}': {str(e)}") from e
        # Generate visualization if enabled
        self._visualize_if_enabled(
            material=material,
            prop_name=prop_name,
            T=T,
            prop_type=prop_type,
            x_data=temp_array,
            y_data=prop_array,
            config=config,
            bounds=(np.min(temp_array), np.max(temp_array))
        )
        # Track processed property
        self.processed_properties.add(prop_name)
        logger.debug(f"Successfully finalized property '{prop_name}'")
        return False

    def _visualize_if_enabled(self, material: Material, prop_name: str,
                              T: Union[float, sp.Symbol], prop_type: str,
                              x_data: Optional[np.ndarray] = None,
                              y_data: Optional[np.ndarray] = None,
                              config: Optional[Dict] = None,
                              bounds: Optional[Tuple[float, float]] = None) -> None:
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
        # Check if visualizer is available
        if self.visualizer is None:
            logger.debug(f"No visualizer available for property '{prop_name}'")
            return
        # Check if visualization is enabled
        if not hasattr(self.visualizer, 'is_visualization_enabled') or \
                not self.visualizer.is_visualization_enabled():
            logger.debug(f"Visualization disabled for property '{prop_name}'")
            return
        try:
            # Initialize default visualization parameters
            has_regression = False
            simplify_type = None
            degree = 1
            segments = 1
            lower_bound_type = 'constant'
            upper_bound_type = 'constant'
            # Extract visualization parameters from config
            if config:
                bounds_config = config.get('bounds', ['constant', 'constant'])
                lower_bound_type, upper_bound_type = bounds_config
                # Check for regression configuration
                if 'regression' in config:
                    has_regression = True
                    regression_config = config['regression']
                    simplify_type = regression_config.get('simplify', 'pre')
                    degree = regression_config.get('degree', 1)
                    segments = regression_config.get('segments', 1)
            # Set temperature bounds if available
            lower_bound = bounds[0] if bounds else None
            upper_bound = bounds[1] if bounds else None
            # Call visualizer with extracted parameters
            self.visualizer.visualize_property(
                material=material,
                prop_name=prop_name,
                T=T,
                prop_type=prop_type,
                x_data=x_data,
                y_data=y_data,
                has_regression=has_regression,
                simplify_type=simplify_type,
                degree=degree,
                segments=segments,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                lower_bound_type=lower_bound_type,
                upper_bound_type=upper_bound_type
            )
            logger.debug(f"Generated visualization for property '{prop_name}'")
        except Exception as e:
            logger.warning(f"Failed to generate visualization for '{prop_name}': {e}")
            # Don't raise exception - visualization failure shouldn't stop processing

    def set_visualizer(self, visualizer) -> None:
        """
        Set the visualizer instance for property plotting.
        Args:
            visualizer: Visualizer instance that implements visualization methods
        """
        self.visualizer = visualizer
        logger.debug("Visualizer set for PropertyProcessorBase")

    def reset_processing_state(self) -> None:
        """
        Reset the processing state for reuse of the processor instance.

        This method clears the set of processed properties, allowing the same
        processor instance to be used for processing multiple materials.
        """
        self.processed_properties.clear()
        logger.debug("Processing state reset")

    def get_processed_properties(self) -> set:
        """
        Get the set of properties that have been processed.
        Returns:
            set: Set of property names that have been successfully processed
        """
        return self.processed_properties.copy()

    def is_property_processed(self, prop_name: str) -> bool:
        """
        Check if a specific property has been processed.
        Args:
            prop_name: Name of the property to check
        Returns:
            bool: True if the property has been processed, False otherwise
        """
        return prop_name in self.processed_properties
