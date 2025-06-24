import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import sympy as sp

from pymatlib.core.materials import Material
from pymatlib.parsing.validation.property_type_detector import PropertyType
from pymatlib.parsing.processors.property_processor_base import PropertyProcessorBase
from pymatlib.parsing.processors.property_handlers import (
    ConstantPropertyHandler,
    StepFunctionPropertyHandler,
    FilePropertyHandler,
    KeyValPropertyHandler,
    PiecewiseEquationPropertyHandler,
    ComputedPropertyHandler
)
from pymatlib.parsing.processors.post_processor import PropertyPostProcessor

logger = logging.getLogger(__name__)


class PropertyProcessor(PropertyProcessorBase):
    """
    Main orchestrator for processing different property types for material objects.

    This class coordinates the processing of various property types by delegating
    to specialized handlers for each property type.
    """

    def __init__(self) -> None:
        """Initialize processor with specialized handlers."""
        super().__init__()
        # Initialize property handlers
        self.handlers = {
            PropertyType.CONSTANT: ConstantPropertyHandler(),
            PropertyType.STEP_FUNCTION: StepFunctionPropertyHandler(),
            PropertyType.FILE: FilePropertyHandler(),
            PropertyType.KEY_VAL: KeyValPropertyHandler(),
            PropertyType.PIECEWISE_EQUATION: PiecewiseEquationPropertyHandler(),
            PropertyType.COMPUTE: ComputedPropertyHandler()
        }
        # Initialize post-processor
        self.post_processor = PropertyPostProcessor()
        # Processing state
        self.properties: Optional[Dict[str, Any]] = None
        self.categorized_properties: Optional[Dict[PropertyType, List[Tuple[str, Any]]]] = None
        self.base_dir: Optional[Path] = None
        logger.debug("PropertyProcessor initialized")

    def process_properties(self, material: Material, T: Union[float, sp.Symbol],
                           properties: Dict[str, Any],
                           categorized_properties: Dict[PropertyType, List[Tuple[str, Any]]],
                           base_dir: Path, visualizer) -> None:
        """Process all properties for the material."""
        logger.info(f"Starting property processing for material: {material.name}")
        # Set processing context
        self._initialize_processing_context(material, T, properties, categorized_properties, base_dir, visualizer)
        try:
            # Process properties by type
            self._process_by_category(material, T)
            # Post-process properties (regression, etc.)
            self.post_processor.post_process_properties(
                material, T, self.properties, self.categorized_properties, self.processed_properties
            )
            logger.info(f"Successfully processed all properties for material: {material.name}")
        except Exception as e:
            logger.error(f"Property processing failed for material '{material.name}': {e}", exc_info=True)
            raise ValueError(f"Failed to process properties \n -> {str(e)}") from e

    def _initialize_processing_context(self, material: Material, T: Union[float, sp.Symbol],
                                       properties: Dict[str, Any],
                                       categorized_properties: Dict[PropertyType, List[Tuple[str, Any]]],
                                       base_dir: Path, visualizer) -> None:
        """Initialize processing context and handler dependencies."""
        self.properties = properties
        self.categorized_properties = categorized_properties
        self.base_dir = base_dir
        self.visualizer = visualizer
        self.processed_properties = set()
        # Set context for all handlers
        for handler in self.handlers.values():
            handler.set_processing_context(self.base_dir, visualizer, self.processed_properties)
        # Initialize dependency processor for computed properties
        computed_handler = self.handlers.get(PropertyType.COMPUTE)
        if computed_handler:
            computed_handler.set_dependency_processor(properties)

    def _process_by_category(self, material: Material, T: Union[float, sp.Symbol]) -> None:
        """Process properties grouped by category."""
        total_properties = sum(len(prop_list) for prop_list in self.categorized_properties.values())
        logger.info(f"Processing {total_properties} properties across {len(self.categorized_properties)} categories")
        for prop_type, prop_list in self.categorized_properties.items():
            if not prop_list:
                continue
            logger.debug(f"Processing {len(prop_list)} properties of type: {prop_type.name}")
            handler = self.handlers.get(prop_type)
            if handler is None:
                raise ValueError(f"No handler available for property type: {prop_type.name}")
            # Sort properties to prioritize temperature references
            sorted_props = self._sort_properties_by_priority(prop_list)
            for prop_name, config in sorted_props:
                logger.debug(f"Processing property: {prop_name}")
                handler.process_property(material, prop_name, config, T)
                logger.debug(f"Successfully processed property: {prop_name}")

    @staticmethod
    def _sort_properties_by_priority(prop_list: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
        """Sort properties to process temperature references first."""
        from pymatlib.parsing.config.yaml_keys import (
            MELTING_TEMPERATURE_KEY, BOILING_TEMPERATURE_KEY,
            SOLIDUS_TEMPERATURE_KEY, LIQUIDUS_TEMPERATURE_KEY,
            INITIAL_BOILING_TEMPERATURE_KEY, FINAL_BOILING_TEMPERATURE_KEY
        )
        priority_props = {
            MELTING_TEMPERATURE_KEY, BOILING_TEMPERATURE_KEY,
            SOLIDUS_TEMPERATURE_KEY, LIQUIDUS_TEMPERATURE_KEY,
            INITIAL_BOILING_TEMPERATURE_KEY, FINAL_BOILING_TEMPERATURE_KEY
        }
        return sorted(prop_list, key=lambda x: 0 if x[0] in priority_props else 1)
