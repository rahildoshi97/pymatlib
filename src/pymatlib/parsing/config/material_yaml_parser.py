import logging
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import sympy as sp
from ruamel.yaml import YAML, constructor, scanner

from pymatlib.core.materials import Material
from pymatlib.parsing.processors.property_processor import PropertyProcessor
from pymatlib.parsing.validation.property_type_detector import PropertyType, PropertyTypeDetector
from pymatlib.visualization.plotters import PropertyVisualizer
from pymatlib.parsing.config.yaml_keys import PROPERTIES_KEY, MATERIAL_TYPE_KEY, \
    COMPOSITION_KEY, PURE_METAL_KEY, MELTING_TEMPERATURE_KEY, BOILING_TEMPERATURE_KEY, SOLIDUS_TEMPERATURE_KEY, \
    LIQUIDUS_TEMPERATURE_KEY, INITIAL_BOILING_TEMPERATURE_KEY, FINAL_BOILING_TEMPERATURE_KEY, ALLOY_KEY, NAME_KEY

logger = logging.getLogger(__name__)

class BaseFileParser:
    """Base class for parsing configuration files."""

    def __init__(self, config_path: Union[str, Path]) -> None:
        self.config_path = Path(config_path)
        self.base_dir = self.config_path.parent
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement _load_config method")

class YAMLFileParser(BaseFileParser):
    """Parser for YAML configuration files."""

    def _load_config(self) -> Dict[str, Any]:
        yaml = YAML(typ='safe')
        yaml.allow_duplicate_keys = False
        try:
            with open(self.config_path, 'r') as f:
                return yaml.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"YAML file not found: {self.config_path}") from e
        except constructor.DuplicateKeyError as e:
            raise constructor.DuplicateKeyError(f"Duplicate key in {self.config_path}: {str(e)}") from e
        except scanner.ScannerError as e:
            raise scanner.ScannerError(f"YAML syntax error in {self.config_path}: {str(e)}") from e
        except Exception as e:
            raise ValueError(f"Error parsing {self.config_path}: {str(e)}") from e

class MaterialYAMLParser(YAMLFileParser):
    """Parser for material configuration files in YAML format."""

    VALID_YAML_PROPERTIES = {
        'density',
        'dynamic_viscosity',
        'energy_density',
        'heat_capacity',
        'heat_conductivity',
        'kinematic_viscosity',
        'latent_heat_of_fusion',
        'latent_heat_of_vaporization',
        'specific_enthalpy',
        'surface_tension',
        'thermal_diffusivity',
        'thermal_expansion_coefficient',
    }

    # --- Constructor ---
    def __init__(self, yaml_path: Union[str, Path]) -> None:
        super().__init__(yaml_path)
        self._validate_config()
        self.categorized_properties = self._analyze_and_categorize_properties(self.config[PROPERTIES_KEY])
        self.property_processor = PropertyProcessor()
        self.visualizer = PropertyVisualizer(self)

    # --- Public API ---
    def create_material(self, T: Union[float, sp.Symbol], enable_plotting: bool = True) -> Material:
        """Create a Material instance from the parsed configuration and temperature."""
        logger.info(f"Creating material from configuration: {self.config_path}")
        try:
            name = self.config.get(NAME_KEY, "Unnamed Material")
            material_type = self.config[MATERIAL_TYPE_KEY]
            logger.debug(f"Material name: {name}, type: {material_type}")
            elements = self._get_elements()
            composition = [val for val in self.config[COMPOSITION_KEY].values()]
            # Create material with different parameters based on material_type
            if material_type == PURE_METAL_KEY:
                material = Material(
                    name=name,
                    elements=elements,
                    composition=composition,
                    material_type=material_type,
                    melting_temperature=sp.Float(self.config[MELTING_TEMPERATURE_KEY]),
                    boiling_temperature=sp.Float(self.config[BOILING_TEMPERATURE_KEY]),
                )
            else:  # alloy
                material = Material(
                    name=name,
                    elements=elements,
                    composition=composition,
                    material_type=material_type,
                    solidus_temperature=sp.Float(self.config[SOLIDUS_TEMPERATURE_KEY]),
                    liquidus_temperature=sp.Float(self.config[LIQUIDUS_TEMPERATURE_KEY]),
                    initial_boiling_temperature=sp.Float(self.config[INITIAL_BOILING_TEMPERATURE_KEY]),
                    final_boiling_temperature=sp.Float(self.config[FINAL_BOILING_TEMPERATURE_KEY]),
                )
            # Initialize visualizer only if plotting is enabled
            visualizer = None
            if enable_plotting:
                self.visualizer.initialize_plots()
                self.visualizer.reset_visualization_tracking()
                visualizer = self.visualizer
            self.property_processor.process_properties(
                material=material,
                T=T,
                properties=self.config[PROPERTIES_KEY],
                categorized_properties=self.categorized_properties,
                base_dir=self.base_dir,
                visualizer=visualizer
            )
            # Save plots only if plotting was enabled
            if enable_plotting:
                self.visualizer.save_property_plots()
            logger.info(f"Successfully created material: {name}")
            return material
        except KeyError as e:
            logger.error(f"Configuration error for material creation: Missing {str(e)}", exc_info=True)
            raise ValueError(f"Configuration error: Missing {str(e)}") from e
        except Exception as e:
            logger.error(f"Failed to create material from {self.config_path}: {e}", exc_info=True)
            raise ValueError(f"Failed to create material \n -> {str(e)}") from e

    # --- Validation Methods ---
    def _validate_config(self) -> None:
        if not isinstance(self.config, dict):
            raise ValueError("The YAML file must start with a dictionary/object structure with key-value pairs, not a list or scalar value")
        self._validate_required_fields()
        properties = self.config.get(PROPERTIES_KEY, {})
        if not isinstance(properties, dict):
            raise ValueError("The 'properties' section in your YAML file must be a dictionary with key-value pairs")
        self._validate_property_names(properties)

    def _validate_required_fields(self) -> None:
        # Check for material_type first
        if MATERIAL_TYPE_KEY not in self.config:
            raise ValueError("Missing required field: material_type")
        material_type = self.config[MATERIAL_TYPE_KEY]
        if material_type not in [ALLOY_KEY, PURE_METAL_KEY]:
            raise ValueError(f"Invalid material_type: {material_type}. Must be {PURE_METAL_KEY} or {ALLOY_KEY}")
        # Common required fields
        common_fields = {NAME_KEY, MATERIAL_TYPE_KEY, COMPOSITION_KEY, PROPERTIES_KEY}
        # Material-type specific fields
        if material_type == PURE_METAL_KEY:
            required_fields = common_fields | {MELTING_TEMPERATURE_KEY, BOILING_TEMPERATURE_KEY}
        else:  # alloy
            required_fields = common_fields | {SOLIDUS_TEMPERATURE_KEY, LIQUIDUS_TEMPERATURE_KEY,
                                               INITIAL_BOILING_TEMPERATURE_KEY, FINAL_BOILING_TEMPERATURE_KEY}
        missing_fields = required_fields - set(self.config.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields for {material_type}: {', '.join(missing_fields)}")
        extra_fields = set(self.config.keys()) - required_fields
        if extra_fields:
            suggestions = {
                field: get_close_matches(field, required_fields, n=1, cutoff=0.6)
                for field in extra_fields
            }
            error_msg = "Extra fields found in configuration: \n ->"
            for field, matches in suggestions.items():
                suggestion = f" (did you mean '{matches[0]}'?)" if matches else ""
                error_msg += f" - '{field}'{suggestion}\n"
            raise ValueError(error_msg)
        self._validate_composition()

    def _validate_composition(self) -> None:
        """Validate composition for both pure metals and alloys."""
        composition = self.config.get(COMPOSITION_KEY, {})
        material_type = self.config[MATERIAL_TYPE_KEY]
        if not isinstance(composition, dict):
            raise ValueError("Composition must be a dictionary")
        if not composition:
            raise ValueError("Composition cannot be empty")
        # Check that all fractions are valid numbers
        for element, fraction in composition.items():
            if not isinstance(fraction, (int, float)):
                raise ValueError(f"Composition fraction for '{element}' must be a number, got {type(fraction).__name__}")
            if fraction < 0:
                raise ValueError(f"Composition fraction for '{element}' cannot be negative, got {fraction}")
            if fraction > 1:
                raise ValueError(f"Composition fraction for '{element}' cannot exceed 1.0, got {fraction}")
        # Check that fractions sum to 1.0
        total = sum(composition.values())
        if not abs(total - 1.0) < 1e-10:
            raise ValueError(f"Composition fractions must sum to 1.0, got {total}")
        # Material-type specific validation
        if material_type == PURE_METAL_KEY:
            self._validate_pure_metal_composition_rules(composition)
        else:  # alloy
            self._validate_alloy_composition_rules(composition)

    @staticmethod
    def _validate_pure_metal_composition_rules(composition: dict) -> None:
        """Validate composition rules specific to pure metals."""
        # Count non-zero elements
        non_zero_elements = {element: fraction for element, fraction in composition.items()
                             if fraction > 1e-10}
        if len(non_zero_elements) == 0:
            raise ValueError("Pure metals must have at least one element with non-zero composition")

        if len(non_zero_elements) > 1:
            element_list = ", ".join(f"{elem}: {frac}" for elem, frac in non_zero_elements.items())
            raise ValueError(
                f"Pure metals must contain exactly one element with composition 1.0. "
                f"Found multiple non-zero elements: {element_list}. "
                f"Use material_type: 'alloy' for multi-element materials."
            )
        # Check that the single element has composition 1.0
        single_element, single_fraction = list(non_zero_elements.items())[0]
        if not abs(single_fraction - 1.0) < 1e-10:
            raise ValueError(
                f"Pure metal element '{single_element}' must have composition 1.0, "
                f"got {single_fraction}. Use material_type: 'alloy' for fractional compositions."
            )
        # ERROR for zero-valued elements in pure metals
        zero_elements = [element for element, fraction in composition.items() if fraction == 0.0]
        if zero_elements:
            raise ValueError(
                f"Pure metal composition should not include zero-valued elements: {zero_elements}. "
                f"Remove these elements from the composition dictionary."
            )

    @staticmethod
    def _validate_alloy_composition_rules(composition: dict) -> None:
        """Validate composition rules specific to alloys."""
        non_zero_elements = {element: fraction for element, fraction in composition.items()
                             if fraction > 1e-10}
        if len(non_zero_elements) < 2:
            if len(non_zero_elements) == 1:
                single_element = list(non_zero_elements.keys())[0]
                raise ValueError(
                    f"Alloys must have at least 2 elements with non-zero composition. "
                    f"Found only '{single_element}'. Use material_type: 'pure_metal' for single elements."
                )
            else:
                raise ValueError("Alloys must have at least 2 elements with non-zero composition")
        # Warning for zero-valued elements in alloys - they might be intentional
        zero_elements = [element for element, fraction in composition.items() if fraction == 0.0]
        if zero_elements:
            logger.warning(
                f"Alloy composition includes zero-valued elements: {zero_elements}. "
                f"Consider removing them if they're not needed."
            )

    def _validate_property_names(self, properties: Dict[str, Any]) -> None:
        invalid_props = set(properties.keys()) - self.VALID_YAML_PROPERTIES
        if invalid_props:
            suggestions = {
                prop: get_close_matches(prop, self.VALID_YAML_PROPERTIES, n=1, cutoff=0.6)
                for prop in invalid_props
            }
            error_msg = "Invalid properties found: \n ->"
            for prop, matches in suggestions.items():
                suggestion = f" (did you mean '{matches[0]}'?)" if matches else ""
                error_msg += f" - '{prop}'{suggestion}\n"
            raise ValueError(error_msg)

    # --- Processing Methods ---
    def _get_elements(self) -> List:
        from pymatlib.data.elements.element_data import element_map
        try:
            return [element_map[sym] for sym in self.config[COMPOSITION_KEY].keys()]
        except KeyError as e:
            raise ValueError(f"Invalid element symbol: {str(e)}") from e

    @staticmethod
    def _analyze_and_categorize_properties(properties: Dict[str, Any]) -> Dict[PropertyType, List[Tuple[str, Any]]]:
        """Categorizes properties after detecting and validating their types."""
        categorized_properties: Dict[PropertyType, List[Tuple[str, Any]]] = {
            prop_type: [] for prop_type in PropertyType
        }
        for prop_name, config in properties.items():
            try:
                prop_type = PropertyTypeDetector.determine_property_type(prop_name, config)
                PropertyTypeDetector.validate_property_config(prop_name, config, prop_type)
                categorized_properties[prop_type].append((prop_name, config))
            except ValueError as e:
                raise ValueError(f"Configuration error for property '{prop_name}': {str(e)}") from e
        return categorized_properties
