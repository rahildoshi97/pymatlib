import numpy as np
import sympy as sp
from pathlib import Path
from difflib import get_close_matches
from ruamel.yaml import YAML, constructor, scanner
from typing import Dict, Any, Union, List, Tuple

from pymatlib.core.alloy import Alloy
from pymatlib.core.yaml_parser.property_types import PropertyType, PropertyTypeDetector
from pymatlib.core.yaml_parser.property_processing import PropertyProcessor
from pymatlib.core.yaml_parser.visualization import PropertyVisualizer
from pymatlib.core.yaml_parser.common_utils import ensure_ascending_order

import logging
logger = logging.getLogger(__name__)

class ConfigParser:
    """Base class for parsing configuration files."""

    def __init__(self, config_path: Union[str, Path]) -> None:
        logger.debug("""ConfigParser: __init__:
            config_path: %r""", config_path)
        self.config_path = Path(config_path)
        self.base_dir = self.config_path.parent
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement _load_config method")

class YAMLConfigParser(ConfigParser):
    """Parser for YAML configuration files."""

    def _load_config(self) -> Dict[str, Any]:
        logger.debug("""YAMLConfigParser: _load_config:
            config_path: %r""", self.config_path)
        yaml = YAML(typ='safe')
        yaml.allow_duplicate_keys = False
        try:
            with open(self.config_path, 'r') as f:
                return yaml.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML file not found: {self.config_path}")
        except constructor.DuplicateKeyError as e:
            raise constructor.DuplicateKeyError(f"Duplicate key in {self.config_path}: {e}")
        except scanner.ScannerError as e:
            raise scanner.ScannerError(f"YAML syntax error in {self.config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error parsing {self.config_path}: {str(e)}")

class MaterialConfigParser(YAMLConfigParser):
    """
    Parser for material configuration files in YAML format.
    """

    # --- Class Constants ---
    MIN_POINTS = 2
    EPSILON = 1e-10
    ABSOLUTE_ZERO = 0.0
    VALID_YAML_PROPERTIES = {
        'boiling_temperature',
        'density',
        'dynamic_viscosity',
        'energy_density',
        'heat_capacity',
        'heat_conductivity',
        'kinematic_viscosity',
        'latent_heat_of_fusion',
        'latent_heat_of_vaporization',
        'melting_temperature',
        'specific_enthalpy',
        'surface_tension',
        'thermal_diffusivity',
        'thermal_expansion_coefficient',
    }

    # --- Constructor ---
    def __init__(self, yaml_path: Union[str, Path]) -> None:
        logger.debug("""MaterialConfigParser: __init__:
            yaml_path: %r""", yaml_path)
        super().__init__(yaml_path)
        self._validate_config()
        self.temperature_array = self._process_temperature_range(self.config['temperature_range'])
        self.categorized_properties = self._categorize_properties(self.config['properties'])
        self.property_processor = PropertyProcessor()
        self.visualizer = PropertyVisualizer(self)
        logger.debug(f"Finished loading configuration from {yaml_path}")

    # --- Public API ---
    def create_alloy(self, T: Union[float, sp.Symbol]) -> Alloy:
        """
        Create an Alloy instance from the parsed configuration and temperature.
        """
        print("\n")
        logger.debug("""MaterialConfigParser: create_alloy:
            T: %r""", T)
        try:
            alloy = Alloy(
                elements=self._get_elements(),
                composition=[val for val in self.config['composition'].values()],
                solidus_temperature=sp.Float(self.config['solidus_temperature']),
                liquidus_temperature=sp.Float(self.config['liquidus_temperature'])
            )
            self.visualizer.initialize_plots()
            self.visualizer.reset_visualization_tracking()
            self.property_processor.process_properties(
                alloy=alloy,
                T=T,
                properties=self.config['properties'],
                categorized_properties=self.categorized_properties,
                temperature_array=self.temperature_array,
                base_dir=self.base_dir,
                visualizer=self.visualizer
            )
            self.visualizer.save_property_plots()
            return alloy
        except KeyError as e:
            raise ValueError(f"Configuration error: Missing {e}")
        except Exception as e:
            raise ValueError(f"Failed to create alloy \n -> {e}")

    # --- Validation Methods ---
    def _validate_config(self) -> None:
        logger.debug("""MaterialConfigParser: _validate_config""")
        if not isinstance(self.config, dict):
            raise ValueError("The YAML file must start with a dictionary/object structure with key-value pairs, not a list or scalar value")
        self._validate_required_fields()
        properties = self.config.get('properties', {})
        if not isinstance(properties, dict):
            raise ValueError("The 'properties' section in your YAML file must be a dictionary with key-value pairs")
        self._validate_property_names(properties)

    def _validate_required_fields(self) -> None:
        logger.debug(f"MaterialConfigParser: _validate_required_fields")
        required_fields = {'name', 'composition', 'solidus_temperature', 'liquidus_temperature', 'temperature_range', 'properties'}
        missing_fields = required_fields - set(self.config.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        extra_fields = set(self.config.keys()) - required_fields
        if extra_fields:
            suggestions = {
                field: get_close_matches(field, required_fields, n=1, cutoff=0.6)
                for field in extra_fields
            }
            error_msg = "Extra fields found in configuration:\n"
            for field, matches in suggestions.items():
                suggestion = f" (did you mean '{matches[0]}'?)" if matches else ""
                error_msg += f" - '{field}'{suggestion}\n"
            raise ValueError(error_msg)

    def _validate_property_names(self, properties: Dict[str, Any]) -> None:
        logger.debug("""MaterialConfigParser: _validate_property_names:
            properties: %r""", properties)
        invalid_props = set(properties.keys()) - self.VALID_YAML_PROPERTIES
        if invalid_props:
            suggestions = {
                prop: get_close_matches(prop, self.VALID_YAML_PROPERTIES, n=1, cutoff=0.6)
                for prop in invalid_props
            }
            error_msg = "Invalid properties found:\n"
            for prop, matches in suggestions.items():
                suggestion = f" (did you mean '{matches[0]}'?)" if matches else ""
                error_msg += f" - '{prop}'{suggestion}\n"
            raise ValueError(error_msg)

    # --- Processing Methods ---
    def _process_temperature_range(self, array_def: List[Union[int, float]]) -> np.ndarray:
        logger.debug("MaterialConfigParser: _process_temperature_range: array_def: %r", array_def)

        # Validate input format
        if not (isinstance(array_def, list) and len(array_def) == 3):
            raise ValueError("Temperature array must be defined as [start, end, points/delta]")

        # Extract and validate values
        start, end, step = float(array_def[0]), float(array_def[1]), array_def[2]

        # Check for temperatures below absolute zero
        if start <= self.ABSOLUTE_ZERO or end <= self.ABSOLUTE_ZERO:
            raise ValueError(f"Temperature must be above absolute zero ({self.ABSOLUTE_ZERO}K), got {start}K and {end}K")

        # Check for zero step
        if isinstance(step, (int, float)) and abs(float(step)) <= self.EPSILON:
            raise ValueError("Delta or number of points cannot be zero")

        # Create array based on step type
        if isinstance(step, int):
            # Handle number of points
            if step <= 0:
                raise ValueError(f"Number of points must be positive, got {step}")
            if step < self.MIN_POINTS:
                raise ValueError(f"Number of points must be at least {self.MIN_POINTS}, got {step}")

            temperature_array = np.linspace(start, end, step)
        else:
            # Handle step size
            if (start < end and step <= 0) or (start > end and step >= 0):
                raise ValueError("Delta sign must match range direction (positive for increasing, negative for decreasing)")

            if abs(step) > abs(end - start):
                raise ValueError(f"Absolute value of delta ({abs(step)}) is too large for the range. It should be <= {abs(end - start)}")

            temperature_array = np.arange(start, end + step/2, step)

        # Ensure array is in ascending order
        temperature_array, = ensure_ascending_order(temperature_array)

        return temperature_array

    def _get_elements(self) -> List:
        from pymatlib.data.element_data import element_map
        logger.debug("""MaterialConfigParser: _get_elements:
            composition: %r""", self.config['composition'])
        try:
            return [element_map[sym] for sym in self.config['composition'].keys()]
        except KeyError as e:
            raise ValueError(f"Invalid element symbol: {e}")

    @staticmethod
    def _categorize_properties(properties: Dict[str, Any]) -> Dict[PropertyType, List[Tuple[str, Any]]]:
        logger.debug("""MaterialConfigParser: _categorize_properties:
            properties: %r""", properties)
        categorized_properties: Dict[PropertyType, List[Tuple[str, Any]]] = {
            prop_type: [] for prop_type in PropertyType
        }
        for prop_name, config in properties.items():
            try:
                prop_type = PropertyTypeDetector.determine_property_type(prop_name, config)
                if prop_type == PropertyType.INVALID:
                    raise ValueError(f"Invalid configuration format for property '{prop_name}': {config}")
                categorized_properties[prop_type].append((prop_name, config))
            except Exception as e:
                raise ValueError(f"Failed to categorize properties \n -> {e}")
        return categorized_properties
