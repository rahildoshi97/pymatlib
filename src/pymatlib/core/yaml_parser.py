import os
import re
import pwlf
import numpy as np
import sympy as sp
from pathlib import Path
from enum import Enum, auto
from difflib import get_close_matches
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, Any, Union, List, Tuple, Set
from ruamel.yaml import YAML, constructor, scanner
from pymatlib.core.alloy import Alloy
from pymatlib.core.elements import ChemicalElement
from pymatlib.core.data_handler import read_data_from_file
from pymatlib.core.pwlfsympy import get_symbolic_conditions
from pymatlib.core.symbol_registry import SymbolRegistry


class PropertyType(Enum):
    CONSTANT = auto()
    FILE = auto()
    KEY_VAL = auto()
    COMPUTE = auto()
    INVALID = auto()


class MaterialConfigParser:

    ################################################## ##################################################
    # Class Constants and Attributes
    ################################################## ##################################################

    MIN_POINTS = 2
    EPSILON = 1e-10  # Small value to handle floating point comparisons
    ABSOLUTE_ZERO = 0.0  # Kelvin

    # Define valid properties as class-level constants
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

    ################################################## ##################################################
    # Initialization and YAML Loading
    ################################################## ##################################################

    def __init__(self, yaml_path: str | Path) -> None:
        """Initialize parser with YAML file path.
        Args:
            yaml_path: Path to the YAML configuration file
        Raises:
            FileNotFoundError: If YAML file is not found
            ValueError: If configuration is invalid
            constructor.DuplicateKeyError: If duplicate keys are found
            scanner.ScannerError: If YAML syntax is invalid
        """
        print("__init__")
        self.yaml_path = Path(yaml_path)
        self.base_dir = Path(yaml_path).parent
        self.config: Dict[str, Any] = self._load_yaml()
        self._validate_config()
        self.temperature_array = self._process_temperature_range(self.config['temperature_range'])
        # print(self.temperature_array)
        # Initialize with an empty dict for each property type
        self.categorized_properties = {prop_type: [] for prop_type in PropertyType}  #TODO: Do we need this?

    def _load_yaml(self) -> Dict[str, Any]:
        """Load and parse YAML file.
        Returns:
            Dict containing parsed YAML content
        Raises:
            FileNotFoundError: If YAML file is not found
            constructor.DuplicateKeyError: If duplicate keys are found
            scanner.ScannerError: If YAML syntax is invalid
        """
        print("_load_yaml")
        yaml = YAML(typ='safe')
        yaml.allow_duplicate_keys = False
        try:
            with open(self.yaml_path, 'r') as f:
                return yaml.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML file not found: {self.yaml_path}")
        except constructor.DuplicateKeyError as e:
            raise constructor.DuplicateKeyError(f"Duplicate key in {self.yaml_path}: {e}")
        except scanner.ScannerError as e:
            raise scanner.ScannerError(f"YAML syntax error in {self.yaml_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error parsing {self.yaml_path}: {str(e)}")

    ################################################## ##################################################
    # Configuration Validation
    ################################################## ##################################################

    def _validate_config(self) -> None:
        """
        Validate YAML configuration structure and content.
        This method checks the overall structure of the configuration,
        validates property names, required fields, and property values.
        Raises:
            ValueError: If any part of the configuration is invalid.
        """
        print("_validate_config")
        if not isinstance(self.config, dict):
            # raise ValueError("Root YAML element must be a mapping")
            raise ValueError("The YAML file must start with a dictionary/object structure with key-value pairs, not a list or scalar value")
        self._validate_required_fields()
        properties = self.config.get('properties', {})
        if not isinstance(properties, dict):
            raise ValueError("The 'properties' section in your YAML file must be a dictionary with key-value pairs")
        self._validate_property_names(properties)

    def _validate_required_fields(self) -> None:
        """
        Validate required configuration fields.
        Raises:
            ValueError: If any required field is missing.
        """
        print("_validate_required_fields")
        required_fields = {'name', 'composition', 'solidus_temperature', 'liquidus_temperature', 'temperature_range', 'properties'}
        missing_fields = required_fields - set(self.config.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        # Check for extra fields
        extra_fields = set(self.config.keys()) - required_fields
        if extra_fields:
            suggestions = {
                field: get_close_matches(field, required_fields, n=1, cutoff=0.6)
                for field in extra_fields
            }
            error_msg = "Extra fields found in configuration:\n"
            for field, matches in suggestions.items():
                suggestion = f" (did you mean '{matches[0]}'?)" if matches else ""
                error_msg += f"  - '{field}'{suggestion}\n"
            raise ValueError(error_msg)

    def _validate_property_names(self, properties: Dict[str, Any]) -> None:
        """
        Validate property names against the allowed set.
        Args:
            properties (Dict[str, Any]): Dictionary of properties to validate.
        Raises:
            ValueError: If any property name is not in VALID_PROPERTIES.
        """
        print("_validate_property_names")
        invalid_props = set(properties.keys()) - self.VALID_YAML_PROPERTIES
        if invalid_props:
            suggestions = {
                prop: get_close_matches(prop, self.VALID_YAML_PROPERTIES, n=1, cutoff=0.6)
                for prop in invalid_props
            }
            error_msg = "Invalid properties found:\n"
            for prop, matches in suggestions.items():
                suggestion = f" (did you mean '{matches[0]}'?)" if matches else ""
                error_msg += f"  - '{prop}'{suggestion}\n"
            raise ValueError(error_msg)

    ################################################## ##################################################
    # Temperature Array Processing
    ################################################## ##################################################

    def _process_temperature_range(self, array_def: List[Union[float, int]]) -> np.ndarray:
        """
        Process temperature array definition with format [start, end, points/delta].
        Args:
            array_def (List[Union[float, int]]): A list defining the temperature array in the format
                                                 [start, end, points/delta].
        Returns:
            np.ndarray: An array of temperature values.
        Raises:
            ValueError: If the array definition is invalid, improperly formatted,
                        or contains physically impossible temperatures.
        Examples:
            >>> self._process_temperature_range([300, 3000, 5])
            array([ 300., 975., 1650., 2325., 3000.])
            >>> self._process_temperature_range([3000, 300, -1350.])
            array([3000., 1650., 300.])
        """
        print("_process_temperature_range")
        if not (isinstance(array_def, list) and len(array_def) == 3):
            raise ValueError("Temperature array must be defined as [start, end, points/delta]")
        try:
            start, end, step = float(array_def[0]), float(array_def[1]), array_def[2]
            if start <= self.ABSOLUTE_ZERO or end <= self.ABSOLUTE_ZERO:
                raise ValueError(f"Temperature must be above absolute zero ({self.ABSOLUTE_ZERO}K)")
            if abs(float(step)) < self.EPSILON:
                raise ValueError("Delta or number of points cannot be zero.")
            # Check if the step represents delta (float) or points (int)
            if isinstance(step, float):
                temperature_array = self._process_float_step(start, end, step)
            else:
                temperature_array = self._process_int_step(start, end, int(step))
            # Ensure the temperature array is in ascending order
            if not np.all(np.diff(temperature_array) >= 0):
                # print("_process_temperature_range -> Flipping temperature array")
                temperature_array = np.flip(temperature_array)
            return temperature_array
        except ValueError as e:
            raise ValueError(f"Invalid temperature array definition \n -> {e}")

    @staticmethod
    def _process_float_step(start: float, end: float, delta: float) -> np.ndarray:
        """Process the temperature array with a float step (delta)."""
        print("_process_float_step")
        # print(f"Processing TA as float: start={start}, end={end}, delta={delta}")
        if start < end and delta <= 0:
            raise ValueError("Delta must be positive for increasing range")
        if start > end and delta >= 0:
            raise ValueError("Delta must be negative for decreasing range")
        max_delta = abs(end - start)
        if abs(delta) > max_delta:
            raise ValueError(f"Absolute value of delta ({abs(delta)}) is too large for the range. It should be <= {max_delta}")
        return np.arange(start, end + delta/2, delta)

    def _process_int_step(self, start: float, end: float, points: int) -> np.ndarray:
        """Process the temperature array with an integer step (number of points)."""
        print("_process_int_step")
        # print(f"Processing TA as int: start={start}, end={end}, points={points}")
        if points <= 0:
            raise ValueError(f"Number of points must be positive, got {points}!")
        if points < self.MIN_POINTS:
            raise ValueError(f"Number of points must be at least {self.MIN_POINTS}, got {points}!")
        return np.linspace(start, end, points)

    def _validate_temperature_range(self, prop: str, temp_array: np.ndarray) -> None:
        """
        Validate that temperature arrays from properties fall within the global temperature range.
        Args:
            prop (str): The name of the property being validated
            temp_array (np.ndarray): The temperature array to validate
        Raises:
            ValueError: If temperature values fall outside the global temperature range
        """
        print("_validate_temperature_range")
        # Extract global temperature range from config
        # temp_range = self.config.get('temperature_range', [])
        temp_range = self.temperature_array
        min_temp, max_temp = np.min(temp_range), np.max(temp_range)
        # Check for temperatures outside the global range
        if ((temp_array < min_temp) | (temp_array > max_temp)).any():
            out_of_range = np.where((temp_array < min_temp) | (temp_array > max_temp))[0]
            out_values = temp_array[out_of_range]
            # Create a more detailed error message
            if len(out_of_range) > 5:
                # Show just a few examples if there are many out-of-range values
                sample_indices = out_of_range[:5]
                sample_values = temp_array[sample_indices]
                raise ValueError(f"Property '{prop}' contains temperature values outside global range [{min_temp}, {max_temp}] "
                                 f"\n -> Found {len(out_of_range)} out-of-range values, first 5 at indices {sample_indices}: {sample_values}"
                                 f"\n -> Min value: {temp_array.min()}, Max value: {temp_array.max()}")
            else:
                # Show all out-of-range values if there aren't too many
                raise ValueError(f"Property '{prop}' contains temperature values outside global range [{min_temp}, {max_temp}] "
                                 f"\n -> Found {len(out_of_range)} out-of-range values at indices {out_of_range}: {out_values}")

    ################################################## ##################################################
    # Alloy Creation
    ################################################## ##################################################

    def create_alloy(self, T: Union[float, sp.Symbol]) -> Alloy:
        """
        Creates an Alloy instance from YAML configuration.
        Args:
            T (Union[float, sp.Symbol]): Temperature value or symbol.
        Returns:
            Alloy: An instance of the Alloy class.
        Raises:
            ValueError: If there's an error in creating the alloy.
        """
        try:
            print("create_alloy")
            alloy = Alloy(
                elements=self._get_elements(),
                # composition=list(self.config['composition'].values()),
                composition=[val for val in self.config['composition'].values()],
                solidus_temperature=sp.Float(self.config['solidus_temperature']),
                liquidus_temperature=sp.Float(self.config['liquidus_temperature'])
            )
            self._process_properties(alloy, T)
            return alloy
        except KeyError as e:
            raise ValueError(f"Configuration error: Missing {e}")
        except Exception as e:
            raise ValueError(f"Failed to create alloy \n -> {e}")

    def _get_elements(self) -> List[ChemicalElement]:
        """
        Convert element symbols to ChemicalElement instances.
        Returns:
            List[ChemicalElement]: List of ChemicalElement instances.
        Raises:
            ValueError: If an invalid element symbol is encountered.
        """
        from pymatlib.data.element_data import element_map
        print("_get_elements")
        try:
            return [element_map[sym] for sym in self.config['composition'].keys()]
        except KeyError as e:
            raise ValueError(f"Invalid element symbol: {e}")

    ################################################## ##################################################
    # Property Type Checking
    ################################################## ##################################################
    @staticmethod
    def _is_numeric(value: Any) -> bool:
        """
        Check if a string represents a float number (including scientific notation).
        Args:
            value (Any): The value to check.
        Returns:
            bool: True if the value represents a float, False otherwise.
        """
        print(f"_is_numeric: {value}")
        if isinstance(value, float):
            return True
        if isinstance(value, str):
            try:
                float(value)
                # return True
                # Ensure it contains a decimal point or is in scientific notation
                return '.' in value or 'e' in value.lower()
            except ValueError:
                return False
        return False

    @staticmethod
    def _validate_keys(value: Dict, required_keys: Set[str], optional_keys: Set[str], context: str) -> None:
        """
        Validate that a dictionary contains all required keys and no unexpected keys.
        Args:
            value (Dict): The dictionary to validate.
            required_keys (Set[str]): Set of required keys.
            optional_keys (Set[str]): Set of optional keys.
            context (str): Context for error messages.
        Raises:
            ValueError: If required keys are missing or unexpected keys are present.
        """
        print(f"_validate_keys: {value}")
        value_keys = set(value.keys())
        # Check for missing required keys
        missing_keys = required_keys - value_keys
        if missing_keys:
            raise ValueError(f"Missing required keys for {context}: {missing_keys}")
        # Check for extra keys (excluding optional keys)
        extra_keys = value_keys - required_keys - optional_keys
        if extra_keys:
            raise ValueError(f"Extra keys found in {context}: {extra_keys}. "
                             f"Allowed keys are: {required_keys | optional_keys}")

    @staticmethod
    def _validate_bounds(bounds, context: str = "bound") -> None:
        """
        Validate bounds configuration structure.
        Args:
            bounds: The bounds configuration to validate.
            context (str): Context for error messages.
        Raises:
            ValueError: If the bounds configuration is invalid.
        """
        print(f"_validate_bounds: {bounds}")
        if not isinstance(bounds, list):
            raise ValueError(f"{context}s must be a list")
        if len(bounds) != 2:
            raise ValueError(f"{context} must have exactly two elements")
        valid_bound_types = {'constant', 'extrapolate'}
        if bounds[0] not in valid_bound_types:
            raise ValueError(f"Lower {context} type must be one of: {valid_bound_types}, got '{bounds[0]}'")
        if bounds[1] not in valid_bound_types:
            raise ValueError(f"Upper {context} type must be one of: {valid_bound_types}, got '{bounds[1]}'")

    @staticmethod
    def _validate_regression(regression: Union[Dict, Any]) -> None:
        """
        Validate regression configuration structure.
        Args:
            regression (Dict): The regression configuration to validate.
        Raises:
            ValueError: If the regression configuration is invalid.
        """
        print(f"_validate_regression: {regression}")
        # Ensure regression is a dictionary
        if not isinstance(regression, dict):
            raise ValueError(f"Regression must be a dictionary, got {type(regression).__name__}")
        # Use _validate_keys for key validation
        required_keys = {'simplify', 'degree', 'segments'}
        optional_keys = set()  # No optional keys for regression
        MaterialConfigParser._validate_keys(regression, required_keys, optional_keys, "regression")
        # Validate regression values
        if not isinstance(regression['simplify'], str) or regression['simplify'] not in {'pre', 'post'}:
            raise ValueError(f"Invalid regression simplify type '{regression['simplify']}'. Must be 'pre', or 'post'")
        if not isinstance(regression['degree'], int) or regression['degree'] < 1:
            raise ValueError("Regression degree must be a positive integer")
        if not isinstance(regression['segments'], int) or regression['segments'] < 1:
            raise ValueError("Regression segments must be an integer >= 1")

    @staticmethod
    def _is_data_file(value: str | Dict[str, str]) -> bool:
        """
        Check if the value represents a valid data file configuration.
        Args:
            value (Union[str, Dict[str, str]]): The value to check.
        Returns:
            bool: True if it's a valid data file configuration, False otherwise.
        Raises:
            ValueError: If the file configuration is invalid or contains extra keys.
        """
        print(f"_is_data_file: {value}")
        # Simple format: property_name: "filename.txt"  #TODO: Deprecated!
        if isinstance(value, str):
            return value.endswith(('.txt', '.csv', '.xlsx'))
        # Advanced format: property_name: { file: "filename", temp_col: "col1", prop_col: "col2" }
        if isinstance(value, dict) and 'file' in value:
            required_keys = {'file', 'temp_col', 'prop_col', 'bounds'}
            optional_keys = {'regression'}
            MaterialConfigParser._validate_keys(value, required_keys, optional_keys, "file configuration")
            # Validate bounds if present
            if 'bounds' in value:
                MaterialConfigParser._validate_bounds(value['bounds'], "file configuration bound")
            # Validate regression structure if present
            if 'regression' in value:
                if not isinstance(value['regression'], dict):
                    raise ValueError(f"Regression must be a dictionary, got {type(value['regression']).__name__} instead")
                MaterialConfigParser._validate_regression(value['regression'])
            return True
        return False

    @staticmethod
    def _is_key_val_property(value: Dict) -> bool:
        """
        Check if a property is defined with key-val pairs.
        Args:
            value (Any): The value to check.
        Returns:
            bool: True if it's a key-val property, False otherwise.
        Raises:
            ValueError: If the key-val property configuration is invalid.
        """
        print(f"_is_key_val_property: {value}")
        required_keys = {'key', 'val', 'bounds'}
        # Check if it looks like it's trying to be a key-val property
        if isinstance(value, dict) and all(k in value for k in required_keys):
            optional_keys = {'regression'}
            MaterialConfigParser._validate_keys(value, required_keys, optional_keys, "key-val configuration")
            # Validate bounds if present
            if 'bounds' in value:
                MaterialConfigParser._validate_bounds(value['bounds'], "key-val configuration bound")
            # Validate regression structure if present
            if 'regression' in value:
                MaterialConfigParser._validate_regression(value['regression'])
            return True
        return False

    @staticmethod
    def _is_compute_property(value: Any) -> bool:
        """
        Check if a property should be computed using any valid format.
        Args:
            value (Any): The value to check.
        Returns:
            bool: True if it's a compute property, False otherwise.
        Raises:
            ValueError: If the compute property configuration is invalid.
        """
        print(f"_is_compute_property: {value}")
        # Simple format: property_name: compute equation
        if isinstance(value, str):
            # New format: direct mathematical expression as string
            # Check if it contains any mathematical operators or function calls
            math_operators = ['+', '-', '*', '/', '**']  #TODO: maybe extend with '(', ')', ' '?
            return any(op in value for op in math_operators)
        # Advanced format: property_name: { equation: compute equation }
        elif isinstance(value, dict) and 'equation' in value:
            # Validate the structure
            required_keys = {'equation', 'bounds'}
            optional_keys = {'regression'}
            # Validate keys
            MaterialConfigParser._validate_keys(value, required_keys, optional_keys, "compute configuration")
            # Validate bounds if present
            if 'bounds' in value:
                MaterialConfigParser._validate_bounds(value['bounds'], "compute configuration bound")
            # Validate regression structure if present
            if 'regression' in value:
                MaterialConfigParser._validate_regression(value['regression'])
            return True
        return False

    ################################################## ##################################################
    # Property Categorization
    ################################################## ##################################################

    def _determine_property_type(self, prop_name: str, config: Any) -> PropertyType:
        """
        Determine the type of property based on its configuration.
        Args:
            prop_name (str): The name of the property.
            config (Any): The configuration of the property.
        Returns:
            PropertyType: The determined property type.
        """
        print(f"_determine_property_type: {prop_name}, {config}")
        try:
            if isinstance(config, int):
                raise ValueError(f"Property '{prop_name}' must be defined as a float, got {config} of type {type(config).__name__}")
            if self._is_numeric(config):
                return PropertyType.CONSTANT
            elif self._is_data_file(config):
                return PropertyType.FILE
            elif self._is_key_val_property(config):
                return PropertyType.KEY_VAL
            elif self._is_compute_property(config):
                return PropertyType.COMPUTE
            else:
                return PropertyType.INVALID
        except Exception as e:
            raise ValueError(f"Failed to determine property type \n -> {e}")

    def _categorize_properties(self, properties: Dict[str, Any]) -> Dict[PropertyType, List[Tuple[str, Any]]]:
        """
        Categorize properties based on their types.
        Args:
            properties (Dict[str, Any]): Dictionary of properties to categorize.
        Returns:
            Dict[PropertyType, List[Tuple[str, Any]]]: Categorized properties.
        Raises:
            ValueError: If an invalid property configuration is found.
        """
        print(f"_categorize_properties: {properties}")
        categorized_properties: Dict[PropertyType, List[Tuple[str, Any]]] = {
            PropertyType.CONSTANT: [],
            PropertyType.FILE: [],
            PropertyType.KEY_VAL: [],
            PropertyType.COMPUTE: [],
        }
        for prop_name, config in properties.items():
            try:
                prop_type = self._determine_property_type(prop_name, config)
                if prop_type == PropertyType.INVALID:
                    raise ValueError(f"Invalid configuration format for property '{prop_name}': {config}")
                categorized_properties[prop_type].append((prop_name, config))
            except Exception as e:
                # Provide more context in the error message
                raise ValueError(f"Failed to categorize properties \n -> {e}")
        return categorized_properties

    ################################################## ##################################################
    # Property Plotting
    ################################################## ##################################################

    def _initialize_plots(self):
        """Initialize the figure and subplots for property visualization"""
        print(f"_initialize_plots: {self.config['name']}")
        if self.categorized_properties is None:
            print("Warning: categorized_properties is None. Skipping plot initialization.")
            raise ValueError("No properties to plot.")
            # return

        # Count properties of each type to determine subplot layout
        property_count = sum(len(props) for props in self.categorized_properties.values()) + 1  #TODO: Remove +1?
        # print(f"property_count: {property_count}")

        # Create a figure with the appropriate size
        self.fig = plt.figure(figsize=(12, 4 * property_count))
        self.gs = GridSpec(property_count, 1, figure=self.fig)
        self.current_subplot = 0
        self.plot_directory = "property_plots"
        os.makedirs(self.plot_directory, exist_ok=True)

    def _save_property_plots(self):
        """Save all property plots to a single file"""
        print(f"_save_property_plots: {self.config['name']}")
        if hasattr(self, 'fig'):
            # Add overall title
            self.fig.suptitle(f"Material Properties: {self.config['name']}", fontsize=16)
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            # Save the figure
            filepath = os.path.join(self.plot_directory, f"{self.config['name'].replace(' ', '_')}_properties.png")
            self.fig.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"All properties plot saved as {filepath}")
            # Show the plot
            # plt.show()
            # Close the figure to free memory
            plt.close(self.fig)

    ################################################## ##################################################
    # Property Processing
    ################################################## ##################################################

    def _process_properties(self, alloy: Alloy, T: Union[float, sp.Symbol]) -> None:
        """
        Process all properties for the alloy.
        Args:
            alloy (Alloy): The alloy object to process properties for.
            T (Union[float, sp.Symbol]): Temperature value or symbol.
        Raises:
            ValueError: If there's an error processing any property.
        """
        print(f"_process_properties: {alloy}, {T}")
        properties = self.config['properties']
        try:
            self.categorized_properties = self._categorize_properties(properties)
            # print(self.categorized_properties)
            # print(type(self.categorized_properties))

            # Always initialize plots, regardless of temperature type
            self._initialize_plots()

            for prop_type, prop_list in self.categorized_properties.items():
                for prop_name, config in prop_list:
                    # print(f"SymbolRegistry.get_all(): {SymbolRegistry.get_all()}")
                    # Create a SymPy symbol for the property
                    # prop_symbol = SymbolRegistry.get(prop_name)
                    # print(f"prop_name: {prop_name}, type: {type(prop_name)}")
                    # print(f"prop_symbol: {prop_symbol}, type: {type(prop_symbol)}")
                    # print(f"SymbolRegistry.get_all(): {SymbolRegistry.get_all()}")

                    # quit()
                    if prop_type == PropertyType.CONSTANT and prop_name in ['latent_heat_of_fusion', 'latent_heat_of_vaporization']:
                        self._process_latent_heat_constant(alloy, prop_name, config, T)  # Ends up in _process_key_val_property
                    elif prop_type == PropertyType.CONSTANT:
                        self._process_constant_property(alloy, prop_name, config, T)
                    elif prop_type == PropertyType.FILE:
                        self._process_file_property(alloy, prop_name, config, T)
                    elif prop_type == PropertyType.KEY_VAL:
                        self._process_key_val_property(alloy, prop_name, config, T)
                    elif prop_type == PropertyType.COMPUTE:
                        print(f"_process_computed_property for {prop_name}")
                        self._process_computed_property(alloy, prop_name, T)

            # Perform post-processing for 'post' simplification
            self._post_process_properties(alloy, T)

            # Save all plots if they were created
            if hasattr(self, 'fig'):
                self._save_property_plots()

            # print(f"prop_symbol: {SymbolRegistry.get('thermal_diffusivity')}, type: {type(SymbolRegistry.get('thermal_diffusivity'))}")
            # print(f"SymbolRegistry.get_all(): {SymbolRegistry.get_all()}")
            # SymbolRegistry.clear()
            # print(f"SymbolRegistry.clear(): {SymbolRegistry.get_all()}")
        except Exception as e:
            raise ValueError(f"Failed to process properties \n -> {e}")

    ########################################################################################################################

    def _process_latent_heat_constant(self, alloy: Alloy, prop_name: str, prop_config: Union[float, str], T: Union[float, sp.Symbol]) -> None:
        """
        Process latent heat properties when provided as constants.
        This automatically expands them to key-val pairs using solidus and liquidus temperatures.
        Args:
            alloy (Alloy): The alloy object to update.
            prop_name (str): The name of the property to set ('latent_heat_of_fusion' or 'latent_heat_of_vaporization').
            prop_config (Union[float, str]): The constant latent heat value.
            T (Union[float, sp.Symbol]): Temperature value or symbol.
        """
        print(f"_process_latent_heat_constant: {prop_name}, {prop_config}")
        try:
            # Convert to float
            latent_heat_value = float(prop_config)
            # print(latent_heat_value)
            # Create an expanded key-val configuration
            if prop_name == 'latent_heat_of_fusion':
                # For fusion, heat is absorbed between solidus and liquidus
                expanded_config = {
                    'key': ['solidus_temperature', 'liquidus_temperature'],
                    'val': [0, latent_heat_value],
                    'bounds': ['constant', 'constant'],
                    'regression': {
                        'simplify': 'pre',
                        'degree': 1,
                        'segments': 1,
                    },
                }
            elif prop_name == 'latent_heat_of_vaporization':
                # For vaporization, heat is absorbed at boiling point
                # Assume boiling happens after liquidus temperature
                expanded_config = {
                    'key': ['boiling_temperature-10', 'boiling_temperature+10'],
                    'val': [0, latent_heat_value],
                    'bounds': ['constant', 'constant'],
                    'regression': {
                        'simplify': 'pre',
                        'degree': 1,
                        'segments': 1,
                    },
                }
            else:
                raise ValueError(f"Unsupported latent heat configuration: {prop_name}")
            # Process using the standard key-val method
            self._process_key_val_property(alloy, prop_name, expanded_config, T)
        except (ValueError, TypeError) as e:
            error_msg = f"Failed to process {prop_name} constant \n -> {e}"
            raise ValueError(error_msg) from e

    ########################################################################################################################

    # @staticmethod
    def _process_constant_property(self, alloy: Alloy, prop_name: str, prop_config: Union[float, str], T: Union[float, sp.Symbol]) -> None:
        """
        Process constant float property.
        Args:
            alloy (Alloy): The alloy object to update.
            prop_name (str): The name of the property to set.
            prop_config (Union[float, str]): The property value or string representation.
            T (Union[float, sp.Symbol]): Temperature value or symbol.
        Raises:
            ValueError: If the property value cannot be converted to float or violates constraints.
        """
        print(f"_process_constant_property: {prop_name}, {prop_config}")
        try:
            # print(f"{prop_name}, {type(prop_name)} = {prop_config}, {type(prop_config)}")
            value = float(prop_config)
            # print(f"{prop_name}, {type(prop_name)} = {value}, {type(value)}")
            setattr(alloy, prop_name, sp.Float(value))  # save as sympy Float

            # Visualization for constant properties
            self._visualize_property(
                alloy=alloy,
                prop_name=prop_name,
                T=T,
                prop_type='Constant',
                lower_bound=min(self.temperature_array),
                upper_bound=max(self.temperature_array)
                # No need to pass x_data or y_data for constant properties
                # No need to pass regression parameters for constant properties
            )
        except (ValueError, TypeError) as e:
            error_msg = f"Failed to process constant property \n -> {e}"
            raise ValueError(error_msg) from e

    ########################################################################################################################

    @staticmethod
    def _create_raw_piecewise(temp_array, prop_array, T, lower_bound_type, upper_bound_type):
        """Create a piecewise function using all data points."""
        print(f"_create_raw_piecewise: {temp_array}, {prop_array}, {T}, {lower_bound_type}, {upper_bound_type}")
        # Create a piecewise function with one segment per data point
        '''segments = len(temp_array) - 1
        v_pwlf = pwlf.PiecewiseLinFit(temp_array, prop_array, degree=1)
        print(f"segments: {segments}")
        if segments > 8:
            segments = 8
        v_pwlf.fit(n_segments=segments)
        print(f"v_pwlf: {v_pwlf}")
        return sp.Piecewise(*get_symbolic_conditions(v_pwlf, T, lower_bound_type, upper_bound_type))'''

        if temp_array[0] > temp_array[-1]:
            temp_array = np.flip(temp_array)
            prop_array = np.flip(prop_array)

        # TODO: Use temp values as break points and use fit_with_breaks
        # https://jekel.me/piecewise_linear_fit_py/examples.html
        '''my_pwlf = pwlf.PiecewiseLinFit(temp_array, prop_array, degree=1)
        my_pwlf.fit_with_breaks(temp_array)
        pw = sp.Piecewise(*get_symbolic_conditions(my_pwlf, T, lower_bound_type, upper_bound_type))'''  # Works but too slow

        conditions = []

        # Handle lower boundary
        if lower_bound_type == 'constant':
            conditions.append((prop_array[0], T < temp_array[0]))
        else:  # 'extrapolate'
            if len(temp_array) >= 2:
                slope = (prop_array[1] - prop_array[0]) / (temp_array[1] - temp_array[0])
                extrapolated_expr = prop_array[0] + slope * (T - temp_array[0])
                conditions.append((extrapolated_expr, T < temp_array[0]))

        # Handle upper boundary
        if upper_bound_type == 'constant':
            conditions.append((prop_array[-1], T >= temp_array[-1]))
        else:  # 'extrapolate'
            if len(temp_array) >= 2:
                slope = (prop_array[-1] - prop_array[-2]) / (temp_array[-1] - temp_array[-2])
                extrapolated_expr = prop_array[-1] + slope * (T - temp_array[-1])
                conditions.append((extrapolated_expr, T >= temp_array[-1]))

        # Add conditions for each segment between data points
        for i in range(len(temp_array) - 1):
            interp_expr = (
                    prop_array[i] + (prop_array[i + 1] - prop_array[i]) /
                    (temp_array[i + 1] - temp_array[i]) * (T - temp_array[i])
            )
            # Add condition for this segment - always use a bounded condition
            conditions.append(
                (interp_expr, sp.And(T >= temp_array[i], T < temp_array[i + 1])))

        # Create and return the piecewise function
        pw = sp.Piecewise(*conditions)
        # print(f"Created piecewise function with {len(conditions)} conditions")
        print(f"pw (_create_raw_piecewise): {pw}")
        return pw

    @staticmethod
    def _interpolate_value(T, x_array, y_array, lower_bound_type, upper_bound_type):
        """
        Interpolate a value at temperature T using the provided data arrays.
        Args:
            T (float): Temperature value
            x_array (np.ndarray): Temperature array
            y_array (np.ndarray): Property value array
            lower_bound_type (str): Type of lower bound ('constant' or 'extrapolate')
            upper_bound_type (str): Type of upper bound ('constant' or 'extrapolate')
        Returns:
            float: Interpolated value
        """
        print(f"_interpolate_value: {T}, {x_array}, {y_array}")
        # Handle interpolation/extrapolation
        if T < x_array[0]:
            if lower_bound_type == 'constant':
                return y_array[0]
            else:  # 'extrapolate'
                slope = (y_array[1] - y_array[0]) / (x_array[1] - x_array[0])
                return y_array[0] + slope * (T - x_array[0])
        elif T >= x_array[-1]:
            if upper_bound_type == 'constant':
                return y_array[-1]
            else:  # 'extrapolate'
                slope = (y_array[-1] - y_array[-2]) / (x_array[-1] - x_array[-2])
                return y_array[-1] + slope * (T - x_array[-1])
        else:
            return np.interp(T, x_array, y_array)

    @staticmethod
    def _process_regression_params(prop_config, prop_name, data_length):
        """
        Process regression parameters from configuration.
        Args:
            prop_config (dict): Property configuration
            prop_name (str): Property name
            data_length (int): Length of a data array
        Returns:
            tuple: (has_regression, simplify_type, degree, segments)
        """
        print(f"_process_regression_params: {prop_name}, {prop_config}")
        # Check if regression is specified
        has_regression = isinstance(prop_config, dict) and 'regression' in prop_config

        if not has_regression:
            print(f"_process_regression_params: False, None, None, None")
            return False, None, None, None

        # Only process if regression is specified
        regression_config = prop_config['regression']
        simplify_type = regression_config.get('simplify', 'before')
        degree = regression_config.get('degree', 1)
        segments = regression_config.get('segments', 3)

        # Validate segments
        if segments >= data_length:
            raise ValueError(f"Number of segments ({segments}) must be less than number of data points ({data_length}) ")
        if segments > 8:
            raise ValueError(f"Number of segments ({segments}) is too high for {prop_name}. Please reduce it.")
        elif segments > 6:
            print(f"Warning: Number of segments ({segments}) for {prop_name} may lead to overfitting.")

        return has_regression, simplify_type, degree, segments

    def _process_file_property(self, alloy: Alloy, prop_name: str, file_config: Union[str, Dict[str, Any]], T: Union[float, sp.Symbol]) -> None:
        """
        Process property data from a file configuration.
        Args:
            alloy (Alloy): The alloy object to update.
            prop_name (str): The name of the property to set.
            file_config (Union[str, Dict[str, Any]]): File path or configuration dictionary.
            T (Union[float, sp.Symbol]): Temperature value or symbol.
        Raises:
            ValueError: If there's an error processing the file data.
        """
        print(f"_process_file_property: {prop_name}, {file_config}")
        try:
            # Get the directory containing the YAML file
            yaml_dir = self.base_dir

            # Construct a path relative to the YAML file location
            if isinstance(file_config, dict) and 'file' in file_config:
                file_config['file'] = str(yaml_dir / file_config['file'])
                temp_array, prop_array = read_data_from_file(file_config)
            else:
                # For string configuration, construct the full path
                file_path = str(yaml_dir / file_config)
                temp_array, prop_array = read_data_from_file(file_path)
            # Check for nan or inf in temp_array or prop_array
            if not (np.all(np.isfinite(temp_array)) and np.all(np.isfinite(prop_array))):
                bad_temps = np.where(~np.isfinite(temp_array))[0]
                bad_props = np.where(~np.isfinite(prop_array))[0]
                msg = []
                if bad_temps.size > 0:
                    msg.append(f"temp_array contains non-finite values at indices: {bad_temps.tolist()}")
                if bad_props.size > 0:
                    msg.append(f"prop_array contains non-finite values at indices: {bad_props.tolist()}")
                raise ValueError(f"Non-finite values detected in property '{prop_name}': " + "; ".join(msg))
            # Validate the temperature array
            self._validate_temperature_range(prop_name, temp_array)

            # Ensure the temperature array is in ascending order
            if not np.all(np.diff(temp_array) >= 0):
                # print("Flipping temperature array")
                temp_array = np.flip(temp_array)
                prop_array = np.flip(prop_array)

            # Extract bound types (always specified for FILE properties)
            lower_bound_type = file_config['bounds'][0]
            upper_bound_type = file_config['bounds'][1]

            # Get the bounds of the data
            lower_bound = np.min(temp_array)
            upper_bound = np.max(temp_array)

            # Check if T is a symbolic variable or a numeric value
            is_symbolic = isinstance(T, sp.Symbol)

            # CASE 1: Numeric Temperature (T is a float)
            if not is_symbolic:
                # Interpolate value
                interpolated_value = self._interpolate_value(
                    T, temp_array, prop_array, lower_bound_type, upper_bound_type)

                # Set property value
                setattr(alloy, prop_name, sp.Float(interpolated_value))

                # No visualization for numeric temperature
                return

            # CASE 2: Symbolic Temperature (T is a sp.Symbol)
            # Process regression parameters - only needed for symbolic temperature - will return None values if regression not specified
            has_regression, simplify_type, degree, segments = self._process_regression_params(
                file_config, prop_name, len(temp_array))

            # Create symbolic representation based on regression parameters
            if has_regression:
                # Use regression parameters for simplification
                if simplify_type == 'pre':
                    # Simplify immediately
                    v_pwlf = pwlf.PiecewiseLinFit(temp_array, prop_array, degree=degree, seed=13579)
                    v_pwlf.fit(n_segments=segments)
                    pw = sp.Piecewise(*get_symbolic_conditions(v_pwlf, T, lower_bound_type, upper_bound_type))
                    setattr(alloy, prop_name, pw)
                else:  # simplify_type == 'post'
                    # Use raw data for now, simplify later
                    raw_pw = self._create_raw_piecewise(temp_array, prop_array, T, lower_bound_type, upper_bound_type)
                    setattr(alloy, prop_name, raw_pw)
            else:  # Regression is not specified
                # Always use raw data when regression is not specified
                raw_pw = self._create_raw_piecewise(temp_array, prop_array, T, lower_bound_type, upper_bound_type)
                setattr(alloy, prop_name, raw_pw)

            # Visualization for symbolic temperature only
            self._visualize_property(
                alloy=alloy,
                prop_name=prop_name,
                T=T,
                prop_type='File',
                x_data=temp_array,
                y_data=prop_array,
                has_regression=has_regression,
                simplify_type=simplify_type,
                degree=degree,
                segments=segments,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                lower_bound_type=lower_bound_type,
                upper_bound_type=upper_bound_type
            )

        except Exception as e:
            error_msg = f"Failed to process file property {prop_name} \n -> {str(e)}"
            raise ValueError(error_msg) from e

    ########################################################################################################################

    def _process_key_val_property(self, alloy: Alloy, prop_name: str, prop_config: Dict, T: Union[float, sp.Symbol]) -> None:
        """
        Process property defined with key-val pairs.
        Args:
            alloy (Alloy): The alloy object to update.
            prop_name (str): The name of the property to set.
            prop_config (Dict[str, Any]): The property configuration dictionary.
            T (Union[float, sp.Symbol]): Temperature value or symbol.
        Raises:
            ValueError: If there's an error processing the key-val property.
        """
        try:
            print(f"Process key val property: {prop_name}")
            key_array = self._process_key_definition(prop_config['key'], prop_config['val'], alloy)
            print(f"key-val property: {key_array}")
            val_array = np.array(prop_config['val'], dtype=float)
            print(f"val array: {val_array}")

            if len(key_array) != len(val_array):
                raise ValueError(f"Length mismatch in {prop_name}: key and val arrays must have same length")

            # Validate the temperature array
            # self._validate_temperature_range(prop_name, key_array)  #TODO: Commented out -> boiling temperature in latent_heat_of_vaporization is higher than 3000. causes error

            # Ensure the temperature array is in ascending order
            if not np.all(np.diff(key_array) >= 0):
                print("Flipping temperature array")
                key_array = np.flip(key_array)
                val_array = np.flip(val_array)

            # Extract bound types (always specified for KEY_VAL properties)
            lower_bound_type = prop_config['bounds'][0]
            upper_bound_type = prop_config['bounds'][1]

            # Get the bounds of the data
            lower_bound = np.min(key_array)
            upper_bound = np.max(key_array)

            # Check if T is a symbolic variable or a numeric value
            is_symbolic = isinstance(T, sp.Symbol)

            # CASE 1: Numeric Temperature (T is a float)
            if not is_symbolic:
                # Interpolate value
                interpolated_value = self._interpolate_value(
                    T, key_array, val_array, lower_bound_type, upper_bound_type)

                # Set property value
                setattr(alloy, prop_name, sp.Float(interpolated_value))

                # No visualization for numeric temperature
                return

            # CASE 2: Symbolic Temperature (T is a sp.Symbol)
            # Process regression parameters - only needed for symbolic temperature - will return None values if regression not specified
            has_regression, simplify_type, degree, segments = self._process_regression_params(
                prop_config, prop_name, len(key_array))

            # Create symbolic representation based on regression parameters
            if has_regression:
                # Use regression parameters for simplification
                if simplify_type == 'pre':
                    # Simplify immediately
                    v_pwlf = pwlf.PiecewiseLinFit(key_array, val_array, degree=degree, seed=13579)
                    v_pwlf.fit(n_segments=segments)
                    pw = sp.Piecewise(*get_symbolic_conditions(v_pwlf, T, lower_bound_type, upper_bound_type))
                    setattr(alloy, prop_name, pw)
                else:  # simplify_type == 'post'
                    # Use raw data now, simplify later
                    raw_pw = self._create_raw_piecewise(key_array, val_array, T, lower_bound_type, upper_bound_type)
                    setattr(alloy, prop_name, raw_pw)
            else:  # Regression is not specified
                # Always use raw data when regression is not specified
                raw_pw = self._create_raw_piecewise(key_array, val_array, T, lower_bound_type, upper_bound_type)
                setattr(alloy, prop_name, raw_pw)

            # Visualization for both symbolic and numeric temperature
            self._visualize_property(
                alloy=alloy,
                prop_name=prop_name,
                T=T,
                prop_type='Key-Value',
                x_data=key_array,
                y_data=val_array,
                has_regression=has_regression,
                simplify_type=simplify_type,
                degree=degree,
                segments=segments,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                lower_bound_type=lower_bound_type,
                upper_bound_type=upper_bound_type
            )

        except Exception as e:
            error_msg = f"Failed to process key-val property '{prop_name}' \n -> {str(e)}"
            raise ValueError(error_msg) from e

    def _process_key_definition(self, key_def, val_array, alloy: Alloy) -> np.ndarray:
        """
        Process temperature key definition.
        Args:
            key_def (Union[str, List[Union[str, float]]]): The key definition.
            val_array (List[float]): The value array.
            alloy (Alloy): The alloy object.
        Returns:
            np.ndarray: Processed key array.
        Raises:
            ValueError: If there's an error processing the key definition.
        """
        try:
            if isinstance(key_def, str) and key_def.startswith('(') and key_def.endswith(')'):
                return self._process_equidistant_key(key_def, len(val_array))
            elif isinstance(key_def, list):
                return self._process_list_key(key_def, alloy)
            else:
                raise ValueError(f"Invalid key definition: {key_def}")
        except Exception as e:
            error_msg = f"Failed to process key definition \n -> {str(e)}"
            raise ValueError(error_msg) from e

    @staticmethod
    def _process_equidistant_key(key_def: str, n_points: int) -> np.ndarray:
        """
        Process equidistant key definition.
        Args:
            key_def (str): The equidistant key definition string.
            n_points (int): Number of points.
        Returns:
            np.ndarray: Equidistant key array.
        Raises:
            ValueError: If there's an error processing the equidistant key.
        """
        try:
            values = [float(x.strip()) for x in key_def.strip('()').split(',')]
            if len(values) != 2:
                raise ValueError("Equidistant definition must have exactly two values: (start, increment)")
            start, increment = values
            key_array = np.arange(start, start + increment * n_points, increment)
            return key_array
        except Exception as e:
            error_msg = f"Invalid equidistant format: {key_def} \n -> {str(e)}"
            raise ValueError(error_msg) from e

    @staticmethod
    def _process_list_key(key_def: List, alloy: Alloy) -> np.ndarray:
        """
        Process list key definition.
        Args:
            key_def (List[Union[str, float]]): The list key definition.
            alloy (Alloy): The alloy object.
        Returns:
            np.ndarray: Processed list key array.
        Raises:
            ValueError: If there's an error processing the list key.
        """
        print(f"processing list key: {key_def}")
        try:
            processed_key = []
            for k in key_def:
                print(f"processing key: {k}")
                if isinstance(k, str):
                    # Handle base temperature references
                    if k == 'solidus_temperature':
                        processed_key.append(alloy.solidus_temperature)
                    elif k == 'liquidus_temperature':
                        processed_key.append(alloy.liquidus_temperature)
                    elif k == 'boiling_temperature':
                        processed_key.append(alloy.boiling_temperature)
                    # Handle temperature expressions like 'liquidus_temperature+300'
                    elif '+' in k:
                        # Split the string into base and offset
                        base, offset = k.split('+')
                        offset_value = float(offset)
                        # Get the base temperature
                        if base == 'solidus_temperature':
                            base_value = alloy.solidus_temperature
                        elif base == 'liquidus_temperature':
                            base_value = alloy.liquidus_temperature
                        elif base == 'boiling_temperature':
                            base_value = alloy.boiling_temperature
                        else:
                            base_value = float(base)
                        # Calculate the final temperature
                        processed_key.append(base_value + offset_value)
                    elif '-' in k:
                        # Split the string into base and offset
                        base, offset = k.split('-')
                        offset_value = -float(offset)
                        # Get the base temperature
                        if base == 'solidus_temperature':
                            base_value = alloy.solidus_temperature
                        elif base == 'liquidus_temperature':
                            base_value = alloy.liquidus_temperature
                        elif base == 'boiling_temperature':
                            base_value = alloy.boiling_temperature
                        else:
                            base_value = float(base)
                        # Calculate the final temperature
                        processed_key.append(base_value + offset_value)
                    else:
                        processed_key.append(float(k))
                else:
                    processed_key.append(float(k))
            key_array = np.array(processed_key, dtype=float)
            return key_array
        except Exception as e:
            error_msg = f"Error processing list key \n -> {str(e)}"
            raise ValueError(error_msg) from e

    ################################################## ##################################################
    # Computed Property Handling
    ################################################## ##################################################

    def _process_computed_property(self, alloy: Alloy, prop_name: str, T: Union[float, sp.Symbol]) -> None:
        """
        Process computed properties using predefined models with dependency checking.
        Args:
            alloy (Alloy): The alloy object to process.
            prop_name (str): The name of the property to compute.
            T (Union[float, Symbol]): The temperature value or symbol.
        Raises:
            ValueError: If no computation method is defined for the property or if the method is unknown.
        """
        try:
            prop_config = self.config['properties'][prop_name]
            print(f"prop_config: {prop_config}")
            # Create a symbol for the property
            # prop_symbol = SymbolRegistry.get(prop_name)
            # print(f"prop_symbol: {prop_symbol}")
            # Handle direct expression format
            if isinstance(prop_config, str):
                # This is a direct mathematical expression
                expression = prop_config
                print(f"_process_computed_property for property: {prop_name} with expression: {expression}")
                material_property = self._parse_and_process_expression(expression, alloy, T)
                print(f"material_property {prop_name}: {material_property}")
            elif isinstance(prop_config, dict) and 'equation' in prop_config:
                expression = prop_config['equation']
                print(f"_process_computed_property for property: {prop_name} with expression: {expression}")
                material_property = self._parse_and_process_expression(expression, alloy, T)
                print(f"material_property {prop_name}: {material_property}")
            else:
                raise ValueError(f"Unsupported property configuration format for {prop_name}: {prop_config}")

            setattr(alloy, prop_name, material_property)

            # Extract bound types
            lower_bound_type = 'constant'
            upper_bound_type = 'constant'
            if isinstance(prop_config, dict) and 'bounds' in prop_config:
                if isinstance(prop_config['bounds'], list) and len(prop_config['bounds']) == 2:
                    lower_bound_type = prop_config['bounds'][0]
                    upper_bound_type = prop_config['bounds'][1]

            # Get the bounds of the data
            lower_bound = np.min(self.temperature_array)
            upper_bound = np.max(self.temperature_array)

            # Process regression parameters
            data_length = len(self.temperature_array)
            if isinstance(T, sp.Symbol):
                f = sp.lambdify(T, material_property, 'numpy')
                try:
                    prop_array = f(self.temperature_array)
                    valid_indices = np.isfinite(prop_array)
                    data_length = np.sum(valid_indices)
                except Exception as e:
                    print(f"Warning: Could not evaluate {prop_name} at all temperature points: {e}")

            has_regression, simplify_type, degree, segments = self._process_regression_params(
                prop_config, prop_name, data_length)

            # For computed properties, default to 'post' if no regression
            if not has_regression:
                simplify_type = 'post'

            # --- THIS BLOCK SHOULD NOT BE NESTED UNDER "if not has_regression" ---
            # For 'pre' simplification, apply simplification immediately
            if has_regression and simplify_type == 'pre' and isinstance(T, sp.Symbol):
                temp_array = self.temperature_array
                f = sp.lambdify(T, material_property, 'numpy')
                try:
                    prop_array = f(temp_array)
                    valid_indices = np.isfinite(prop_array)
                    if not np.all(valid_indices):
                        print(f"Warning: Found {np.sum(~valid_indices)} non-finite values in {prop_name}. Filtering them out.")
                        temp_array = temp_array[valid_indices]
                        prop_array = prop_array[valid_indices]
                    if len(temp_array) < 2:
                        print(f"Warning: Not enough valid points to fit {prop_name}. Using original expression.")
                    else:
                        v_pwlf = pwlf.PiecewiseLinFit(temp_array, prop_array, degree=degree, seed=13579)
                        v_pwlf.fit(n_segments=segments)
                        pw = sp.Piecewise(*get_symbolic_conditions(v_pwlf, T, lower_bound_type, upper_bound_type))
                        material_property = pw
                        print(f"Simplified {prop_name} with 'pre' simplification")
                except Exception as e:
                    print(f"Warning: Failed to simplify {prop_name} with 'pre': {e}")

            # Set the property on the alloy (again, in case it was simplified)
            setattr(alloy, prop_name, material_property)

            # Visualization (unchanged)
            if isinstance(T, sp.Symbol):
                self._visualize_property(
                    alloy=alloy,
                    prop_name=prop_name,
                    T=T,
                    prop_type='Computed',
                    has_regression=has_regression,
                    simplify_type=simplify_type,
                    degree=degree,
                    segments=segments,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    lower_bound_type=lower_bound_type,
                    upper_bound_type=upper_bound_type
                )
        except Exception as e:
            error_msg = f"Failed to process computed property '{prop_name}' \n -> {str(e)}"
            raise ValueError(error_msg) from e

    def _parse_and_process_expression(self, expression: str, alloy: Alloy, T: Union[float, sp.Symbol]) -> sp.Expr:
        """
        Parse and process a mathematical expression string into a SymPy expression.
        Args:
            expression (str): The mathematical expression as a string.
            alloy (Alloy): The alloy object containing property values.
            T (Union[float, sp.Symbol]): The temperature value or symbol.
        Returns:
            sp.Expr: The processed SymPy expression.
        Raises:
            ValueError: If the expression cannot be parsed, or if dependencies are missing.
        """
        try:
            # Convert the expression to a SymPy expression
            sympy_expr = sp.sympify(expression)
            print(f"sympy_expr: {sympy_expr}, type: {type(sympy_expr)}")

            # Extract dependencies (free symbols)
            dependencies = [str(symbol) for symbol in sympy_expr.free_symbols]
            print(f"dependencies: {dependencies}")

            # Special handling for temperature symbol
            if 'T' in dependencies and isinstance(T, sp.Symbol):
                print(f"removing T from {dependencies}")
                dependencies.remove('T')
                print(f"new dependencies: {dependencies}")

            # Check for circular dependencies
            self._check_circular_dependencies(prop_name=None, current_deps=dependencies, visited=set())

            # Process dependencies
            for dep in dependencies:
                if not hasattr(alloy, dep) or getattr(alloy, dep) is None:
                    print(f"processing dependencies for dep: {dep}")
                    if dep in self.config['properties']:
                        dep_config = self.config['properties'][dep]
                        print(f"dep_config: {dep_config}, dep: {dep}")
                        self._process_computed_property(alloy, dep, T)
                    else:
                        raise ValueError(f"Dependency '{dep}' not found in properties configuration")

            # Verify all dependencies are available
            missing_deps = [dep for dep in dependencies if not hasattr(alloy, dep) or getattr(alloy, dep) is None]
            if missing_deps:
                raise ValueError(f"Cannot compute expression. Missing dependencies: {missing_deps}")

            # Substitute dependencies with their values
            substitutions = {}
            for dep in dependencies:
                print(f"dep: {dep}, type: {type(dep)}")
                dep_value = getattr(alloy, dep)
                print(f"dep_value: {dep_value}, type: {type(dep_value)}, dep: {dep}")
                dep_symbol = SymbolRegistry.get(dep)
                print(f"dep_symbol: {dep_symbol}, type: {type(dep_symbol)}, dep: {dep}")
                substitutions[dep_symbol] = dep_value
                print(f"substitutions: {substitutions}")

            # If T is a symbol, keep it as a symbol in the expression
            if isinstance(T, sp.Symbol):
                substitutions[sp.Symbol('T')] = T
                print(f"substitutions: {substitutions}")

            # Substitute the values
            result_expr = sympy_expr.subs(substitutions)
            print(f"result_expr: {result_expr}")
            return result_expr

        except Exception as e:
            raise ValueError(f"Failed to parse and process expression: {expression} \n -> {str(e)}")

    def _check_circular_dependencies(self, prop_name, current_deps, visited, path=None):
        """
        Check for circular dependencies in property definitions.
        Args:
            prop_name: The current property being checked
            current_deps: List of dependencies for the current property
            visited: Set of properties already visited in this branch
            path: Current dependency path for error reporting
        Raises:
            ValueError: If a circular dependency is detected
        """
        if path is None:
            path = []

        if prop_name is not None:
            if prop_name in visited:
                cycle_path = path + [prop_name]
                raise ValueError(f"Circular dependency detected: {' -> '.join(cycle_path)}")

            visited.add(prop_name)
            path = path + [prop_name]

        for dep in current_deps:
            if dep in self.config['properties']:
                dep_config = self.config['properties'][dep]

                # Extract dependencies for this property
                if isinstance(dep_config, str):
                    # Direct expression
                    expr = sp.sympify(dep_config)
                    dep_deps = [str(symbol) for symbol in expr.free_symbols if str(symbol) != 'T']
                elif isinstance(dep_config, dict) and 'equation' in dep_config:
                    # Dictionary with equation
                    expr = sp.sympify(dep_config['equation'])
                    dep_deps = [str(symbol) for symbol in expr.free_symbols if str(symbol) != 'T']
                else:
                    # Not a computed property, no dependencies
                    dep_deps = []

                # Recursively check dependencies
                if dep_deps:
                    self._check_circular_dependencies(dep, dep_deps, visited.copy(), path)

    ################################################## ##################################################
    # Post-Processing for 'post' Simplification
    ################################################## ##################################################

    def _post_process_properties(self, alloy: Alloy, T: Union[float, sp.Symbol]) -> None:
        """Perform post-processing on properties after all have been initially processed."""
        print(f"_post_process_properties: {self.config['properties']}")
        # Skip post-processing entirely if T is a float
        if not isinstance(T, sp.Symbol):
            print("Skipping post-processing for numeric temperature")
            return

        properties = self.config['properties']

        for prop_name, prop_config in properties.items():
            try:
                # Check if this property needs post-processing
                if not isinstance(prop_config, dict) or 'regression' not in prop_config:
                    continue

                regression_config = prop_config['regression']
                simplify_type = regression_config.get('simplify', 'pre')

                # Determine if this property should use 'post' simplification
                is_after = False
                if simplify_type == 'post':
                    is_after = True

                # Skip if not using 'post' simplification
                if not is_after:
                    continue

                # Get the property value
                prop_value = getattr(alloy, prop_name)

                # Skip if it's not a symbolic expression
                if not isinstance(prop_value, sp.Expr):
                    print(f"Skipping {prop_name} - not a symbolic expression (type: {type(prop_value)})")
                    continue

                # Get regression parameters
                degree = regression_config.get('degree', 1)
                segments = regression_config.get('segments', 3)

                # Get bound types
                lower_bound_type = 'constant'
                upper_bound_type = 'constant'
                if 'bounds' in prop_config:
                    if isinstance(prop_config['bounds'], list) and len(prop_config['bounds']) == 2:
                        lower_bound_type = prop_config['bounds'][0]
                        upper_bound_type = prop_config['bounds'][1]

                # Create a simplified version
                temp_array = self.temperature_array
                f = sp.lambdify(T, prop_value, 'numpy')

                try:
                    prop_array = f(temp_array)

                    # Check for NaN or Inf values
                    valid_indices = np.isfinite(prop_array)
                    if not np.all(valid_indices):
                        print(f"Warning: Found {np.sum(~valid_indices)} non-finite values in {prop_name}. Filtering them out.")
                        temp_array = temp_array[valid_indices]
                        prop_array = prop_array[valid_indices]

                        # Skip if not enough valid points
                        if len(temp_array) < 2:
                            print(f"Warning: Not enough valid points to fit {prop_name}. Skipping post-processing.")
                            continue

                    # Special handling for latent heat properties
                    if prop_name in ['latent_heat_of_fusion', 'latent_heat_of_vaporization']:
                        # Use only 1 segment for latent heat properties
                        segments = 1

                    v_pwlf = pwlf.PiecewiseLinFit(temp_array, prop_array, degree=degree, seed=13579)
                    v_pwlf.fit(n_segments=segments)
                    pw = sp.Piecewise(*get_symbolic_conditions(v_pwlf, T, lower_bound_type, upper_bound_type))

                    # Update the property
                    setattr(alloy, prop_name, pw)
                    print(f"Post-processed {prop_name} with simplify type 'post'")

                except Exception as e:
                    print(f"Warning: Failed to post-process {prop_name}: {e}")
                    # Continue with other properties instead of failing completely
                    continue

            except Exception as e:
                print(f"Warning: Error processing {prop_name} in post-processing: {e}")
                # Continue with other properties
                continue

    def _visualize_property(self, alloy, prop_name, T, prop_type,
                            x_data=None, y_data=None,
                            has_regression=False, simplify_type=None,
                            degree=1, segments=3,
                            lower_bound=None, upper_bound=None,
                            lower_bound_type='constant', upper_bound_type='constant'):
        """
        Unified visualization function for all property types and temperature formats.
        """
        print(f"_visualize_property for property: {prop_name} with type: {prop_type}")
        # Skip if visualization is not enabled
        if not hasattr(self, 'fig'):
            print(f"Skipping visualization for property {prop_name} - no figure available")
            return

        # Create subplot
        ax = self.fig.add_subplot(self.gs[self.current_subplot])
        self.current_subplot += 1
        # print(self.current_subplot, prop_name)

        # Get the current property from the alloy
        current_prop = getattr(alloy, prop_name)

        # Determine if T is symbolic or numeric
        is_symbolic = isinstance(T, sp.Symbol)

        # Set bounds if not provided
        if lower_bound is None or upper_bound is None:
            lower_bound = np.min(self.temperature_array)
            upper_bound = np.max(self.temperature_array)

        # Create an extended temperature range
        padding = (upper_bound - lower_bound) * 0.2  # 20% padding
        if x_data is not None:
            num_points = np.size(x_data) * (100 if prop_type == 'Key-Value' or prop_type == 'Computed' else 2)
        else:
            num_points = np.size(self.temperature_array) * 100

        extended_temp = np.linspace(lower_bound - padding, upper_bound + padding, num_points)

        # Handle different property types
        if prop_type == 'Constant':
            # For constant properties
            value = float(current_prop)
            ax.axhline(y=value, color='blue', linestyle='-', linewidth=1.5, label='constant')

            # If T is a numeric value, highlight that specific point
            if not is_symbolic:
                ax.plot(T, value, 'ro', markersize=6, label=f'T={T}K')

            # Add text annotation for the constant value
            ax.text(0.5, 0.9, f"Value: {value}", transform=ax.transAxes,
                    horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

            # Set y-axis limits to show the constant value clearly
            ax.set_ylim(value * 0.9, value * 1.1)

            # Set y-value for bound annotations
            y_value = value

        else:
            # For non-constant properties

            # Plot raw data points if available
            if x_data is not None and y_data is not None:
                marker_size = 2 if prop_type == 'Key-Value' else 1
                ax.plot(x_data, y_data, linewidth=1, marker='o', markersize=marker_size, label='measurement')

            # For numeric temperature, show the interpolated value
            if not is_symbolic:
                property_value = float(current_prop)
                ax.plot(T, property_value, 'ro', markersize=6, label=f'Value at T={T}K')

                # Add a text annotation with the exact value
                ax.text(0.005, 0.45, f"{prop_name} at T={T}K:\n{property_value:.6g}",
                        transform=ax.transAxes, fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.8))

                # Set y-value for bound annotations
                y_value = property_value

                # For numeric temperature, still show the full curve if data is available
                if x_data is not None and y_data is not None:
                    # For 'pre' simplification, show a simplified model
                    if has_regression and simplify_type == 'pre':
                        # Create a model to show the context
                        v_pwlf = pwlf.PiecewiseLinFit(x_data, y_data, degree=degree, seed=13579)
                        v_pwlf.fit(n_segments=segments)
                        context_pw = sp.Piecewise(*get_symbolic_conditions(v_pwlf, sp.Symbol('T_context'),
                                                                           lower_bound_type, upper_bound_type))
                        f_context = sp.lambdify(sp.Symbol('T_context'), context_pw, 'numpy')
                        ax.plot(extended_temp, f_context(extended_temp), linestyle='-', linewidth=1,
                                color='blue', alpha=0.5, label='property curve')
                    else:
                        # For 'post' or no regression, show raw curve
                        # Use piecewise interpolation for context
                        if len(x_data) > 1:
                            context_pw = self._create_raw_piecewise(x_data, y_data, sp.Symbol('T_context'),
                                                                    lower_bound_type, upper_bound_type)
                            f_context = sp.lambdify(sp.Symbol('T_context'), context_pw, 'numpy')
                            ax.plot(extended_temp, f_context(extended_temp), linestyle='-', linewidth=1,
                                    color='blue', alpha=0.5, label='property curve')

            # For symbolic temperature, show the full temperature-dependent behavior
            else:
                # Convert property to numerical function
                f_current = sp.lambdify(T, current_prop, 'numpy')

                if prop_type == 'Computed':
                    # For computed properties, show the symbolic model
                    ax.plot(extended_temp, f_current(extended_temp), linestyle='-', linewidth=1, label='symb')

                    # Create and plot PWLF approximation
                    v_pwlf = pwlf.PiecewiseLinFit(extended_temp, f_current(extended_temp), degree=degree, seed=13579)
                    v_pwlf.fit(n_segments=segments)
                    pw = sp.Piecewise(*get_symbolic_conditions(v_pwlf, T, lower_bound_type, upper_bound_type))
                    g = sp.lambdify(T, pw, 'numpy')
                    ax.plot(extended_temp, g(extended_temp), linestyle=':', linewidth=1, label='approx')

                    # Plot PWLF model with boundary handling
                    pwlf_extended = np.zeros_like(extended_temp)
                    for i, temp in enumerate(extended_temp):
                        if temp < lower_bound and lower_bound_type == 'constant':
                            pwlf_extended[i] = v_pwlf.predict(np.array([lower_bound]))[0]
                        elif temp > upper_bound and upper_bound_type == 'constant':
                            pwlf_extended[i] = v_pwlf.predict(np.array([upper_bound]))[0]
                        else:
                            pwlf_extended[i] = v_pwlf.predict(np.array([temp]))[0]

                    ax.plot(extended_temp, pwlf_extended, linestyle='--', linewidth=2, label='pwlf')

                    # Get y-value for annotations
                    y_value = current_prop.subs(T, 3000).evalf()

                elif has_regression and simplify_type == 'pre':
                    # For 'pre', we're already using the simplified model
                    ax.plot(extended_temp, f_current(extended_temp), linestyle='-', linewidth=1, label='simplified')
                    y_value = np.max(y_data) if y_data is not None else f_current(upper_bound)

                else:
                    # For 'post' or no regression, show the raw model
                    ax.plot(extended_temp, f_current(extended_temp), linestyle='-', linewidth=1, label='raw')
                    y_value = np.max(y_data) if y_data is not None else f_current(upper_bound)
                    print('For post or no regression, show the raw model')

                    # If regression is specified with 'post', also show what the simplified model would look like
                    if has_regression and simplify_type == 'post' and x_data is not None and y_data is not None:
                        # Create a preview of the simplified model
                        v_pwlf = pwlf.PiecewiseLinFit(x_data, y_data, degree=degree, seed=13579)
                        v_pwlf.fit(n_segments=segments)
                        preview_pw = sp.Piecewise(*get_symbolic_conditions(v_pwlf, T, lower_bound_type, upper_bound_type))
                        f_preview = sp.lambdify(T, preview_pw, 'numpy')
                        ax.plot(extended_temp, f_preview(extended_temp), linestyle=':', linewidth=1, label='will simplify to')

        # Always add vertical lines and annotations for bounds (for all property types and temperature formats)
        ax.axvline(x=lower_bound, color='brown', linestyle='--', alpha=0.5, label='_nolegend_')
        ax.axvline(x=upper_bound, color='brown', linestyle='--', alpha=0.5, label='_nolegend_')

        # Always add bound type annotations
        ax.text(lower_bound, y_value, f' {lower_bound_type}',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.5))
        ax.text(upper_bound, y_value, f' {upper_bound_type}',
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.5))

        # Add regression info if present (for both symbolic and numeric T)
        if has_regression:
            # Make sure simplify_type is not None for computed properties
            if prop_type == 'Computed' and simplify_type is None:
                simplify_type = 'post'  # Default for computed properties

            ax.text(0.5, 0.95, f"Simplify: {simplify_type} | Degree: {degree} | Segments: {segments}",
                    transform=ax.transAxes, horizontalalignment='center',
                    bbox=dict(facecolor='white', alpha=0.5))

        # Set labels and title
        title_suffix = f" at T={T}K" if not is_symbolic else ""
        ax.set_title(f"{prop_name} ({prop_type} Property){title_suffix}")
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(f"{prop_name}")
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.5)

##################################################
# External Function
##################################################

def create_alloy_from_yaml(yaml_path: Union[str, Path], T: Union[float, sp.Symbol]) -> Tuple[Alloy, np.ndarray]:
    """
    Create alloy instance from YAML configuration file
    Args:
        yaml_path: Path to the YAML configuration file
        T: Temperature value or symbol for property evaluation
    Returns:
        Tuple containing the alloy instance and the temperature array
    """
    parser = MaterialConfigParser(yaml_path)
    print(parser.config['name'])
    # print(parser.temperature_array)
    alloy = parser.create_alloy(T)
    temperature_array = parser.temperature_array
    return alloy, temperature_array
