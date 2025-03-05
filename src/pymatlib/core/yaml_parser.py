from pathlib import Path
import numpy as np
from typing import Dict, Any, Union, List, Tuple
import sympy as sp
from pymatlib.core.alloy import Alloy
from pymatlib.core.elements import ChemicalElement
from pymatlib.core.interpolators import interpolate_property
from pymatlib.core.data_handler import read_data_from_file
from pymatlib.core.models import (density_by_thermal_expansion,
                                  thermal_diffusivity_by_heat_conductivity,
                                  energy_density_standard, energy_density_enthalpy_based, energy_density_total_enthalpy)
from pymatlib.core.typedefs import MaterialProperty
from ruamel.yaml import YAML, constructor, scanner
from difflib import get_close_matches
from enum import Enum, auto


class PropertyType(Enum):
    CONSTANT = auto()
    FILE = auto()
    KEY_VAL = auto()
    COMPUTE = auto()
    TUPLE_STRING = auto()
    INVALID = auto()


class MaterialConfigParser:

    ##################################################
    # Class Constants and Attributes
    ##################################################

    MIN_POINTS = 2
    EPSILON = 1e-10  # Small value to handle floating point comparisons
    ABSOLUTE_ZERO = 0.0  # Kelvin

    # Define valid properties as class-level constants
    VALID_PROPERTIES = {
        'base_temperature',
        'base_density',
        'density',
        'dynamic_viscosity',
        'energy_density',
        'energy_density_solidus',
        'energy_density_liquidus',
        'energy_density_temperature_array',
        'energy_density_array',
        'heat_capacity',
        'heat_conductivity',
        'kinematic_viscosity',
        'latent_heat_of_fusion',
        'latent_heat_of_vaporization',
        'specific_enthalpy',
        'surface_tension',
        'temperature',
        'temperature_array',
        'thermal_diffusivity',
        'thermal_expansion_coefficient',
    }

    ##################################################
    # Initialization and YAML Loading
    ##################################################

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
        self.yaml_path = Path(yaml_path)
        self.base_dir = Path(yaml_path).parent
        self.config: Dict[str, Any] = self._load_yaml()
        self._validate_config()

    def _load_yaml(self) -> Dict[str, Any]:
        """Load and parse YAML file.
        Returns:
            Dict containing parsed YAML content
        Raises:
            FileNotFoundError: If YAML file is not found
            constructor.DuplicateKeyError: If duplicate keys are found
            scanner.ScannerError: If YAML syntax is invalid
        """
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

    ##################################################
    # Configuration Validation
    ##################################################

    def _validate_config(self) -> None:
        """
        Validate YAML configuration structure and content.
        This method checks the overall structure of the configuration,
        validates property names, required fields, and property values.
        Raises:
            ValueError: If any part of the configuration is invalid.
        """
        if not isinstance(self.config, dict):
            # raise ValueError("Root YAML element must be a mapping")
            raise ValueError("The YAML file must start with a dictionary/object structure with key-value pairs, not a list or scalar value")
        if 'properties' not in self.config:
            raise ValueError("Missing 'properties' section in configuration")
        properties = self.config.get('properties', {})
        if not isinstance(properties, dict):
            # raise ValueError("'properties' must be a mapping")
            raise ValueError("The 'properties' section in your YAML file must be a dictionary with key-value pairs")
        self._validate_property_names(properties)
        self._validate_required_fields()
        self._validate_property_values(properties)

    def _validate_property_names(self, properties: Dict[str, Any]) -> None:
        """
        Validate property names against the allowed set.
        Args:
            properties (Dict[str, Any]): Dictionary of properties to validate.
        Raises:
            ValueError: If any property name is not in VALID_PROPERTIES.
        """
        invalid_props = set(properties.keys()) - self.VALID_PROPERTIES
        if invalid_props:
            suggestions = {
                prop: get_close_matches(prop, self.VALID_PROPERTIES, n=1, cutoff=0.6)
                for prop in invalid_props
            }
            error_msg = "Invalid properties found:\n"
            for prop, matches in suggestions.items():
                suggestion = f" (did you mean '{matches[0]}'?)" if matches else ""
                error_msg += f"  - '{prop}'{suggestion}\n"
            raise ValueError(error_msg)

    def _validate_required_fields(self) -> None:
        """
        Validate required configuration fields.
        Raises:
            ValueError: If any required field is missing.
        """
        required_fields = {'name', 'composition', 'solidus_temperature', 'liquidus_temperature'}
        missing_fields = required_fields - set(self.config.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    @staticmethod
    def _validate_property_values(properties: Dict[str, Any]) -> None:
        """
        Validate property values for type and range constraints.
        Args:
            properties (Dict[str, Any]): Dictionary of properties to validate.
        Raises:
            ValueError: If any property value is invalid.
        """
        for prop_name, prop_value in properties.items():
            BASE_PROPERTIES = {'base_temperature', 'base_density'}
            POSITIVE_PROPERTIES = {'density', 'heat_capacity', 'heat_conductivity', 'specific_enthalpy'}
            NON_NEGATIVE_PROPERTIES = {'latent_heat'}
            if prop_value is None or (isinstance(prop_value, str) and prop_value.strip() == ''):
                raise ValueError(f"Property '{prop_name}' has an empty or undefined value")
            if prop_name in BASE_PROPERTIES:
                if not isinstance(prop_value, float) or prop_value <= 0:
                    raise ValueError(f"'{prop_name}' must be a positive number of type float, "
                                     f"got {prop_value} of type {type(prop_value).__name__}")
            if prop_name in POSITIVE_PROPERTIES:
                if isinstance(prop_value, float) and prop_value <= 0:
                    raise ValueError(f"'{prop_name}' must be positive, got {prop_value}")
            if prop_name in NON_NEGATIVE_PROPERTIES:
                if isinstance(prop_value, float) and prop_value < 0:
                    raise ValueError(f"'{prop_name}' cannot be negative, got {prop_value}")
            if prop_name == 'thermal_expansion_coefficient':
                if isinstance(prop_value, float) and prop_value < -3e-5 or prop_value > 0.001:
                    raise ValueError(f"'{prop_name}' value {prop_value} is outside the expected range (-3e-5/K to 0.001/K)")
            if prop_name == 'energy_density_temperature_array':
                if not (isinstance(prop_value, str) and prop_value.startswith('(') and prop_value.endswith(')')):
                    raise ValueError(f"'{prop_name}' must be a tuple of three comma-separated values representing (start, end, points/step)")
            if prop_name in ['energy_density_solidus', 'energy_density_liquidus']:
                raise ValueError(f"{prop_name} cannot be set directly. It is computed from other properties")

    ##################################################
    # Alloy Creation
    ##################################################

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
            alloy = Alloy(
                elements=self._get_elements(),
                composition=list(self.config['composition'].values()),
                temperature_solidus=self.config['solidus_temperature'],
                temperature_liquidus=self.config['liquidus_temperature']
            )
            self._process_properties(alloy, T)
            return alloy
        except KeyError as e:
            raise ValueError(f"Configuration error: Missing {e}")
        except Exception as e:
            raise ValueError(f"Failed to create alloy: {e}")

    def _get_elements(self) -> List[ChemicalElement]:
        """
        Convert element symbols to ChemicalElement instances.
        Returns:
            List[ChemicalElement]: List of ChemicalElement instances.
        Raises:
            ValueError: If an invalid element symbol is encountered.
        """
        from pymatlib.data.element_data import element_map
        try:
            return [element_map[sym] for sym in self.config['composition'].keys()]
        except KeyError as e:
            raise ValueError(f"Invalid element symbol: {e}")

    ##################################################
    # Property Type Checking
    ##################################################
    @staticmethod
    def _is_numeric(value: str) -> bool:
        """
        Check if string represents a number (including scientific notation).
        Args:
            value (str): The string to check.
        Returns:
            bool: True if the string represents a number, False otherwise.
        """
        try:
            float(value)
            return True
        except ValueError:
            return False

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
        # Simple format: property_name: "filename.txt"
        if isinstance(value, str):
            return value.endswith(('.txt', '.csv', '.xlsx'))
        # Advanced format: property_name: { file: "filename", temp_col: "col1", prop_col: "col2" }
        if isinstance(value, dict) and 'file' in value and 'temp_col' in value and 'prop_col' in value:
            required_keys = {'file', 'temp_col', 'prop_col'}
            value_keys = set(value.keys())
            if not required_keys.issubset(value_keys):
                missing_keys = required_keys - value_keys
                raise ValueError(f"Missing required keys for file configuration: {missing_keys}")
            if value_keys != required_keys:
                extra_keys = value_keys - required_keys
                raise ValueError(f"Extra keys found in file configuration: {extra_keys}")
            return True
        return False

    @staticmethod
    def _is_key_val_property(value: Any) -> bool:
        """
        Check if property is defined with key-val pairs.
        Args:
            value (Any): The value to check.
        Returns:
            bool: True if it's a key-val property, False otherwise.
        Raises:
            ValueError: If the key-val property configuration is invalid.
        """
        if isinstance(value, dict) and 'key' in value and 'val' in value:
            required_keys = {'key', 'val'}
            value_keys = set(value.keys())

            if value_keys != required_keys:
                missing_keys = required_keys - value_keys
                extra_keys = value_keys - required_keys
                if missing_keys:
                    raise ValueError(f"Missing required keys for key-val property: {missing_keys}")
                if extra_keys:
                    raise ValueError(f"Extra keys found in key-val property: {extra_keys}")
            return True
        return False

    @staticmethod
    def _is_compute_property(value: Any) -> bool:
        """
        Check if property should be computed using any valid format.
        Args:
            value (Any): The value to check.
        Returns:
            bool: True if it's a compute property, False otherwise.
        Raises:
            ValueError: If the compute property configuration is invalid.
        """
        if isinstance(value, str) and value == 'compute':
            return True
        elif isinstance(value, dict) and 'compute' in value:
            required_keys = {'compute'}
            value_keys = set(value.keys())

            if value_keys != required_keys:
                missing_keys = required_keys - value_keys
                extra_keys = value_keys - required_keys
                if missing_keys:
                    raise ValueError(f"Missing required key for compute property: {missing_keys}")
                if extra_keys:
                    raise ValueError(f"Extra keys found in compute property: {extra_keys}")
            return True
        return False

    def _determine_property_type(self, prop_name: str, config: Any) -> PropertyType:
        """
        Determine the type of property based on its configuration.
        Args:
            prop_name (str): The name of the property.
            config (Any): The configuration of the property.
        Returns:
            PropertyType: The determined property type.
        """
        if isinstance(config, float) or (isinstance(config, str) and self._is_numeric(config)):
            return PropertyType.CONSTANT
        elif self._is_data_file(config):
            return PropertyType.FILE
        elif self._is_key_val_property(config):
            return PropertyType.KEY_VAL
        elif self._is_compute_property(config):
            return PropertyType.COMPUTE
        elif prop_name == 'energy_density_temperature_array' and isinstance(config, str) and config.startswith('(') and config.endswith(')'):
            return PropertyType.TUPLE_STRING
        else:
            return PropertyType.INVALID

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
        categorized_properties: Dict[PropertyType, List[Tuple[str, Any]]] = {
            PropertyType.CONSTANT: [],
            PropertyType.FILE: [],
            PropertyType.KEY_VAL: [],
            PropertyType.COMPUTE: [],
            PropertyType.TUPLE_STRING: []
        }
        for prop_name, config in properties.items():
            prop_type = self._determine_property_type(prop_name, config)
            if prop_type == PropertyType.INVALID:
                raise ValueError(f"Invalid configuration for property '{prop_name}': {config}")
            categorized_properties[prop_type].append((prop_name, config))
        return categorized_properties

    ##################################################
    # Property Processing
    ##################################################

    def _process_properties(self, alloy: Alloy, T: Union[float, sp.Symbol]) -> None:
        """
        Process all properties for the alloy.
        Args:
            alloy (Alloy): The alloy object to process properties for.
            T (Union[float, sp.Symbol]): Temperature value or symbol.
        Raises:
            ValueError: If there's an error processing any property.
        """
        properties = self.config['properties']
        try:
            categorized_properties = self._categorize_properties(properties)
            for prop_type, prop_list in categorized_properties.items():
                for prop_name, config in prop_list:
                    if prop_type == PropertyType.CONSTANT:
                        self._process_constant_property(alloy, prop_name, config)
                    elif prop_type == PropertyType.FILE:
                        self._process_file_property(alloy, prop_name, config, T)
                    elif prop_type == PropertyType.KEY_VAL:
                        self._process_key_val_property(alloy, prop_name, config, T)
                    elif prop_type == PropertyType.COMPUTE:
                        self._process_computed_property(alloy, prop_name, T)
                    elif prop_type == PropertyType.TUPLE_STRING:
                        # Handle tuple string properties if needed
                        pass
        except Exception as e:
            raise ValueError(f"Failed to process properties: {e}")

########################################################################################################################

    @staticmethod
    def _process_constant_property(alloy: Alloy, prop_name: str, prop_config: Union[int, float, str]) -> None:
        """
        Process constant float property.
        Args:
            alloy (Alloy): The alloy object to update.
            prop_name (str): The name of the property to set.
            prop_config (Union[int, float, str]): The property value or string representation.
        Raises:
            ValueError: If the property value cannot be converted to float.
        """
        try:
            value = float(prop_config)
            setattr(alloy, prop_name, value)
        except (ValueError, TypeError) as e:
            error_msg = f"Invalid number format for {prop_name}: {prop_config}"
            raise ValueError(error_msg) from e

########################################################################################################################

    def _process_file_property(self, alloy: Alloy, prop_name: str, file_config: Union[str, Dict[str, Any]], T: Union[float, sp.Symbol]) -> None:
        """
        Process property data from file configuration.
        Args:
            alloy (Alloy): The alloy object to update.
            prop_name (str): The name of the property to set.
            file_config (Union[str, Dict[str, Any]]): File path or configuration dictionary.
            T (Union[float, sp.Symbol]): Temperature value or symbol.
        Raises:
            ValueError: If there's an error processing the file data.
        """
        try:
            # Get the directory containing the YAML file
            yaml_dir = self.base_dir
            # Construct path relative to YAML file location
            if isinstance(file_config, dict) and 'file' in file_config:
                file_config['file'] = str(yaml_dir / file_config['file'])
                temp_array, prop_array = read_data_from_file(file_config)
            else:
                # For string configuration, construct the full path
                file_path = str(yaml_dir / file_config)
                temp_array, prop_array = read_data_from_file(file_path)
            self._process_property_data(alloy, prop_name, T, temp_array, prop_array)
        except Exception as e:
            error_msg = f"Error processing file property {prop_name}: {str(e)}"
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
            key_array = self._process_key_definition(prop_config['key'], prop_config['val'], alloy)
            val_array = np.array(prop_config['val'], dtype=float)
            if len(key_array) != len(val_array):
                raise ValueError(f"Length mismatch in {prop_name}: key and val arrays must have same length")
            self._process_property_data(alloy, prop_name, T, key_array, val_array)
        except Exception as e:
            error_msg = f"Error processing key-val property {prop_name}: {str(e)}"
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
            error_msg = f"Error processing key definition: {str(e)}"
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
            error_msg = f"Invalid equidistant format: {key_def}. Error: {str(e)}"
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
        try:
            processed_key = []
            for k in key_def:
                if isinstance(k, str):
                    if k == 'solidus_temperature':
                        processed_key.append(alloy.temperature_solidus)
                    elif k == 'liquidus_temperature':
                        processed_key.append(alloy.temperature_liquidus)
                    else:
                        processed_key.append(float(k))
                else:
                    processed_key.append(float(k))
            key_array = np.array(processed_key, dtype=float)
            return key_array
        except Exception as e:
            error_msg = f"Error processing list key: {str(e)}"
            raise ValueError(error_msg) from e

    ##################################################
    # Property Data Processing
    ##################################################

    def _process_property_data(self, alloy: Alloy, prop_name: str, T: Union[float, sp.Symbol], temp_array: np.ndarray, prop_array: np.ndarray) -> None:
        """
        Process property data and set it on the alloy object.
        Args:
            alloy (Alloy): The alloy object to update.
            prop_name (str): The name of the property to set.
            T (Union[float, sp.Symbol]): Temperature value or symbol.
            temp_array (np.ndarray): Array of temperature values.
            prop_array (np.ndarray): Array of property values.
        Raises:
            ValueError: If there's an error processing the property data.
        """
        try:
            if isinstance(T, sp.Symbol):
                self._process_symbolic_temperature(alloy, prop_name, T, temp_array, prop_array)
            elif isinstance(T, (float, int)):
                self._process_constant_temperature(alloy, prop_name, T, temp_array, prop_array)
            else:
                raise ValueError(f"Unexpected type for T: {type(T)}")
        except Exception as e:
            error_msg = f"Error processing property data for {prop_name}: {str(e)}"
            raise ValueError(error_msg) from e

    def _process_symbolic_temperature(self, alloy: Alloy, prop_name: str, T: sp.Symbol, temp_array: np.ndarray, prop_array: np.ndarray) -> None:
        """
        Process property data for symbolic temperature.
        Args:
            alloy (Alloy): The alloy object to update.
            prop_name (str): The name of the property to set.
            T (sp.Symbol): Symbolic temperature.
            temp_array (np.ndarray): Array of temperature values.
            prop_array (np.ndarray): Array of property values.
        """
        # If T is symbolic, store the full temperature array if not already set then interpolate
        if getattr(alloy, 'temperature_array', None) is None or len(alloy.temperature_array) == 0:
            alloy.temperature_array = temp_array
        material_property = interpolate_property(T, temp_array, prop_array)
        setattr(alloy, prop_name, material_property)
        if prop_name == 'energy_density':
            self._process_energy_density(alloy, material_property, T, temp_array, prop_array)

    @staticmethod
    def _process_constant_temperature(alloy: Alloy, prop_name: str, T: Union[float, int], temp_array: np.ndarray, prop_array: np.ndarray) -> None:
        """
        Process property data for constant temperature.
        Args:
            alloy (Alloy): The alloy object to update.
            prop_name (str): The name of the property to set.
            T (Union[float, int]): Constant temperature value.
            temp_array (np.ndarray): Array of temperature values.
            prop_array (np.ndarray): Array of property values.
        """
        # If T is a constant, store just that value if not already set then interpolate
        if getattr(alloy, 'temperature', None) is None:
            alloy.temperature = float(T)
        material_property = interpolate_property(T, temp_array, prop_array)
        setattr(alloy, prop_name, material_property)

    @staticmethod
    def _process_energy_density(alloy: Alloy, material_property: Any, T: sp.Symbol, temp_array: np.ndarray, prop_array: np.ndarray) -> None:
        """
        Process additional properties for energy density.
        Args:
            alloy (Alloy): The alloy object to update.
            material_property (Any): The interpolated material property.
            T (sp.Symbol): Symbolic temperature.
            temp_array (np.ndarray): Array of temperature values.
            prop_array (np.ndarray): Array of property values.
        """
        alloy.energy_density_temperature_array = temp_array
        alloy.energy_density_array = prop_array
        alloy.energy_density_solidus = material_property.evalf(T, alloy.temperature_solidus)
        alloy.energy_density_liquidus = material_property.evalf(T, alloy.temperature_liquidus)

    ##################################################
    # Computed Property Handling
    ##################################################

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
        computation_methods = self._get_computation_methods(alloy, T)
        dependencies = self._get_dependencies()
        # Check if property has computation methods
        if prop_name not in computation_methods:
            raise ValueError(f"No computation method defined for property: {prop_name}")
        # Determine which computation method to use
        prop_config = self.config['properties'][prop_name]
        method = 'default'
        if isinstance(prop_config, dict) and 'compute' in prop_config:
            method = prop_config['compute']
        # Validate method exists
        if method not in computation_methods[prop_name]:
            available_methods = list(computation_methods[prop_name].keys())
            raise ValueError(f"Unknown computation method '{method}' for {prop_name}. Available: {available_methods}")
        # Get dependencies for selected method
        method_dependencies = dependencies[prop_name][method]
        # Process dependencies
        self._process_dependencies(alloy, prop_name, method_dependencies, T)
        # Compute property
        material_property = computation_methods[prop_name][method]()
        setattr(alloy, prop_name, material_property)
        # Handle special case for energy_density
        if prop_name == 'energy_density' and isinstance(T, sp.Symbol):
            self._handle_energy_density(alloy, material_property, T, method_dependencies)

    @staticmethod
    def _get_computation_methods(alloy: Alloy, T: Union[float, sp.Symbol]):
        """
        Get the computation methods for various properties of the alloy.
        Args:
            alloy (Alloy): The alloy object containing property data.
            T (Union[float, sp.Symbol]): The temperature value or symbol.
        Returns:
            dict: A dictionary of computation methods for different properties.
        """
        return {
            'density': {
                'default': lambda: density_by_thermal_expansion(
                    T,
                    alloy.base_temperature,
                    alloy.base_density,
                    alloy.thermal_expansion_coefficient
                )
            },
            'thermal_diffusivity': {
                'default': lambda: thermal_diffusivity_by_heat_conductivity(
                    alloy.heat_conductivity,
                    alloy.density,
                    alloy.heat_capacity
                )
            },
            'energy_density': {
                'default': lambda: energy_density_standard(
                    T,
                    alloy.density,
                    alloy.heat_capacity,
                    alloy.latent_heat_of_fusion
                ),
                'enthalpy_based': lambda: energy_density_enthalpy_based(
                    alloy.density,
                    alloy.specific_enthalpy,
                    alloy.latent_heat_of_fusion
                ),
                'total_enthalpy': lambda: energy_density_total_enthalpy(
                    alloy.density,
                    alloy.specific_enthalpy
                ),
            },
        }

    @staticmethod
    def _get_dependencies():
        """
        Get the dependencies for each computation method.
        Returns:
            dict: A nested dictionary specifying the dependencies for each
                  computation method of each property.
        """
        return {
            'density': {
                'default': ['base_temperature', 'base_density', 'thermal_expansion_coefficient'],
            },
            'thermal_diffusivity': {
                'default': ['heat_conductivity', 'density', 'heat_capacity'],
            },
            'energy_density': {
                'default': ['density', 'heat_capacity', 'latent_heat_of_fusion'],
                'enthalpy_based': ['density', 'specific_enthalpy', 'latent_heat_of_fusion'],
                'total_enthalpy': ['density', 'specific_enthalpy'],
            },
        }

    def _process_dependencies(self, alloy: Alloy, prop_name: str, dependencies: List[str], T: Union[float, sp.Symbol]):
        """
        Process and compute the dependencies required for a given property.
        This method checks if each dependency is already computed for the alloy.
        If not, it attempts to compute the dependency if a computation method is defined.
        Args:
            alloy (Alloy): The alloy object to process.
            prop_name (str): The name of the property being computed.
            dependencies (List[str]): List of dependency names for the property.
            T (Union[float, sp.Symbol]): The temperature value or symbol.
        Raises:
            ValueError: If any required dependency cannot be computed or is missing.
        """
        for dep in dependencies:
            if getattr(alloy, dep, None) is None:
                if dep in self.config['properties']:
                    dep_config = self.config['properties'][dep]
                    if dep_config == 'compute' or (isinstance(dep_config, dict) and 'compute' in dep_config):
                        self._process_computed_property(alloy, dep, T)
        # Verify all dependencies are available
        missing_deps = [dep for dep in dependencies if getattr(alloy, dep, None) is None]
        if missing_deps:
            raise ValueError(f"Cannot compute {prop_name}. Missing dependencies: {missing_deps}")

    def _handle_energy_density(self, alloy: Alloy, material_property: MaterialProperty, T: sp.Symbol, dependencies: List[str]):
        """
        Handle the special case of energy density computation.
        This method computes additional properties related to energy density when T is symbolic.
        It computes the energy density array, solidus, and liquidus values.
        Args:
            alloy (Alloy): The alloy object to process.
            material_property (MaterialProperty): The computed energy density property.
            T (sp.Symbol): The symbolic temperature variable.
            dependencies (List[str]): List of dependencies for energy density computation.
        Raises:
            ValueError: If T is not symbolic or if energy_density_temperature_array is not defined in the config.
        """
        # Ensure T is symbolic
        if not isinstance(T, sp.Symbol):
            raise ValueError("_handle_energy_density should only be called with symbolic T")
        # Check dependencies
        deps_to_check = [getattr(alloy, dep) for dep in dependencies if hasattr(alloy, dep)]
        if any(isinstance(dep, MaterialProperty) for dep in deps_to_check):
            if 'energy_density_temperature_array' not in self.config['properties']:
                raise ValueError(f"energy_density_temperature_array must be defined when energy_density is computed with symbolic T")
            # Process energy_density_temperature_array
            edta = self.config['properties']['energy_density_temperature_array']
            alloy.energy_density_temperature_array = self._process_edta(edta)
        if len(alloy.energy_density_temperature_array) >= 2:
            alloy.energy_density_array = material_property.evalf(T, alloy.energy_density_temperature_array)
            alloy.energy_density_solidus = material_property.evalf(T, alloy.temperature_solidus)
            alloy.energy_density_liquidus = material_property.evalf(T, alloy.temperature_liquidus)

    ##################################################
    # Energy Density Temperature Array Processing
    ##################################################

    def _process_edta(self, array_def: str) -> np.ndarray:
        """
        Process temperature array definition with format (start, end, points/delta).
        Args:
            array_def (str): A string defining the temperature array in the format
                             "(start, end, points/delta)".
        Returns:
            np.ndarray: An array of temperature values.
        Raises:
            ValueError: If the array definition is invalid, improperly formatted,
                        or contains physically impossible temperatures.
        Examples:
            >>> self._process_edta("(300, 3000, 5)")
            array([ 300., 975., 1650., 2325., 3000.])
            >>> self._process_edta("(3000, 300, -1350.)")
            array([3000., 1650., 300.])
        """
        if not (isinstance(array_def, str) and array_def.startswith('(') and array_def.endswith(')')):
            raise ValueError("Temperature array must be defined as (start, end, points/delta)")
        try:
            # Parse the tuple string
            values = [v.strip() for v in array_def.strip('()').split(',')]
            if len(values) != 3:
                raise ValueError("'energy_density_temperature_array' must be a tuple of three comma-separated values representing (start, end, points/step)")
            start, end, step = float(values[0]), float(values[1]), values[2]
            if start <= self.ABSOLUTE_ZERO or end <= self.ABSOLUTE_ZERO:
                raise ValueError(f"Temperature must be above absolute zero ({self.ABSOLUTE_ZERO}K)")
            if abs(float(step)) < self.EPSILON:
                raise ValueError("Delta or number of points cannot be zero.")
            # Check if step represents delta (float) or points (int)
            if '.' in step or 'e' in step.lower():
                return self._process_float_step(start, end, float(step))
            else:
                return self._process_int_step(start, end, int(step))
        except ValueError as e:
            raise ValueError(f"Invalid temperature array definition: {e}")

    @staticmethod
    def _process_float_step(start: float, end: float, delta: float) -> np.ndarray:
        """Process temperature array with float step (delta)."""
        if start < end and delta <= 0:
            raise ValueError("Delta must be positive for increasing range")
        if start > end and delta >= 0:
            raise ValueError("Delta must be negative for decreasing range")
        max_delta = abs(end - start)
        if abs(delta) > max_delta:
            raise ValueError(f"Absolute value of delta ({abs(delta)}) is too large for the range. It should be <= {max_delta}")
        return np.arange(start, end + delta/2, delta)

    def _process_int_step(self, start: float, end: float, points: int) -> np.ndarray:
        """Process temperature array with integer step (number of points)."""
        if points <= 0:
            raise ValueError(f"Number of points must be positive, got {points}!")
        if points < self.MIN_POINTS:
            raise ValueError(f"Number of points must be at least {self.MIN_POINTS}, got {points}!")
        return np.linspace(start, end, points)

##################################################
# External Function
##################################################

def create_alloy_from_yaml(yaml_path: Union[str, Path], T: Union[float, sp.Symbol]) -> Alloy:
    """Create alloy instance from YAML configuration file"""
    parser = MaterialConfigParser(yaml_path)
    return parser.create_alloy(T)
