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


class MaterialConfigParser:
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
        'temperature_array',
        'thermal_diffusivity',
        'thermal_expansion_coefficient',
    }

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

    def _validate_config(self) -> None:
        """Validate YAML configuration structure and content."""
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
        """Validate property names against allowed set."""
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
        """Validate required configuration fields."""
        required_fields = {'name', 'composition', 'solidus_temperature', 'liquidus_temperature'}
        missing_fields = required_fields - set(self.config.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    def _validate_property_values(self, properties: Dict[str, Any]) -> None:
        """Validate property values for type and range constraints."""
        if 'density' in properties:
            density = properties['density']
            if isinstance(density, (int, float)) and density <= 0:
                raise ValueError("Density must be positive")

        if 'energy_density_temperature_array' in properties:
            edta = properties['energy_density_temperature_array']
            if isinstance(edta, tuple) and len(edta) != 3:
                raise ValueError("Temperature array must be a tuple of (start, end, points/step)")

########################################################################################################################

    def create_alloy(self, T: Union[float, sp.Symbol]) -> Alloy:
        """Creates Alloy instance from YAML configuration"""
        alloy = Alloy(
            elements=self._get_elements(),
            composition=list(self.config['composition'].values()),
            temperature_solidus=self.config['solidus_temperature'],
            temperature_liquidus=self.config['liquidus_temperature']
        )
        self._process_properties(alloy, T)
        return alloy

    def _get_elements(self) -> List[ChemicalElement]:
        """Convert element symbols to ChemicalElement instances"""
        from pymatlib.data.element_data import element_map
        return [element_map[sym] for sym in self.config['composition'].keys()]

    def _is_numeric(self, value: str) -> bool:
        """Check if string represents a number (including scientific notation)"""
        try:
            float(value)
            return True
        except ValueError:
            return False

    def _is_data_file(self, value: Dict[str, str]) -> bool:
        """Check if dictionary represents a valid data file configuration"""
        # Simple format: property_name: "filename.txt"
        if isinstance(value, str) and (value.endswith('.txt') or value.endswith('.csv') or value.endswith('.xlsx')):
            return True
        # Advanced format: property_name: { file: "filename", temp_col: "col1", prop_col: "col2" }
        if isinstance(value, dict) and 'file' in value:
            # Required keys for advanced format
            required_keys = ['file', 'temp_col', 'prop_col']
            missing_keys = [k for k in required_keys if k not in value]

            if missing_keys:
                raise ValueError(f"Missing required keys for file configuration: {missing_keys}")
            return True
        return False

    def _is_key_val_property(self, value: Dict) -> bool:
        """Check if property is defined with key-val pairs"""
        return isinstance(value, dict) and 'key' in value and 'val' in value

    def _is_compute_property(self, value: Any) -> bool:
        """Check if property should be computed using any valid format"""
        # Simple format: property_name: compute
        if isinstance(value, str) and value == 'compute':
            return True
        # Advanced format: property_name: { compute: "method_name" }
        elif isinstance(value, dict) and 'compute' in value:
            # Ensure no other keys are present
            invalid_keys = [k for k in value.keys() if k != 'compute']
            if invalid_keys:
                raise ValueError(f"Invalid keys in compute property: {invalid_keys}. Only 'compute' is allowed.")
            return True
        return False

    def _process_properties(self, alloy: Alloy, T: Union[float, sp.Symbol]):
        """Process all material properties in correct order"""
        properties = self.config['properties']

        # Step 1: Process constant float properties
        for name, config in properties.items():
            if isinstance(config, (int, float)) or (isinstance(config, str) and self._is_numeric(config)):
                self._process_constant_property(alloy, name, config)

        # Step 2: Process file-based properties
        for name, config in properties.items():
            if self._is_data_file(config):
                self._process_file_property(alloy, name, config, T)

        # Step 3: Process key-val pair properties
        for name, config in properties.items():
            if self._is_key_val_property(config):
                self._process_key_val_property(alloy, name, config, T)

        # Step 4: Process computed properties
        for name, config in properties.items():
            if self._is_compute_property(config):
                self._process_computed_property(alloy, name, T)

########################################################################################################################

    def _process_constant_property(self, alloy: Alloy, prop_name: str, prop_config: Union[int, float, str]):
        """Process constant float property"""
        try:
            value = float(prop_config)
            setattr(alloy, prop_name, value)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid number format for {prop_name}: {prop_config}")

########################################################################################################################

    def _process_file_property(self, alloy: Alloy, prop_name: str, file_config: Union[str, Dict], T: Union[float, sp.Symbol]):
        """Process property data from file configuration"""
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

        # Store temperature array if not already set
        if not hasattr(alloy, 'temperature_array') or len(alloy.temperature_array) == 0:
            alloy.temperature_array = temp_array

        material_property = interpolate_property(T, temp_array, prop_array)
        setattr(alloy, prop_name, material_property)

        # Store additional properties if this is energy_density
        if prop_name == 'energy_density':
            alloy.energy_density_temperature_array = temp_array
            alloy.energy_density_array = prop_array
            alloy.energy_density_solidus = material_property.evalf(T, alloy.temperature_solidus)
            alloy.energy_density_liquidus = material_property.evalf(T, alloy.temperature_liquidus)

########################################################################################################################

    def _process_key_val_property(self, alloy: Alloy, prop_name: str, prop_config: Dict, T: Union[float, sp.Symbol]):
        """Process property defined with key-val pairs"""
        key_array = self._process_key_definition(prop_config['key'], prop_config['val'], alloy)
        val_array = np.array(prop_config['val'], dtype=float)

        if len(key_array) != len(val_array):
            raise ValueError(f"Length mismatch in {prop_name}: key and val arrays must have same length")

        # Store temperature array if not already set
        if not hasattr(alloy, 'temperature_array') or len(alloy.temperature_array) == 0:
            alloy.temperature_array = key_array

        material_property = interpolate_property(T, key_array, val_array)
        setattr(alloy, prop_name, material_property)

        # Store additional properties if this is energy_density
        if prop_name == 'energy_density':
            alloy.energy_density_temperature_array = key_array
            alloy.energy_density_array = val_array
            alloy.energy_density_solidus = material_property.evalf(T, alloy.temperature_solidus)
            alloy.energy_density_liquidus = material_property.evalf(T, alloy.temperature_liquidus)

    def _process_key_definition(self, key_def, val_array, alloy: Alloy) -> np.ndarray:
        """Process temperature key definition"""
        if isinstance(key_def, str) and key_def.startswith('(') and key_def.endswith(')'):
            return self._process_equidistant_key(key_def, len(val_array))
        elif isinstance(key_def, list):
            return self._process_list_key(key_def, alloy)
        else:
            raise ValueError(f"Invalid key definition: {key_def}")

    def _process_equidistant_key(self, key_def: str, n_points: int) -> np.ndarray:
        """Process equidistant key definition"""
        try:
            values = [float(x.strip()) for x in key_def.strip('()').split(',')]
            if len(values) != 2:
                raise ValueError("Equidistant definition must have exactly two values: (start, increment)")
            start, increment = values
            return np.arange(start, start + increment * n_points, increment)
        except ValueError as e:
            raise ValueError(f"Invalid equidistant format: {key_def}. Error: {str(e)}")

    def _process_list_key(self, key_def: List, alloy: Alloy) -> np.ndarray:
        """Process list key definition"""
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
        return np.array(processed_key, dtype=float)

########################################################################################################################

    def _process_computed_property(self, alloy: Alloy, prop_name: str, T: Union[float, sp.Symbol]):
        """Process computed properties using predefined models with dependency checking"""

        # Define property dependencies and their computation methods
        computation_methods = {
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
                )
            },
        }

        # Define property dependencies for each computation method
        dependencies = {
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

        # Process dependencies first if they're marked for computation
        if prop_name == 'energy_density':
            if 'energy_density_temperature_array' not in self.config['properties']:
                raise ValueError(f"energy_density_temperature_array must be defined when energy_density is computed")

            # Process energy_density_temperature_array
            edta = self.config['properties']['energy_density_temperature_array']
            alloy.energy_density_temperature_array = self._process_edta(edta)

        # Process other dependencies
        for dep in method_dependencies:
            if hasattr(alloy, dep) and getattr(alloy, dep) is None:
                if dep in self.config['properties']:
                    dep_config = self.config['properties'][dep]
                    if dep_config == 'compute' or (isinstance(dep_config, dict) and 'compute' in dep_config):
                        self._process_computed_property(alloy, dep, T)

        # Verify all dependencies are available
        missing_deps = [dep for dep in method_dependencies
                        if not hasattr(alloy, dep) or getattr(alloy, dep) is None]
        if missing_deps:
            raise ValueError(f"Cannot compute {prop_name}. Missing dependencies: {missing_deps}")

        # Compute property
        material_property = computation_methods[prop_name][method]()
        setattr(alloy, prop_name, material_property)

        # Handle special case for energy_density arrays and phase points
        if prop_name == 'energy_density':
            # Check if any dependency is a MaterialProperty
            deps_to_check = [getattr(alloy, dep) for dep in method_dependencies if hasattr(alloy, dep)]
            if any(isinstance(dep, MaterialProperty) for dep in deps_to_check):
                if hasattr(alloy, 'energy_density_temperature_array') and len(alloy.energy_density_temperature_array) > 0:
                    alloy.energy_density_array = np.array([
                        material_property.evalf(T, temp) for temp in alloy.energy_density_temperature_array
                    ])
                    alloy.energy_density_solidus = material_property.evalf(T, alloy.temperature_solidus)
                    alloy.energy_density_liquidus = material_property.evalf(T, alloy.temperature_liquidus)

    def _process_edta(self, array_def: str) -> np.ndarray:
        """Process temperature array definition with format (start, end, points/delta)"""
        if not (isinstance(array_def, str) and array_def.startswith('(') and array_def.endswith(')')):
            raise ValueError("Temperature array must be defined as (start, end, points/delta)")

        try:
            # Parse the tuple string
            values = [v.strip() for v in array_def.strip('()').split(',')]
            if len(values) != 3:
                raise ValueError("Temperature array definition must have exactly three values")

            start = float(values[0])
            end = float(values[1])
            step = values[2]

            # Check if step represents delta (float) or points (int)
            if '.' in step or 'e' in step.lower():
                delta = float(step)
                return np.arange(start, end + delta/2, delta)  # delta/2 ensures end point inclusion
            else:
            # Handle as points (int)
                points = int(step)
                if points <= 0:
                    raise ValueError(f"Number of points must be a positive integer, got {points}")
                return np.linspace(start, end, points)

        except ValueError as e:
            raise ValueError(f"Invalid temperature array definition: {e}")

########################################################################################################################

def create_alloy_from_yaml(yaml_path: Union[str, Path], T: Union[float, sp.Symbol]) -> Alloy:
    """Create alloy instance from YAML configuration file"""
    parser = MaterialConfigParser(yaml_path)
    return parser.create_alloy(T)
