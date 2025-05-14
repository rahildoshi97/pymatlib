import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import sympy as sp

from pymatlib.core.material import Material
from pymatlib.core.data_handler import read_data_from_file
from pymatlib.core.symbol_registry import SymbolRegistry
from pymatlib.core.yaml_parser.common_utils import (
    _process_regression,
    create_piecewise_from_formulas,
    create_raw_piecewise,
    ensure_ascending_order,
    get_symbolic_conditions,
    interpolate_value,
    process_regression_params,
    validate_temperature_range
)
from pymatlib.core.yaml_parser.property_types import PropertyType

logger = logging.getLogger(__name__)

seed = 13579

class PropertyProcessor:
    """Handles processing of different property types for material objects."""

    # --- Constructor ---
    def __init__(self) -> None:
        """Initialize processor state."""
        logger.debug("""PropertyProcessor: __init__:""")
        self.properties: Optional[Dict[str, Any]] = None
        self.categorized_properties: Optional[Dict[PropertyType, List[Tuple[str, Any]]]] = None
        self.temperature_array: Optional[np.ndarray] = None
        self.base_dir: Optional[Path] = None
        self.visualizer = None
        self.processed_properties: set = set()

    # --- Public API ---
    def process_properties(
            self,
            material: Material,
            T: Union[float, sp.Symbol],
            properties: Dict[str, Any],
            categorized_properties: Dict[PropertyType, List[Tuple[str, Any]]],
            temperature_array: np.ndarray,
            base_dir: Path,
            visualizer) -> None:
        """Process all properties for the material."""
        logger.debug("""PropertyProcessor: process_properties:
            material: %r
            T: %r
            properties: %r
            categorized_properties: %r
            temperature_array: %r
            base_dir: %r""",
                     material,
                     T,
                     properties,
                     categorized_properties,
                     temperature_array.shape,
                     base_dir)
        self.properties = properties
        self.categorized_properties = categorized_properties
        self.temperature_array = temperature_array
        self.base_dir = base_dir
        self.visualizer = visualizer
        self.processed_properties = set()
        try:
            for prop_type, prop_list in self.categorized_properties.items():
                # Sort properties to ensure certain temperature intervals are processed first
                sorted_props = sorted(prop_list, key=lambda x: 0 if x[0] in ['melting_temperature', 'boiling_temperature',
                                                                             'solidus_temperature', 'liquidus_temperature',
                                                                             'initial_boiling_temperature', 'final_boiling_temperature'] else 1)
                print(f"sorted_props={sorted_props}")
                for prop_name, config in sorted_props:
                    if prop_type == PropertyType.CONSTANT and prop_name not in ['latent_heat_of_fusion', 'latent_heat_of_vaporization']:
                        print(f"PropertyType.CONSTANT not in latent_heat: prop_name={prop_name}, config={config}")
                        self._process_constant_property(material, prop_name, config, T)
                    elif prop_type == PropertyType.CONSTANT and prop_name in ['latent_heat_of_fusion', 'latent_heat_of_vaporization']:
                        print(f"PropertyType.CONSTANT in latent_heat: prop_name={prop_name}, config={config}")
                        self._process_latent_heat_constant(material, prop_name, config, T)
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
            raise ValueError(f"Failed to process properties \n -> {e}")

    # --- Property-Type Processing Methods ---
    def _process_constant_property(self, material: Material, prop_name: str, prop_config: Union[float, str], T: Union[float, sp.Symbol]) -> None:
        """Process constant float property."""
        logger.debug("""PropertyProcessor: _process_constant_property:
            material: %r
            prop_name: %r
            prop_config: %r
            T: %r""", material, prop_name, prop_config, T)
        try:
            value = float(prop_config)
            setattr(material, prop_name, sp.Float(value))
            if self.visualizer is not None:
                self.visualizer.visualize_property(
                    material=material,
                    prop_name=prop_name,
                    T=T,
                    prop_type='CONSTANT',
                    lower_bound=min(self.temperature_array),
                    upper_bound=max(self.temperature_array)
                )
            self.processed_properties.add(prop_name)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to process constant property \n -> {e}") from e

    def _process_latent_heat_constant(self, material: Material, prop_name: str, prop_config: Union[float, str], T: Union[float, sp.Symbol]) -> None:
        """Process latent heat properties when provided as constants."""
        logger.debug("""PropertyProcessor: _process_latent_heat_constant:
            material: %r
            prop_name: %r
            prop_config: %r
            T: %r""", material, prop_name, prop_config, T)
        try:
            latent_heat_value = float(prop_config)
            # Different configuration based on material type
            if material.material_type == 'pure_metal':
                # For pure metals: step function at the transition temperature
                if prop_name == 'latent_heat_of_fusion':
                    expanded_config = {
                        'key': 'melting_temperature',
                        'val': [0, latent_heat_value],
                        'bounds': ['constant', 'constant'],
                    }
                elif prop_name == 'latent_heat_of_vaporization':
                    expanded_config = {
                        'key': 'boiling_temperature',
                        'val': [0, latent_heat_value],
                        'bounds': ['constant', 'constant'],
                    }
                else:
                    raise ValueError(f"Unsupported latent heat property: {prop_name}")
            elif material.material_type == 'alloy':
                # For alloys: linear curve between temperature points
                if prop_name == 'latent_heat_of_fusion':
                    expanded_config = {
                        'key': ['solidus_temperature', 'liquidus_temperature'],
                        'val': [0, latent_heat_value],
                        'bounds': ['constant', 'constant'],
                        'regression': {'simplify': 'pre', 'degree': 1, 'segments': 1},
                    }
                elif prop_name == 'latent_heat_of_vaporization':
                    expanded_config = {
                        'key': ['initial_boiling_temperature', 'final_boiling_temperature'],
                        'val': [0, latent_heat_value],
                        'bounds': ['constant', 'constant'],
                        'regression': {'simplify': 'pre', 'degree': 1, 'segments': 1},
                    }
                else:
                    raise ValueError(f"Unsupported latent heat property: {prop_name}")
            else:
                raise ValueError(f"Unsupported latent heat configuration: {prop_name}")
            self._process_key_val_property(material, prop_name, expanded_config, T)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to process {prop_name} constant \n -> {e}") from e

    def _process_file_property(self, material: Material, prop_name: str, file_config: Union[str, Dict[str, Any]], T: Union[float, sp.Symbol]) -> None:
        """Process property data from a file configuration."""
        logger.debug("""PropertyProcessor: _process_file_property:
            material: %r
            prop_name: %r
            file_config: %r
            T: %r""", material, prop_name, file_config, T)
        try:
            yaml_dir = self.base_dir
            FILE_PATH_KEY = "file_path"
            if isinstance(file_config, dict) and FILE_PATH_KEY in file_config:
                file_config[FILE_PATH_KEY] = str(yaml_dir / file_config[FILE_PATH_KEY])
                temp_array, prop_array = read_data_from_file(file_config)
            else:
                file_path = str(yaml_dir / file_config)
                temp_array, prop_array = read_data_from_file(file_path)
            if not (np.all(np.isfinite(temp_array)) and np.all(np.isfinite(prop_array))):
                bad_temps = np.where(~np.isfinite(temp_array))[0]
                bad_props = np.where(~np.isfinite(prop_array))[0]
                msg = []
                if bad_temps.size > 0:
                    msg.append(f"temp_array contains non-finite values at indices: {bad_temps.tolist()}")
                if bad_props.size > 0:
                    msg.append(f"prop_array contains non-finite values at indices: {bad_props.tolist()}")
                raise ValueError(f"Non-finite values detected in property '{prop_name}': " + "; ".join(msg))
            self._validate_temperature_range(prop_name, temp_array)
            temp_array, prop_array = ensure_ascending_order(temp_array, prop_array)
            lower_bound_type, upper_bound_type = file_config['bounds']
            lower_bound = np.min(temp_array)
            upper_bound = np.max(temp_array)
            if not isinstance(T, sp.Symbol):
                interpolated_value = self._interpolate_value(T, temp_array, prop_array, lower_bound_type, upper_bound_type)
                setattr(material, prop_name, sp.Float(interpolated_value))
                return
            has_regression, simplify_type, degree, segments = self._process_regression_params(file_config, prop_name, len(temp_array))
            if has_regression and simplify_type == 'pre':
                pw = _process_regression(temp_array, prop_array, T, lower_bound_type, upper_bound_type, degree, segments, seed)
                setattr(material, prop_name, pw)
            else:  # No regression OR not pre
                raw_pw = self._create_raw_piecewise(temp_array, prop_array, T, lower_bound_type, upper_bound_type)
                setattr(material, prop_name, raw_pw)
            if self.visualizer is not None:
                self.visualizer.visualize_property(
                    material=material,
                    prop_name=prop_name,
                    T=T,
                    prop_type='FILE',
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
            self.processed_properties.add(prop_name)
        except Exception as e:
            raise ValueError(f"Failed to process file property {prop_name} \n -> {str(e)}") from e

    def _process_key_val_property(self, material: Material, prop_name: str, prop_config: Dict[str, Any], T: Union[float, sp.Symbol]) -> None:
        """Process property defined with key-val pairs."""
        logger.debug("""PropertyProcessor: _process_key_val_property:
            material: %r
            prop_name: %r
            prop_config: %r
            T: %r""", material, prop_name, prop_config, T)
        try:
            # Handle step function case for pure metals
            if material.material_type == 'pure_metal' and isinstance(prop_config['key'], str) and prop_config['key'] in ['melting_temperature', 'boiling_temperature'] and len(prop_config['val']) == 2:
                # Get the transition temperature
                if prop_config['key'] == 'melting_temperature':
                    transition_temp = float(material.melting_temperature)
                elif prop_config['key'] == 'boiling_temperature':
                    transition_temp = float(material.boiling_temperature)
                else:
                    raise ValueError(f"Invalid key '{prop_config['key']}' for pure metal")
                # Create step function
                T_sym = T if isinstance(T, sp.Symbol) else sp.Symbol('T')
                val_array = np.array(prop_config['val'], dtype=float)
                # Create step function: val[0] if T < transition_temp else val[1]
                step_function = sp.Piecewise((val_array[0], T_sym < transition_temp), (val_array[1], True))
                setattr(material, prop_name, step_function)
                if not isinstance(T, sp.Symbol):
                    value = float(step_function.subs(T_sym, T).evalf())
                    setattr(material, prop_name, sp.Float(value))
                    return
                f_pw = sp.lambdify(T_sym, step_function, 'numpy')
                diff = abs(self.temperature_array[1] - self.temperature_array[0])
                temp_dense = np.arange(np.min(self.temperature_array), np.max(self.temperature_array)+diff/2, diff)
                y_dense = f_pw(temp_dense)
                # Visualize if needed
                if self.visualizer is not None:
                    self.visualizer.visualize_property(
                        material=material,
                        prop_name=prop_name,
                        T=T,
                        prop_type='KEY_VAL',
                        x_data=temp_dense,
                        y_data=y_dense,
                        has_regression=False,
                        lower_bound=np.min(self.temperature_array),
                        upper_bound=np.max(self.temperature_array),
                        lower_bound_type=prop_config['bounds'][0],
                        upper_bound_type=prop_config['bounds'][1]
                    )
                self.processed_properties.add(prop_name)
                return
            # Alloy case
            key_array = self._process_key_definition(prop_config['key'], prop_config['val'], material)
            val_array = np.array(prop_config['val'], dtype=float)
            if len(key_array) != len(val_array):
                raise ValueError(f"Length mismatch in {prop_name}: key and val arrays must have same length")
            # Skip temperature validation for latent heat properties
            if prop_name not in ['latent_heat_of_fusion', 'latent_heat_of_vaporization']:
                self._validate_temperature_range(prop_name, key_array)
            key_array, val_array = ensure_ascending_order(key_array, val_array)
            lower_bound_type, upper_bound_type = prop_config['bounds']
            lower_bound = np.min(key_array)
            upper_bound = np.max(key_array)
            if not isinstance(T, sp.Symbol):
                interpolated_value = self._interpolate_value(T, key_array, val_array, lower_bound_type, upper_bound_type)
                setattr(material, prop_name, sp.Float(interpolated_value))
                return
            has_regression, simplify_type, degree, segments = self._process_regression_params(prop_config, prop_name, len(key_array))
            if has_regression and simplify_type == 'pre':
                pw = _process_regression(key_array, val_array, T, lower_bound_type, upper_bound_type, degree, segments, seed)
                print(f"pw={pw}\ntype(pw)={type(pw)}")
                setattr(material, prop_name, pw)
            else:  # No regression OR not pre
                raw_pw = self._create_raw_piecewise(key_array, val_array, T, lower_bound_type, upper_bound_type)
                setattr(material, prop_name, raw_pw)
            if self.visualizer is not None:
                self.visualizer.visualize_property(
                    material=material,
                    prop_name=prop_name,
                    T=T,
                    prop_type='KEY_VAL',
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
            self.processed_properties.add(prop_name)
        except Exception as e:
            raise ValueError(f"Failed to process key-val property '{prop_name}' \n -> {str(e)}") from e

    # --- Key-Value Helpers ---
    def _process_key_definition(self, key_def: Union[str, List[Union[str, float]]], val_array: List[float], material: Material) -> np.ndarray:
        """Process temperature key definition."""
        logger.debug("""PropertyProcessor: _process_key_definition:
            key_def: %r
            val_array: %r
            material: %r""", key_def, val_array, material)
        try:
            if isinstance(key_def, str) and key_def.startswith('(') and key_def.endswith(')'):
                return self._process_equidistant_key(key_def, len(val_array))
            elif isinstance(key_def, list):
                return self._process_list_key(key_def, material)
            else:
                raise ValueError(f"Invalid key definition: {key_def}")
        except Exception as e:
            raise ValueError(f"Failed to process key definition \n -> {str(e)}") from e

    @staticmethod
    def _process_equidistant_key(key_def: str, n_points: int) -> np.ndarray:
        """Process equidistant key definition."""
        logger.debug("""PropertyProcessor: _process_equidistant_key:
            key_def: %r
            n_points: %r""", key_def, n_points)
        try:
            values = [float(x.strip()) for x in key_def.strip('()').split(',')]
            if len(values) != 2:
                raise ValueError("Equidistant definition must have exactly two values: (start, increment)")
            start, increment = values
            key_array = np.arange(start, start + increment * n_points, increment)
            return key_array
        except Exception as e:
            raise ValueError(f"Invalid equidistant format: {key_def} \n -> {str(e)}") from e

    @staticmethod
    def _process_list_key(key_def: List[Union[str, float]], material: Material) -> np.ndarray:
        """Process list key definition."""
        logger.debug("""PropertyProcessor: _process_list_key:
            key_def: %r
            material: %r""", key_def, material)
        try:
            processed_key = []
            for k in key_def:
                if isinstance(k, str):
                    # Handle pure metal properties
                    if k == 'melting_temperature' and hasattr(material, 'melting_temperature'):
                        processed_key.append(material.melting_temperature)
                    elif k == 'boiling_temperature' and hasattr(material, 'boiling_temperature'):
                        processed_key.append(material.boiling_temperature)
                    # Handle alloy properties
                    elif k == 'solidus_temperature' and hasattr(material, 'solidus_temperature'):
                        processed_key.append(material.solidus_temperature)
                    elif k == 'liquidus_temperature' and hasattr(material, 'liquidus_temperature'):
                        processed_key.append(material.liquidus_temperature)
                    elif k == 'initial_boiling_temperature' and hasattr(material, 'initial_boiling_temperature'):
                        processed_key.append(material.initial_boiling_temperature)
                    elif k == 'final_boiling_temperature' and hasattr(material, 'final_boiling_temperature'):
                        processed_key.append(material.final_boiling_temperature)
                    # Handle offset notation (e.g., "melting_temperature+10")
                    elif '+' in k:
                        base, offset = k.split('+')
                        offset_value = float(offset)
                        # Get base value based on material type
                        if base == 'melting_temperature' and hasattr(material, 'melting_temperature'):
                            base_value = material.melting_temperature
                        elif base == 'boiling_temperature' and hasattr(material, 'boiling_temperature'):
                            base_value = material.boiling_temperature
                        elif base == 'solidus_temperature' and hasattr(material, 'solidus_temperature'):
                            base_value = material.solidus_temperature
                        elif base == 'liquidus_temperature' and hasattr(material, 'liquidus_temperature'):
                            base_value = material.liquidus_temperature
                        elif base == 'initial_boiling_temperature' and hasattr(material, 'initial_boiling_temperature'):
                            base_value = material.initial_boiling_temperature
                        elif base == 'final_boiling_temperature' and hasattr(material, 'final_boiling_temperature'):
                            base_value = material.final_boiling_temperature
                        else:
                            base_value = float(base)
                        processed_key.append(base_value + offset_value)
                    # Similar handling for subtraction
                    elif '-' in k:
                        # Similar implementation as above for subtraction
                        base, offset = k.split('-')
                        offset_value = float(offset)
                        # Get base value based on material type
                        if base == 'melting_temperature' and hasattr(material, 'melting_temperature'):
                            base_value = material.melting_temperature
                        elif base == 'boiling_temperature' and hasattr(material, 'boiling_temperature'):
                            base_value = material.boiling_temperature
                        elif base == 'solidus_temperature' and hasattr(material, 'solidus_temperature'):
                            base_value = material.solidus_temperature
                        elif base == 'liquidus_temperature' and hasattr(material, 'liquidus_temperature'):
                            base_value = material.liquidus_temperature
                        elif base == 'initial_boiling_temperature' and hasattr(material, 'initial_boiling_temperature'):
                            base_value = material.initial_boiling_temperature
                        elif base == 'final_boiling_temperature' and hasattr(material, 'final_boiling_temperature'):
                            base_value = material.final_boiling_temperature
                        else:
                            base_value = float(base)
                        processed_key.append(base_value - offset_value)
                    else:
                        processed_key.append(float(k))
                else:
                    processed_key.append(float(k))
            key_array = np.array(processed_key, dtype=float)
            return key_array
        except Exception as e:
            raise ValueError(f"Error processing list key \n -> {str(e)}") from e

    def _process_piecewise_equation_property(self, material: Material, prop_name: str, prop_config: Dict[str, Any], T: Union[float, sp.Symbol]) -> None:
        """Process piecewise equation property."""
        logger.debug("""PropertyProcessor: _process_piecewise_equation_property:
            material: %r
            prop_name: %r
            prop_config: %r
            T: %r""", material, prop_name, prop_config, T)
        temp_points = np.array(prop_config['temperature'], dtype=float)
        eqn_strings = prop_config['equation']
        # Validate that only 'T' is used in equations
        for eq in eqn_strings:
            expr = sp.sympify(eq)
            symbols = expr.free_symbols
            for sym in symbols:
                if str(sym) != 'T':
                    raise ValueError(f"Unsupported symbol '{sym}' found in equation '{eq}' for property '{prop_name}'. Only 'T' is allowed.")
        lower_bound_type, upper_bound_type = prop_config['bounds']
        T_sym = T if isinstance(T, sp.Symbol) else sp.Symbol('T')
        temp_points, eqn_strings = ensure_ascending_order(temp_points, eqn_strings)
        eqn_exprs = [sp.sympify(eq, locals={'T': T_sym}) for eq in eqn_strings]
        pw = self._create_piecewise_from_formulas(temp_points, eqn_exprs, T_sym, lower_bound_type, upper_bound_type)
        # print(f"pw={pw}, type(pw)={type(pw)}\neqn_strings={eqn_strings},\ntemp_points={temp_points}")
        if not isinstance(T, sp.Symbol):
            value = float(pw.subs(T_sym, T).evalf())
            setattr(material, prop_name, sp.Float(value))
            self.processed_properties.add(prop_name)
            return
        has_regression, simplify_type, degree, segments = self._process_regression_params(prop_config, prop_name, len(temp_points))
        # if not has_regression:
            # simplify_type = 'post'
        f_pw = sp.lambdify(T_sym, pw, 'numpy')
        diff = abs(self.temperature_array[1] - self.temperature_array[0])
        print(f"diff={diff}")
        temp_dense = np.arange(temp_points[0], temp_points[-1]+diff/2, diff)
        print(f"temp_dense.shape={temp_dense.shape}")
        y_dense = f_pw(temp_dense)
        if has_regression and simplify_type == 'pre':
            pw_reg = _process_regression(temp_dense, y_dense, T_sym, lower_bound_type, upper_bound_type, degree, segments, seed)
            setattr(material, prop_name, pw_reg)
        else:  # No regression OR not pre
            raw_pw = pw
            print(f"raw_pw={raw_pw}")
            setattr(material, prop_name, raw_pw)
        self.processed_properties.add(prop_name)
        if self.visualizer is not None:
            self.visualizer.visualize_property(
                material=material,
                prop_name=prop_name,
                T=T,
                prop_type='PIECEWISE_EQUATION',
                x_data=temp_dense,
                y_data=y_dense,
                has_regression=has_regression,
                simplify_type=simplify_type,
                degree=degree,
                segments=segments,
                lower_bound=temp_points[0],
                upper_bound=temp_points[-1],
                lower_bound_type=lower_bound_type,
                upper_bound_type=upper_bound_type
            )

    def _process_computed_property(self, material: Material, prop_name: str, T: Union[float, sp.Symbol]) -> None:
        """Process computed properties using predefined models with dependency checking."""
        logger.debug("PropertyProcessor: _process_computed_property - prop_name: %r, T: %r", prop_name, T)
        # Skip if already processed
        if prop_name in self.processed_properties:
            return
        try:
            # Get property configuration
            prop_config = self.properties.get(prop_name)
            if prop_config is None:
                raise ValueError(f"Property '{prop_name}' not found in configuration")
            # Extract expression from config
            if isinstance(prop_config, str):
                expression = prop_config
            elif isinstance(prop_config, dict) and 'equation' in prop_config:
                expression = prop_config['equation']
            else:
                raise ValueError(f"Unsupported property configuration format for {prop_name}: {prop_config}")
            # Process the expression
            material_property = self._parse_and_process_expression(expression, material, T)
            # Set property on material
            setattr(material, prop_name, material_property)
            # Handle non-symbolic temperature case
            T_sym = T if isinstance(T, sp.Symbol) else sp.Symbol('T')
            if not isinstance(T, sp.Symbol):
                value = float(material_property.subs(T_sym, T).evalf())
                setattr(material, prop_name, sp.Float(value))
                self.processed_properties.add(prop_name)
                return
            # Extract bounds and regression parameters
            lower_bound_type, upper_bound_type = 'constant', 'constant'
            if isinstance(prop_config, dict) and 'bounds' in prop_config:
                if isinstance(prop_config['bounds'], list) and len(prop_config['bounds']) == 2:
                    lower_bound_type, upper_bound_type = prop_config['bounds']
            temp_array = self.temperature_array
            lower_bound = np.min(temp_array)
            upper_bound = np.max(temp_array)
            # Get regression parameters
            has_regression, simplify_type, degree, segments = self._process_regression_params(prop_config, prop_name, len(temp_array))
            if not has_regression:
                simplify_type = 'post'
            # Create function from symbolic expression
            f_pw = sp.lambdify(T_sym, material_property, 'numpy')
            y_dense = f_pw(temp_array)  # TODO: <lambdifygenerated-16>:2: RuntimeWarning: divide by zero encountered in reciprocal
            # Apply regression if needed
            if has_regression and simplify_type == 'pre':
                pw_reg = _process_regression(temp_array, y_dense, T_sym, lower_bound_type, upper_bound_type, degree, segments, seed)
                setattr(material, prop_name, pw_reg)
            else:  # No regression OR not pre
                raw_pw = self._create_raw_piecewise(temp_array, y_dense, T, lower_bound_type, upper_bound_type)
                setattr(material, prop_name, raw_pw)
            self.processed_properties.add(prop_name)
            if self.visualizer is not None:
                self.visualizer.visualize_property(
                    material=material,
                    prop_name=prop_name,
                    T=T,
                    prop_type='COMPUTE',
                    x_data=temp_array,
                    y_data=y_dense,
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
            raise ValueError(f"Failed to process computed property '{prop_name}' \n -> {str(e)}") from e

    # --- Expression/Dependency Helpers ---
    def _parse_and_process_expression(self, expression: str, material: Material, T: Union[float, sp.Symbol]) -> sp.Expr:
        """Parse and process a mathematical expression string into a SymPy expression."""
        logger.debug("PropertyProcessor: _parse_and_process_expression - expression: %r, T: %r", expression, T)

        try:
            sympy_expr = sp.sympify(expression, evaluate=False)

            # Extract dependencies (excluding temperature symbol if symbolic)
            dependencies = [str(symbol) for symbol in sympy_expr.free_symbols
                            if not (str(symbol) == 'T' and isinstance(T, sp.Symbol))]

            # Check for circular dependencies
            self._check_circular_dependencies(prop_name=None, current_deps=dependencies, visited=set())

            # Process dependencies first
            for dep in dependencies:
                if not hasattr(material, dep) or getattr(material, dep) is None:
                    if dep in self.properties:
                        self._process_computed_property(material, dep, T)
                    else:
                        raise ValueError(f"Dependency '{dep}' not found in properties configuration")

            # Verify all dependencies are now available
            missing_deps = [dep for dep in dependencies if not hasattr(material, dep) or getattr(material, dep) is None]
            if missing_deps:
                raise ValueError(f"Cannot compute expression. Missing dependencies: {missing_deps}")

            # Create substitution dictionary
            substitutions = {}
            for dep in dependencies:
                dep_value = getattr(material, dep)
                dep_symbol = SymbolRegistry.get(dep)
                if dep_symbol is None:
                    raise ValueError(f"Symbol '{dep}' not found in symbol registry")
                substitutions[dep_symbol] = dep_value

            # Add temperature symbol if needed
            if isinstance(T, sp.Symbol):
                substitutions[sp.Symbol('T')] = T

            # Perform substitution and evaluate integrals if present
            result_expr = sympy_expr.subs(substitutions)
            if isinstance(result_expr, sp.Integral):
                result_expr = result_expr.doit()

            return result_expr

        except Exception as e:
            raise ValueError(f"Failed to parse and process expression: {expression}") from e

    def _check_circular_dependencies(self, prop_name, current_deps, visited, path=None):
        """Check for circular dependencies in property definitions."""
        logger.debug("""PropertyProcessor: _check_circular_dependencies:
            prop_name: %r
            current_deps: %r
            visited: %r
            path: %r""", prop_name, current_deps, visited, path)
        if path is None:
            path = []
        if prop_name is not None:
            if prop_name in visited:
                cycle_path = path + [prop_name]
                raise ValueError(f"Circular dependency detected: {' -> '.join(cycle_path)}")
            visited.add(prop_name)
            path = path + [prop_name]
        for dep in current_deps:
            if dep in self.properties:
                dep_config = self.properties[dep]
                if isinstance(dep_config, str):
                    expr = sp.sympify(dep_config)
                    dep_deps = [str(symbol) for symbol in expr.free_symbols if str(symbol) != 'T']
                elif isinstance(dep_config, dict) and 'equation' in dep_config:
                    eqn = dep_config['equation']
                    if isinstance(eqn, list):
                        symbols = set()
                        for eq in eqn:
                            expr = sp.sympify(eq)
                            symbols.update(expr.free_symbols)
                        dep_deps = [str(symbol) for symbol in symbols if str(symbol) != 'T']
                    else:
                        expr = sp.sympify(eqn)
                        dep_deps = [str(symbol) for symbol in expr.free_symbols if str(symbol) != 'T']
                else:
                    dep_deps = []
                if dep_deps:
                    self._check_circular_dependencies(dep, dep_deps, visited.copy(), path)

    # --- Post-Processing ---
    def _post_process_properties(self, material: Material, T: Union[float, sp.Symbol]) -> None:
        """Perform post-processing on properties after all have been initially processed."""
        logger.debug("""PropertyProcessor: _post_process_properties:
            material: %r
            T: %r""", material, T)
        if not isinstance(T, sp.Symbol):
            return
        properties = self.properties
        errors = []
        for prop_name, prop_config in properties.items():
            try:
                if not isinstance(prop_config, dict) or 'regression' not in prop_config:
                    continue
                regression_config = prop_config['regression']
                simplify_type = regression_config['simplify']
                if simplify_type != 'post':
                    continue
                prop_value = getattr(material, prop_name)
                print(prop_value)
                if isinstance(prop_value, sp.Integral):
                    raise ValueError(f"Property '{prop_name}' is an integral and cannot be post-processed.")
                    result = prop_value.doit()
                    if isinstance(result, sp.Integral):
                        temp_array = self.temperature_array
                        values = np.array([float(prop_value.evalf(subs={T: t})) for t in temp_array], dtype=float)
                        degree = regression_config['degree']
                        segments = regression_config['segments']
                        lower_bound_type, upper_bound_type = prop_config['bounds']
                        pw = _process_regression(temp_array, values, T, lower_bound_type, upper_bound_type, degree, segments, seed)
                        prop_value = pw
                    else:
                        prop_value = result
                    setattr(material, prop_name, prop_value)
                if not isinstance(prop_value, sp.Expr):
                    logger.debug("""PropertyVisualizer: _post_process_properties:
                    Skipping - not symbolic for property: %r
                    type: %r""", prop_name, type(prop_value))
                    continue
                degree = regression_config['degree']
                segments = regression_config['segments']
                lower_bound_type, upper_bound_type = prop_config['bounds']
                temp_array = self.temperature_array
                f = sp.lambdify(T, prop_value, 'numpy')
                prop_array = f(temp_array)
                valid_indices = np.isfinite(prop_array)
                if not np.all(valid_indices):
                    logger.warning(f"Found {np.sum(~valid_indices)} non-finite values in {prop_name}. Filtering them out.")
                    temp_array = temp_array[valid_indices]
                    prop_array = prop_array[valid_indices]
                if len(temp_array) < 2:
                    logger.warning(f"Not enough valid points to fit {prop_name}. Skipping post-processing.")
                    continue
                pw = _process_regression(temp_array, prop_array, T, lower_bound_type, upper_bound_type, degree, segments, seed)
                setattr(material, prop_name, pw)
                logger.debug("""PropertyVisualizer: _post_process_properties:
                    Post-processed property with simplify type 'post': %r""", prop_name)
            except Exception as e:
                error_msg = f"Failed to post-process {prop_name}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        if errors:
            error_msg = "\n".join(errors)
            raise ValueError(f"Post-processing errors occurred:\n{error_msg}")

    def _validate_temperature_range(self, prop: str, temp_array: np.ndarray) -> None:
        from pymatlib.core.yaml_parser.common_utils import validate_temperature_range
        return validate_temperature_range(prop, temp_array, self.temperature_array)

    @staticmethod
    def _interpolate_value(T: float, x_array: np.ndarray, y_array: np.ndarray, lower_bound_type: str, upper_bound_type: str) -> float:
        from pymatlib.core.yaml_parser.common_utils import interpolate_value
        return interpolate_value(T, x_array, y_array, lower_bound_type, upper_bound_type)

    @staticmethod
    def _process_regression_params(prop_config: dict, prop_name: str, data_length: int):
        from pymatlib.core.yaml_parser.common_utils import process_regression_params
        return process_regression_params(prop_config, prop_name, data_length)

    @staticmethod
    def _create_raw_piecewise(temp_array: np.ndarray, prop_array: np.ndarray, T: sp.Symbol, lower_bound_type: str, upper_bound_type: str) -> sp.Piecewise:
        from pymatlib.core.yaml_parser.common_utils import create_raw_piecewise
        return create_raw_piecewise(temp_array, prop_array, T, lower_bound_type, upper_bound_type)

    @staticmethod
    def _create_piecewise_from_formulas(temp_points: np.ndarray, eqn_exprs: List[sp.Expr], T: sp.Symbol, lower_bound_type: str, upper_bound_type: str) -> sp.Piecewise:
        from pymatlib.core.yaml_parser.common_utils import create_piecewise_from_formulas
        return create_piecewise_from_formulas(temp_points, eqn_exprs, T, lower_bound_type, upper_bound_type)
