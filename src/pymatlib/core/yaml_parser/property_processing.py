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
from pymatlib.core.yaml_parser.custom_error import DependencyError, CircularDependencyError
from pymatlib.core.yaml_parser.property_types import PropertyType
from pymatlib.core.yaml_parser.yaml_keys import MELTING_TEMPERATURE_KEY, BOILING_TEMPERATURE_KEY, \
    SOLIDUS_TEMPERATURE_KEY, LIQUIDUS_TEMPERATURE_KEY, INITIAL_BOILING_TEMPERATURE_KEY, FINAL_BOILING_TEMPERATURE_KEY, \
    PURE_METAL_KEY, TEMPERATURE_KEY, VALUE_KEY, BOUNDS_KEY, CONSTANT_KEY, ALLOY_KEY, REGRESSION_KEY, SIMPLIFY_KEY, \
    PRE_KEY, DEGREE_KEY, SEGMENTS_KEY, FILE_PATH_KEY, EQUATION_KEY, POST_KEY

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
                sorted_props = sorted(prop_list, key=lambda x: 0 if x[0] in [MELTING_TEMPERATURE_KEY, BOILING_TEMPERATURE_KEY,
                                                                             SOLIDUS_TEMPERATURE_KEY, LIQUIDUS_TEMPERATURE_KEY,
                                                                             INITIAL_BOILING_TEMPERATURE_KEY, FINAL_BOILING_TEMPERATURE_KEY] else 1)
                for prop_name, config in sorted_props:
                    if prop_type == PropertyType.CONSTANT and prop_name not in ['latent_heat_of_fusion', 'latent_heat_of_vaporization']:
                        self._process_constant_property(material, prop_name, config, T)
                    elif prop_type == PropertyType.CONSTANT and prop_name in ['latent_heat_of_fusion', 'latent_heat_of_vaporization']:
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
            raise ValueError(f"Failed to process properties \n -> {str(e)}") from e

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
            raise ValueError(f"Failed to process constant property \n -> {str(e)}") from e

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
            if material.material_type == PURE_METAL_KEY:
                # For pure metals: step function at the transition temperature
                if prop_name == 'latent_heat_of_fusion':
                    expanded_config = {
                        TEMPERATURE_KEY: MELTING_TEMPERATURE_KEY,
                        VALUE_KEY: [0, latent_heat_value],
                        BOUNDS_KEY: [CONSTANT_KEY, CONSTANT_KEY],
                    }
                elif prop_name == 'latent_heat_of_vaporization':
                    expanded_config = {
                        TEMPERATURE_KEY: BOILING_TEMPERATURE_KEY,
                        VALUE_KEY: [0, latent_heat_value],
                        BOUNDS_KEY: [CONSTANT_KEY, CONSTANT_KEY],
                    }
                else:
                    raise ValueError(f"Unsupported latent heat property: {prop_name}")
            elif material.material_type == ALLOY_KEY:
                # For alloys: linear curve between temperature points
                if prop_name == 'latent_heat_of_fusion':
                    expanded_config = {
                        TEMPERATURE_KEY: [SOLIDUS_TEMPERATURE_KEY, LIQUIDUS_TEMPERATURE_KEY],
                        VALUE_KEY: [0, latent_heat_value],
                        BOUNDS_KEY: [CONSTANT_KEY, CONSTANT_KEY],
                        REGRESSION_KEY: {SIMPLIFY_KEY: PRE_KEY, DEGREE_KEY: 1, SEGMENTS_KEY: 1},
                    }
                elif prop_name == 'latent_heat_of_vaporization':
                    expanded_config = {
                        TEMPERATURE_KEY: [INITIAL_BOILING_TEMPERATURE_KEY, FINAL_BOILING_TEMPERATURE_KEY],
                        VALUE_KEY: [0, latent_heat_value],
                        BOUNDS_KEY: [CONSTANT_KEY, CONSTANT_KEY],
                        REGRESSION_KEY: {SIMPLIFY_KEY: PRE_KEY, DEGREE_KEY: 1, SEGMENTS_KEY: 1},
                    }
                else:
                    raise ValueError(f"Unsupported latent heat property: {prop_name}")
            else:
                raise ValueError(f"Unsupported latent heat configuration: {prop_name}")
            self._process_key_val_property(material, prop_name, expanded_config, T)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to process {prop_name} constant \n -> {str(e)}") from e

    def _process_file_property(self, material: Material, prop_name: str, file_config: Union[str, Dict[str, Any]], T: Union[float, sp.Symbol]) -> None:
        """Process property data from a file configuration."""
        logger.debug("""PropertyProcessor: _process_file_property:
            material: %r
            prop_name: %r
            file_config: %r
            T: %r""", material, prop_name, file_config, T)
        try:
            yaml_dir = self.base_dir
            file_config[FILE_PATH_KEY] = str(yaml_dir / file_config[FILE_PATH_KEY])
            temp_array, prop_array = read_data_from_file(file_config)
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
            lower_bound_type, upper_bound_type = file_config[BOUNDS_KEY]
            lower_bound = np.min(temp_array)
            upper_bound = np.max(temp_array)
            if not isinstance(T, sp.Symbol):
                interpolated_value = self._interpolate_value(T, temp_array, prop_array, lower_bound_type, upper_bound_type)
                setattr(material, prop_name, sp.Float(interpolated_value))
                return
            has_regression, simplify_type, degree, segments = self._process_regression_params(file_config, prop_name, len(temp_array))
            if has_regression and simplify_type == PRE_KEY:
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
            if (material.material_type == PURE_METAL_KEY and isinstance(prop_config[TEMPERATURE_KEY], str)
                    and prop_config[TEMPERATURE_KEY] in [MELTING_TEMPERATURE_KEY, BOILING_TEMPERATURE_KEY] and len(prop_config[VALUE_KEY]) == 2):
                # Get the transition temperature
                if prop_config[TEMPERATURE_KEY] == MELTING_TEMPERATURE_KEY:
                    transition_temp = float(material.melting_temperature)
                elif prop_config[TEMPERATURE_KEY] == BOILING_TEMPERATURE_KEY:
                    transition_temp = float(material.boiling_temperature)
                else:
                    raise ValueError(f"Invalid key '{prop_config[TEMPERATURE_KEY]}' for {PURE_METAL_KEY}")
                # Create step function
                T_sym = T if isinstance(T, sp.Symbol) else sp.Symbol('T')
                val_array = np.array(prop_config[VALUE_KEY], dtype=float)
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
            key_array = self._process_key_definition(prop_config[TEMPERATURE_KEY], prop_config[VALUE_KEY], material)
            val_array = np.array(prop_config[VALUE_KEY], dtype=float)
            if len(key_array) != len(val_array):
                raise ValueError(f"Length mismatch in {prop_name}: key and val arrays must have same length")
            # Skip temperature validation for latent heat properties
            if prop_name not in ['latent_heat_of_fusion', 'latent_heat_of_vaporization']:
                self._validate_temperature_range(prop_name, key_array)
            key_array, val_array = ensure_ascending_order(key_array, val_array)
            lower_bound_type, upper_bound_type = prop_config[BOUNDS_KEY]
            lower_bound = np.min(key_array)
            upper_bound = np.max(key_array)
            if not isinstance(T, sp.Symbol):
                interpolated_value = self._interpolate_value(T, key_array, val_array, lower_bound_type, upper_bound_type)
                setattr(material, prop_name, sp.Float(interpolated_value))
                return
            has_regression, simplify_type, degree, segments = self._process_regression_params(prop_config, prop_name, len(key_array))
            if has_regression and simplify_type == PRE_KEY:
                pw = _process_regression(key_array, val_array, T, lower_bound_type, upper_bound_type, degree, segments, seed)
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
                    if k == MELTING_TEMPERATURE_KEY and hasattr(material, MELTING_TEMPERATURE_KEY):
                        processed_key.append(material.melting_temperature)
                    elif k == BOILING_TEMPERATURE_KEY and hasattr(material, BOILING_TEMPERATURE_KEY):
                        processed_key.append(material.boiling_temperature)
                    # Handle alloy properties
                    elif k == SOLIDUS_TEMPERATURE_KEY and hasattr(material, SOLIDUS_TEMPERATURE_KEY):
                        processed_key.append(material.solidus_temperature)
                    elif k == LIQUIDUS_TEMPERATURE_KEY and hasattr(material, LIQUIDUS_TEMPERATURE_KEY):
                        processed_key.append(material.liquidus_temperature)
                    elif k == INITIAL_BOILING_TEMPERATURE_KEY and hasattr(material, INITIAL_BOILING_TEMPERATURE_KEY):
                        processed_key.append(material.initial_boiling_temperature)
                    elif k == FINAL_BOILING_TEMPERATURE_KEY and hasattr(material, FINAL_BOILING_TEMPERATURE_KEY):
                        processed_key.append(material.final_boiling_temperature)
                    # Handle offset notation (e.g., "melting_temperature+10")
                    elif '+' in k:
                        base, offset = k.split('+')
                        offset_value = float(offset)
                        # Get base value based on material type
                        if base == MELTING_TEMPERATURE_KEY and hasattr(material, MELTING_TEMPERATURE_KEY):
                            base_value = material.melting_temperature
                        elif base == BOILING_TEMPERATURE_KEY and hasattr(material, BOILING_TEMPERATURE_KEY):
                            base_value = material.boiling_temperature
                        elif base == SOLIDUS_TEMPERATURE_KEY and hasattr(material, SOLIDUS_TEMPERATURE_KEY):
                            base_value = material.solidus_temperature
                        elif base == LIQUIDUS_TEMPERATURE_KEY and hasattr(material, LIQUIDUS_TEMPERATURE_KEY):
                            base_value = material.liquidus_temperature
                        elif base == INITIAL_BOILING_TEMPERATURE_KEY and hasattr(material, INITIAL_BOILING_TEMPERATURE_KEY):
                            base_value = material.initial_boiling_temperature
                        elif base == FINAL_BOILING_TEMPERATURE_KEY and hasattr(material, FINAL_BOILING_TEMPERATURE_KEY):
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
                        if base == MELTING_TEMPERATURE_KEY and hasattr(material, MELTING_TEMPERATURE_KEY):
                            base_value = material.melting_temperature
                        elif base == BOILING_TEMPERATURE_KEY and hasattr(material, BOILING_TEMPERATURE_KEY):
                            base_value = material.boiling_temperature
                        elif base == SOLIDUS_TEMPERATURE_KEY and hasattr(material, SOLIDUS_TEMPERATURE_KEY):
                            base_value = material.solidus_temperature
                        elif base == LIQUIDUS_TEMPERATURE_KEY and hasattr(material, LIQUIDUS_TEMPERATURE_KEY):
                            base_value = material.liquidus_temperature
                        elif base == INITIAL_BOILING_TEMPERATURE_KEY and hasattr(material, INITIAL_BOILING_TEMPERATURE_KEY):
                            base_value = material.initial_boiling_temperature
                        elif base == FINAL_BOILING_TEMPERATURE_KEY and hasattr(material, FINAL_BOILING_TEMPERATURE_KEY):
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

        temp_points = np.array(prop_config[TEMPERATURE_KEY], dtype=float)
        eqn_strings = prop_config[EQUATION_KEY]

        # Validate that only 'T' is used in equations
        for eq in eqn_strings:
            expr = sp.sympify(eq)
            symbols = expr.free_symbols
            for sym in symbols:
                if str(sym) != 'T':
                    raise ValueError(f"Unsupported symbol '{sym}' found in equation '{eq}' for property '{prop_name}'. Only 'T' is allowed.")

        lower_bound_type, upper_bound_type = prop_config[BOUNDS_KEY]

        # Use standard symbol for parsing
        T_standard = sp.Symbol('T')
        temp_points, eqn_strings = ensure_ascending_order(temp_points, eqn_strings)

        # Parse expressions with standard symbol
        eqn_exprs = [sp.sympify(eq, locals={'T': T_standard}) for eq in eqn_strings]

        # Create piecewise function with standard symbol
        pw_standard = self._create_piecewise_from_formulas(temp_points, eqn_exprs, T_standard, lower_bound_type, upper_bound_type)

        # If T is a different symbol, substitute T_standard with the actual symbol
        if isinstance(T, sp.Symbol) and str(T) != 'T':
            pw = pw_standard.subs(T_standard, T)
        else:
            pw = pw_standard

        # Handle numeric temperature
        if not isinstance(T, sp.Symbol):
            value = float(pw_standard.subs(T_standard, T).evalf())
            setattr(material, prop_name, sp.Float(value))
            self.processed_properties.add(prop_name)
            return

        has_regression, simplify_type, degree, segments = self._process_regression_params(prop_config, prop_name, len(temp_points))

        f_pw = sp.lambdify(T, pw, 'numpy')
        diff = abs(self.temperature_array[1] - self.temperature_array[0])
        temp_dense = np.arange(temp_points[0], temp_points[-1]+diff/2, diff)
        y_dense = f_pw(temp_dense)

        if has_regression and simplify_type == PRE_KEY:
            # Use T_standard for regression and then substitute if needed
            pw_reg = _process_regression(temp_dense, y_dense, T_standard, lower_bound_type, upper_bound_type, degree, segments, seed)
            # If T is a different symbol, substitute T_standard with the actual symbol
            if isinstance(T, sp.Symbol) and str(T) != 'T':
                pw_reg = pw_reg.subs(T_standard, T)
            setattr(material, prop_name, pw_reg)
        else: # No regression OR not pre
            raw_pw = pw
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
            elif isinstance(prop_config, dict) and EQUATION_KEY in prop_config:
                expression = prop_config[EQUATION_KEY]
            else:
                raise ValueError(f"Unsupported property configuration format for {prop_name}: {prop_config}")

            # Process the expression
            try:
                material_property = self._parse_and_process_expression(expression, material, T)
            except CircularDependencyError:
                # Re-raise CircularDependencyError without wrapping it
                raise
            except Exception as e:
                # Wrap other exceptions with more context
                raise ValueError(f"Failed to process computed property '{prop_name}' \n -> {str(e)}") from e

            # Set property on material
            setattr(material, prop_name, material_property)

            # Handle non-symbolic temperature case
            if not isinstance(T, sp.Symbol):
                # For numeric T, we've already substituted the value in _parse_and_process_expression
                # Just ensure it's a float value
                if hasattr(material_property, 'evalf'):
                    value = float(material_property.evalf())
                else:
                    value = float(material_property)
                setattr(material, prop_name, sp.Float(value))

            self.processed_properties.add(prop_name)

            # Extract bounds and regression parameters for visualization
            lower_bound_type, upper_bound_type = CONSTANT_KEY, CONSTANT_KEY
            if isinstance(prop_config, dict) and BOUNDS_KEY in prop_config:
                if isinstance(prop_config[BOUNDS_KEY], list) and len(prop_config[BOUNDS_KEY]) == 2:
                    lower_bound_type, upper_bound_type = prop_config[BOUNDS_KEY]

            temp_array = self.temperature_array
            lower_bound = np.min(temp_array)
            upper_bound = np.max(temp_array)

            # Get regression parameters
            has_regression, simplify_type, degree, segments = self._process_regression_params(prop_config, prop_name, len(temp_array))

            # Create function from symbolic expression
            # Use standard symbol for lambdification if T is a different symbol
            T_standard = sp.Symbol('T')
            if isinstance(T, sp.Symbol) and str(T) != 'T':
                standard_expr = material_property.subs(T, T_standard)
                f_pw = sp.lambdify(T_standard, standard_expr, 'numpy')
            else:
                f_pw = sp.lambdify(T, material_property, 'numpy')

            # Evaluate the function over the temperature range
            try:
                y_dense = f_pw(temp_array)

                # Check for invalid values
                if not np.all(np.isfinite(y_dense)):
                    invalid_count = np.sum(~np.isfinite(y_dense))
                    logger.warning(
                        f"Property '{prop_name}' has {invalid_count} non-finite values. "
                        f"This may indicate issues with the expression: {expression}"
                    )

                # Apply regression if needed
                if has_regression and simplify_type == PRE_KEY:
                    # Use T_standard for regression and then substitute if needed
                    T_for_regression = T_standard if isinstance(T, sp.Symbol) and str(T) != 'T' else T
                    pw_reg = _process_regression(temp_array, y_dense, T_for_regression, lower_bound_type, upper_bound_type, degree, segments, seed)

                    # If T is a different symbol, substitute T_standard with the actual symbol
                    if isinstance(T, sp.Symbol) and str(T) != 'T':
                        pw_reg = pw_reg.subs(T_standard, T)
                    setattr(material, prop_name, pw_reg)
                else:  # No regression OR not pre
                    raw_pw = self._create_raw_piecewise(temp_array, y_dense, T, lower_bound_type, upper_bound_type)
                    setattr(material, prop_name, raw_pw)

                self.processed_properties.add(prop_name)

                # Visualize the property if a visualizer is available
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
                raise ValueError(f"Error evaluating expression for '{prop_name}': {str(e)}") from e

        except CircularDependencyError:
            # Re-raise CircularDependencyError without wrapping it
            raise
        except Exception as e:
            raise ValueError(f"Failed to process computed property '{prop_name}' \n -> {str(e)}") from e

    # --- Expression/Dependency Helpers ---
    def _parse_and_process_expression(self, expression: str, material: Material, T: Union[float, sp.Symbol]) -> sp.Expr:
        """Parse and process a mathematical expression string into a SymPy expression."""
        logger.debug("PropertyProcessor: _parse_and_process_expression - expression: %r, T: %r", expression, T)

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
            self._check_circular_dependencies(prop_name=None, current_deps=dependencies, visited=set())

            # Process dependencies first
            for dep in dependencies:
                if not hasattr(material, dep) or getattr(material, dep) is None:
                    if dep in self.properties:
                        self._process_computed_property(material, dep, T)
                    else:
                        # This should not happen due to the check above, but just in case
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
            if isinstance(T, sp.Symbol):
                # If T is a symbolic variable, substitute the standard 'T' with it
                substitutions[T_standard] = T
            else:
                # For numeric T, substitute with the value directly
                substitutions[T_standard] = T

            # Perform substitution and evaluate integrals if present
            result_expr = sympy_expr.subs(substitutions)
            if isinstance(result_expr, sp.Integral):
                result_expr = result_expr.doit()

            return result_expr

        except CircularDependencyError:
            # Re-raise the circular dependency error directly without wrapping it
            raise
        except DependencyError as e:
            # Re-raise with the original exception as the cause
            raise e
        except Exception as e:
            # Wrap other exceptions with more context
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
                if not isinstance(prop_config, dict) or REGRESSION_KEY not in prop_config:
                    continue
                regression_config = prop_config[REGRESSION_KEY]
                simplify_type = regression_config[SIMPLIFY_KEY]
                if simplify_type != POST_KEY:
                    continue
                prop_value = getattr(material, prop_name)
                if isinstance(prop_value, sp.Integral):
                    raise ValueError(f"Property '{prop_name}' is an integral and cannot be post-processed.")
                    result = prop_value.doit()
                    if isinstance(result, sp.Integral):
                        temp_array = self.temperature_array
                        values = np.array([float(prop_value.evalf(subs={T: t})) for t in temp_array], dtype=float)
                        degree = regression_config[DEGREE_KEY]
                        segments = regression_config[SEGMENTS_KEY]
                        lower_bound_type, upper_bound_type = prop_config[BOUNDS_KEY]
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
                degree = regression_config[DEGREE_KEY]
                segments = regression_config[SEGMENTS_KEY]
                lower_bound_type, upper_bound_type = prop_config[BOUNDS_KEY]
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
                logger.debug(f"""PropertyVisualizer: _post_process_properties:
                    Post-processed property with simplify type {POST_KEY}: %r""", prop_name)
            except Exception as e:
                error_msg = f"Failed to post-process {prop_name}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        if errors:
            error_msg = "\n".join(errors)
            raise ValueError(f"Post-processing errors occurred: \n -> {error_msg}")

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
