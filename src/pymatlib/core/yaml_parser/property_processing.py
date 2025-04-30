import pwlf
import numpy as np
import sympy as sp
from pathlib import Path
from typing import Dict, Any, Union, List, Tuple, Optional

from pymatlib.core.alloy import Alloy
from pymatlib.core.yaml_parser.property_types import PropertyType
from pymatlib.core.data_handler import read_data_from_file
from pymatlib.core.pwlfsympy import get_symbolic_conditions
from pymatlib.core.symbol_registry import SymbolRegistry
from pymatlib.core.yaml_parser.common_utils import (
    ensure_ascending_order,
    validate_temperature_range,
    interpolate_value,
    process_regression_params,
    create_raw_piecewise,
    create_piecewise_from_formulas,)

import logging
logger = logging.getLogger(__name__)

seed = 13579

class PropertyProcessor:
    """Handles processing of different property types for alloy objects."""

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
            alloy: Alloy,
            T: Union[float, sp.Symbol],
            properties: Dict[str, Any],
            categorized_properties: Dict[PropertyType, List[Tuple[str, Any]]],
            temperature_array: np.ndarray,
            base_dir: Path,
            visualizer) -> None:
        """
        Process all properties for the alloy.
        """
        logger.debug("""PropertyProcessor: process_properties:
            alloy: %r
            T: %r
            properties: %r
            categorized_properties: %r
            temperature_array: %r
            base_dir: %r""",
                     alloy,
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
                for prop_name, config in prop_list:
                    if prop_type == PropertyType.CONSTANT and prop_name in ['latent_heat_of_fusion', 'latent_heat_of_vaporization']:
                        self._process_latent_heat_constant(alloy, prop_name, config, T)
                    elif prop_type == PropertyType.CONSTANT:
                        self._process_constant_property(alloy, prop_name, config, T)
                    elif prop_type == PropertyType.FILE:
                        self._process_file_property(alloy, prop_name, config, T)
                    elif prop_type == PropertyType.KEY_VAL:
                        self._process_key_val_property(alloy, prop_name, config, T)
                    elif prop_type == PropertyType.PIECEWISE_EQUATION:
                        self._process_piecewise_equation_property(alloy, prop_name, config, T)
                    elif prop_type == PropertyType.COMPUTE:
                        self._process_computed_property(alloy, prop_name, T)
                        print(f"_process_computed_property: prop_name={prop_name}, T={T}")
            self._post_process_properties(alloy, T)
        except Exception as e:
            raise ValueError(f"Failed to process properties \n -> {e}")

    # --- Property-Type Processing Methods ---
    def _process_latent_heat_constant(self, alloy: Alloy, prop_name: str, prop_config: Union[float, str], T: Union[float, sp.Symbol]) -> None:
        """Process latent heat properties when provided as constants."""
        logger.debug("""PropertyProcessor: _process_latent_heat_constant:
            alloy: %r
            prop_name: %r
            prop_config: %r
            T: %r""", alloy, prop_name, prop_config, T)
        try:
            latent_heat_value = float(prop_config)
            if prop_name == 'latent_heat_of_fusion':
                expanded_config = {
                    'key': ['solidus_temperature', 'liquidus_temperature'],
                    'val': [0, latent_heat_value],
                    'bounds': ['constant', 'constant'],
                    'regression': {'simplify': 'pre', 'degree': 1, 'segments': 1},
                }
            elif prop_name == 'latent_heat_of_vaporization':
                expanded_config = {
                    'key': ['boiling_temperature-10', 'boiling_temperature+10'],
                    'val': [0, latent_heat_value],
                    'bounds': ['constant', 'constant'],
                    'regression': {'simplify': 'pre', 'degree': 1, 'segments': 1},
                }
            else:
                raise ValueError(f"Unsupported latent heat configuration: {prop_name}")
            self._process_key_val_property(alloy, prop_name, expanded_config, T)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to process {prop_name} constant \n -> {e}") from e

    def _process_constant_property(self, alloy: Alloy, prop_name: str, prop_config: Union[float, str], T: Union[float, sp.Symbol]) -> None:
        """Process constant float property."""
        logger.debug("""PropertyProcessor: _process_constant_property:
            alloy: %r
            prop_name: %r
            prop_config: %r
            T: %r""", alloy, prop_name, prop_config, T)
        try:
            value = float(prop_config)
            setattr(alloy, prop_name, sp.Float(value))
            if self.visualizer is not None:
                self.visualizer.visualize_property(
                    alloy=alloy,
                    prop_name=prop_name,
                    T=T,
                    prop_type='CONSTANT',
                    lower_bound=min(self.temperature_array),
                    upper_bound=max(self.temperature_array)
                )
            self.processed_properties.add(prop_name)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to process constant property \n -> {e}") from e

    def _process_file_property(self, alloy: Alloy, prop_name: str, file_config: Union[str, Dict[str, Any]], T: Union[float, sp.Symbol]) -> None:
        """Process property data from a file configuration."""
        logger.debug("""PropertyProcessor: _process_file_property:
            alloy: %r
            prop_name: %r
            file_config: %r
            T: %r""", alloy, prop_name, file_config, T)
        try:
            yaml_dir = self.base_dir
            if isinstance(file_config, dict) and 'file' in file_config:
                file_config['file'] = str(yaml_dir / file_config['file'])
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
            lower_bound_type = file_config['bounds'][0]
            upper_bound_type = file_config['bounds'][1]
            lower_bound = np.min(temp_array)
            upper_bound = np.max(temp_array)
            is_symbolic = isinstance(T, sp.Symbol)
            if not is_symbolic:
                interpolated_value = self._interpolate_value(T, temp_array, prop_array, lower_bound_type, upper_bound_type)
                setattr(alloy, prop_name, sp.Float(interpolated_value))
                return
            has_regression, simplify_type, degree, segments = self._process_regression_params(file_config, prop_name, len(temp_array))
            if has_regression:
                if simplify_type == 'pre':
                    v_pwlf = pwlf.PiecewiseLinFit(temp_array, prop_array, degree=degree, seed=seed)
                    v_pwlf.fit(n_segments=segments)
                    print(f"_process_file_property: prop_name={prop_name}, T={T}, lower_bound_type={lower_bound_type}, upper_bound_type={upper_bound_type}")
                    pw = sp.Piecewise(*get_symbolic_conditions(v_pwlf, T, lower_bound_type, upper_bound_type))
                    setattr(alloy, prop_name, pw)
                else:
                    raw_pw = self._create_raw_piecewise(temp_array, prop_array, T, lower_bound_type, upper_bound_type)
                    setattr(alloy, prop_name, raw_pw)
            else:
                raw_pw = self._create_raw_piecewise(temp_array, prop_array, T, lower_bound_type, upper_bound_type)
                setattr(alloy, prop_name, raw_pw)
            if self.visualizer is not None:
                self.visualizer.visualize_property(
                    alloy=alloy,
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

    def _process_key_val_property(self, alloy: Alloy, prop_name: str, prop_config: Dict[str, Any], T: Union[float, sp.Symbol]) -> None:
        """Process property defined with key-val pairs."""
        logger.debug("""PropertyProcessor: _process_key_val_property:
            alloy: %r
            prop_name: %r
            prop_config: %r
            T: %r""", alloy, prop_name, prop_config, T)
        try:
            key_array = self._process_key_definition(prop_config['key'], prop_config['val'], alloy)
            val_array = np.array(prop_config['val'], dtype=float)
            if len(key_array) != len(val_array):
                raise ValueError(f"Length mismatch in {prop_name}: key and val arrays must have same length")
            if prop_name not in ['latent_heat_of_fusion', 'latent_heat_of_vaporization']:
                self._validate_temperature_range(prop_name, key_array)
            key_array, val_array = ensure_ascending_order(key_array, val_array)
            lower_bound_type = prop_config['bounds'][0]
            upper_bound_type = prop_config['bounds'][1]
            lower_bound = np.min(key_array)
            upper_bound = np.max(key_array)
            is_symbolic = isinstance(T, sp.Symbol)
            if not is_symbolic:
                interpolated_value = self._interpolate_value(T, key_array, val_array, lower_bound_type, upper_bound_type)
                setattr(alloy, prop_name, sp.Float(interpolated_value))
                return
            has_regression, simplify_type, degree, segments = self._process_regression_params(prop_config, prop_name, len(key_array))
            if has_regression:
                if simplify_type == 'pre':
                    v_pwlf = pwlf.PiecewiseLinFit(key_array, val_array, degree=degree, seed=seed)
                    v_pwlf.fit(n_segments=segments)
                    print(f"_process_key_val_property: prop_name={prop_name}, T={T}, lower_bound_type={lower_bound_type}, upper_bound_type={upper_bound_type}")
                    pw = sp.Piecewise(*get_symbolic_conditions(v_pwlf, T, lower_bound_type, upper_bound_type))
                    setattr(alloy, prop_name, pw)
                else:
                    raw_pw = self._create_raw_piecewise(key_array, val_array, T, lower_bound_type, upper_bound_type)
                    setattr(alloy, prop_name, raw_pw)
            else:
                raw_pw = self._create_raw_piecewise(key_array, val_array, T, lower_bound_type, upper_bound_type)
                setattr(alloy, prop_name, raw_pw)
            if self.visualizer is not None:
                self.visualizer.visualize_property(
                    alloy=alloy,
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
    def _process_key_definition(self, key_def: Union[str, List[Union[str, float]]], val_array: List[float], alloy: Alloy) -> np.ndarray:
        """Process temperature key definition."""
        logger.debug("""PropertyProcessor: _process_key_definition:
            key_def: %r
            val_array: %r
            alloy: %r""", key_def, val_array, alloy)
        try:
            if isinstance(key_def, str) and key_def.startswith('(') and key_def.endswith(')'):
                return self._process_equidistant_key(key_def, len(val_array))
            elif isinstance(key_def, list):
                return self._process_list_key(key_def, alloy)
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
    def _process_list_key(key_def: List[Union[str, float]], alloy: Alloy) -> np.ndarray:
        """Process list key definition."""
        logger.debug("""PropertyProcessor: _process_list_key:
            key_def: %r
            alloy: %r""", key_def, alloy)
        try:
            processed_key = []
            for k in key_def:
                if isinstance(k, str):
                    if k == 'solidus_temperature':
                        processed_key.append(alloy.solidus_temperature)
                    elif k == 'liquidus_temperature':
                        processed_key.append(alloy.liquidus_temperature)
                    elif k == 'boiling_temperature':
                        processed_key.append(alloy.boiling_temperature)
                    elif '+' in k:
                        base, offset = k.split('+')
                        offset_value = float(offset)
                        if base == 'solidus_temperature':
                            base_value = alloy.solidus_temperature
                        elif base == 'liquidus_temperature':
                            base_value = alloy.liquidus_temperature
                        elif base == 'boiling_temperature':
                            base_value = alloy.boiling_temperature
                        else:
                            base_value = float(base)
                        processed_key.append(base_value + offset_value)
                    elif '-' in k:
                        base, offset = k.split('-')
                        offset_value = -float(offset)
                        if base == 'solidus_temperature':
                            base_value = alloy.solidus_temperature
                        elif base == 'liquidus_temperature':
                            base_value = alloy.liquidus_temperature
                        elif base == 'boiling_temperature':
                            base_value = alloy.boiling_temperature
                        else:
                            base_value = float(base)
                        processed_key.append(base_value + offset_value)
                    else:
                        processed_key.append(float(k))
                else:
                    processed_key.append(float(k))
            key_array = np.array(processed_key, dtype=float)
            return key_array
        except Exception as e:
            raise ValueError(f"Error processing list key \n -> {str(e)}") from e

    def _process_piecewise_equation_property(self, alloy: Alloy, prop_name: str, prop_config: Dict[str, Any], T: Union[float, sp.Symbol]) -> None:
        """Process piecewise equation property."""
        logger.debug("""PropertyProcessor: _process_piecewise_equation_property:
            alloy: %r
            prop_name: %r
            prop_config: %r
            T: %r""", alloy, prop_name, prop_config, T)
        temp_points = np.array(prop_config['temperature'], dtype=float)
        eqn_strings = prop_config['equation']
        lower_bound_type, upper_bound_type = prop_config.get('bounds', ['constant', 'constant'])
        T_sym = T if isinstance(T, sp.Symbol) else sp.Symbol('T')
        temp_points, eqn_strings = ensure_ascending_order(temp_points, eqn_strings)
        eqn_exprs = [sp.sympify(eq, locals={'T': T_sym}) for eq in eqn_strings]
        pw = self._create_piecewise_from_formulas(temp_points, eqn_exprs, T_sym, lower_bound_type, upper_bound_type)
        print(f"pw={pw},\neqn_strings={eqn_strings},\ntemp_points={temp_points}")
        is_symbolic = isinstance(T, sp.Symbol)
        if not is_symbolic:
            value = float(pw.subs(T_sym, T).evalf())
            setattr(alloy, prop_name, sp.Float(value))
            self.processed_properties.add(prop_name)
            return
        has_regression, simplify_type, degree, segments = self._process_regression_params(prop_config, prop_name, len(temp_points))
        if not has_regression:
            simplify_type = 'post'
        f_pw = sp.lambdify(T_sym, pw, 'numpy')
        diff = abs(self.temperature_array[1] - self.temperature_array[0])
        print(f"diff={diff}")
        # temp_dense = np.linspace(temp_points[0], temp_points[-1], max(200, len(temp_points) * 100))  # TODO: Replace with self.temperature_array?
        temp_dense = np.arange(temp_points[0], temp_points[-1]+diff/2, diff)
        print(f"temp_dense.shape={temp_dense.shape}")
        y_dense = f_pw(temp_dense)
        if has_regression:
            if simplify_type == 'pre':
                v_pwlf = pwlf.PiecewiseLinFit(temp_dense, y_dense, degree=degree, seed=seed)
                v_pwlf.fit(n_segments=segments)
                print(f"_process_piecewise_equation_property: prop_name={prop_name}, T={T}, lower_bound_type={lower_bound_type}, upper_bound_type={upper_bound_type}")
                pw_reg = sp.Piecewise(*get_symbolic_conditions(v_pwlf, T_sym, lower_bound_type, upper_bound_type))
                print(f"pw_reg={pw_reg}")
                setattr(alloy, prop_name, pw_reg)
            else:
                # raw_pw = self._create_piecewise_from_formulas(temp_points, eqn_exprs, T, lower_bound_type, upper_bound_type)
                raw_pw = pw
                print(f"raw_pw={raw_pw}")
                setattr(alloy, prop_name, raw_pw)
        else:
            # raw_pw = self._create_piecewise_from_formulas(temp_points, eqn_exprs, T, lower_bound_type, upper_bound_type)
            raw_pw = pw
            print(f"raw_pw={raw_pw}")
            setattr(alloy, prop_name, raw_pw)
        self.processed_properties.add(prop_name)
        if self.visualizer is not None and isinstance(T, sp.Symbol):
            self.visualizer.visualize_property(
                alloy=alloy,
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

    def _process_computed_property(self, alloy: Alloy, prop_name: str, T: Union[float, sp.Symbol]) -> None:
        """Process computed properties using predefined models with dependency checking."""
        logger.debug("""PropertyProcessor: _process_computed_property:
            alloy: %r
            prop_name: %r
            T: %r""", alloy, prop_name, T)
        if prop_name in self.processed_properties:
            return
        try:
            prop_config = self.properties.get(prop_name)
            if prop_config is None:
                raise ValueError(f"Property '{prop_name}' not found in configuration")
            if isinstance(prop_config, str):
                expression = prop_config
                material_property = self._parse_and_process_expression(expression, alloy, T)
                print(f"prop_name 1={prop_name},\nT={T},\nmaterial_property={material_property}\n")
            elif isinstance(prop_config, dict) and 'equation' in prop_config:
                expression = prop_config['equation']
                material_property = self._parse_and_process_expression(expression, alloy, T)
                print(f"prop_name 2={prop_name},\nT={T},\nmaterial_property={material_property}\n")
            else:
                raise ValueError(f"Unsupported property configuration format for {prop_name}: {prop_config}")

            if isinstance(material_property, sp.Integral):
                result = material_property.doit()
                if isinstance(result, sp.Integral):
                    temp_array = self.temperature_array
                    values = np.array([float(material_property.evalf(subs={T: t})) for t in temp_array], dtype=float)
                    degree, segments = 2, 3
                    if isinstance(prop_config, dict) and 'regression' in prop_config:
                        degree = prop_config['regression'].get('degree', 2)
                        segments = prop_config['regression'].get('segments', 3)
                    lower_bound_type = prop_config.get('bounds', ['constant', 'constant'])[0] if isinstance(prop_config, dict) else 'constant'
                    upper_bound_type = prop_config.get('bounds', ['constant', 'constant'])[1] if isinstance(prop_config, dict) else 'constant'
                    v_pwlf = pwlf.PiecewiseLinFit(temp_array, values, degree=degree, seed=seed)
                    v_pwlf.fit(n_segments=segments)
                    print(f"_process_computed_property 1: prop_name={prop_name}, T={T}, lower_bound_type={lower_bound_type}, upper_bound_type={upper_bound_type}")
                    pw = sp.Piecewise(*get_symbolic_conditions(v_pwlf, T, lower_bound_type, upper_bound_type))
                    material_property = pw
                else:
                    material_property = result
            setattr(alloy, prop_name, material_property)
            T_sym = T if isinstance(T, sp.Symbol) else sp.Symbol('T')
            lower_bound_type = 'constant'
            upper_bound_type = 'constant'
            if isinstance(prop_config, dict) and 'bounds' in prop_config:
                if isinstance(prop_config['bounds'], list) and len(prop_config['bounds']) == 2:
                    lower_bound_type = prop_config['bounds'][0]
                    upper_bound_type = prop_config['bounds'][1]
            lower_bound = np.min(self.temperature_array)
            upper_bound = np.max(self.temperature_array)
            is_symbolic = isinstance(T, sp.Symbol)
            if not is_symbolic:
                value = float(material_property.subs(T_sym, T).evalf())
                setattr(alloy, prop_name, sp.Float(value))
                self.processed_properties.add(prop_name)
                return
            temp_array = self.temperature_array
            has_regression, simplify_type, degree, segments = self._process_regression_params(prop_config, prop_name, len(temp_array))
            if not has_regression:
                simplify_type = 'post'
            print("11")
            print(f"material_property={prop_name},\nexpr={material_property},\nT_sym={T_sym}")
            f_pw = sp.lambdify(T_sym, material_property, 'numpy')
            print("12")
            temp_dense = temp_array
            print("13")
            print(f"temp_dense.shape={temp_dense.shape}")
            y_dense = f_pw(temp_dense)  # TODO: <lambdifygenerated-16>:2: RuntimeWarning: divide by zero encountered in reciprocal
            print("14")
            print(f"y_dense={y_dense},\ny_dense.shape={y_dense.shape}")
            if has_regression:
                if simplify_type == 'pre':
                    v_pwlf = pwlf.PiecewiseLinFit(temp_dense, y_dense, degree=degree, seed=seed)
                    v_pwlf.fit(n_segments=segments)
                    pw_reg = sp.Piecewise(*get_symbolic_conditions(v_pwlf, T_sym, lower_bound_type, upper_bound_type))
                    setattr(alloy, prop_name, pw_reg)
                else:
                    raw_pw = self._create_raw_piecewise(temp_dense, y_dense, T, lower_bound_type, upper_bound_type)
                    setattr(alloy, prop_name, raw_pw)
            else:
                raw_pw = self._create_raw_piecewise(temp_dense, y_dense, T, lower_bound_type, upper_bound_type)
                setattr(alloy, prop_name, raw_pw)
            self.processed_properties.add(prop_name)
            if isinstance(T, sp.Symbol) and self.visualizer is not None:
                self.visualizer.visualize_property(
                    alloy=alloy,
                    prop_name=prop_name,
                    T=T,
                    prop_type='COMPUTE',
                    x_data=temp_dense,
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
                print(f"_process_computed_property 7: prop_name={prop_name}")
        except Exception as e:
            raise ValueError(f"Failed to process computed property '{prop_name}' \n -> {str(e)}") from e

    # --- Expression/Dependency Helpers ---
    def _parse_and_process_expression(self, expression: str, alloy: Alloy, T: Union[float, sp.Symbol]) -> sp.Expr:
        """Parse and process a mathematical expression string into a SymPy expression."""
        logger.debug("""PropertyProcessor: _parse_and_process_expression:
            expression: %r
            alloy: %r
            T: %r""", expression, alloy, T)
        try:
            sympy_expr = sp.sympify(expression, evaluate=False)
            dependencies = []
            for symbol in sympy_expr.free_symbols:
                symbol_str = str(symbol)
                if symbol_str == 'T' and isinstance(T, sp.Symbol):
                    continue
                dependencies.append(symbol_str)
            self._check_circular_dependencies(prop_name=None, current_deps=dependencies, visited=set())
            for dep in dependencies:
                if not hasattr(alloy, dep) or getattr(alloy, dep) is None:
                    if dep in self.properties:
                        # dep_config = self.properties[dep]
                        self._process_computed_property(alloy, dep, T)
                    else:
                        raise ValueError(f"Dependency '{dep}' not found in properties configuration")
            missing_deps = [dep for dep in dependencies if not hasattr(alloy, dep) or getattr(alloy, dep) is None]
            if missing_deps:
                raise ValueError(f"Cannot compute expression. Missing dependencies: {missing_deps}")
            substitutions = {}
            for dep in dependencies:
                dep_value = getattr(alloy, dep)
                dep_symbol = SymbolRegistry.get(dep)
                substitutions[dep_symbol] = dep_value
            if isinstance(T, sp.Symbol):
                substitutions[sp.Symbol('T')] = T
            result_expr = sympy_expr.subs(substitutions)
            if isinstance(result_expr, sp.Integral):
                result_expr = result_expr.doit()
            return result_expr
        except Exception as e:
            raise ValueError(f"Failed to parse and process expression: {expression} \n -> {str(e)}")

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
    def _post_process_properties(self, alloy: Alloy, T: Union[float, sp.Symbol]) -> None:
        """Perform post-processing on properties after all have been initially processed."""
        logger.debug("""PropertyProcessor: _post_process_properties:
            alloy: %r
            T: %r""", alloy, T)
        if not isinstance(T, sp.Symbol):
            return
        properties = self.properties
        errors = []
        for prop_name, prop_config in properties.items():
            try:
                if not isinstance(prop_config, dict) or 'regression' not in prop_config:
                    continue
                regression_config = prop_config['regression']
                simplify_type = regression_config.get('simplify', 'pre')
                if simplify_type != 'post':
                    continue
                prop_value = getattr(alloy, prop_name)
                if isinstance(prop_value, sp.Integral):
                    result = prop_value.doit()
                    if isinstance(result, sp.Integral):
                        temp_array = self.temperature_array
                        values = np.array([float(prop_value.evalf(subs={T: t})) for t in temp_array], dtype=float)
                        degree = regression_config.get('degree', 2)
                        segments = regression_config.get('segments', 3)
                        lower_bound_type = prop_config.get('bounds', ['constant', 'constant'])[0]
                        upper_bound_type = prop_config.get('bounds', ['constant', 'constant'])[1]
                        v_pwlf = pwlf.PiecewiseLinFit(temp_array, values, degree=degree, seed=seed)
                        v_pwlf.fit(n_segments=segments)
                        print(f"_post_process_properties: prop_name={prop_name}, T={T}, lower_bound_type={lower_bound_type}, upper_bound_type={upper_bound_type}")
                        pw = sp.Piecewise(*get_symbolic_conditions(v_pwlf, T, lower_bound_type, upper_bound_type))
                        prop_value = pw
                    else:
                        prop_value = result
                    setattr(alloy, prop_name, prop_value)
                if not isinstance(prop_value, sp.Expr):
                    logger.debug("""PropertyVisualizer: _post_process_properties:
                    Skipping - not symbolic for property: %r
                    type: %r""", prop_name, type(prop_value))
                    continue
                degree = regression_config.get('degree', 1)
                segments = regression_config.get('segments', 3)
                lower_bound_type = prop_config['bounds'][0]
                upper_bound_type = prop_config['bounds'][1]
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
                v_pwlf = pwlf.PiecewiseLinFit(temp_array, prop_array, degree=degree, seed=seed)
                v_pwlf.fit(n_segments=segments)
                print(f"_post_process_properties: prop_name={prop_name}, T={T}, lower_bound_type={lower_bound_type}, upper_bound_type={upper_bound_type}")
                pw = sp.Piecewise(*get_symbolic_conditions(v_pwlf, T, lower_bound_type, upper_bound_type))
                setattr(alloy, prop_name, pw)
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
