import numpy as np
import sympy as sp
from typing import Any, Tuple, List, Dict, Union

import logging
logger = logging.getLogger(__name__)

# --- Shared/Utility Methods ---
def ensure_ascending_order(temp_array: np.ndarray, *value_arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    Ensure temperature array is in ascending order, flipping all provided arrays if needed.
    """
    logger.debug("""ensure_ascending_order:
            temp_array: %r""",
                 temp_array.shape if temp_array is not None else None)
    diffs = np.diff(temp_array)
    if np.all(diffs > 0):
        return (temp_array,) + value_arrays
    elif np.all(diffs < 0):
        logger.debug("ensure_ascending_order: Temperature array is descending, flipping arrays")
        flipped_temp = np.flip(temp_array)
        flipped_values = tuple(np.flip(arr) for arr in value_arrays)
        return (flipped_temp,) + flipped_values
    else:
        raise ValueError(f"Array is not strictly ascending or strictly descending: {temp_array}")

def validate_temperature_range(
        prop: str,
        temp_array: np.ndarray,
        global_temp_array: np.ndarray) -> None:
    """
    Validate that temperature arrays from properties fall within the global temperature range.
    """
    logger.debug("""validate_temperature_range:
            prop: %r
            temp_array: %r
            global_temp_array: %r""",
                 prop,
                 temp_array.shape if temp_array is not None else None,
                 global_temp_array.shape if global_temp_array is not None else None)
    logger.debug(f"validate_temperature_range: Validating property '{prop}' with temperature array of shape {temp_array.shape} against global temperature array of shape {global_temp_array.shape}")
    min_temp, max_temp = np.min(global_temp_array), np.max(global_temp_array)
    if ((temp_array < min_temp) | (temp_array > max_temp)).any():
        out_of_range = np.where((temp_array < min_temp) | (temp_array > max_temp))[0]
        out_values = temp_array[out_of_range]
        if len(out_of_range) > 5:
            sample_indices = out_of_range[:5]
            sample_values = temp_array[sample_indices]
            raise ValueError(
                f"Property '{prop}' contains temperature values outside global range [{min_temp}, {max_temp}] "
                f"\n -> Found {len(out_of_range)} out-of-range values, first 5 at indices {sample_indices}: {sample_values}"
                f"\n -> Min value: {temp_array.min()}, Max value: {temp_array.max()}"
            )
        else:
            raise ValueError(
                f"Property '{prop}' contains temperature values outside global range [{min_temp}, {max_temp}] "
                f"\n -> Found {len(out_of_range)} out-of-range values at indices {out_of_range}: {out_values}"
            )

def interpolate_value(
        T: float,
        x_array: np.ndarray,
        y_array: np.ndarray,
        lower_bound_type: str,
        upper_bound_type: str) -> float:
    """
    Interpolate a value at temperature T using the provided data arrays.
    """
    logger.debug("""interpolate_value:
            T: %r
            x_array: %r
            y_array: %r
            lower_bound_type: %r
            upper_bound_type: %r""", T, x_array, y_array, lower_bound_type, upper_bound_type)
    if T < x_array[0]:
        if lower_bound_type == 'constant':
            return float(y_array[0])
        else:  # 'extrapolate'
            denominator = x_array[1] - x_array[0]
            if denominator == 0:
                raise ValueError("Cannot extrapolate: first two temperature values are equal.")
            slope = (y_array[1] - y_array[0]) / denominator
            return float(y_array[0] + slope * (T - x_array[0]))
    elif T >= x_array[-1]:
        if upper_bound_type == 'constant':
            return float(y_array[-1])
        else:  # 'extrapolate'
            denominator = x_array[-1] - x_array[-2]
            if denominator == 0:
                raise ValueError("Cannot extrapolate: last two temperature values are equal.")
            slope = (y_array[-1] - y_array[-2]) / denominator
            return float(y_array[-1] + slope * (T - x_array[-1]))
    else:
        return float(np.interp(T, x_array, y_array))

def process_regression_params(
        prop_config: dict,
        prop_name: str,
        data_length: int) -> Tuple[bool, Union[str, None], Union[int, None], Union[int, None]]:
    """
    Process regression parameters from configuration.
    """
    logger.debug("""process_regression_params:
            prop_config: %r
            prop_name: %r
            data_length: %r""", prop_config, prop_name, data_length)
    has_regression = isinstance(prop_config, dict) and 'regression' in prop_config
    if not has_regression:
        return False, None, None, None
    regression_config = prop_config['regression']
    simplify_type = regression_config['simplify']
    degree = regression_config['degree']
    segments = regression_config['segments']
    if segments >= data_length:
        raise ValueError(f"Number of segments ({segments}) must be less than number of data points ({data_length})")
    if segments > 8:
        raise ValueError(f"Number of segments ({segments}) is too high for {prop_name}. Please reduce it.")
    elif segments > 6:
        logger.warning(f"Number of segments ({segments}) for {prop_name} may lead to overfitting.")
    return has_regression, simplify_type, degree, segments



def create_raw_piecewise(temp_array: np.ndarray, prop_array: np.ndarray, T: sp.Symbol, lower: str = Union['constant', 'extrapolate'], upper: str = Union['constant', 'extrapolate'],):
    logger.debug("""create_raw_piecewise:
            temp_array: %r
            prop_array: %r
            T: %r
            lower: %r
            upper: %r""",
                 temp_array.shape if temp_array is not None else None,
                 prop_array.shape if prop_array is not None else None,
                 T, lower, upper)
    # if temp_array[0] > temp_array[-1]: temp_array, prop_array = np.flip(temp_array), np.flip(prop_array)
    if temp_array[0] > temp_array[-1]: raise ValueError("Temperature array is not in ascending order.")
    conditions = [((prop_array[0] if lower=='constant' else prop_array[0]+(prop_array[1]-prop_array[0])/(temp_array[1]-temp_array[0])*(T-temp_array[0])), T<temp_array[0])] + \
                 [(prop_array[i]+(prop_array[i+1]-prop_array[i])/(temp_array[i+1]-temp_array[i])*(T-temp_array[i]), sp.And(T>=temp_array[i], T<temp_array[i+1])) for i in range(len(temp_array)-1)] + \
                 [((prop_array[-1] if upper=='constant' else prop_array[-1]+(prop_array[-1]-prop_array[-2])/(temp_array[-1]-temp_array[-2])*(T-temp_array[-1])), T>=temp_array[-1])]
    return sp.Piecewise(*conditions)

def create_raw_piecewise1(
        temp_array: np.ndarray,
        prop_array: np.ndarray,
        T: sp.Symbol,
        lower_bound_type: str,
        upper_bound_type: str) -> sp.Piecewise:
    """
    Create a piecewise function using all data points.
    """
    logger.debug("""create_raw_piecewise1:
            temp_array: %r
            prop_array: %r
            T: %r
            lower_bound_type: %r
            upper_bound_type: %r""",
                 temp_array.shape if temp_array is not None else None,
                 prop_array.shape if prop_array is not None else None,
                 T, lower_bound_type, upper_bound_type)
    temp_array = np.asarray(temp_array)
    prop_array = np.asarray(prop_array)
    logger.debug(f"create_raw_piecewise: Creating piecewise function with temperature array of shape {temp_array.shape} and property array of shape {prop_array.shape}")
    if len(temp_array) != len(prop_array):
        raise ValueError("Temperature and property arrays must have the same length.")
    temp_array, prop_array = ensure_ascending_order(temp_array, prop_array)
    conditions = []
    # Lower boundary
    if lower_bound_type == 'constant':
        conditions.append((prop_array[0], T < temp_array[0]))
    else:  # 'extrapolate'
        if len(temp_array) >= 2:
            denominator = temp_array[1] - temp_array[0]
            if denominator == 0:
                raise ValueError("Cannot extrapolate: first two temperature values are equal.")
            slope = (prop_array[1] - prop_array[0]) / denominator
            extrapolated_expr = prop_array[0] + slope * (T - temp_array[0])
            conditions.append((extrapolated_expr, T < temp_array[0]))
        else:
            raise ValueError("Not enough points for extrapolation at lower bound.")
    # Segments
    for i in range(len(temp_array) - 1):
        denominator = temp_array[i + 1] - temp_array[i]
        if denominator == 0:
            raise ValueError(f"Temperature values at indices {i} and {i+1} are equal; cannot interpolate.")
        interp_expr = (
                prop_array[i] + (prop_array[i + 1] - prop_array[i]) /
                denominator * (T - temp_array[i])
        )
        conditions.append((interp_expr, sp.And(T >= temp_array[i], T < temp_array[i + 1])))
    # Upper boundary
    if upper_bound_type == 'constant':
        conditions.append((prop_array[-1], T >= temp_array[-1]))
    else:  # 'extrapolate'
        if len(temp_array) >= 2:
            denominator = temp_array[-1] - temp_array[-2]
            if denominator == 0:
                raise ValueError("Cannot extrapolate: last two temperature values are equal.")
            slope = (prop_array[-1] - prop_array[-2]) / denominator
            extrapolated_expr = prop_array[-1] + slope * (T - temp_array[-1])
            conditions.append((extrapolated_expr, T >= temp_array[-1]))
        else:
            raise ValueError("Not enough points for extrapolation at upper bound.")
    pw = sp.Piecewise(*conditions)
    return pw

def create_piecewise_from_formulas(
        temp_points: np.ndarray,
        eqn_exprs: List[Union[str, sp.Expr]],
        T: sp.Symbol,
        lower_bound_type: str = 'constant',
        upper_bound_type: str = 'constant') -> sp.Piecewise:
    """
    Create a SymPy Piecewise function from breakpoints and symbolic expressions.
    """
    logger.debug("""create_piecewise_from_formulas:
            temp_points: %r
            eqn_exprs: %r
            T: %r
            lower_bound_type: %r
            upper_bound_type: %r""", temp_points, eqn_exprs, T, lower_bound_type, upper_bound_type)
    temp_points = np.asarray(temp_points, dtype=float)
    eqn_exprs = [
        sp.sympify(expr, locals={'T': T}) if isinstance(expr, str) else expr
        for expr in eqn_exprs
    ]
    if len(eqn_exprs) < len(temp_points) - 1:
        eqn_exprs += [eqn_exprs[-1]] * (len(temp_points) - 1 - len(eqn_exprs))
    elif len(eqn_exprs) > len(temp_points) - 1:
        raise ValueError(
            f"Number of formulas ({len(eqn_exprs)}) must be one less than number of breakpoints ({len(temp_points)})"
        )
    conditions = []
    # Lower bound
    if lower_bound_type == 'constant':
        conditions.append((eqn_exprs[0].subs(T, temp_points[0]), T < temp_points[0]))
    else:  # 'extrapolate'
        slope = sp.diff(eqn_exprs[0], T)
        extrap_expr = eqn_exprs[0] + slope * (T - temp_points[0])
        conditions.append((extrap_expr, T < temp_points[0]))
    # Intervals
    for i, expr in enumerate(eqn_exprs):
        cond = (expr, sp.And(T >= temp_points[i], T < temp_points[i+1]))
        conditions.append(cond)
    # Upper bound
    if upper_bound_type == 'constant':
        conditions.append((eqn_exprs[-1].subs(T, temp_points[-1]), T >= temp_points[-1]))
    else:  # 'extrapolate'
        slope = sp.diff(eqn_exprs[-1], T)
        extrap_expr = eqn_exprs[-1] + slope * (T - temp_points[-1])
        conditions.append((extrap_expr, T >= temp_points[-1]))
    return sp.Piecewise(*conditions)