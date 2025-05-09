import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pwlf
import sympy as sp

logger = logging.getLogger(__name__)

# --- Shared/Utility Methods ---
def ensure_ascending_order(temp_array: np.ndarray, *value_arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Ensure temperature array is in ascending order, flipping all provided arrays if needed."""
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
    """Validate that temperature arrays from properties fall within the global temperature range."""
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
    """Interpolate a value at temperature T using the provided data arrays."""
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
    """Process regression parameters from configuration."""
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

def _process_regression(temp_array, prop_array, T, lower_bound_type, upper_bound_type, degree, segments, seed=13579):
    """Centralized regression processing logic."""
    v_pwlf = pwlf.PiecewiseLinFit(temp_array, prop_array, degree=degree, seed=seed)
    v_pwlf.fit(n_segments=segments)
    return sp.Piecewise(*get_symbolic_conditions(v_pwlf, T, lower_bound_type, upper_bound_type))

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

def create_piecewise_from_formulas(
        temp_points: np.ndarray,
        eqn_exprs: List[Union[str, sp.Expr]],
        T: sp.Symbol,
        lower_bound_type: str = 'constant',
        upper_bound_type: str = 'constant') -> sp.Piecewise:
    """Create a SymPy Piecewise function from breakpoints and symbolic expressions."""
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
    if len(eqn_exprs) != len(temp_points) - 1:
        raise ValueError(
            f"Number of formulas ({len(eqn_exprs)}) must be one less than number of breakpoints ({len(temp_points)})"
        )
    # Special case: single expression with extrapolation at both ends
    if len(eqn_exprs) == 1 and lower_bound_type == 'extrapolate' and upper_bound_type == 'extrapolate':
        logger.warning(
            "Using a single expression with extrapolation at both ends. "
            "Consider simplifying your YAML definition to use a direct equation."
        )
        return eqn_exprs[0]
    conditions = []
    # Handle lower bound
    if lower_bound_type == 'constant':
        conditions.append((eqn_exprs[0].subs(T, temp_points[0]), T < temp_points[0]))
    # Handle intervals (including special cases for first and last)
    for i, expr in enumerate(eqn_exprs):
        if i == 0 and lower_bound_type == 'extrapolate':
            # First segment with extrapolation
            conditions.append((expr, T < temp_points[i+1]))
        elif i == len(eqn_exprs) - 1 and upper_bound_type == 'extrapolate':
            # Last segment with extrapolation
            conditions.append((expr, T >= temp_points[i]))
        else:
            # Regular interval
            conditions.append((expr, sp.And(T >= temp_points[i], T < temp_points[i+1])))
    # Handle upper bound
    if upper_bound_type == 'constant':
        conditions.append((eqn_exprs[-1].subs(T, temp_points[-1]), T >= temp_points[-1]))

    return sp.Piecewise(*conditions)

#https://github.com/cjekel/piecewise_linear_fit_py/blob/master/examples/understanding_higher_degrees/polynomials_in_pwlf.ipynb
def get_symbolic_eqn(pwlf_: pwlf.PiecewiseLinFit, segment_number: int, x: Union[float, sp.Symbol]):
    if pwlf_.degree < 1:
        raise ValueError('Degree must be at least 1')
    if segment_number < 1 or segment_number > pwlf_.n_segments:
        raise ValueError('segment_number not possible')
    # Check if x is a symbolic variable
    is_symbolic = isinstance(x, (sp.Symbol, sp.Expr))
    my_eqn = 0  # Initialize my_eqn before the loop
    # assemble degree = 1 first
    for line in range(segment_number):
        if line == 0:
            my_eqn = pwlf_.beta[0] + (pwlf_.beta[1])*(x-pwlf_.fit_breaks[0])
        else:
            my_eqn += (pwlf_.beta[line+1])*(x-pwlf_.fit_breaks[line])
    # assemble all other degrees
    if pwlf_.degree > 1:
        for k in range(2, pwlf_.degree + 1):
            for line in range(segment_number):
                beta_index = pwlf_.n_segments*(k-1) + line + 1
                my_eqn += (pwlf_.beta[beta_index])*(x-pwlf_.fit_breaks[line])**k
    # Only call simplify if x is symbolic
    if is_symbolic:
        return my_eqn.simplify()
    else:
        # For numeric x, just return the equation
        return my_eqn

def get_symbolic_conditions(pwlf_: pwlf.PiecewiseLinFit, x: sp.Symbol, lower_: str, upper_: str):
    """Create symbolic conditions for a piecewise function from a pwlf fit."""
    conditions = []

    # Special case: single segment with extrapolation at both ends
    if pwlf_.n_segments == 1 and lower_ == "extrapolate" and upper_ == "extrapolate":
        eqn = get_symbolic_eqn(pwlf_, 1, x)
        conditions.append((eqn, True))
        return conditions

    # Handle lower bound for all cases
    if lower_ == "constant":
        eqn = get_symbolic_eqn(pwlf_, 1, x)
        conditions.append((eqn.evalf(subs={x: pwlf_.fit_breaks[0]}), x < pwlf_.fit_breaks[0]))

    # Process all segments
    for i in range(pwlf_.n_segments):
        eqn = get_symbolic_eqn(pwlf_, i + 1, x)

        # First segment with extrapolation
        if i == 0 and lower_ == "extrapolate":
            conditions.append((eqn, x < pwlf_.fit_breaks[i+1]))
        # Last segment with extrapolation
        elif i == pwlf_.n_segments - 1 and upper_ == "extrapolate":
            conditions.append((eqn, x >= pwlf_.fit_breaks[i]))
        # Regular intervals (including first/last with constant bounds)
        else:
            conditions.append((eqn, sp.And(x >= pwlf_.fit_breaks[i], x < pwlf_.fit_breaks[i+1])))

    # Handle upper bound
    if upper_ == "constant":
        eqn = get_symbolic_eqn(pwlf_, pwlf_.n_segments, x)
        conditions.append((eqn.evalf(subs={x: pwlf_.fit_breaks[-1]}), x >= pwlf_.fit_breaks[-1]))

    return conditions
