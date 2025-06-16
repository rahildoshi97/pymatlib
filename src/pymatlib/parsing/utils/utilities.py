import logging
from typing import List, Tuple, Union

import numpy as np
import sympy as sp

from pymatlib.algorithms import interpolate_value
from pymatlib.core.material import Material
from pymatlib.data.constants import ProcessingConstants

logger = logging.getLogger(__name__)

# --- Core Utility Functions ---
def is_monotonic(arr: np.ndarray, name: str = "Array",
                 mode: str = "strictly_increasing",
                 threshold: float = ProcessingConstants.MONOTONICITY_THRESHOLD,
                 raise_error: bool = True) -> bool:
    """Universal monotonicity checker supporting multiple modes."""
    for i in range(1, len(arr)):
        diff = arr[i] - arr[i-1]
        violation = False
        if mode == "strictly_increasing" and diff <= threshold:
            violation = True
        elif mode == "non_decreasing" and diff < -threshold:
            violation = True
        elif mode == "strictly_decreasing" and diff >= -threshold:
            violation = True
        elif mode == "non_increasing" and diff > threshold:
            violation = True
        if violation:
            start_idx = max(0, i-2)
            end_idx = min(len(arr), i+3)
            context = "\nSurrounding values:\n"
            for j in range(start_idx, end_idx):
                context += f"Index {j}: {arr[j]:.10e}\n"
            error_msg = (
                f"{name} is not {mode.replace('_', ' ')} at index {i}:\n"
                f"Previous value ({i-1}): {arr[i-1]:.10e}\n"
                f"Current value ({i}): {arr[i]:.10e}\n"
                f"Difference: {diff:.10e}\n"
                f"{context}"
            )
            if raise_error:
                raise ValueError(error_msg)
            else:
                logger.warning(f"Warning: {error_msg}")
                return False
    # logger.info(f"{name} is {mode.replace('_', ' ')}")
    return True

def validate_energy_density_monotonicity(prop_name: str, temp_array: np.ndarray,
                                         prop_array: np.ndarray,
                                         tolerance: float = ProcessingConstants.DEFAULT_TOLERANCE) -> None:
    """Validate that energy_density is monotonically non-decreasing with temperature."""
    if prop_name != 'energy_density':
        return
    if len(temp_array) < ProcessingConstants.MIN_DATA_POINTS:
        return
    try:
        is_monotonic(prop_array, "Energy density", "non_decreasing", tolerance, True)
    except ValueError as e:
        # Add domain-specific context
        diffs = np.diff(prop_array)
        decreasing_indices = np.where(diffs < -tolerance)[0]
        if len(decreasing_indices) > 0:
            problem_temps = temp_array[decreasing_indices]
            problem_values = prop_array[decreasing_indices]
            next_values = prop_array[decreasing_indices + 1]
            error_details = []
            for i, (temp, val, next_val) in enumerate(zip(problem_temps[:3], problem_values[:3], next_values[:3])):
                decrease = val - next_val
                error_details.append(f"T={temp:.2f}K: {val:.6e} → {next_val:.6e} (decrease: {decrease:.6e})")
            enhanced_msg = (
                    f"Energy density must be monotonically non-decreasing with temperature. "
                    f"Found {len(decreasing_indices)} decreasing points:\n"
                    + "\n".join(error_details)
            )
            if len(decreasing_indices) > 3:
                enhanced_msg += f"\n... and {len(decreasing_indices) - 3} more violations"
            raise ValueError(enhanced_msg) from e

def evaluate_numeric_temperature(material: Material, prop_name: str,
                                 T: Union[float, sp.Symbol], processor_instance,
                                 piecewise_expr: sp.Expr = None,
                                 interpolation_data: Tuple = None) -> bool:
    """Handle the case where T is numeric - evaluate and set property."""
    if not isinstance(T, sp.Symbol):
        try:
            if interpolation_data is not None:
                x_array, y_array, lower_bound_type, upper_bound_type = interpolation_data
                value = interpolate_value(T, x_array, y_array, lower_bound_type, upper_bound_type)
            elif piecewise_expr is not None:
                T_standard = sp.Symbol('T')
                value = float(piecewise_expr.subs(T_standard, T).evalf())
            else:
                raise ValueError("Either piecewise_expr or interpolation_data must be provided")
            setattr(material, prop_name, sp.Float(value))
            processor_instance.processed_properties.add(prop_name)
            return True
        except Exception as e:
            raise ValueError(f"Failed to evaluate {prop_name} at T={T}: {str(e)}")
    return False

def generate_step_plot_data(transition_temp: float, val_array: List[float],
                            temp_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create proper step function visualization data."""
    margin = (np.max(temp_range) - np.min(temp_range)) * ProcessingConstants.TEMPERATURE_PADDING_FACTOR
    epsilon = ProcessingConstants.TEMPERATURE_EPSILON
    x_data = np.array([
        np.min(temp_range) - margin,
        transition_temp - epsilon,
        transition_temp,
        transition_temp + epsilon,
        np.max(temp_range) + margin
    ])
    y_data = np.array([
        val_array[0],
        val_array[0],
        val_array[0],
        val_array[1],
        val_array[1]
    ])
    return x_data, y_data
