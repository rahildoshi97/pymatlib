import numpy as np
from typing import Tuple

from pymatlib.parsing.config.yaml_keys import CONSTANT_KEY


def interpolate_value(T: float, x_array: np.ndarray, y_array: np.ndarray,
                      lower_bound_type: str, upper_bound_type: str) -> float:
    """Interpolate a value at temperature T using the provided data arrays."""
    if T < x_array[0]:
        if lower_bound_type == CONSTANT_KEY:
            return float(y_array[0])
        else:  # 'extrapolate'
            denominator = x_array[1] - x_array[0]
            if denominator == 0:
                raise ValueError("Cannot extrapolate: first two temperature values are equal.")
            slope = (y_array[1] - y_array[0]) / denominator
            return float(y_array[0] + slope * (T - x_array[0]))
    elif T >= x_array[-1]:
        if upper_bound_type == CONSTANT_KEY:
            return float(y_array[-1])
        else:  # 'extrapolate'
            denominator = x_array[-1] - x_array[-2]
            if denominator == 0:
                raise ValueError("Cannot extrapolate: last two temperature values are equal.")
            slope = (y_array[-1] - y_array[-2]) / denominator
            return float(y_array[-1] + slope * (T - x_array[-1]))
    else:
        return float(np.interp(T, x_array, y_array))


def ensure_ascending_order(temp_array: np.ndarray, *value_arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Ensure temperature array is in ascending order, flipping all provided arrays if needed."""
    diffs = np.diff(temp_array)
    if np.all(diffs > 0):
        return (temp_array,) + value_arrays
    elif np.all(diffs < 0):
        flipped_temp = np.flip(temp_array)
        flipped_values = tuple(np.flip(arr) for arr in value_arrays)
        return (flipped_temp,) + flipped_values
    else:
        raise ValueError(f"Array is not strictly ascending or strictly descending: {temp_array}")