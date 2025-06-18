import logging
from typing import List, Tuple, Union

import numpy as np
import sympy as sp

from pymatlib.algorithms import interpolate_value
from pymatlib.core.materials import Material
from pymatlib.data.constants import ProcessingConstants

logger = logging.getLogger(__name__)

# --- Core Utility Functions ---
def handle_numeric_temperature(material: Material, prop_name: str,
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

def create_step_visualization_data(transition_temp: float, val_array: List[float],
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
