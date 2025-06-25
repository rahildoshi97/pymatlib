import logging
from typing import List, Tuple, Union

import numpy as np
import sympy as sp

from pymatlib.algorithms import interpolate_value
from pymatlib.core.materials import Material
from pymatlib.data.constants import ProcessingConstants

logger = logging.getLogger(__name__)


# --- Core Utility Functions ---
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
