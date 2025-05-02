import os
import pwlf
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Union, Optional

from pymatlib.core.alloy import Alloy
from pymatlib.core.pwlfsympy import get_symbolic_conditions

import logging
logger = logging.getLogger(__name__)

class PropertyVisualizer:
    """Handles visualization of material properties."""

    # --- Constructor ---
    def __init__(self, parser) -> None:
        logger.debug("""PropertyVisualizer: __init__""")
        self.parser = parser
        self.fig = None
        self.gs = None
        self.current_subplot = 0
        self.plot_directory = "property_plots"
        self.visualized_properties = set()

    # --- Public API Methods ---
    def initialize_plots(self) -> None:
        logger.debug("""PropertyVisualizer: initialize_plots:
            material name: %r""", self.parser.config['name'])
        if self.parser.categorized_properties is None:
            logger.warning("categorized_properties is None. Skipping plot initialization.")
            raise ValueError("No properties to plot.")
        property_count = sum(len(props) for props in self.parser.categorized_properties.values())
        self.fig = plt.figure(figsize=(12, 4 * property_count))
        self.gs = GridSpec(property_count, 1, figure=self.fig)
        self.current_subplot = 0
        os.makedirs(self.plot_directory, exist_ok=True)

    def reset_visualization_tracking(self) -> None:
        logger.debug("""PropertyVisualizer: reset_visualization_tracking""")
        self.visualized_properties = set()

    def visualize_property(
            self,
            alloy: Alloy,
            prop_name: str,
            T: Union[float, sp.Symbol],
            prop_type: str,
            x_data: Optional[np.ndarray] = None,
            y_data: Optional[np.ndarray] = None,
            has_regression: bool = False,
            simplify_type: Optional[str] = None,
            degree: int = 1,
            segments: int = 3,
            lower_bound: Optional[float] = None,
            upper_bound: Optional[float] = None,
            lower_bound_type: str = 'constant',
            upper_bound_type: str = 'constant') -> None:
        """Visualize a single property."""
        logger.debug("""PropertyVisualizer: visualize_property:
            alloy: %r
            prop_name: %r
            T: %r
            prop_type: %r
            x_data: %r
            y_data: %r
            has_regression: %r
            simplify_type: %r
            degree: %r
            segments: %r
            lower_bound: %r
            upper_bound: %r""",
                     alloy,
                     prop_name,
                     T,
                     prop_type,
                     x_data.shape if x_data is not None else None,
                     y_data.shape if y_data is not None else None,
                     has_regression,
                     simplify_type,
                     degree,
                     segments,
                     lower_bound,
                     upper_bound)
        if prop_name in self.visualized_properties:
            logger.debug("""PropertyVisualizer: visualize_property:
            Skipping - already visualized for property: %r""", prop_name)
            return
        if not hasattr(self, 'fig') or self.fig is None:
            logger.debug("""PropertyVisualizer: visualize_property:
            Skipping - no figure available for property: %r""", prop_name)
            return
        if not isinstance(T, sp.Symbol):
            logger.debug("""PropertyVisualizer: visualize_property:
            Skipping - T is not symbolic for property: %r""", prop_name)
            return
        try:
            ax = self.fig.add_subplot(self.gs[self.current_subplot])
            self.current_subplot += 1
            current_prop = getattr(alloy, prop_name)
            print(current_prop)
            print(type(current_prop))
            temp_array = self.parser.temperature_array
            step = temp_array[1] - temp_array[0]
            if lower_bound is None or upper_bound is None:
                lower_bound = np.min(temp_array)
                upper_bound = np.max(temp_array)
            padding = (upper_bound - lower_bound) * 0.1
            ABSOLUTE_ZERO = 0.0
            padded_lower = max(lower_bound - padding, ABSOLUTE_ZERO)
            padded_upper = upper_bound + padding
            num_points = int(np.ceil((padded_upper - padded_lower) / step)) + 1
            extended_temp = np.linspace(padded_lower, padded_upper, num_points)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_title(f"{prop_name} ({prop_type} Property)")
            ax.set_xlabel("Temperature (K)")
            ax.set_ylabel(f"{prop_name}")

            if prop_type == 'CONSTANT':  # isinstance(current_prop, float):
                value = float(current_prop)
                ax.axhline(y=value, color='blue', linestyle='-', linewidth=1.5, label='constant')
                ax.text(0.5, 0.9, f"Value: {value}", transform=ax.transAxes,
                        horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.5))
                ax.set_ylim(value * 0.9, value * 1.1)
                ax.legend(loc='best')
                y_value = value
            else:
                f_current = sp.lambdify(T, current_prop, 'numpy')
                y_value = None
                # Plot raw data if available (for data-driven property types)
                if x_data is not None and y_data is not None and prop_type in ['FILE', 'KEY_VAL', 'PIECEWISE_EQUATION', 'COMPUTE']:
                    marker_size = 2.5 if prop_type == 'Key-Value' else 2.
                    ax.plot(x_data, y_data, linewidth=1.0, marker='o', markersize=marker_size, label='raw')
                    # Main line: either pre-regression or raw (extended)
                    if has_regression and simplify_type == 'pre':
                        ax.plot(extended_temp, f_current(extended_temp), linestyle='-', linewidth=1, label='regression (pre)')
                    else:  # No regression OR not pre: plot the raw extended function
                        ax.plot(extended_temp, f_current(extended_temp), linestyle='-', linewidth=1, label='raw (extended)')
                    # For legend/annotation purposes
                    y_value = np.max(y_data) if y_data is not None else f_current(upper_bound)
                    # Overlay post-regression fit if requested
                    if has_regression and simplify_type == 'post' and degree is not None and segments is not None:
                        v_pwlf = pwlf.PiecewiseLinFit(x_data, y_data, degree=degree, seed=13579)
                        v_pwlf.fit(n_segments=segments)
                        preview_pw = sp.Piecewise(*get_symbolic_conditions(v_pwlf, T, lower_bound_type, upper_bound_type))
                        f_preview = sp.lambdify(T, preview_pw, 'numpy')
                        ax.plot(extended_temp, f_preview(extended_temp), linestyle=':', linewidth=1, label='regression (post)')
                else:  # INVALID property type
                    ax.plot(extended_temp, f_current(extended_temp), linestyle='-', linewidth=1, label='property')
                    try:
                        y_value = float(f_current(upper_bound))
                    except Exception:
                        y_value = float(np.nanmax(f_current(extended_temp)))
            ax.axvline(x=lower_bound, color='brown', linestyle='--', alpha=0.5, label='_nolegend_')
            ax.axvline(x=upper_bound, color='brown', linestyle='--', alpha=0.5, label='_nolegend_')
            if y_value is None or not np.isfinite(y_value):
                try:
                    y_value = float(current_prop.subs(T, lower_bound).evalf())
                except Exception:
                    y_value = 0.0
            ax.text(lower_bound, y_value, f' {lower_bound_type}',
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.5))
            ax.text(upper_bound, y_value, f' {upper_bound_type}',
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.5))
            if has_regression and degree is not None:
                # if prop_type == 'Computed' and simplify_type is None:
                    # simplify_type = 'post'
                ax.text(0.5, 0.95, f"Simplify: {simplify_type} | Degree: {degree} | Segments: {segments}",
                        transform=ax.transAxes, horizontalalignment='center',
                        bbox=dict(facecolor='white', alpha=0.5))
            ax.legend(loc='best')
            self.visualized_properties.add(prop_name)
        except Exception as e:
            logger.error(f"Error visualizing property {prop_name}: {str(e)}")
            raise ValueError(f"Error visualizing property {prop_name}: {str(e)}")

    def save_property_plots(self) -> None:
        logger.debug("""PropertyVisualizer: save_property_plots:
            material name: %r""", self.parser.config['name'])
        if hasattr(self, 'fig') and self.fig is not None:
            self.fig.suptitle(f"Material Properties: {self.parser.config['name']}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            filepath = os.path.join(self.plot_directory, f"{self.parser.config['name'].replace(' ', '_')}_properties.png")
            self.fig.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"All properties plot saved as {filepath}")
            plt.close(self.fig)