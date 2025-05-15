import logging
import os
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.gridspec import GridSpec

from pymatlib.core.material import Material
from pymatlib.core.yaml_parser.common_utils import _process_regression
from pymatlib.core.yaml_parser.yaml_keys import CONSTANT_KEY, PRE_KEY, POST_KEY, NAME_KEY, MATERIAL_TYPE_KEY

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
        self.plot_directory = "pymatlib_plots"
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
            material: Material,
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
            lower_bound_type: str = CONSTANT_KEY,
            upper_bound_type: str = CONSTANT_KEY) -> None:
        """Visualize a single property."""
        logger.debug("""PropertyVisualizer: visualize_property:
            material: %r
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
                     material,
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
            # Create subplot
            ax = self.fig.add_subplot(self.gs[self.current_subplot])
            self.current_subplot += 1

            # Get property and prepare temperature array
            current_prop = getattr(material, prop_name)
            temp_array = self.parser.temperature_array
            step = temp_array[1] - temp_array[0]

            # Set bounds
            if lower_bound is None or upper_bound is None:
                lower_bound = np.min(temp_array)
                upper_bound = np.max(temp_array)

            padding = (upper_bound - lower_bound) * 0.1
            ABSOLUTE_ZERO = 0.0
            padded_lower = max(lower_bound - padding, ABSOLUTE_ZERO)
            padded_upper = upper_bound + padding
            num_points = int(np.ceil((padded_upper - padded_lower) / step)) + 1
            extended_temp = np.linspace(padded_lower, padded_upper, num_points)

            # Set up plot style
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_title(f"{prop_name} ({prop_type} Property)", fontweight='bold')
            ax.set_xlabel("Temperature (K)", fontweight='bold')
            ax.set_ylabel(f"{prop_name}", fontweight='bold')

            # Define a color palette for better visualization
            colors = {
                'constant': '#1f77b4',          # blue
                'raw': '#ff7f0e',               # orange
                'regression_pre': '#2ca02c',    # green
                'regression_post': '#d62728',   # red
                'bounds': '#9467bd',            # purple
                'extended': '#8c564b',          # brown
            }

            if prop_type == 'CONSTANT':
                print(f"property {prop_name} is constant")
                value = float(current_prop)
                ax.axhline(y=value, color=colors['constant'], linestyle='-', linewidth=2.5, label='constant')
                ax.text(0.5, 0.9, f"Value: {value}", transform=ax.transAxes,
                        horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
                ax.set_ylim(value * 0.9, value * 1.1)
                y_value = value
            else:
                print(f"property {prop_name} is not constant")
                try:
                    f_current = sp.lambdify(T, current_prop, 'numpy')
                    y_value = None

                    # Plot raw data if available
                    if x_data is not None and y_data is not None and prop_type in ['FILE', 'KEY_VAL', 'PIECEWISE_EQUATION', 'COMPUTE']:
                        print(f"property {prop_name} has raw data")
                        # marker_size = 6 if prop_type == 'Key-Value' else 3
                        # ax.scatter(x_data, y_data, color=colors['raw'], marker='o', s=marker_size**2,
                                   # alpha=0.7, label='data points', zorder=3)

                    # Main line: either pre-regression or raw (extended)
                    if has_regression and simplify_type == PRE_KEY:
                        if prop_name in ['latent_heat_of_fusion', 'latent_heat_of_vaporization']:
                            # Calculate padding based on x_data range (15% on each side)
                            x_min, x_max = np.min(x_data), np.max(x_data)
                            x_range = x_max - x_min
                            padding = x_range * 0.15
                            extended_temp = np.array([
                                x_data[0] - padding,
                                x_data[0],
                                x_data[-1],
                                x_data[-1] + padding
                            ])

                        ax.plot(extended_temp, f_current(extended_temp), color=colors['regression_pre'],
                                linestyle='-', linewidth=2.5, label='regression (pre)', zorder=2)
                    else:
                        # No regression OR not pre: plot the raw extended function
                        ax.plot(extended_temp, f_current(extended_temp), color=colors['extended'],
                                linestyle='-', linewidth=2, label='raw (extended)', zorder=2)  # function

                    # For legend/annotation
                    y_value = np.max(y_data) if y_data is not None else f_current(upper_bound)

                    # Overlay post-regression fit if requested
                    if has_regression and simplify_type == POST_KEY and degree is not None and segments is not None:
                        try:
                            preview_pw = _process_regression(
                                temp_array=x_data, prop_array=y_data, T=T,
                                lower_bound_type=lower_bound_type, upper_bound_type=upper_bound_type,
                                degree=degree, segments=segments, seed=13579
                            )
                            f_preview = sp.lambdify(T, preview_pw, 'numpy')
                            ax.plot(extended_temp, f_preview(extended_temp), color=colors['regression_post'],
                                    linestyle='--', linewidth=2, label='regression (post)', zorder=4)
                        except (ValueError, TypeError, np.linalg.LinAlgError) as e:
                            logger.warning(f"Could not generate post-regression preview: {str(e)}")
                except (ValueError, TypeError, AttributeError) as e:
                    logger.error(f"Error creating function for property {prop_name}: {str(e)}")
                    ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes,
                            horizontalalignment='center', bbox=dict(facecolor='red', alpha=0.2))
                    y_value = 0

            # Add boundary lines
            ax.axvline(x=lower_bound, color=colors['bounds'], linestyle='--', alpha=0.7, linewidth=1.5, label='_nolegend_')
            ax.axvline(x=upper_bound, color=colors['bounds'], linestyle='--', alpha=0.7, linewidth=1.5, label='_nolegend_')

            # Handle y_value for annotations
            if y_value is None or not np.isfinite(y_value):
                try:
                    y_value = float(current_prop.subs(T, lower_bound).evalf())
                except (ValueError, TypeError, AttributeError):
                    y_value = 0.0

            # Add boundary type annotations
            ax.text(lower_bound, y_value, f' {lower_bound_type}',
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
            ax.text(upper_bound, y_value, f' {upper_bound_type}',
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

            # Add regression info if applicable
            if has_regression and degree is not None:
                ax.text(0.5, 0.95, f"Simplify: {simplify_type} | Degree: {degree} | Segments: {segments}",
                        transform=ax.transAxes, horizontalalignment='center',
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

            # Add legend with nice styling
            ax.legend(loc='best', framealpha=0.8, fancybox=True, shadow=True)

            # Add property to visualized set
            self.visualized_properties.add(prop_name)

        except ValueError as e:
            logger.error(f"ValueError visualizing property {prop_name}: {str(e)}")
            raise ValueError(f"Error visualizing property {prop_name}: {str(e)}")
        except TypeError as e:
            logger.error(f"TypeError visualizing property {prop_name}: {str(e)}")
            raise ValueError(f"Type error in property {prop_name}: {str(e)}")
        except AttributeError as e:
            logger.error(f"AttributeError visualizing property {prop_name}: {str(e)}")
            raise ValueError(f"Missing attribute for property {prop_name}: {str(e)}")
        except np.linalg.LinAlgError as e:
            logger.error(f"Linear algebra error visualizing property {prop_name}: {str(e)}")
            raise ValueError(f"Numerical error in property {prop_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error visualizing property {prop_name}: {str(e)}")
            raise ValueError(f"Unexpected error in property {prop_name}: {str(e)}")

    def save_property_plots(self) -> None:
        logger.debug("""PropertyVisualizer: save_property_plots:
            material name: %r""", self.parser.config[NAME_KEY])
        if hasattr(self, 'fig') and self.fig is not None:
            material_type = self.parser.config[MATERIAL_TYPE_KEY]
            title = f"Material Properties: {self.parser.config[NAME_KEY]} ({material_type})"
            self.fig.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            filepath = os.path.join(self.plot_directory, f"{self.parser.config[NAME_KEY].replace(' ', '_')}_properties.png")
            self.fig.savefig(filepath, dpi=300, bbox_inches="tight")
            logger.info(f"All properties plot saved as {filepath}")
            plt.close(self.fig)