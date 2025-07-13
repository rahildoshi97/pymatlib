import logging
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.gridspec import GridSpec

from pymatlib.core.materials import Material
from pymatlib.algorithms.regression_processor import RegressionProcessor
from pymatlib.parsing.config.yaml_keys import CONSTANT_KEY, PRE_KEY, POST_KEY, NAME_KEY, MATERIAL_TYPE_KEY
from pymatlib.data.constants import PhysicalConstants, ProcessingConstants

logger = logging.getLogger(__name__)


class PropertyVisualizer:
    """Handles visualization of material properties."""

    # --- Constructor ---
    def __init__(self, parser) -> None:
        self.parser = parser
        self.fig = None
        self.gs = None
        self.current_subplot = 0
        yaml_dir = self.parser.base_dir
        self.plot_directory = yaml_dir / "pymatlib_plots"
        self.visualized_properties = set()
        self.is_enabled = True
        logger.debug("PropertyVisualizer initialized for parser: %s", parser.config_path)

    def is_visualization_enabled(self) -> bool:
        """Check if visualization is currently enabled."""
        return self.is_enabled and self.fig is not None

    # --- Public API Methods ---
    def initialize_plots(self) -> None:
        """Initialize plots only if visualization is enabled."""
        if not self.is_enabled:
            logger.debug("Visualization disabled, skipping plot initialization")
            return
        if self.parser.categorized_properties is None:
            logger.error("categorized_properties is None - cannot initialize plots")
            raise ValueError("No properties to plot.")
        property_count = sum(len(props) for props in self.parser.categorized_properties.values())
        logger.info("Initializing visualization for %d properties", property_count)
        self.fig = plt.figure(figsize=(12, 4 * property_count))
        self.gs = GridSpec(property_count, 1, figure=self.fig)
        self.current_subplot = 0
        self.plot_directory.mkdir(exist_ok=True)
        logger.debug("Plot directory created: %s", self.plot_directory)

    def reset_visualization_tracking(self) -> None:
        logger.debug("Resetting visualization tracking - clearing %d tracked properties",
                     len(self.visualized_properties))
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
            segments: int = 1,
            lower_bound: Optional[float] = None,
            upper_bound: Optional[float] = None,
            lower_bound_type: str = CONSTANT_KEY,
            upper_bound_type: str = CONSTANT_KEY) -> None:
        """Visualize a single property."""
        if prop_name in self.visualized_properties:
            logger.debug("Property '%s' already visualized, skipping", prop_name)
            return
        if not hasattr(self, 'fig') or self.fig is None:
            logger.warning("No figure available for property '%s' - visualization skipped", prop_name)
            return
        if not isinstance(T, sp.Symbol):
            logger.debug("Temperature is not symbolic for property '%s' - visualization skipped", prop_name)
            return
        logger.info("Visualizing property: %s (type: %s) for material: %s",
                    prop_name, prop_type, material.name)
        try:
            # Create subplot
            ax = self.fig.add_subplot(self.gs[self.current_subplot])
            self.current_subplot += 1
            # Get property and prepare temperature array
            current_prop = getattr(material, prop_name)
            if x_data is not None and len(x_data) > 0:
                # Use property's own temperature range
                data_lower, data_upper = np.min(x_data), np.max(x_data)
                temp_range = data_upper - data_lower
                step = temp_range / 1000  # Create 1000 points for smooth visualization
            else:
                # Fallback for properties without explicit temperature data
                data_lower, data_upper = (ProcessingConstants.DEFAULT_TEMP_LOWER,
                                          ProcessingConstants.DEFAULT_TEMP_UPPER)
                step = (data_upper - data_lower) / 1000
                logger.debug("Using data temperature range: %.1f - %.1f K", data_lower, data_upper)
            # Set bounds with property-specific defaults
            if lower_bound is None:
                lower_bound = data_lower
            if upper_bound is None:
                upper_bound = data_upper
                logger.debug("Using default temperature range: %.1f - %.1f K", data_lower, data_upper)
            # Create extended temperature range for visualization
            padding = (upper_bound - lower_bound) * ProcessingConstants.TEMPERATURE_PADDING_FACTOR
            ABSOLUTE_ZERO = PhysicalConstants.ABSOLUTE_ZERO
            padded_lower = max(lower_bound - padding, ABSOLUTE_ZERO)
            padded_upper = upper_bound + padding
            num_points = int(np.ceil((padded_upper - padded_lower) / step)) + 1
            extended_temp = np.linspace(padded_lower, padded_upper, num_points)
            # Set up plot style
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_title(f"{prop_name} ({prop_type} Property)", fontweight='bold')
            ax.set_xlabel("Temperature (K)", fontweight='bold')
            ax.set_ylabel(f"{prop_name}", fontweight='bold')
            colors = {
                'constant': '#1f77b4',  # blue
                'raw': '#ff7f0e',  # orange
                'regression_pre': '#2ca02c',  # green
                'regression_post': '#d62728',  # red
                'bounds': '#9467bd',  # purple
                'extended': '#8c564b',  # brown
            }
            # Initialize y_value for annotations
            _y_value = 0.0
            if prop_type == 'CONSTANT':
                value = float(current_prop)
                ax.axhline(y=value, color=colors['constant'], linestyle='-', linewidth=2.5, label='constant')
                ax.text(0.5, 0.9, f"Value: {value}", transform=ax.transAxes,
                        horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
                ax.set_ylim(value * 0.9, value * 1.1)
                _y_value = value
                logger.debug("Plotted constant property '%s' with value: %g", prop_name, value)
            elif prop_type == 'STEP_FUNCTION':
                if x_data is not None and y_data is not None:
                    # Plot step function with proper visualization
                    ax.plot(x_data, y_data, color=colors['raw'], linestyle='-', linewidth=2.5,
                            marker='o', markersize=4, label='step function', zorder=3)
                    # Add vertical line at transition point
                    transition_idx = len(x_data) // 2
                    transition_temp = x_data[transition_idx]
                    ax.axvline(x=transition_temp, color='red', linestyle='--', alpha=0.7,
                               linewidth=1.5, label='transition point')
                    # Add annotations
                    ax.text(transition_temp, y_data[0], f' Before: {y_data[0]:.2e}',
                            verticalalignment='bottom', horizontalalignment='left',
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
                    ax.text(transition_temp, y_data[-1], f' After: {y_data[-1]:.2e}',
                            verticalalignment='top', horizontalalignment='left',
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
                    # Set y_value for boundary annotations
                    _y_value = np.mean(y_data)
                    logger.debug("Plotted step function '%s' with transition at %.1f K",
                                 prop_name, transition_temp)
                else:  # Fallback for step function without data
                    try:
                        f_current = sp.lambdify(T, current_prop, 'numpy')
                        _y_value = f_current(lower_bound)
                    except Exception as e:
                        logger.warning("Could not evaluate step function '%s': %s", prop_name, e)
                        _y_value = 0.0
            else:  # Handle all other property types (FILE, KEY_VAL, PIECEWISE_EQUATION, COMPUTE)
                try:
                    f_current = sp.lambdify(T, current_prop, 'numpy')
                    # Determine the appropriate label and color based on regression status
                    if has_regression and simplify_type == PRE_KEY:
                        main_color = colors['regression_pre']
                        main_label = 'regression (pre)'
                    else:
                        main_color = colors['extended']
                        main_label = 'raw (extended)'
                    try:  # Plot the main function over extended range
                        y_extended = f_current(extended_temp)
                        ax.plot(extended_temp, y_extended, color=main_color,
                                linestyle='-', linewidth=2.5, label=main_label, zorder=2)
                        logger.debug("Plotted extended range for property '%s'", prop_name)
                    except Exception as e:
                        logger.warning("Could not evaluate function over extended range for '%s': %s",
                                       prop_name, e)
                        # Fallback to data range if available
                        if x_data is not None and y_data is not None:
                            ax.plot(x_data, y_data, color=colors['raw'],
                                    linestyle='-', linewidth=2, label='data points', zorder=2)
                    # Plot data points if available (for FILE, KEY_VAL properties)
                    if x_data is not None and y_data is not None and prop_type in ['FILE', 'KEY_VAL']:
                        # marker_size = 6 if prop_type == 'KEY_VAL' else 3
                        # ax.scatter(x_data, y_data, color=colors['raw'], marker='o', s=marker_size**2,
                        #            alpha=0.7, label='data points', zorder=3)
                        pass
                    # Set y_value for boundary annotations
                    if y_data is not None and len(y_data) > 0:
                        _y_value = np.max(y_data)
                    else:
                        try:
                            _y_value = f_current(upper_bound)
                        except (ValueError, TypeError, AttributeError) as e:
                            logger.error("Could not evaluate function at boundary for '%s': %s",
                                         prop_name, e)
                            _y_value = 0.0
                    # Overlay post-regression preview if requested
                    if has_regression and simplify_type == POST_KEY and x_data is not None and y_data is not None:
                        try:
                            preview_pw = RegressionProcessor.process_regression(
                                temp_array=x_data, prop_array=y_data, T=T,
                                lower_bound_type=lower_bound_type, upper_bound_type=upper_bound_type,
                                degree=degree, segments=segments, seed=ProcessingConstants.DEFAULT_REGRESSION_SEED
                            )
                            f_preview = sp.lambdify(T, preview_pw, 'numpy')
                            y_preview = f_preview(extended_temp)
                            ax.plot(extended_temp, y_preview, color=colors['regression_post'],
                                    linestyle='--', linewidth=2, label='regression (post)', zorder=4)
                            logger.debug("Added post-regression preview for property '%s'", prop_name)
                        except Exception as e:
                            logger.warning("Could not generate post-regression preview for '%s': %s",
                                           prop_name, e)
                except Exception as e:
                    logger.error("Error creating function for property '%s': %s", prop_name, e)
                    ax.text(0.5, 0.5, f"Error: {str(e)}", transform=ax.transAxes,
                            horizontalalignment='center', bbox=dict(facecolor='red', alpha=0.2))
                    _y_value = 0.0
            # Add boundary lines and annotations
            ax.axvline(x=lower_bound, color=colors['bounds'], linestyle='--', alpha=0.7,
                       linewidth=1.5, label='_nolegend_')
            ax.axvline(x=upper_bound, color=colors['bounds'], linestyle='--', alpha=0.7,
                       linewidth=1.5, label='_nolegend_')
            # Ensure _y_value is valid for annotations
            if _y_value is None or not np.isfinite(_y_value):
                try:
                    if hasattr(current_prop, 'subs') and hasattr(current_prop, 'evalf'):
                        _y_value = float(current_prop.subs(T, lower_bound).evalf())
                    else:
                        _y_value = float(current_prop) if hasattr(current_prop, '__float__') else 0.0
                except (ValueError, TypeError, AttributeError):
                    _y_value = 0.0
                    logger.warning("Could not determine y_value for annotations for property '%s'", prop_name)
            # Add boundary type annotations
            ax.text(lower_bound, _y_value, f' {lower_bound_type}',
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
            ax.text(upper_bound, _y_value, f' {upper_bound_type}',
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
            # Add regression info if applicable
            if has_regression and degree is not None:
                ax.text(0.5, 0.95, f"Simplify: {simplify_type} | Degree: {degree} | Segments: {segments}",
                        transform=ax.transAxes, horizontalalignment='center',
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
            # Check if there are any labeled artists before creating legend
            handles, labels = ax.get_legend_handles_labels()
            if handles:  # Only create legend if there are plot elements
                ax.legend(loc='best', framealpha=0.8, fancybox=True, shadow=True)
            # Add legend
            # ax.legend(loc='best', framealpha=0.8, fancybox=True, shadow=True)
            # Add property to visualized set
            self.visualized_properties.add(prop_name)
            logger.info("Successfully visualized property: %s", prop_name)
        except Exception as e:
            logger.error("Unexpected error visualizing property '%s': %s", prop_name, e, exc_info=True)
            raise ValueError(f"Unexpected error in property {prop_name}: {e}")

    def save_property_plots(self) -> None:
        """Save plots only if visualization is enabled and plots exist."""
        if not self.is_enabled or not hasattr(self, 'fig') or self.fig is None:
            logger.debug("No plots to save - visualization disabled or no plots created")
            return
        try:
            if hasattr(self, 'fig') and self.fig is not None:
                material_type = self.parser.config[MATERIAL_TYPE_KEY]
                title = f"Material Properties: {self.parser.config[NAME_KEY]} ({material_type})"
                self.fig.suptitle(title, fontsize=16, fontweight='bold')
                try:
                    plt.tight_layout(rect=[0, 0.01, 1, 0.98], pad=1.0)
                except Exception as e:
                    logger.warning("tight_layout failed: %s. Using subplots_adjust as fallback", e)
                    plt.subplots_adjust(
                        left=0.1,  # Left margin
                        bottom=0.1,  # Bottom margin
                        right=0.9,  # Right margin
                        top=0.9,  # Top margin (leave space for subtitle)
                        hspace=0.4  # Height spacing between subplots
                    )
                filename = f"{self.parser.config[NAME_KEY].replace(' ', '_')}_properties.png"
                filepath = self.plot_directory / filename
                self.fig.savefig(
                    str(filepath),
                    dpi=300,
                    bbox_inches="tight",
                    facecolor='white',
                    edgecolor='none'
                )
                total_properties = sum(len(props) for props in self.parser.categorized_properties.values())
                visualized_count = len(self.visualized_properties)
                if visualized_count != total_properties:
                    logger.warning(
                        "Not all properties visualized! Visualized: %d, Total: %d",
                        visualized_count, total_properties
                    )
                else:
                    logger.info("All properties (%d) visualized successfully", total_properties)
                logger.info("Property plots saved to: %s", filepath)
        finally:  # Always close the figure to prevent memory leaks
            if hasattr(self, 'fig') and self.fig is not None:
                plt.close(self.fig)
                self.fig = None
                logger.debug("Figure closed and memory cleaned up")
