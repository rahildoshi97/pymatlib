import logging
from pathlib import Path
from typing import Union, Any
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import yaml
from datetime import datetime
import warnings

from pymatlib.core.materials import Material
from pymatlib.visualization.plotters import PropertyVisualizer
from pymatlib.parsing.config.yaml_keys import (
    CONSTANT_KEY, PRE_KEY, POST_KEY, NAME_KEY, MATERIAL_TYPE_KEY,
    REGRESSION_KEY, SIMPLIFY_KEY, EQUATION_KEY, BOUNDS_KEY, TEMPERATURE_KEY, VALUE_KEY
)
from pymatlib.data.constants import PhysicalConstants, ProcessingConstants

logger = logging.getLogger(__name__)

class PDFReporter:
    """Generates comprehensive PDF reports for material properties with formatting."""

    # Layout constants
    TEXT_FONT_SIZE = 8  # Smaller font to fit more content
    TITLE_FONT_SIZE = 12
    SECTION_TITLE_FONT_SIZE = 9
    MIN_PLOT_HEIGHT = 0.35  # Minimum plot height to ensure visibility
    MAX_TEXT_HEIGHT = 0.25  # Maximum height for any text section
    SECTION_GAP = 0.04  # Gap between sections

    # Color scheme
    COLORS = {
        'config_bg': '#E8F4FD',
        'expression_bg': '#E8F8E8',
        'title_color': '#2C3E50',
        'section_color': '#34495E',
        'border_color': '#BDC3C7'
    }

    def __init__(self, parser):
        self.parser = parser
        self.yaml_dir = self.parser.base_dir
        self.pdf_directory = self.yaml_dir / "pymatlib_reports"
        self.pdf_directory.mkdir(exist_ok=True)
        self.material = None
        self.T_symbol = None

    def generate_pdf_report(self, material: Material, T: Union[float, sp.Symbol]) -> Path:
        """Generate comprehensive PDF report for material properties."""
        self.material = material
        self.T_symbol = T

        material_name = self.parser.config[NAME_KEY].replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{material_name}_report_{timestamp}.pdf"
        filepath = self.pdf_directory / filename

        logger.info(f"Generating PDF report: {filepath}")

        # Set matplotlib parameters
        original_rcParams = plt.rcParams.copy()
        plt.rcParams.update({
            'font.size': self.TEXT_FONT_SIZE,
            'font.family': 'serif',
            'axes.titlesize': self.SECTION_TITLE_FONT_SIZE,
            'axes.labelsize': self.TEXT_FONT_SIZE,
            'figure.titlesize': self.TITLE_FONT_SIZE,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'figure.autolayout': False,
            'figure.constrained_layout.use': False
        })

        with PdfPages(str(filepath)) as pdf:
            # Title page
            self._create_title_page(pdf)
            # YAML configuration pages
            self._create_yaml_config_pages(pdf)
            # Property pages
            self._create_property_pages(pdf)

        # Reset matplotlib parameters
        plt.rcParams.update(original_rcParams)

        logger.info(f"PDF report generated successfully: {filepath}")
        return filepath

    @staticmethod
    def _save_figure_safely(pdf: PdfPages, fig) -> None:
        """Save figure to PDF without layout warnings."""
        try:
            # fig.set_tight_layout(False)
            # Check if set_tight_layout method exists
            if hasattr(fig, 'set_tight_layout'):
                fig.set_tight_layout(False)

            # Alternative approach for older matplotlib versions
            # Disable constrained layout if it exists
            if hasattr(fig, 'set_constrained_layout'):
                fig.set_constrained_layout(False)
            pdf.savefig(fig, dpi=150, facecolor='white', edgecolor='none')
        except Exception as e:
            logger.warning(f"Standard save failed, trying fallback: {e}")
            try:
                pdf.savefig(fig)
            except Exception as e2:
                logger.error(f"All save methods failed: {e2}")
                raise
        finally:
            plt.close(fig)

    def _create_title_page(self, pdf: PdfPages) -> None:
        """Create title page."""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')

        # Title and material info
        ax.text(0.5, 0.85, "PyMatLib Material Report",
                transform=ax.transAxes, fontsize=20, fontweight='bold',
                ha='center', color=self.COLORS['title_color'])

        material_name = self.parser.config[NAME_KEY]
        material_type = self.parser.config[MATERIAL_TYPE_KEY]

        ax.text(0.5, 0.75, f"{material_name}", transform=ax.transAxes,
                fontsize=16, fontweight='bold', ha='center')
        ax.text(0.5, 0.72, f"({material_type.replace('_', ' ').title()})",
                transform=ax.transAxes, fontsize=12, ha='center')

        # Composition
        comp_text = " | ".join([f"{elem}: {frac:.4f}"
                                for elem, frac in self.parser.config['composition'].items()])
        ax.text(0.5, 0.65, "Composition", transform=ax.transAxes,
                fontsize=14, fontweight='bold', ha='center')
        ax.text(0.5, 0.62, comp_text, transform=ax.transAxes,
                fontsize=11, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.COLORS['config_bg']))

        # Temperature properties
        temp_info = self._get_temperature_info(material_type)
        ax.text(0.5, 0.50, "Temperature Properties", transform=ax.transAxes,
                fontsize=14, fontweight='bold', ha='center')
        ax.text(0.5, 0.45, temp_info, transform=ax.transAxes,
                fontsize=9, ha='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor=self.COLORS['expression_bg']))

        # Summary
        total_props = sum(len(props) for props in self.parser.categorized_properties.values())
        summary_y = 0.38 if material_type == 'pure_metal' else 0.32
        ax.text(0.5, summary_y, f"Total Properties: {total_props}",
                transform=ax.transAxes, fontsize=12, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                          edgecolor=self.COLORS['border_color']))

        self._save_figure_safely(pdf, fig)

    def _get_temperature_info(self, material_type: str) -> str:
        """Get formatted temperature information."""
        if material_type == 'pure_metal':
            melting = float(self.material.melting_temperature)
            boiling = float(self.material.boiling_temperature)
            return f"Melting: {melting:.1f} K\nBoiling: {boiling:.1f} K"
        else:
            solidus = float(self.material.solidus_temperature)
            liquidus = float(self.material.liquidus_temperature)
            initial_boil = float(self.material.initial_boiling_temperature)
            final_boil = float(self.material.final_boiling_temperature)
            return (f"Solidus: {solidus:.1f} K | Liquidus: {liquidus:.1f} K\n"
                    f"Initial Boiling: {initial_boil:.1f} K | Final Boiling: {final_boil:.1f} K")

    def _create_yaml_config_pages(self, pdf: PdfPages) -> None:
        """Create YAML configuration pages."""
        yaml_content = yaml.dump(self.parser.config, default_flow_style=False, indent=2)
        lines = yaml_content.split('\n')
        lines_per_page = 65
        page_num = 1
        start_line = 0

        while start_line < len(lines):
            end_line = min(start_line + lines_per_page, len(lines))
            page_content = '\n'.join(lines[start_line:end_line])

            fig, ax = plt.subplots(figsize=(8.5, 11))
            title = f"Material Configuration (Page {page_num})" if len(lines) > lines_per_page else "Material Configuration"
            fig.suptitle(title, fontsize=16, fontweight='bold', color=self.COLORS['title_color'])
            ax.axis('off')

            ax.text(0.05, 0.95, page_content, transform=ax.transAxes, fontsize=8,
                    va='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=self.COLORS['border_color']))

            self._save_figure_safely(pdf, fig)
            start_line = end_line
            page_num += 1

    def _create_property_pages(self, pdf: PdfPages) -> None:
        """Create individual pages for each property with smart layout decisions."""
        for prop_type, prop_list in self.parser.categorized_properties.items():
            for prop_name, prop_config in prop_list:
                self._create_single_property_page(pdf, prop_name, prop_config, prop_type)

    def _create_single_property_page(self, pdf: PdfPages, prop_name: str,
                                     prop_config: Any, prop_type) -> None:
        """Create a single property page with intelligent layout based on content size."""
        # Get text content first
        config_text = self._get_config_text(prop_name, prop_config, prop_type)
        expr_text = self._get_expression_text(prop_name)

        # Estimate content requirements more accurately
        config_lines = self._estimate_text_lines(config_text)
        expr_lines = self._estimate_text_lines(expr_text)

        # Make layout decision based on content size
        if config_lines > 20 or expr_lines > 15:
            # Very long content - use multipage layout
            self._create_multi_page_property(pdf, prop_name, prop_config, prop_type, config_text, expr_text)
        elif config_lines + expr_lines > 25:
            # Combined content too long - split across pages
            self._create_split_layout_property(pdf, prop_name, prop_config, prop_type, config_text, expr_text)
        else:
            # Try single page with dynamic sizing
            self._create_single_page_with_smart_sizing(pdf, prop_name, prop_config, prop_type, config_text, expr_text)

    def _create_single_page_with_smart_sizing(self, pdf: PdfPages, prop_name: str,
                                              prop_config: Any, prop_type,
                                              config_text: str, expr_text: str) -> None:
        """Create single page with smart dynamic sizing to prevent overlaps."""
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle(f"Property Analysis: {prop_name}", fontsize=14, fontweight='bold',
                     color=self.COLORS['title_color'], y=0.95)

        # Calculate required heights based on content
        config_lines = self._estimate_text_lines(config_text)
        expr_lines = self._estimate_text_lines(expr_text)

        # Calculate heights with safety margins
        config_height = min(self.MAX_TEXT_HEIGHT, max(0.08, config_lines * 0.012))
        expr_height = min(self.MAX_TEXT_HEIGHT, max(0.08, expr_lines * 0.012))

        # Available space for content (leaving margins and title space)
        available_height = 0.80  # From y=0.08 to y=0.88
        total_gaps = 2 * self.SECTION_GAP  # Gaps between sections

        # Check if content fits with minimum plot height
        required_text_height = config_height + expr_height + total_gaps
        remaining_for_plot = available_height - required_text_height

        if remaining_for_plot < self.MIN_PLOT_HEIGHT:
            # Not enough space - use split layout
            self._create_split_layout_property(pdf, prop_name, prop_config, prop_type, config_text, expr_text)
            return

        plot_height = remaining_for_plot

        # Calculate positions from bottom up to ensure no overlaps
        plot_bottom = 0.08
        plot_top = plot_bottom + plot_height

        expr_bottom = plot_top + self.SECTION_GAP
        expr_top = expr_bottom + expr_height

        config_bottom = expr_top + self.SECTION_GAP
        config_top = config_bottom + config_height

        # Ensure we don't exceed page bounds
        if config_top > 0.88:
            # Still too tall - force split layout
            self._create_split_layout_property(pdf, prop_name, prop_config, prop_type, config_text, expr_text)
            return

        # Create sections with calculated positions
        ax_config = fig.add_axes([0.08, config_bottom, 0.84, config_height])
        ax_config.axis('off')
        self._add_config_section_manual(ax_config, config_text, prop_type)

        ax_expr = fig.add_axes([0.08, expr_bottom, 0.84, expr_height])
        ax_expr.axis('off')
        self._add_expression_section_manual(ax_expr, expr_text)

        ax_plot = fig.add_axes([0.1, plot_bottom, 0.8, plot_height])
        self._create_plot_using_exact_plotters_logic(ax_plot, prop_name, prop_config, prop_type)

        self._save_figure_safely(pdf, fig)

    def _create_split_layout_property(self, pdf: PdfPages, prop_name: str,
                                      prop_config: Any, prop_type,
                                      config_text: str, expr_text: str) -> None:
        """Create property with config+expression on one page, plot on next."""
        # Page 1: Configuration + Expression with guaranteed separation
        fig1 = plt.figure(figsize=(8.5, 11))
        fig1.suptitle(f"Property Analysis: {prop_name} - Configuration & Expression",
                      fontsize=14, fontweight='bold', color=self.COLORS['title_color'], y=0.95)

        # Use half page for each section with clear separation
        ax_config = fig1.add_axes([0.08, 0.55, 0.84, 0.30])  # Top half
        ax_config.axis('off')
        self._add_config_section_manual(ax_config, config_text, prop_type)

        ax_expr = fig1.add_axes([0.08, 0.15, 0.84, 0.30])  # Bottom half
        ax_expr.axis('off')
        self._add_expression_section_manual(ax_expr, expr_text)

        self._save_figure_safely(pdf, fig1)

        # Page 2: Plot only
        self._create_plot_only_page(pdf, prop_name, prop_config, prop_type)

    def _create_multi_page_property(self, pdf: PdfPages, prop_name: str,
                                    prop_config: Any, prop_type,
                                    config_text: str, expr_text: str) -> None:
        """Create property across multiple pages for very long content."""
        # Page 1: Configuration only
        fig1 = plt.figure(figsize=(8.5, 11))
        fig1.suptitle(f"Property Analysis: {prop_name} - Configuration",
                      fontsize=14, fontweight='bold', color=self.COLORS['title_color'], y=0.95)

        ax1 = fig1.add_subplot(111)
        ax1.axis('off')

        ax1.text(0.05, 0.85, "Configuration Details", fontsize=self.SECTION_TITLE_FONT_SIZE,
                 fontweight='bold', color=self.COLORS['section_color'], transform=ax1.transAxes)
        ax1.text(0.05, 0.78, config_text, transform=ax1.transAxes, fontsize=self.TEXT_FONT_SIZE,
                 va='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=self.COLORS['config_bg'],
                           edgecolor=self.COLORS['border_color']))

        self._save_figure_safely(pdf, fig1)

        # Page 2: Expression only
        fig2 = plt.figure(figsize=(8.5, 11))
        fig2.suptitle(f"Property Analysis: {prop_name} - Expression",
                      fontsize=14, fontweight='bold', color=self.COLORS['title_color'], y=0.95)

        ax2 = fig2.add_subplot(111)
        ax2.axis('off')

        ax2.text(0.05, 0.85, "Final Expression", fontsize=self.SECTION_TITLE_FONT_SIZE,
                 fontweight='bold', color=self.COLORS['section_color'], transform=ax2.transAxes)
        ax2.text(0.05, 0.78, expr_text, transform=ax2.transAxes, fontsize=self.TEXT_FONT_SIZE,
                 va='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=self.COLORS['expression_bg'],
                           edgecolor=self.COLORS['border_color']))

        self._save_figure_safely(pdf, fig2)

        # Page 3: Plot only
        self._create_plot_only_page(pdf, prop_name, prop_config, prop_type)

    def _create_plot_only_page(self, pdf: PdfPages, prop_name: str,
                               prop_config: Any, prop_type) -> None:
        """Create a plot-only page with optimal aspect ratio."""
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle(f"Property Visualization: {prop_name}", fontsize=14, fontweight='bold',
                     color=self.COLORS['title_color'], y=0.95)

        # Create subplot with positioning and PROPER aspect ratio
        ax = fig.add_axes([0.12, 0.25, 0.76, 0.55])

        # Apply EXACT same styling as plotters.py - for consistent appearance
        ax.set_aspect('auto')  # Use auto aspect ratio like plotters.py
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Apply same border styling as plotters.py
        for spine in ax.spines.values():
            spine.set_color('#CCCCCC')
            spine.set_linewidth(1.2)

        self._create_plot_using_exact_plotters_logic(ax, prop_name, prop_config, prop_type)

        self._save_figure_safely(pdf, fig)

    @staticmethod
    def _estimate_text_lines(text: str) -> int:
        """Estimate the number of display lines needed for text content."""
        lines = text.split('\n')
        total_lines = 0

        for line in lines:
            if not line.strip():
                total_lines += 0.5  # Empty lines
            else:
                # Account for line wrapping (assuming ~70 characters per line)
                line_length = len(line)
                wrapped_lines = max(1, (line_length + 69) // 70)
                total_lines += wrapped_lines

        return int(total_lines * 1.3)  # Add 30% safety margin

    @staticmethod
    def _get_config_text(prop_name: str, prop_config: Any, prop_type) -> str:
        """Get formatted configuration text."""
        config_text = f"Property Type: {prop_type.name}\n"
        config_text += f"Property Name: {prop_name}\n\n"

        if prop_type.name == 'CONSTANT':
            config_text += f"Value: {prop_config}"
        else:
            # Limit YAML output length and width
            config_yaml = yaml.dump({prop_name: prop_config}, default_flow_style=False,
                                    indent=2, width=60)
            # Truncate very long configs
            if len(config_yaml) > 1000:
                config_yaml = config_yaml[:997] + "..."
            config_text += config_yaml

        return config_text

    def _get_expression_text(self, prop_name: str) -> str:
        """Get formatted expression text."""
        if hasattr(self.material, prop_name):
            prop_value = getattr(self.material, prop_name)
            if isinstance(prop_value, (sp.Expr, sp.Piecewise)):
                expr_text = f"Expression Type: {type(prop_value).__name__}\n"
                expr_text += f"Temperature Symbol: {self.T_symbol}\n\n"
                if isinstance(prop_value, sp.Piecewise):
                    expr_text += "Piecewise Function Definition:\n"
                    for i, (expr, condition) in enumerate(prop_value.args):
                        if i >= 5:  # Limit to first 5 segments
                            expr_text += f" ... ({len(prop_value.args) - 5} more segments)\n"
                            break
                        expr_text += f" Segment {i+1}:\n"
                        # Truncate long expressions
                        expr_str = str(expr)
                        if len(expr_str) > 60:
                            expr_str = expr_str[:57] + "..."
                        expr_text += f" Expression: {expr_str}\n"
                        condition_str = str(condition)
                        if len(condition_str) > 60:
                            condition_str = condition_str[:57] + "..."
                        expr_text += f" Condition: {condition_str}\n"
                else:
                    # Truncate very long expressions
                    expr_str = str(prop_value)
                    if len(expr_str) > 150:
                        expr_str = expr_str[:147] + "..."
                    expr_text += f"Mathematical Expression:\n{expr_str}"
            else:
                expr_text = f"Value: {prop_value}\n"
                expr_text += f"Type: {type(prop_value).__name__}\n"
                expr_text += f"Temperature Symbol: {self.T_symbol}"
        else:
            expr_text = "Expression not available for this property."

        return expr_text

    def _add_config_section_manual(self, ax, config_text: str, prop_type) -> None:
        """Add formatted configuration section."""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Section title
        ax.text(0.02, 0.95, "Configuration", fontsize=self.SECTION_TITLE_FONT_SIZE,
                fontweight='bold', color=self.COLORS['section_color'], transform=ax.transAxes)

        # Property type badge
        type_colors = {
            'CONSTANT': '#3498DB', 'STEP_FUNCTION': '#E74C3C', 'FILE': '#F39C12',
            'KEY_VAL': '#9B59B6', 'PIECEWISE_EQUATION': '#1ABC9C', 'COMPUTE': '#2ECC71'
        }
        type_color = type_colors.get(prop_type.name, '#95A5A6')

        ax.text(0.98, 0.95, prop_type.name, fontsize=self.TEXT_FONT_SIZE - 1,
                fontweight='bold', color='white', ha='right', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=type_color))

        # Configuration content
        ax.text(0.02, 0.85, config_text, transform=ax.transAxes, fontsize=self.TEXT_FONT_SIZE,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.COLORS['config_bg'],
                          edgecolor=self.COLORS['border_color']))

    def _add_expression_section_manual(self, ax, expr_text: str) -> None:
        """Add formatted expression section."""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Section title
        ax.text(0.02, 0.95, "Final Expression", fontsize=self.SECTION_TITLE_FONT_SIZE,
                fontweight='bold', color=self.COLORS['section_color'], transform=ax.transAxes)

        # Expression content
        ax.text(0.02, 0.85, expr_text, transform=ax.transAxes, fontsize=self.TEXT_FONT_SIZE,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.COLORS['expression_bg'],
                          edgecolor=self.COLORS['border_color']))

    def _create_plot_using_exact_plotters_logic(self, ax, prop_name: str, prop_config: Any, prop_type) -> None:
        """
        Create property plot using EXACTLY the same logic as PropertyVisualizer.visualize_property().
        This ensures 100% consistency with plotters.py while maintaining formatting.
        """
        try:
            # Extract the plotting parameters - Get all parameters as a dict instead of unpacking
            plot_params = self._extract_plotting_parameters(prop_name, prop_config, prop_type)

            # Apply plot styling that matches plotters.py
            ax.set_aspect('auto')  # Use auto aspect ratio like plotters.py
            ax.grid(True, alpha=0.3, linestyle='--')  # Match plotters.py grid style
            ax.set_axisbelow(True)  # Match plotters.py layering

            # Apply border styling
            for spine in ax.spines.values():
                spine.set_color('#CCCCCC')  # Match plotters.py border color
                spine.set_linewidth(1.2)    # Match plotters.py border width

            # Create temporary PropertyVisualizer setup
            temp_visualizer = PropertyVisualizer(self.parser)
            temp_fig = plt.figure(figsize=(1, 1))
            temp_gs = GridSpec(1, 1, figure=temp_fig)

            # Set up visualizer state
            temp_visualizer.fig = temp_fig
            temp_visualizer.gs = temp_gs
            temp_visualizer.current_subplot = 0
            temp_visualizer.is_enabled = True

            # Override add_subplot to use our axis
            original_add_subplot = temp_fig.add_subplot
            temp_fig.add_subplot = lambda *args, **kwargs: ax

            # Suppress legend warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="No artists with labels found")

                # Call visualization method with the extracted parameters
                temp_visualizer.visualize_property(
                    material=self.material,
                    prop_name=prop_name,
                    T=self.T_symbol,
                    **plot_params  # Unpack the dictionary here
                )

            # Check if legend was created, if not add property info
            handles, labels = ax.get_legend_handles_labels()
            if not handles:
                ax.text(0.02, 0.98, f"Property: {prop_name}", transform=ax.transAxes,
                        fontsize=8, va='top', ha='left',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))

            # Restore and cleanup
            temp_fig.add_subplot = original_add_subplot
            plt.close(temp_fig)

        except Exception as e:
            logger.warning(f"Could not create plot for {prop_name}: {e}")
            ax.text(0.5, 0.5, f"Plot Error: {str(e)}", ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.2))

    def _extract_plotting_parameters(self, prop_name: str, prop_config: Any, prop_type) -> dict:
        """
        Extract plotting parameters using the same logic as the property processors.
        This ensures we pass the exact same parameters to the visualizer as the main processing does.
        """
        # Import the necessary modules to extract parameters the same way
        from pymatlib.parsing.processors.temperature_resolver import TemperatureResolver
        from pymatlib.parsing.utils.utilities import create_step_visualization_data
        from pymatlib.algorithms.regression_processor import RegressionProcessor

        # Initialize default values - return as dictionary to match PropertyVisualizer.visualize_property signature
        params = {
            'prop_type': prop_type.name,
            'x_data': None,
            'y_data': None,
            'has_regression': False,
            'simplify_type': None,
            'degree': 1,
            'segments': 1,
            'lower_bound': None,
            'upper_bound': None,
            'lower_bound_type': CONSTANT_KEY,
            'upper_bound_type': CONSTANT_KEY
        }

        if isinstance(prop_config, dict):
            # Extract bounds exactly like the property processors do
            if BOUNDS_KEY in prop_config:
                bounds_config = prop_config[BOUNDS_KEY]
                if isinstance(bounds_config, list) and len(bounds_config) == 2:
                    params['lower_bound_type'], params['upper_bound_type'] = bounds_config

            # Extract regression parameters exactly like the property processors do
            if REGRESSION_KEY in prop_config:
                has_regression, simplify_type, degree, segments = RegressionProcessor.process_regression_params(
                    prop_config, prop_name, 100  # Default length for parameter extraction
                )
                params.update({
                    'has_regression': has_regression,
                    'simplify_type': simplify_type,
                    'degree': degree,
                    'segments': segments
                })

            # Extract temperature data exactly like each property processor does
            try:
                if prop_type.name == 'STEP_FUNCTION':
                    # Use the exact same logic as StepFunctionPropertyHandler
                    temp_key = prop_config[TEMPERATURE_KEY]
                    val_array = prop_config[VALUE_KEY]
                    transition_temp = TemperatureResolver.resolve_temperature_reference(temp_key, self.material)

                    # Create step visualization data exactly like the handler does
                    offset = ProcessingConstants.STEP_FUNCTION_OFFSET
                    val1 = max(transition_temp - offset, PhysicalConstants.ABSOLUTE_ZERO)
                    val2 = transition_temp + offset
                    step_temp_array = np.array([val1, transition_temp, val2])

                    x_data, y_data = create_step_visualization_data(transition_temp, val_array, step_temp_array)
                    params.update({
                        'x_data': x_data,
                        'y_data': y_data,
                        'lower_bound': np.min(x_data),
                        'upper_bound': np.max(x_data)
                    })

                elif TEMPERATURE_KEY in prop_config:
                    temp_def = prop_config[TEMPERATURE_KEY]
                    if prop_type.name == 'KEY_VAL':
                        # Use the exact same logic as KeyValPropertyHandler
                        val_array = prop_config.get(VALUE_KEY, [])
                        temp_array = TemperatureResolver.resolve_temperature_definition(
                            temp_def, len(val_array), self.material)
                        params.update({
                            'x_data': temp_array,
                            'y_data': np.array(val_array),
                            'lower_bound': np.min(temp_array),
                            'upper_bound': np.max(temp_array)
                        })

                    elif prop_type.name in ['COMPUTE', 'PIECEWISE_EQUATION']:
                        # Use the exact same logic as the respective handlers
                        temp_array = TemperatureResolver.resolve_temperature_definition(
                            temp_def, material=self.material)
                        params.update({
                            'x_data': temp_array,
                            'lower_bound': np.min(temp_array),
                            'upper_bound': np.max(temp_array)
                        })

                        # For COMPUTE properties, we need to evaluate the property to get y_data
                        if prop_type.name == 'COMPUTE' and hasattr(self.material, prop_name):
                            current_prop = getattr(self.material, prop_name)
                            if isinstance(current_prop, (sp.Expr, sp.Piecewise)):
                                f_prop = sp.lambdify(self.T_symbol, current_prop, 'numpy')
                                params['y_data'] = f_prop(temp_array)

            except Exception as e:
                logger.warning(f"Could not extract temperature data for {prop_name}: {e}")

        return params
