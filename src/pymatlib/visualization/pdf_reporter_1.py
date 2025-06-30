import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import yaml
from datetime import datetime

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
    STANDARD_SPACING = 0.3  # Increased spacing for better readability
    TEXT_FONT_SIZE = 9  # Slightly larger for better readability
    TITLE_FONT_SIZE = 12
    SECTION_TITLE_FONT_SIZE = 10
    PAGE_MARGINS = {'left': 0.08, 'right': 0.92, 'top': 0.92, 'bottom': 0.08}

    # Color scheme
    COLORS = {
        'config_bg': '#E8F4FD',      # Light blue
        'expression_bg': '#E8F8E8',   # Light green
        'title_color': '#2C3E50',     # Dark blue-gray
        'section_color': '#34495E',   # Medium blue-gray
        'border_color': '#BDC3C7'     # Light gray
    }

    def __init__(self, parser):
        self.parser = parser
        self.yaml_dir = self.parser.base_dir
        self.pdf_directory = self.yaml_dir / "pymatlib_reports"
        self.pdf_directory.mkdir(exist_ok=True)
        self.material = None
        self.T_symbol = None

        # Create a dedicated PropertyVisualizer instance for PDF generation
        self.pdf_visualizer = PropertyVisualizer(parser)
        self.pdf_visualizer.is_enabled = False  # Disable file saving for PDF visualizer

    def generate_pdf_report(self, material: Material, T: Union[float, sp.Symbol]) -> Path:
        """Generate comprehensive PDF report for material properties."""
        self.material = material
        self.T_symbol = T

        material_name = self.parser.config[NAME_KEY].replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{material_name}_report_{timestamp}.pdf"
        filepath = self.pdf_directory / filename

        logger.info(f"Generating PDF report: {filepath}")

        # Set matplotlib parameters with better font fallbacks
        plt.rcParams.update({
            'font.size': self.TEXT_FONT_SIZE,
            'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans', 'Bitstream Vera Sans'],
            'axes.titlesize': self.SECTION_TITLE_FONT_SIZE,
            'axes.labelsize': self.TEXT_FONT_SIZE,
            'xtick.labelsize': self.TEXT_FONT_SIZE - 1,
            'ytick.labelsize': self.TEXT_FONT_SIZE - 1,
            'legend.fontsize': self.TEXT_FONT_SIZE - 1,
            'figure.titlesize': self.TITLE_FONT_SIZE,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.axisbelow': True,
            # 'figure.autolayout': False,  # Disable automatic layout
            # 'figure.constrained_layout.use': False  # Disable constrained layout
        })

        with PdfPages(str(filepath)) as pdf:
            # Title page
            self._create_title_page(pdf)

            # YAML configuration page
            self._create_yaml_config_pages(pdf)

            # Property pages (one per property)
            self._create_property_pages(pdf)

        # Reset matplotlib parameters
        plt.rcdefaults()

        logger.info(f"PDF report generated successfully: {filepath}")
        return filepath

    def _save_figure_safely(self, pdf: PdfPages, fig) -> None:
        """Save figure to PDF without any layout warnings."""
        try:
            # Disable tight_layout completely and use manual layout
            fig.set_tight_layout(False)

            # Save with minimal parameters to avoid any automatic layout adjustments
            pdf.savefig(fig, dpi=150, facecolor='white', edgecolor='none',
                        bbox_inches=None, pad_inches=0)
        except Exception as e:
            logger.warning(f"Standard save failed, trying fallback: {e}")
            try:
                # Fallback without any special parameters
                pdf.savefig(fig)
            except Exception as e2:
                logger.error(f"All save methods failed: {e2}")
                raise
        finally:
            plt.close(fig)

    def _create_title_page(self, pdf: PdfPages) -> None:
        """Create title page with material overview."""
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')

        # Main title
        ax.text(0.5, 0.85, "PyMatLib Material Report",
                transform=ax.transAxes, fontsize=20, fontweight='bold',
                horizontalalignment='center', color=self.COLORS['title_color'])

        # Subtitle line
        ax.plot([0.2, 0.8], [0.82, 0.82], transform=ax.transAxes,
                color=self.COLORS['border_color'], linewidth=2)

        # Material information
        material_type = self.parser.config[MATERIAL_TYPE_KEY]
        material_name = self.parser.config[NAME_KEY]

        # Material info box
        info_y_start = 0.75
        ax.text(0.5, info_y_start, f"{material_name}",
                transform=ax.transAxes, fontsize=16, fontweight='bold',
                horizontalalignment='center', color=self.COLORS['section_color'])

        ax.text(0.5, info_y_start - 0.04, f"({material_type.replace('_', ' ').title()})",
                transform=ax.transAxes, fontsize=12,
                horizontalalignment='center', color=self.COLORS['section_color'])

        # Generation info
        gen_info = f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}\nPyMatLib Version: 1.0.0"
        ax.text(0.5, info_y_start - 0.1, gen_info,
                transform=ax.transAxes, fontsize=10,
                horizontalalignment='center', style='italic')

        # Composition section
        comp_y_start = 0.58
        ax.text(0.5, comp_y_start, "Material Composition",
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                horizontalalignment='center', color=self.COLORS['section_color'])

        # Composition table
        comp_text = ""
        for i, (element, fraction) in enumerate(self.parser.config['composition'].items()):
            comp_text += f"{element}: {fraction:.4f}"
            if i < len(self.parser.config['composition']) - 1:
                comp_text += " | "

        ax.text(0.5, comp_y_start - 0.04, comp_text,
                transform=ax.transAxes, fontsize=11,
                horizontalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=self.COLORS['config_bg'],
                          edgecolor=self.COLORS['border_color'], alpha=0.8))

        # Temperature properties section
        temp_y_start = 0.42
        ax.text(0.5, temp_y_start, "Temperature Properties",
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                horizontalalignment='center', color=self.COLORS['section_color'])

        temp_info = ""
        if material_type == 'pure_metal':
            melting_temp = float(self.material.melting_temperature)
            boiling_temp = float(self.material.boiling_temperature)
            temp_info = f"Melting Temperature: {melting_temp:.1f} K\nBoiling Temperature: {boiling_temp:.1f} K"
        else:
            solidus_temp = float(self.material.solidus_temperature)
            liquidus_temp = float(self.material.liquidus_temperature)
            initial_boiling = float(self.material.initial_boiling_temperature)
            final_boiling = float(self.material.final_boiling_temperature)
            temp_info = (f"Solidus Temperature: {solidus_temp:.1f} K\n"
                         f"Liquidus Temperature: {liquidus_temp:.1f} K\n"
                         f"Initial Boiling Temperature: {initial_boiling:.1f} K\n"
                         f"Final Boiling Temperature: {final_boiling:.1f} K")

        ax.text(0.5, temp_y_start - 0.04, temp_info,
                transform=ax.transAxes, fontsize=11,
                horizontalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=self.COLORS['expression_bg'],
                          edgecolor=self.COLORS['border_color'], alpha=0.8))

        # Summary section
        total_properties = sum(len(props) for props in self.parser.categorized_properties.values())
        summary_y = 0.25
        ax.text(0.5, summary_y, "Report Summary",
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                horizontalalignment='center', color=self.COLORS['section_color'])

        ax.text(0.5, summary_y - 0.04, f"Total Properties Analyzed: {total_properties}",
                transform=ax.transAxes, fontsize=12,
                horizontalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                          edgecolor=self.COLORS['border_color'], alpha=0.8))

        self._save_figure_safely(pdf, fig)

    def _create_yaml_config_pages(self, pdf: PdfPages) -> None:
        """Create YAML configuration pages that can span multiple pages."""
        # Convert config to YAML string
        yaml_content = yaml.dump(self.parser.config, default_flow_style=False, indent=2)

        # Split content into manageable chunks
        lines = yaml_content.split('\n')
        lines_per_page = 55  # Conservative estimate for readability

        page_num = 1
        start_line = 0

        while start_line < len(lines):
            end_line = min(start_line + lines_per_page, len(lines))
            page_content = '\n'.join(lines[start_line:end_line])

            # Create page
            fig = plt.figure(figsize=(8.5, 11))

            # Title with page number if multiple pages
            if len(lines) > lines_per_page:
                title = f"Material Configuration (Page {page_num})"
            else:
                title = "Material Configuration"

            fig.suptitle(title, fontsize=16, fontweight='bold',
                         color=self.COLORS['title_color'], y=0.95)

            ax = fig.add_subplot(111)
            ax.axis('off')

            # Display YAML content with reduced spacing
            ax.text(0.05, 0.88, page_content, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=self.COLORS['border_color'], alpha=0.9))

            self._save_figure_safely(pdf, fig)

            start_line = end_line
            page_num += 1

    def _create_property_pages(self, pdf: PdfPages) -> None:
        """Create individual pages for each property."""
        for prop_type, prop_list in self.parser.categorized_properties.items():
            for prop_name, prop_config in prop_list:
                self._create_single_property_page(pdf, prop_name, prop_config, prop_type)

    def _create_single_property_page(self, pdf: PdfPages, prop_name: str,
                                     prop_config: Any, prop_type) -> None:
        """Create a single page for one property with layout."""
        # Get text content
        config_text = self._get_config_text(prop_name, prop_config, prop_type)
        expr_text = self._get_expression_text(prop_name)

        # Estimate content size more accurately
        config_lines = self._count_display_lines(config_text)
        expr_lines = self._count_display_lines(expr_text)

        # Layout decision with more conservative thresholds
        if config_lines > 12:  # Very conservative for config
            self._create_multi_page_property(pdf, prop_name, prop_config, prop_type,
                                             config_text, expr_text)
        elif (config_lines + expr_lines) > 18:  # Conservative combined threshold
            self._create_split_layout_property(pdf, prop_name, prop_config, prop_type,
                                               config_text, expr_text)
        else:
            self._create_single_page_property_layout(pdf, prop_name, prop_config, prop_type,
                                                     config_text, expr_text)

    def _count_display_lines(self, text: str) -> int:
        """Count lines more accurately considering text wrapping and formatting."""
        lines = text.split('\n')
        total_lines = 0
        for line in lines:
            # More conservative estimate for better formatting
            line_length = len(line)
            wrapped_lines = max(1, (line_length + 69) // 70)  # 70 chars per line for better readability
            total_lines += wrapped_lines
        return total_lines

    def _create_single_page_property_layout(self, pdf: PdfPages, prop_name: str,
                                            prop_config: Any, prop_type,
                                            config_text: str, expr_text: str) -> None:
        """Create property on a single page with better layout."""
        fig = plt.figure(figsize=(8.5, 11))

        # Title
        fig.suptitle(f"Property Analysis: {prop_name}", fontsize=14, fontweight='bold',
                     color=self.COLORS['title_color'], y=0.95)

        # Grid layout with better proportions
        gs = GridSpec(3, 1, height_ratios=[1.2, 1, 2.5], hspace=self.STANDARD_SPACING,
                      left=self.PAGE_MARGINS['left'], right=self.PAGE_MARGINS['right'],
                      top=0.88, bottom=self.PAGE_MARGINS['bottom'])

        self._add_config_section(fig, gs[0], config_text, prop_type)
        self._add_expression_section(fig, gs[1], expr_text)
        self._add_plot_section(fig, gs[2], prop_name, prop_config, prop_type)

        # Save with warning suppression
        self._save_figure_safely(pdf, fig)

    def _create_split_layout_property(self, pdf: PdfPages, prop_name: str,
                                      prop_config: Any, prop_type,
                                      config_text: str, expr_text: str) -> None:
        """Create property with config+expression on one page, plot on next."""
        # Page 1: Configuration + Expression
        fig1 = plt.figure(figsize=(8.5, 11))
        fig1.suptitle(f"Property Analysis: {prop_name} - Configuration & Expression",
                      fontsize=14, fontweight='bold', color=self.COLORS['title_color'], y=0.95)

        gs1 = GridSpec(2, 1, height_ratios=[1, 1], hspace=self.STANDARD_SPACING,
                       left=self.PAGE_MARGINS['left'], right=self.PAGE_MARGINS['right'],
                       top=0.88, bottom=self.PAGE_MARGINS['bottom'])

        self._add_config_section(fig1, gs1[0], config_text, prop_type)
        self._add_expression_section(fig1, gs1[1], expr_text)

        self._save_figure_safely(pdf, fig1)

        # Page 2: Plot only
        self._create_plot_only_page(pdf, prop_name, prop_config, prop_type)

    def _create_multi_page_property(self, pdf: PdfPages, prop_name: str,
                                    prop_config: Any, prop_type,
                                    config_text: str, expr_text: str) -> None:
        """Create property across multiple pages with formatting."""
        # Page 1: Configuration only
        fig1 = plt.figure(figsize=(8.5, 11))
        fig1.suptitle(f"Property Analysis: {prop_name} - Configuration",
                      fontsize=14, fontweight='bold', color=self.COLORS['title_color'], y=0.95)

        ax1 = fig1.add_subplot(111)
        ax1.axis('off')

        # Configuration display
        ax1.text(0.05, 0.85, "Configuration Details", fontsize=self.SECTION_TITLE_FONT_SIZE,
                 fontweight='bold', color=self.COLORS['section_color'], transform=ax1.transAxes)

        ax1.text(0.05, 0.8, config_text, transform=ax1.transAxes, fontsize=self.TEXT_FONT_SIZE,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor=self.COLORS['config_bg'],
                           edgecolor=self.COLORS['border_color'], alpha=0.8))

        self._save_figure_safely(pdf, fig1)

        # Page 2: Expression + Plot
        fig2 = plt.figure(figsize=(8.5, 11))
        fig2.suptitle(f"Property Analysis: {prop_name} - Expression & Visualization",
                      fontsize=14, fontweight='bold', color=self.COLORS['title_color'], y=0.95)

        gs = GridSpec(2, 1, height_ratios=[1, 2.5], hspace=self.STANDARD_SPACING,
                      left=self.PAGE_MARGINS['left'], right=self.PAGE_MARGINS['right'],
                      top=0.88, bottom=self.PAGE_MARGINS['bottom'])

        self._add_expression_section(fig2, gs[0], expr_text)
        self._add_plot_section(fig2, gs[1], prop_name, prop_config, prop_type)

        self._save_figure_safely(pdf, fig2)

    def _create_plot_only_page(self, pdf: PdfPages, prop_name: str,
                               prop_config: Any, prop_type) -> None:
        """Create a plot-only page with optimal aspect ratio."""
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle(f"Property Visualization: {prop_name}", fontsize=14, fontweight='bold',
                     color=self.COLORS['title_color'], y=0.95)

        # Create subplot with positioning and optimal aspect ratio
        ax = fig.add_axes([0.12, 0.25, 0.76, 0.55])  # [left, bottom, width, height]

        # Apply styling
        ax.set_aspect('auto')
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        self._create_plot_using_exact_plotters_logic(ax, prop_name, prop_config, prop_type)

        self._save_figure_safely(pdf, fig)

    def _get_config_text(self, prop_name: str, prop_config: Any, prop_type) -> str:
        """Get formatted configuration text."""
        config_text = f"Property Type: {prop_type.name}\n"
        config_text += f"Property Name: {prop_name}\n\n"

        if prop_type.name == 'CONSTANT':
            config_text += f"Value: {prop_config}"
        else:
            # Convert config to YAML format for display
            config_yaml = yaml.dump({prop_name: prop_config}, default_flow_style=False, indent=2)
            config_text += config_yaml

        return config_text

    def _get_expression_text(self, prop_name: str) -> str:
        """Get formatted expression text."""
        # Get the final property value from material
        if hasattr(self.material, prop_name):
            prop_value = getattr(self.material, prop_name)

            if isinstance(prop_value, (sp.Expr, sp.Piecewise)):
                # Format the expression nicely
                expr_text = f"Expression Type: {type(prop_value).__name__}\n"
                expr_text += f"Temperature Symbol: {self.T_symbol}\n\n"

                if isinstance(prop_value, sp.Piecewise):
                    expr_text += "Piecewise Function Definition:\n"
                    for i, (expr, condition) in enumerate(prop_value.args):
                        expr_text += f"  Segment {i+1}:\n"
                        expr_text += f"    Expression: {expr}\n"
                        expr_text += f"    Condition: {condition}\n"
                else:
                    expr_text += f"Mathematical Expression:\n{prop_value}"
            else:
                expr_text = f"Value: {prop_value}\n"
                expr_text += f"Type: {type(prop_value).__name__}\n"
                expr_text += f"Temperature Symbol: {self.T_symbol}"
        else:
            expr_text = "Expression not available for this property."

        return expr_text

    def _add_config_section(self, fig, grid_spec, config_text: str, prop_type) -> None:
        """Add formatted configuration section."""
        ax = fig.add_subplot(grid_spec)
        ax.axis('off')

        # Section title with styling
        ax.text(0.02, 0.95, "Configuration", fontsize=self.SECTION_TITLE_FONT_SIZE,
                fontweight='bold', color=self.COLORS['section_color'], transform=ax.transAxes)

        # Add property type badge
        type_color = {
            'CONSTANT': '#3498DB',
            'STEP_FUNCTION': '#E74C3C',
            'FILE': '#F39C12',
            'KEY_VAL': '#9B59B6',
            'PIECEWISE_EQUATION': '#1ABC9C',
            'COMPUTE': '#2ECC71'
        }.get(prop_type.name, '#95A5A6')

        ax.text(0.98, 0.95, prop_type.name, fontsize=self.TEXT_FONT_SIZE - 1,
                fontweight='bold', color='white', transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=type_color, alpha=0.9))

        # Configuration content
        ax.text(0.02, 0.85, config_text, transform=ax.transAxes, fontsize=self.TEXT_FONT_SIZE,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.4", facecolor=self.COLORS['config_bg'],
                          edgecolor=self.COLORS['border_color'], alpha=0.8))

    def _add_expression_section(self, fig, grid_spec, expr_text: str) -> None:
        """Add formatted expression section."""
        ax = fig.add_subplot(grid_spec)
        ax.axis('off')

        # Section title
        ax.text(0.02, 0.95, "Final Expression", fontsize=self.SECTION_TITLE_FONT_SIZE,
                fontweight='bold', color=self.COLORS['section_color'], transform=ax.transAxes)

        # Expression content
        ax.text(0.02, 0.85, expr_text, transform=ax.transAxes, fontsize=self.TEXT_FONT_SIZE,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.4", facecolor=self.COLORS['expression_bg'],
                          edgecolor=self.COLORS['border_color'], alpha=0.8))

    def _add_plot_section(self, fig, grid_spec, prop_name: str,
                          prop_config: Any, prop_type) -> None:
        """Add formatted plot section."""
        ax = fig.add_subplot(grid_spec)

        # Apply plot styling that matches plotters.py
        ax.set_aspect('auto')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add subtle border
        for spine in ax.spines.values():
            spine.set_color(self.COLORS['border_color'])
            spine.set_linewidth(1.2)

        self._create_plot_using_exact_plotters_logic(ax, prop_name, prop_config, prop_type)

    def _create_plot_using_exact_plotters_logic(self, ax, prop_name: str, prop_config: Any, prop_type) -> None:
        """
        Create property plot using EXACTLY the same logic as PropertyVisualizer.visualize_property().
        This ensures 100% consistency with plotters.py while maintaining formatting.
        """
        # Extract the exact parameters that plotters.py uses
        x_data, y_data, has_regression, simplify_type, degree, segments, lower_bound, upper_bound, lower_bound_type, upper_bound_type = self._extract_plotting_parameters(prop_name, prop_config, prop_type)

        # Call the EXACT same method from PropertyVisualizer but on our axis
        original_fig = getattr(self.pdf_visualizer, 'fig', None)
        original_gs = getattr(self.pdf_visualizer, 'gs', None)
        original_current_subplot = getattr(self.pdf_visualizer, 'current_subplot', 0)

        try:
            # Create a temporary figure structure that matches PropertyVisualizer expectations
            temp_fig = plt.figure(figsize=(1, 1))  # Temporary figure
            temp_gs = GridSpec(1, 1, figure=temp_fig)

            # Set up the visualizer to use our provided axis instead of creating its own
            self.pdf_visualizer.fig = temp_fig
            self.pdf_visualizer.gs = temp_gs
            self.pdf_visualizer.current_subplot = 0

            # Monkey patch the add_subplot method to return our axis
            original_add_subplot = temp_fig.add_subplot
            temp_fig.add_subplot = lambda *args, **kwargs: ax

            # Now call the exact same visualization method from plotters.py
            self.pdf_visualizer.visualize_property(
                material=self.material,
                prop_name=prop_name,
                T=self.T_symbol,
                prop_type=prop_type.name,
                x_data=x_data,
                y_data=y_data,
                has_regression=has_regression,
                simplify_type=simplify_type,
                degree=degree,
                segments=segments,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                lower_bound_type=lower_bound_type,
                upper_bound_type=upper_bound_type
            )

            # Restore the original add_subplot method
            temp_fig.add_subplot = original_add_subplot

        finally:
            # Restore original PropertyVisualizer state
            self.pdf_visualizer.fig = original_fig
            self.pdf_visualizer.gs = original_gs
            self.pdf_visualizer.current_subplot = original_current_subplot

            # Close the temporary figure
            plt.close(temp_fig)

    def _extract_plotting_parameters(self, prop_name: str, prop_config: Any, prop_type):
        """
        Extract plotting parameters using the same logic as the property processors.
        This ensures we pass the exact same parameters to the visualizer as the main processing does.
        """
        # Import the necessary modules to extract parameters the same way
        from pymatlib.parsing.processors.temperature_resolver import TemperatureResolver
        from pymatlib.parsing.utils.utilities import create_step_visualization_data
        from pymatlib.algorithms.regression_processor import RegressionProcessor

        # Initialize default values
        x_data = None
        y_data = None
        has_regression = False
        simplify_type = None
        degree = 1
        segments = 1
        lower_bound = None
        upper_bound = None
        lower_bound_type = CONSTANT_KEY
        upper_bound_type = CONSTANT_KEY

        if isinstance(prop_config, dict):
            # Extract bounds exactly like the property processors do
            if BOUNDS_KEY in prop_config:
                bounds_config = prop_config[BOUNDS_KEY]
                if isinstance(bounds_config, list) and len(bounds_config) == 2:
                    lower_bound_type, upper_bound_type = bounds_config

            # Extract regression parameters exactly like the property processors do
            if REGRESSION_KEY in prop_config:
                has_regression, simplify_type, degree, segments = RegressionProcessor.process_regression_params(
                    prop_config, prop_name, 100  # Default length for parameter extraction
                )

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
                    lower_bound = np.min(x_data)
                    upper_bound = np.max(x_data)

                elif TEMPERATURE_KEY in prop_config:
                    temp_def = prop_config[TEMPERATURE_KEY]
                    if prop_type.name == 'KEY_VAL':
                        # Use the exact same logic as KeyValPropertyHandler
                        val_array = prop_config.get(VALUE_KEY, [])
                        temp_array = TemperatureResolver.resolve_temperature_definition(
                            temp_def, len(val_array), self.material)
                        x_data = temp_array
                        y_data = np.array(val_array)
                        lower_bound = np.min(temp_array)
                        upper_bound = np.max(temp_array)
                    elif prop_type.name in ['COMPUTE', 'PIECEWISE_EQUATION']:
                        # Use the exact same logic as the respective handlers
                        temp_array = TemperatureResolver.resolve_temperature_definition(
                            temp_def, material=self.material)
                        x_data = temp_array
                        lower_bound = np.min(temp_array)
                        upper_bound = np.max(temp_array)

                        # For COMPUTE properties, we need to evaluate the property to get y_data
                        if prop_type.name == 'COMPUTE' and hasattr(self.material, prop_name):
                            current_prop = getattr(self.material, prop_name)
                            if isinstance(current_prop, (sp.Expr, sp.Piecewise)):
                                f_prop = sp.lambdify(self.T_symbol, current_prop, 'numpy')
                                y_data = f_prop(temp_array)

            except Exception as e:
                logger.warning(f"Could not extract temperature data for {prop_name}: {e}")

        return x_data, y_data, has_regression, simplify_type, degree, segments, lower_bound, upper_bound, lower_bound_type, upper_bound_type
