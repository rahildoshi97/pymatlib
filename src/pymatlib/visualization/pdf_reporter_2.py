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
    STANDARD_SPACING = 0.3  # Reduced spacing for better content fitting
    TEXT_FONT_SIZE = 9
    TITLE_FONT_SIZE = 12
    SECTION_TITLE_FONT_SIZE = 10
    PAGE_MARGINS = {'left': 0.08, 'right': 0.92, 'top': 0.92, 'bottom': 0.08}

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

        # Create a dedicated PropertyVisualizer instance for PDF generation
        self.pdf_visualizer = PropertyVisualizer(parser)
        self.pdf_visualizer.is_enabled = False

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
            'figure.autolayout': False,
            'figure.constrained_layout.use': False
        })

        with PdfPages(str(filepath)) as pdf:
            # Title page
            self._create_title_page(pdf)

            # YAML configuration pages (can span multiple pages)
            self._create_yaml_config_pages(pdf)

            # Property pages (one per property)
            self._create_property_pages(pdf)

        # Reset matplotlib parameters
        plt.rcParams.update(original_rcParams)

        logger.info(f"PDF report generated successfully: {filepath}")
        return filepath

    @staticmethod
    def _save_figure_safely(pdf: PdfPages, fig) -> None:
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
        """Create title page with layout to prevent overlaps."""
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

        ax.text(0.5, info_y_start - 0.03, f"({material_type.replace('_', ' ').title()})",
                transform=ax.transAxes, fontsize=12,
                horizontalalignment='center', color=self.COLORS['section_color'])

        # Generation info
        gen_info = f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}\nPyMatLib Version: 1.0.0"
        ax.text(0.5, info_y_start - 0.08, gen_info,
                transform=ax.transAxes, fontsize=10,
                horizontalalignment='center', style='italic')

        # Composition section
        comp_y_start = 0.60
        ax.text(0.5, comp_y_start, "Material Composition",
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                horizontalalignment='center', color=self.COLORS['section_color'])

        # Composition table
        comp_text = ""
        for i, (element, fraction) in enumerate(self.parser.config['composition'].items()):
            comp_text += f"{element}: {fraction:.4f}"
            if i < len(self.parser.config['composition']) - 1:
                comp_text += " | "

        ax.text(0.5, comp_y_start - 0.03, comp_text,
                transform=ax.transAxes, fontsize=11,
                horizontalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.COLORS['config_bg'],
                          edgecolor=self.COLORS['border_color'], alpha=0.8))

        # Temperature properties section - positioning to prevent overlap
        temp_y_start = 0.50
        ax.text(0.5, temp_y_start, "Temperature Properties",
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                horizontalalignment='center', color=self.COLORS['section_color'])

        temp_info = ""
        if material_type == 'pure_metal':
            melting_temp = float(self.material.melting_temperature)
            boiling_temp = float(self.material.boiling_temperature)
            temp_info = f"Melting Temperature: {melting_temp:.1f} K\nBoiling Temperature: {boiling_temp:.1f} K"
            # PURE METAL: 2 lines - smaller offset for uniform spacing
            content_offset = 0.045
        else:
            solidus_temp = float(self.material.solidus_temperature)
            liquidus_temp = float(self.material.liquidus_temperature)
            initial_boiling = float(self.material.initial_boiling_temperature)
            final_boiling = float(self.material.final_boiling_temperature)
            temp_info = (f"Solidus Temperature: {solidus_temp:.1f} K\n"
                         f"Liquidus Temperature: {liquidus_temp:.1f} K\n"
                         f"Initial Boiling Temperature: {initial_boiling:.1f} K\n"
                         f"Final Boiling Temperature: {final_boiling:.1f} K")
            # ALLOY: 4 lines - larger offset for uniform spacing
            content_offset = 0.075

        # DYNAMIC positioning - uniform spacing between heading and content
        ax.text(0.5, temp_y_start - content_offset, temp_info,
                transform=ax.transAxes, fontsize=9,
                horizontalalignment='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor=self.COLORS['expression_bg'],
                          edgecolor=self.COLORS['border_color'], alpha=0.8))

        # Summary section - DYNAMIC positioning based on content size
        if material_type == 'pure_metal':
            summary_y = 0.38  # Higher position for pure metals (less content above)
        else:
            summary_y = 0.32  # Lower position for alloys (more content above)

        total_properties = sum(len(props) for props in self.parser.categorized_properties.values())
        ax.text(0.5, summary_y, "Report Summary",
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                horizontalalignment='center', color=self.COLORS['section_color'])

        ax.text(0.5, summary_y - 0.03, f"Total Properties Analyzed: {total_properties}",
                transform=ax.transAxes, fontsize=12,
                horizontalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                          edgecolor=self.COLORS['border_color'], alpha=0.8))

        self._save_figure_safely(pdf, fig)

    def _create_yaml_config_pages(self, pdf: PdfPages) -> None:
        """Create YAML configuration pages."""
        # Convert config to YAML string
        yaml_content = yaml.dump(self.parser.config, default_flow_style=False, indent=2)

        # Split content into manageable chunks
        lines = yaml_content.split('\n')
        lines_per_page = 65  # Conservative estimate for readability

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

            # Spacing to reduced gap between title and content
            ax.text(0.05, 0.96, page_content, transform=ax.transAxes, fontsize=8,
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
        """Create a single page for one property with dynamic layout based on content size."""
        # Get text content
        config_text = self._get_config_text(prop_name, prop_config, prop_type)
        expr_text = self._get_expression_text(prop_name)

        # More accurate content size estimation
        config_lines = self._count_display_lines(config_text)
        expr_lines = self._count_display_lines(expr_text)

        # DYNAMIC layout decision based on actual content size
        total_text_lines = config_lines + expr_lines

        if config_lines > 15:  # Very long config
            self._create_multi_page_property(pdf, prop_name, prop_config, prop_type,
                                             config_text, expr_text)
        elif total_text_lines > 25:  # Combined content too long
            self._create_split_layout_property(pdf, prop_name, prop_config, prop_type,
                                               config_text, expr_text)
        elif expr_lines > 12:  # Long expression specifically
            self._create_expression_heavy_layout(pdf, prop_name, prop_config, prop_type,
                                                 config_text, expr_text)
        else:
            self._create_single_page_property_layout(pdf, prop_name, prop_config, prop_type,
                                                     config_text, expr_text)

    def _create_expression_heavy_layout(self, pdf: PdfPages, prop_name: str,
                                        prop_config: Any, prop_type,
                                        config_text: str, expr_text: str) -> None:
        """Create layout optimized for properties with long expressions."""
        fig = plt.figure(figsize=(8.5, 11))

        # Title
        fig.suptitle(f"Property Analysis: {prop_name}", fontsize=14, fontweight='bold',
                     color=self.COLORS['title_color'], y=0.95)

        # DYNAMIC height allocation based on expression length
        expr_lines = self._count_display_lines(expr_text)

        if expr_lines > 15:  # Very long expression
            # Config: small, Expression: large, Plot: separate page
            ax_config = fig.add_axes([0.08, 0.85, 0.84, 0.08])  # Very small config
            ax_config.axis('off')
            self._add_config_section_manual(ax_config, config_text, prop_type)

            ax_expr = fig.add_axes([0.08, 0.15, 0.84, 0.65])    # Large expression area
            ax_expr.axis('off')
            self._add_expression_section_manual(ax_expr, expr_text)

            self._save_figure_safely(pdf, fig)

            # Plot on separate page
            self._create_plot_only_page(pdf, prop_name, prop_config, prop_type)

        else:  # Moderately long expression
            # Allocate more space to expression, less to plot
            ax_config = fig.add_axes([0.08, 0.80, 0.84, 0.10])  # Config
            ax_config.axis('off')
            self._add_config_section_manual(ax_config, config_text, prop_type)

            ax_expr = fig.add_axes([0.08, 0.55, 0.84, 0.20])    # Larger expression
            ax_expr.axis('off')
            self._add_expression_section_manual(ax_expr, expr_text)

            ax_plot = fig.add_axes([0.1, 0.08, 0.8, 0.42])      # Smaller plot
            self._create_plot_using_exact_plotters_logic(ax_plot, prop_name, prop_config, prop_type)

            self._save_figure_safely(pdf, fig)

    def _create_single_page_property_layout1(self, pdf: PdfPages, prop_name: str,
                                            prop_config: Any, prop_type,
                                            config_text: str, expr_text: str) -> None:
        """Create property on a single page with DYNAMIC height allocation."""
        fig = plt.figure(figsize=(8.5, 11))

        # Title
        fig.suptitle(f"Property Analysis: {prop_name}", fontsize=14, fontweight='bold',
                     color=self.COLORS['title_color'], y=0.95)

        # DYNAMIC height calculation based on content
        config_lines = self._count_display_lines(config_text)
        expr_lines = self._count_display_lines(expr_text)

        # Calculate dynamic heights (total available: 0.80)
        total_text_lines = config_lines + expr_lines
        available_height = 0.80  # From y=0.08 to y=0.88

        # Allocate heights proportionally with minimums
        config_height = max(0.08, min(0.25, (config_lines / total_text_lines) * 0.35))
        expr_height = max(0.08, min(0.25, (expr_lines / total_text_lines) * 0.35))
        plot_height = available_height - config_height - expr_height - 0.06  # 0.06 for gaps

        # Ensure minimum plot height
        if plot_height < 0.30:
            # If plot would be too small, move to split layout
            self._create_split_layout_property(pdf, prop_name, prop_config, prop_type,
                                               config_text, expr_text)
            return

        # Calculate positions (bottom-up)
        plot_bottom = 0.08
        plot_top = plot_bottom + plot_height

        expr_bottom = plot_top + 0.03  # 3% gap
        expr_top = expr_bottom + expr_height

        config_bottom = expr_top + 0.03  # 3% gap
        config_top = config_bottom + config_height

        # Create sections with dynamic positioning
        ax_config = fig.add_axes([0.08, config_bottom, 0.84, config_height])
        ax_config.axis('off')
        self._add_config_section_manual(ax_config, config_text, prop_type)

        ax_expr = fig.add_axes([0.08, expr_bottom, 0.84, expr_height])
        ax_expr.axis('off')
        self._add_expression_section_manual(ax_expr, expr_text)

        ax_plot = fig.add_axes([0.1, plot_bottom, 0.8, plot_height])
        self._create_plot_using_exact_plotters_logic(ax_plot, prop_name, prop_config, prop_type)

        self._save_figure_safely(pdf, fig)

    def _create_single_page_property_layout1(self, pdf: PdfPages, prop_name: str,
                                            prop_config: Any, prop_type,
                                            config_text: str, expr_text: str) -> None:
        """Create property on a single page with INCREASED gap between sections."""
        fig = plt.figure(figsize=(8.5, 11))

        # Title
        fig.suptitle(f"Property Analysis: {prop_name}", fontsize=14, fontweight='bold',
                     color=self.COLORS['title_color'], y=0.95)

        # FIXED coordinates with INCREASED gaps between sections:
        # Config: y=0.78 to y=0.88 (height=0.10)
        # Expression: y=0.60 to y=0.70 (height=0.10)
        # LARGE gap: y=0.50 to y=0.60 (gap=0.10)
        # Plot: y=0.08 to y=0.50 (height=0.42)

        ax_config = fig.add_axes([0.08, 0.78, 0.84, 0.10])  # Config section
        ax_config.axis('off')
        self._add_config_section_manual(ax_config, config_text, prop_type)

        ax_expr = fig.add_axes([0.08, 0.60, 0.84, 0.10])    # Expression section - MOVED HIGHER
        ax_expr.axis('off')
        self._add_expression_section_manual(ax_expr, expr_text)

        ax_plot = fig.add_axes([0.1, 0.08, 0.8, 0.42])      # Plot section - REDUCED height
        self._create_plot_using_exact_plotters_logic(ax_plot, prop_name, prop_config, prop_type)

        self._save_figure_safely(pdf, fig)

    def _create_single_page_property_layout(self, pdf: PdfPages, prop_name: str,
                                            prop_config: Any, prop_type,
                                            config_text: str, expr_text: str) -> None:
        """Create property on a single page with DYNAMIC height allocation based on content."""
        fig = plt.figure(figsize=(8.5, 11))

        # Title
        fig.suptitle(f"Property Analysis: {prop_name}", fontsize=14, fontweight='bold',
                     color=self.COLORS['title_color'], y=0.95)

        # DYNAMIC height calculation based on actual content length
        config_lines = self._count_display_lines(config_text)
        expr_lines = self._count_display_lines(expr_text)

        # If content is too long for single page, split it
        if config_lines > 15 or expr_lines > 10 or (config_lines + expr_lines) > 20:
            self._create_split_layout_property(pdf, prop_name, prop_config, prop_type,
                                               config_text, expr_text)
            return

        # Calculate dynamic heights based on content
        # Base heights with minimum guarantees
        min_config_height = 0.08
        min_expr_height = 0.08
        min_plot_height = 0.35

        # Calculate proportional heights
        total_text_lines = max(config_lines + expr_lines, 1)  # Avoid division by zero

        # Allocate heights proportionally but with minimums
        config_ratio = config_lines / total_text_lines
        expr_ratio = expr_lines / total_text_lines

        # Available space for text sections (leaving space for plot and gaps)
        available_text_space = 0.45  # 45% of page for text sections

        config_height = max(min_config_height, config_ratio * available_text_space)
        expr_height = max(min_expr_height, expr_ratio * available_text_space)

        # Ensure we don't exceed available space
        total_text_height = config_height + expr_height
        if total_text_height > available_text_space:
            # Scale down proportionally
            scale_factor = available_text_space / total_text_height
            config_height *= scale_factor
            expr_height *= scale_factor

        # Calculate positions from top down with gaps
        gap = 0.03  # 3% gap between sections

        # Config section (top)
        config_top = 0.88
        config_bottom = config_top - config_height

        # Expression section (middle)
        expr_top = config_bottom - gap
        expr_bottom = expr_top - expr_height

        # Plot section (bottom) - use remaining space
        plot_top = expr_bottom - gap
        plot_bottom = 0.08
        plot_height = plot_top - plot_bottom

        # Ensure minimum plot height
        if plot_height < min_plot_height:
            # If plot would be too small, force split layout
            self._create_split_layout_property(pdf, prop_name, prop_config, prop_type,
                                               config_text, expr_text)
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

    @staticmethod
    def _count_display_lines(text: str) -> int:
        """Enhanced line counting that accurately estimates space needed."""
        lines = text.split('\n')
        total_lines = 0

        for line in lines:
            line_length = len(line.strip())
            if line_length == 0:
                total_lines += 0.3  # Empty lines take less space
            elif line_length < 40:
                total_lines += 0.8  # Short lines
            elif line_length < 70:
                total_lines += 1.0  # Normal lines
            else:
                # Long lines that will wrap
                wrapped_lines = (line_length + 59) // 60
                total_lines += wrapped_lines

        return int(total_lines * 1.2)  # Add 20% buffer for safety

    def _create_split_layout_property(self, pdf: PdfPages, prop_name: str,
                                      prop_config: Any, prop_type,
                                      config_text: str, expr_text: str) -> None:
        """Create property with config+expression on one page, plot on next with FIXED spacing."""
        # Page 1: Configuration + Expression with GUARANTEED no overlap
        fig1 = plt.figure(figsize=(8.5, 11))
        fig1.suptitle(f"Property Analysis: {prop_name} - Configuration & Expression",
                      fontsize=14, fontweight='bold', color=self.COLORS['title_color'], y=0.95)

        # FIXED positioning with clear separation:
        # Config: y=0.65 to y=0.85 (height=0.20)
        # Clear gap: y=0.40 to y=0.65 (gap=0.25)
        # Expression: y=0.20 to y=0.40 (height=0.20)

        ax_config = fig1.add_axes([0.08, 0.65, 0.84, 0.20])
        ax_config.axis('off')
        self._add_config_section_manual(ax_config, config_text, prop_type)

        ax_expr = fig1.add_axes([0.08, 0.20, 0.84, 0.20])
        ax_expr.axis('off')
        self._add_expression_section_manual(ax_expr, expr_text)

        self._save_figure_safely(pdf, fig1)

        # Page 2: Plot only
        self._create_plot_only_page(pdf, prop_name, prop_config, prop_type)

    def _create_multi_page_property(self, pdf: PdfPages, prop_name: str,
                                    prop_config: Any, prop_type,
                                    config_text: str, expr_text: str) -> None:
        """Create property across multiple pages with GUARANTEED no overlap."""
        # Page 1: Configuration only
        fig1 = plt.figure(figsize=(8.5, 11))
        fig1.suptitle(f"Property Analysis: {prop_name} - Configuration",
                      fontsize=14, fontweight='bold', color=self.COLORS['title_color'], y=0.95)

        ax1 = fig1.add_subplot(111)
        ax1.axis('off')

        # Configuration display
        ax1.text(0.05, 0.85, "Configuration Details", fontsize=self.SECTION_TITLE_FONT_SIZE,
                 fontweight='bold', color=self.COLORS['section_color'], transform=ax1.transAxes)

        ax1.text(0.05, 0.78, config_text, transform=ax1.transAxes, fontsize=self.TEXT_FONT_SIZE,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=self.COLORS['config_bg'],
                           edgecolor=self.COLORS['border_color'], alpha=0.8))

        self._save_figure_safely(pdf, fig1)

        # Page 2: Expression + Plot with GUARANTEED no overlap
        fig2 = plt.figure(figsize=(8.5, 11))
        fig2.suptitle(f"Property Analysis: {prop_name} - Expression & Visualization",
                      fontsize=14, fontweight='bold', color=self.COLORS['title_color'], y=0.95)

        # FIXED coordinates with LARGE separation:
        # Expression: y=0.80 to y=0.88 (height=0.08)
        # LARGE gap: y=0.65 to y=0.80 (gap=0.15)
        # Plot: y=0.08 to y=0.65 (height=0.57)

        ax_expr = fig2.add_axes([0.08, 0.80, 0.84, 0.08])  # Expression section - SMALLER height
        ax_expr.axis('off')
        self._add_expression_section_manual(ax_expr, expr_text)

        ax_plot = fig2.add_axes([0.1, 0.08, 0.8, 0.57])    # Plot section - REDUCED height
        self._create_plot_using_exact_plotters_logic(ax_plot, prop_name, prop_config, prop_type)

        self._save_figure_safely(pdf, fig2)

    def _create_plot_only_page(self, pdf: PdfPages, prop_name: str,
                               prop_config: Any, prop_type) -> None:
        """Create a plot-only page with optimal aspect ratio."""
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle(f"Property Visualization: {prop_name}", fontsize=14, fontweight='bold',
                     color=self.COLORS['title_color'], y=0.95)

        # Create subplot with positioning and optimal aspect ratio
        ax = fig.add_axes([0.12, 0.25, 0.76, 0.55])

        # Apply styling
        ax.set_aspect('auto')
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        self._create_plot_using_exact_plotters_logic(ax, prop_name, prop_config, prop_type)

        self._save_figure_safely(pdf, fig)

    @staticmethod
    def _get_config_text(prop_name: str, prop_config: Any, prop_type) -> str:
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

    def _add_config_section_manual(self, ax, config_text: str, prop_type) -> None:
        """Add formatted configuration section with manual positioning."""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Section title with styling
        ax.text(0.02, 0.98, "Configuration", fontsize=self.SECTION_TITLE_FONT_SIZE,
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

        ax.text(0.98, 0.98, prop_type.name, fontsize=self.TEXT_FONT_SIZE - 1,
                fontweight='bold', color='white', transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=type_color, alpha=0.9))

        # Configuration content with better positioning
        ax.text(0.02, 0.88, config_text, transform=ax.transAxes, fontsize=self.TEXT_FONT_SIZE,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.COLORS['config_bg'],
                          edgecolor=self.COLORS['border_color'], alpha=0.8))

    def _add_expression_section_manual(self, ax, expr_text: str) -> None:
        """Add formatted expression section with manual positioning."""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Section title
        ax.text(0.02, 0.98, "Final Expression", fontsize=self.SECTION_TITLE_FONT_SIZE,
                fontweight='bold', color=self.COLORS['section_color'], transform=ax.transAxes)

        # Expression content with better positioning
        ax.text(0.02, 0.88, expr_text, transform=ax.transAxes, fontsize=self.TEXT_FONT_SIZE,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self.COLORS['expression_bg'],
                          edgecolor=self.COLORS['border_color'], alpha=0.8))

    def _create_plot_using_exact_plotters_logic(self, ax, prop_name: str, prop_config: Any, prop_type) -> None:
        """
        Create property plot using EXACTLY the same logic as PropertyVisualizer.visualize_property().
        This ensures 100% consistency with plotters.py while maintaining formatting.
        """
        # Extract the exact parameters that plotters.py uses
        x_data, y_data, has_regression, simplify_type, degree, segments, lower_bound, upper_bound, lower_bound_type, upper_bound_type = self._extract_plotting_parameters(prop_name, prop_config, prop_type)

        # Apply plot styling that matches plotters.py
        ax.set_aspect('auto')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Add subtle border
        for spine in ax.spines.values():
            spine.set_color(self.COLORS['border_color'])
            spine.set_linewidth(1.2)

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
