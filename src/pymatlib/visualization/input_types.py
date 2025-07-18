import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def create_split_figure(yaml_text, x_data, y_data, property_name, x_label='Temperature (K)', y_label='Property Value'):
    """
    Create a figure with YAML snippet on left and property plot on right

    Parameters:
    - yaml_text: String containing the YAML configuration
    - x_data: Temperature data array
    - y_data: Property values array
    - property_name: Name of the property being plotted
    - x_label: Label for x-axis
    - y_label: Label for y-axis
    """

    # Create figure with wider aspect ratio
    fig = plt.figure(figsize=(8, 3))

    # Create grid layout with equal width columns
    gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1, 1.], figure=fig)

    # Left panel: YAML configuration text
    ax_text = fig.add_subplot(gs[0])
    ax_text.axis('off')  # Remove axes

    # Add YAML text with proper formatting
    ax_text.text(0.05, 0.95, yaml_text,
                 fontsize=8,
                 family='monospace',
                 verticalalignment='top',
                 horizontalalignment='left',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgray", alpha=0.2))

    # Add title for YAML section
    ax_text.set_title(f"YAML Configuration\n{property_name}", fontsize=10, fontweight='bold', pad=10)

    # Right panel: Property plot
    ax_plot = fig.add_subplot(gs[1])
    ax_plot.plot(x_data, y_data, 'b-', linewidth=2, marker='o', markersize=4)
    ax_plot.set_xlabel(x_label, fontsize=10)
    ax_plot.set_ylabel(y_label, fontsize=10)
    ax_plot.grid(True, alpha=0.2)
    ax_plot.set_title(f"{property_name} vs Temperature", fontsize=10, fontweight='bold')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)

    return fig

# Example usage function
def plot_property_example():
    """Example of how to use the split figure function"""

    # Example YAML snippet
    yaml_snippet_1 = """density: 1.71401E5"""

    yaml_snippet_2 = """density:
      temperature: melting_temperature - 1
      value: [0.0, 10790.0]
      bounds: [constant, constant]"""

    yaml_snippet_3 = """density:
      temperature: [300, 400, 500, 600, 700]
      value: [2.38e-05, 2.55e-05, 2.75e-05, 2.95e-05, 3.15e-05]
      bounds: [constant, constant]"""

    yaml_snippet_4 = """density:
      file_path: ./SS304L.xlsx
      temperature_header: T (K)
      value_header: Density (kg/m³)
      bounds: [constant, constant]
      regression:
        simplify: post
        degree: 2
        segments: 3"""

    yaml_snippet_5 = """density:
      temperature: (300, 3000, 5.0)
      equation: heat_conductivity / (density * heat_capacity)
      bounds: [constant, constant]"""

    # Example data (replace with your actual property data)
    temperatures = np.linspace(300, 1500, 50)
    density_values = 8000 - 0.5 * temperatures  # Example density decreasing with temperature

    # Create the figure
    fig = create_split_figure(
        yaml_text=yaml_snippet_2,
        x_data=temperatures,
        y_data=density_values,
        property_name="Density",
        y_label="Density (kg/m³)"
    )

    # Save the figure
    plt.savefig('property_with_yaml.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_property_example()
