def create_yaml_config_images(output_dir: str = "./yaml_configs"):
    """Create 6 separate YAML configuration images."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Your YAML templates
    yaml_templates = {
        'CONSTANT': '''density: 6950.0''',

        'STEP_FUNCTION': '''density:
  temperature: melting_temperature
  value: [7500.0, 6500.0]
  bounds: [extrapolate, extrapolate]''',

        'KEY_VAL': '''density:
  temperature: [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1500, 2000, 2500, 3000]
  value: [7747, 7716, 7685, 7652, 7617, 7582, 7545, 7508, 7469, 7305, 6825, 6388, 5925]
  bounds: [extrapolate, extrapolate]''',

        'FILE': '''density:
  file_path: ./SS304L.xlsx
  temperature_header: T (K)
  value_header: Density (kg/(m)^3)
  bounds: [extrapolate, extrapolate]''',

        'PIECEWISE_EQUATION': '''density:
  temperature: [300, 1660, 1736, 3000]
  equation: [7877.39163826692 - 0.377781577789007*T, 11816.6337569868 - 2.74041499241854*T, 8596.40178865677 - 0.885849240373116*T]
  bounds: [extrapolate, extrapolate]''',

        'COMPUTE': '''density:
  temperature: (300, 3000, 5.0)
  equation: 2700 * (1 - 3*thermal_expansion_coefficient * (T - 293))
  bounds: [extrapolate, extrapolate]'''
    }

    # Method colors
    method_colors = {
        'CONSTANT': '#FF6B6B',
        'STEP_FUNCTION': '#4ECDC4',
        'KEY_VAL': '#45B7D1',
        'FILE': '#96CEB4',
        'PIECEWISE_EQUATION': '#FFEAA7',
        'COMPUTE': '#DDA0DD'
    }

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    count = 1

    # Generate each image
    for prop_type, yaml_text in yaml_templates.items():
        fig, ax = plt.subplots(figsize=(10, 0.8))
        ax.axis('off')

        # Add YAML text
        ax.text(0.01, 0.99, yaml_text,
                fontsize=8,
                family='monospace',
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="lightgray",
                          alpha=0.2,
                          edgecolor=method_colors[prop_type],
                          linewidth=2))

        # Add method label
        """ax.text(0.05, 0.02, f"Method: {prop_type}",
                fontsize=11,
                fontweight='bold',
                color=method_colors[prop_type],
                verticalalignment='bottom',
                horizontalalignment='left')"""

        # Add title
        ax.set_title(f"{count}. {prop_type} Configuration",
                     fontsize=8, fontweight='bold', pad=10, loc='left')
        count += 1
        # Save
        filename = f"yaml_config_{prop_type.lower()}.png"
        filepath = output_path / filename

        plt.savefig(str(filepath), dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none', pad_inches=0.2)
        plt.close(fig)

        print(f"Created: {filepath}")

    print("All 6 YAML configuration images created successfully!")

# Call the function
create_yaml_config_images()
