# Getting Started with pymatlib

This tutorial will guide you through the basics of using pymatlib to model material properties for simulations.

## Prerequisites

Before starting, ensure you have:

- Python 3.10 or newer
- Basic knowledge of material properties
- Basic familiarity with Python

## Installation

Install pymatlib using pip:

```bash
pip install "git+https://i10git.cs.fau.de/rahil.doshi/pymatlib.git"
```


For development, clone the repository and install in development mode:

```bash
git clone https://i10git.cs.fau.de/rahil.doshi/pymatlib.git
cd pymatlib
pip install -e .
```

## Basic Concepts

pymatlib organizes material data around a few key concepts:

1. **Alloys**: Compositions of multiple elements with specific properties
2. **Material Properties**: Physical characteristics that can vary with temperature
3. **Interpolation**: Methods to estimate property values between known data points

## Your First Alloy

Let's create a simple alloy definition:

1. Create a file named `simple_steel.yaml` with the following content:
```python
name: SimpleSteel

composition:
    Fe: 0.98
    C: 0.02

solidus_temperature: 1450
liquidus_temperature: 1520

properties:
    density:
        key:[300, 800, 1300, 1800]
        val:[7850, 7800, 7750, 7700]

    heat_conductivity:
        key: [300, 800, 1300, 1800]
        val: [18.5, 25, 32, 36.5]

    heat_capacity:
        key: [300, 800, 1300, 1800]
        val: [450, 500, 550, 600]

    thermal_diffusivity: compute    
```

2. Load the alloy in Python:

```python
from pymatlib.core.yaml_parser import create_alloy_from_yaml

# Load the alloy definition
alloy = create_alloy_from_yaml("simple_steel.yaml")

# Print basic information
print(f"Alloy: {alloy.name}")
print(f"Composition: {alloy.composition}")
print(f"Temperature range: {alloy.solidus_temperature}K - {alloy.liquidus_temperature}K")

# Get property values at specific temperatures
temp = 500  # Kelvin
density = alloy.get_property("density", temp)
conductivity = alloy.get_property("heat_conductivity", temp)
capacity = alloy.get_property("heat_capacity", temp)

print(f"At {temp}K:")
print(f" Density: {density} kg/m³")
print(f" Thermal Conductivity: {conductivity} W/(m·K)")
print(f" Heat Capacity: {capacity} J/(kg·K)")
```

## Next Steps

Now that you've created your first alloy, you can:

- Learn how to [create your first simulation](first_simulation.md)
- Explore [defining custom material properties](../how-to/define_materials.md)
- Understand [interpolation methods](../explanation/interpolation_methods.md)
