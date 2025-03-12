# pymatlib

# pymatlib

A high-performance Python library for material simulation and analysis with a focus on temperature-dependent properties.
pymatlib enables efficient modeling of alloys, interpolation of material properties, 
and robust energy-temperature conversions for scientific and engineering applications.

[![Pipeline Status](https://i10git.cs.fau.de/rahil.doshi/pymatlib/badges/master/pipeline.svg)](https://i10git.cs.fau.de/rahil.doshi/pymatlib/-/pipelines)
![License](https://img.shields.io/badge/license-GPLv3-blue)

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

To install `pymatlib`, use pip in a [virtual environment](https://docs.python.org/3/library/venv.html):
```bash
pip install "git+https://i10git.cs.fau.de/rahil.doshi/pymatlib.git"
```

## Quick Start

### Creating and Using an Alloy
```python
import pystencils as ps
from pystencilssfg import SourceFileGenerator
from pymatlib.core.typedefs import MaterialProperty
from pymatlib.core.yaml_parser import create_alloy_from_yaml

# Define a temperature field
with SourceFileGenerator() as sfg:
    u = ps.fields("u: float64[2D]", layout='fzyx')

    # Create an alloy from YAML configuration
    alloy = create_alloy_from_yaml("path/to/alloy.yaml", u.center())
    
    # Now you can use the alloy in your simulation
    # Get material properties at specific temperature
    temperature = 1500.0  # Kelvin
    thermal_diffusivity_value = alloy.thermal_diffusivity.evalf(u.center(), temperature)
    print(f"Thermal diffusivity at {temperature}K: {thermal_diffusivity_value} W/(m·K)")
```
See the [tutorials](#documentation) for more detailed examples.

## Documentation

Our documentation follows the Diátaxis framework with four distinct types:

### Tutorials - Learning-oriented guides
- [Getting Started with pymatlib](docs/tutorials/getting_started.md)
- [Creating Your First Material Simulation](docs/tutorials/first_simulation.md)

### How-to Guides - Problem-oriented instructions
- [Defining Custom Material Properties](docs/how-to/define_materials.md)
- [Converting Between Energy Density and Temperature](docs/how-to/energy_temperature_conversion.md)

### Reference - Information-oriented documentation
- [API Reference](docs/reference/api/)
- [YAML Configuration Schema](docs/reference/yaml_schema.md)

### Explanation - Understanding-oriented discussions
- [Interpolation Methods](docs/explanation/interpolation_methods.md)
- [Material Properties Concepts](docs/explanation/material_properties.md)
- [Design Philosophy](docs/explanation/design_philosophy.md)

## Key Features

| Feature | Description |
|---------|-------------|
| **Material Modeling** | Define complex alloys and their properties using simple YAML configurations |
| **Temperature-dependent Properties** | Model how material properties change with temperature through various interpolation methods |
| **Energy-Temperature Conversion** | Perform efficient bilateral conversion between energy density and temperature |
| **High Performance** | Generate optimized C++ code for computationally intensive simulations |
| **Extensible Design** | Easily add custom material properties and interpolation methods |

## Core Components

### Alloy

A dataclass representing a material alloy with properties like:
- Elements and composition
- Solidus and liquidus temperatures
- Temperature-dependent material properties

### MaterialProperty

A dataclass for material properties that can be:
- Defined as constant values
- Specified as key-value pairs for interpolation
- Loaded from external data files
- Computed from other properties

### InterpolationArrayContainer

Manages data for efficient interpolation between energy density and temperature using:
- Binary search interpolation for non-uniform data
- Double lookup interpolation for uniform data

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Clone your fork locally: `git clone https://github.com/yourusername/pymatlib.git`
3. Create a new branch: `git checkout -b feature/your-feature`
4. Make your changes and commit: `git commit -m "Add new feature"`
5. Push to the branch: `git push origin feature/your-feature`
6. Open a pull request

## License

This project is licensed under the GNU General Public License v3 (GPLv3). See the [LICENSE](https://i10git.cs.fau.de/rahil.doshi/pymatlib/-/blob/master/LICENSE?ref_type=heads) file for details.

## Contact

- **Author**: Rahil Doshi
- **Email**: rahil.doshi@fau.de
- **Project Homepage**: [pymatlib](https://i10git.cs.fau.de/rahil.doshi/pymatlib)
- **Bug Tracker**: [Issues](https://i10git.cs.fau.de/rahil.doshi/pymatlib/-/issues)
