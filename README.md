# PyMatLib - Python Material Properties Library

A high-performance Python library for material simulation and analysis with a focus on temperature-dependent properties. PyMatLib enables efficient modeling of pure metals and alloys through YAML configuration files, providing symbolic and numerical property evaluation for various material properties.

<<<<<<< HEAD
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Latest Release](https://i10git.cs.fau.de/rahil.doshi/pymatlib/-/badges/release.svg)](https://i10git.cs.fau.de/rahil.doshi/pymatlib/-/releases)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)
||||||| 5cbcbff
=======
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Latest Release](https://i10git.cs.fau.de/rahil.doshi/pymatlib/-/badges/release.svg)](https://i10git.cs.fau.de/rahil.doshi/pymatlib/-/releases)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](LICENSE)
>>>>>>> release/v0.4.1
[![Pipeline Status](https://i10git.cs.fau.de/rahil.doshi/pymatlib/badges/master/pipeline.svg)](https://i10git.cs.fau.de/rahil.doshi/pymatlib/-/pipelines)
[![Code Coverage](https://i10git.cs.fau.de/rahil.doshi/pymatlib/badges/master/coverage.svg)](https://i10git.cs.fau.de/rahil.doshi/pymatlib/-/commits/master)

## Table of Contents
- [Key Features](#-key-features)
- [Installation](#-installation)
- [YAML Configuration Format](#-yaml-configuration-format)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [Known Limitations](#-known-limitations)
- [License](#-license)
- [Support](#-support)

## 🚀 Key Features
- **Modular Architecture**: Clean separation with algorithms, parsing, and visualization modules
- **Flexible Material Definition**: Support for both pure metals and alloys
- **YAML-Driven Configuration**: Define materials using intuitive YAML files
- **Temperature-Dependent Properties**: Support for complex temperature-dependent material properties
- **Symbolic Mathematics**: Built on SymPy for precise mathematical expressions
- **Piecewise Functions**: Advanced piecewise function support with regression capabilities
- **Property Inversion**: Create inverse functions for energy density and other properties
- **Visualization**: Automatic plotting of material properties with customizable options
- **Material Types**: Support for both pure metals and alloys with appropriate phase transition temperatures
- **Multiple Property Types**: Support for constants, step functions, file-based data, key-value pairs, and computed properties
- **Regression Analysis**: Built-in piecewise linear fitting with configurable parameters

## 📦 Installation
### Prerequisites
- Python 3.10 or higher
- Required dependencies: `numpy`, `sympy`, `matplotlib`, `pandas`, `ruamel.yaml`

### Install from Git Repository
```
pip install "git+https://i10git.cs.fau.de/rahil.doshi/pymatlib.git"
```
### Development Installation
```bash
git clone https://i10git.cs.fau.de/rahil.doshi/pymatlib.git
cd pymatlib
pip install -e .[dev]
```

## 🏃 Quick Start
### Basic Material Creation

```python
import sympy as sp
from pymatlib.parsing.api import create_material

# Create a material with symbolic temperature
T = sp.Symbol('T')
material_T = create_material('path/to/material.yaml', T)

# Create a material at specific temperature
material_500 = create_material('path/to/material.yaml', 500.0)

# Access properties
print(f"Heat capacity: {material_T.heat_capacity}")
print(f"Density: {material_500.density}")

# Evaluate at specific temperature
temp_value = 1500.0  # Kelvin
density_at_temp = float(material_T.density.subs(T, temp_value))
print(f"Density at {temp_value}K: {density_at_temp:.2f} kg/m³")

# For numerical evaluation (no plots generated)
material_800 = create_material('aluminum.yaml', 800.0)

# For symbolic expressions with automatic plotting
material_with_plot = create_material('steel.yaml', T, enable_plotting=True)
```
### Working with Piecewise Inverse Functions

```python
from pymatlib.algorithms.piecewise_inverter import PiecewiseInverter

# Create inverse energy density function: T = f_inv(E)
if hasattr(material, 'energy_density'):
    E = sp.Symbol('E')
    inverse_func = PiecewiseInverter.create_energy_density_inverse(material, 'E')

    # Test round-trip accuracy
    test_temp = 500.0
    energy_val = float(material.energy_density.subs(T, test_temp))
    recovered_temp = float(inverse_func.subs(E, energy_val))
    print(f"Round-trip: T={test_temp} -> E={energy_val:.2e} -> T={recovered_temp:.2f}")
```

## 📋 YAML Configuration Format
### Supported Property Types
- **CONSTANT**: Simple numeric values
- **FILE**: Data loaded from CSV/Excel files
- **KEY_VAL**: Temperature-value pairs
- **STEP_FUNCTION**: Discontinuous transitions
- **PIECEWISE_EQUATION**: Symbolic equations over temperature ranges
- **COMPUTE**: Properties calculated from other properties

See [the YAML schema documentation](docs/reference/yaml_schema.md) for detailed configuration options.
YAML configuration examples can be found here:
- [Pure Metals](src/pymatlib/data/materials/pure_metals/Al/Al.yaml)
- [Alloys](src/pymatlib/data/materials/alloys/SS304L/SS304L.yaml)

## 📚 Documentation
Our documentation follows the _Diátaxis_ framework with four distinct types:
### Tutorials - Learning-oriented guides
- [Getting Started with pymatlib](docs/tutorials/getting_started.md)
- [Creating Your First Material Simulation](docs/tutorials/first_simulation.md)
### How-to Guides - Problem-oriented instructions
- [Defining Custom Material Properties](docs/how-to/define_materials.md)
- [Converting Between Energy Density and Temperature](docs/how-to/energy_temperature_conversion.md)
### Reference - Information-oriented documentation
- [API Reference](docs/reference/api)
- [YAML Configuration Schema](docs/reference/yaml_schema.md)
### Explanation - Understanding-oriented discussions
- [Material Properties Concepts](docs/explanation/material_properties.md)
- [Design Philosophy](docs/explanation/design_philosophy.md)

<<<<<<< HEAD
## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🐛 Known Limitations
- **Piecewise Inverter**: Currently supports only linear piecewise functions
- **File Formats**: Limited to CSV, Excel, and text files
- **Memory Usage**: Large datasets may require optimization for very high-resolution data
- **Regression**: Maximum 8 segments recommended for stability

## 📄 License
This project is licensed under the GNU General Public License v3 (GPLv3). See the [LICENSE](https://i10git.cs.fau.de/rahil.doshi/pymatlib/-/blob/master/LICENSE?ref_type=heads) file for details.

## 📖 Citation
If you use PyMatLib in your research, please cite it as:
```bibtex
@software{doshi2025pymatlib,
author = {Doshi, Rahil},
title = {PyMatLib: A Python Library for Temperature-Dependent Materials Property Modeling},
version = {0.3.0},
year = {2025},
url = {https://i10git.cs.fau.de/rahil.doshi/pymatlib}
}
```
For the latest citation information, see [CITATION.cff](CITATION.cff).
||||||| 5cbcbff
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
=======
## 🤝 Contributing
Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

## 🐛 Known Limitations
- **Piecewise Inverter**: Currently supports only linear piecewise functions
- **File Formats**: Limited to CSV, Excel, and text files
- **Memory Usage**: Large datasets may require optimization for very high-resolution data
- **Regression**: Maximum 8 segments recommended for stability

## 📄 License
This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

## 📖 Citation
If you use PyMatLib in your research, please cite it using the information in our [CITATION.cff](CITATION.cff) file.
>>>>>>> release/v0.4.1

## 📞 Support
- **Author**: Rahil Doshi
- **Email**: rahil.doshi@fau.de
- **Project Homepage**: [pymatlib](https://i10git.cs.fau.de/rahil.doshi/pymatlib)
- **Bug Tracker**: [Issues](https://i10git.cs.fau.de/rahil.doshi/pymatlib/-/issues)
<<<<<<< HEAD

## 🙏 Acknowledgments
- Built with [SymPy](https://www.sympy.org/) for symbolic mathematics
- Data handling powered by [pandas](https://pandas.pydata.org/)
- Uses [pwlf](https://github.com/cjekel/piecewise_linear_fit_py) for piecewise linear fitting
- Visualization powered by [Matplotlib](https://matplotlib.org/)
- YAML parsing with [ruamel.yaml](https://yaml.readthedocs.io/)

#### PyMatLib - Empowering material simulation with Python 🚀

---
||||||| 5cbcbff
=======

## 🙏 Acknowledgments
- Built with [SymPy](https://www.sympy.org/) for symbolic mathematics
- Data handling powered by [pandas](https://pandas.pydata.org/)
- Uses [pwlf](https://github.com/cjekel/piecewise_linear_fit_py) for piecewise linear fitting
- Visualization powered by [Matplotlib](https://matplotlib.org/)
- YAML parsing with [ruamel.yaml](https://yaml.dev/doc/ruamel.yaml/)

#### PyMatLib - Empowering material simulation with Python 🚀

---
>>>>>>> release/v0.4.1
