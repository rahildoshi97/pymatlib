# YAML Schema for Material Definition
This document defines the schema for material definition YAML files in pymatlib.

## Schema Overview

### A valid `pure metal` definition must include:
- `name`: String identifier for the material
- `composition`: Map of element symbols to their mass fractions
- `melting_temperature`: Numeric value in Kelvin
- `boiling_temperature`: Numeric value in Kelvin
- `properties`: Map of property names to their definitions
### A valid `alloy` definition must include:
- `name`: String identifier for the material
- `composition`: Map of element symbols to their mass fractions
- `solidus_temperature`: Numeric value in Kelvin
- `liquidus_temperature`: Numeric value in Kelvin
- `initial_boiling_temperature`: Numeric value in Kelvin
- `final_boiling_temperature`: Numeric value in Kelvin
- `properties`: Map of property names to their definitions

## 🔧 Property Definition Types
Properties can be defined in six different ways:
### 1. Constant Value
Simple numeric values, for properties that don't vary with temperature:
```yaml
thermal_expansion_coefficient: 16.3e-6 # Single numeric value
```
### 2. Step Function
Phase transitions with before/after values:
```yaml
latent_heat_of_fusion:
  temperature: melting_temperature - 10
  value: [0.0, 10790.0]
  bounds: [constant, constant]
```
### 3. File
Data from Excel, CSV, or text files:
```yaml
heat_capacity:
  file_path: ./data.xlsx
  temperature_header: T (K)
  value_header: Cp (J/kg·K)
  bounds: [constant, constant]
```
### 4. Key-Value Pairs
Explicit temperature-property pairs:
```yaml
thermal_expansion_coefficient:
  temperature: [300, 400, 500, 600]
  value: [1.2e-5, 1.4e-5, 1.6e-5, 1.8e-5]
  bounds: [constant, constant]
```
### 5. Piecewise Equation
Multiple equations for different temperature ranges:
```yaml
heat_conductivity:
  temperature: [500, 1700, 3000]
  equation: ["0.012*T + 13", "0.015*T + 5"]
  bounds: [constant, constant]
```
### 6. Compute
Properties calculated from other properties:
```yaml
thermal_diffusivity:
  temperature: (300, 3000, 5.0)
  equation: heat_conductivity / (density * heat_capacity)
  bounds: [extrapolate, extrapolate]
```

## 📊 Temperature Definition Formats
### Explicit Lists
```yaml
temperature: [300, 400, 500, 600]  # Explicit values
```
### Tuple Formats
```yaml
# (start, increment) - requires matching value list length
temperature: (300, 50)  # 300, 350, 400, ... (length from values)

# (start, stop, step) - step size
temperature: (300, 1000, 10.0)  # 300, 310, 320, ..., 1000

# (start, stop, points) - number of points
temperature: (300, 1000, 71)  # 71 evenly spaced points

# Decreasing temperature
temperature: (1000, 300, -5.0)  # 1000, 995, 990, ..., 300
```
### Temperature References
```yaml
# Direct references
temperature: melting_temperature
temperature: solidus_temperature

# Arithmetic expressions
temperature: melting_temperature + 50
temperature: liquidus_temperature - 10
```

## 🎯 Advanced Features
### Regression Configuration
Control data simplification and memory usage:
```yam
regression:
  simplify: pre    # Apply before symbolic processing
  degree: 1        # Linear regression
  segments: 3      # Number of piecewise segments
```
- `simplify: pre`: Apply regression to raw data before processing
- `simplify: post`: Apply regression after symbolic expressions are evaluated
- `degree`: Polynomial degree (1=linear, 2=quadratic, etc.)
- `segments`: Number of segments for piecewise functions
### Boundary Behavior
Control extrapolation outside data range:
```yaml
bounds: [constant, extrapolate]
```
- `constant`: Use boundary values as constants outside range
- `extrapolate`: Linear extrapolation outside range
### Dependency Resolution
Properties are automatically processed in correct order:
```yaml# These will be processed in dependency order automatically
specific_enthalpy:
  equation: Integral(heat_capacity, T)

energy_density:
  equation: density * specific_enthalpy  # Depends on specific_enthalpy

thermal_diffusivity:
  equation: heat_conductivity / (density * heat_capacity)  # Multiple dependencies
```

## 📈 Visualization
Automatic plot generation when using symbolic temperature:
```python
import sympy as sp
from pymatlib.core.yaml_parser.api import create_material_from_yaml
T = sp.Symbol('T')
material = create_material_from_yaml('steel.yaml', T, enable_plotting=True)
# Plots automatically saved to 'pymatlib_plots/' directory
```

## 🧪 Energy-Temperature Inversion
For applications requiring temperature from energy density:
```python
import sympy as sp
from pymatlib.core.yaml_parser.api import create_material_from_yaml
from pymatlib.core.piecewise_inverter import create_energy_density_inverse

# Create inverse function T = f_inv(E)
T = sp.Symbol('T')
E = sp.Symbol('E')
material = create_material_from_yaml('steel.yaml', T)

# Create inverse (only for linear piecewise functions)
inverse_func = create_energy_density_inverse(material, 'E')

# Use inverse function
energy_value = 1.5e9  # J/m³
temperature = float(inverse_func.subs(E, energy_value))
```

## 🔍 Supported Properties
```python
from pymatlib.core.yaml_parser.api import get_supported_properties
print(get_supported_properties())
```

## Validation Rules

1. All required top-level fields must be present
2. Composition fractions must sum to approximately 1.0
3. Liquidus temperature must be greater than or equal to solidus temperature
4. Properties cannot be defined in multiple ways or multiple times
5. Required dependencies for computed properties must be present
6. Temperature arrays must be monotonic
7. Energy density arrays must be monotonic with respect to temperature
8. File paths must be valid and files must exist
9. For key-value pairs, key and value must have the same length
10. When using tuple notation for temperature arrays, the increment must be non-zero

## Important Notes

1. All numerical values must use period (.) as decimal separator, not comma
2. Interpolation between data points is performed automatically for file-based and key-val properties
3. Properties will be computed in the correct order regardless of their position in the file
4. To retrieve temperature from energy_density, use the default "interpolate" method from within the generated class
