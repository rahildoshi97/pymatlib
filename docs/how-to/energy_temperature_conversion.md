# Converting Between Energy Density and Temperature

This guide explains how to perform bilateral conversions between energy density and temperature in pymatlib.

## Why Energy-Temperature Conversion Matters

In many material simulations, particularly those involving heat transfer and phase changes, you need to convert between:

- **Temperature**: The conventional measure of thermal state
- **Energy Density**: The amount of thermal energy per unit volume

This conversion is essential because:
- Energy is conserved in physical processes
- Phase transitions involve latent heat where temperature remains constant
- Many numerical methods work better with energy as the primary variable

## Basic Conversion Process

### 1. Generate the Interpolation Container

First, create an interpolation container that stores the energy-temperature relationship:

```python
import pystencils as ps
from pystencilssfg import SourceFileGenerator
from pymatlib.core.yaml_parser import create_alloy_from_yaml
from pymatlib.core.interpolators import InterpolationArrayContainer

with SourceFileGenerator() as sfg:
u = ps.fields("u: float64[2D]", layout='fzyx')

# Create an alloy
alloy = create_alloy_from_yaml("path/to/alloy.yaml", u.center())

# Generate the interpolation container
arr_container = InterpolationArrayContainer.from_material("SS304L", alloy)
sfg.generate(arr_container)
```

This generates C++ code with the `InterpolationArrayContainer` class that contains:
- Temperature array
- Energy density array
- Methods for interpolation

### 2. Using the Generated Code

In your C++ application, you can use the generated code:

```c++
// Energy to temperature conversion using binary search
double energy_density = 16950000000.0;  // J/m³
SS304L material;
double temp = material.interpolateBS(energy_density);

// Energy to temperature conversion using double lookup
double temp2 = material.interpolateDL(energy_density);

// Using the automatic method selection
double temp3 = material.interpolate(energy_density);
```

Note that the temperature to energy density conversion is handled in Python using the evalf method:

```python
import sympy as sp

# Create symbolic temperature variable
T = sp.Symbol('T')

# Temperature to energy conversion in Python
temperature = 1500.0  # Kelvin
energy = alloy.energy_density.evalf(T, temperature)

```

## Interpolation Methods

pymatlib provides two interpolation methods:

### Binary Search Interpolation

```c++
double temp = material.interpolateBS(energy_density);
```

- Best for non-uniform temperature distributions
- O(log n) lookup complexity
- Robust for any monotonically increasing data

### Double Lookup Interpolation

```c++
double temp = material.interpolateDL(energy_density);
```

- Optimized for uniform temperature distributions
- O(1) lookup complexity
- Faster but requires pre-processing

### Automatic Method Selection

```c++
double temp = material.interpolate(energy_density);
```

## Custom Interpolation Arrays

You can create custom interpolation containers for specific temperature ranges:

```python
import numpy as np
from pymatlib.core.interpolators import InterpolationArrayContainer

# Create custom temperature and energy arrays
T_array = np.linspace(300, 3000, 5)
E_array = np.array([...]) # Your energy values

# Create a custom container
custom_container = InterpolationArrayContainer("CustomMaterial", T_array, E_array)
sfg.generate(custom_container)
```

## Performance Considerations

- **Binary Search**: Use for general-purpose interpolation
- **Double Lookup**: Use for uniform temperature distributions
- **Array Size**: Balance between accuracy and memory usage
- **Density**: Add more points around phase transitions for accuracy

## Complete Example

Here's a complete example showing both methods:

```python
import numpy as np
from pystencilssfg import SourceFileGenerator
from pymatlib.core.interpolators import InterpolationArrayContainer

with SourceFileGenerator() as sfg:
    # Create temperature and energy arrays
    T_bs = np.array([3243.15, 3248.15, 3258.15, 3278.15], dtype=np.float64)
    E_bs = np.array([1.68e10, 1.69e10, 1.70e10, 1.71e10], dtype=np.float64)

    # Binary search container
    binary_search_container = InterpolationArrayContainer("BinarySearchTests", T_bs, E_bs)
    sfg.generate(binary_search_container)
    
    # Double lookup container
    T_eq = np.array([3243.15, 3253.15, 3263.15, 3273.15], dtype=np.float64)
    E_neq = np.array([1.68e10, 1.69e10, 1.70e10, 1.71e10], dtype=np.float64)
    double_lookup_container = InterpolationArrayContainer("DoubleLookupTests", T_eq, E_neq)
    sfg.generate(double_lookup_container)
```
