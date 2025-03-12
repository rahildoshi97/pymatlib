# YAML Schema for Material Definition

This document defines the schema for material definition YAML files in pymatlib.

## Schema Overview

A valid material definition must include:
- `name`: String identifier for the material
- `composition`: Map of element symbols to their mass fractions
- `solidus_temperature`: Numeric value in Kelvin
- `liquidus_temperature`: Numeric value in Kelvin
- `properties`: Map of property names to their definitions

## Top-Level Structure

```yaml
name: <string> # Required: Material name
composition: # Required: Chemical composition
<element>: <mass_fraction> # At least one element required
solidus_temperature: <float> # Required: Solidus temperature in K
liquidus_temperature: <float> # Required: Liquidus temperature in K
properties: # Required: Material properties
<property_name>: <definition> # At least one property required
```

## Property Definition Types

Properties can be defined in four different ways:

### 1. Constant Value

For properties that don't vary with temperature:

```yaml
properties:
    thermal_expansion_coefficient: 16.3e-6 # Single numeric value
```

### 2. File-Based Properties

#### 2.1 Simple Format

```yaml
properties:
    density: ./density_temperature.txt # Path to file with two columns
```

The simple format assigns the first column to temperature data (K) and the second column to the corresponding property values.

#### 2.2 Advanced Format

```yaml
properties:
    density:
        file: ./path/to/file.xlsx
        temp_col: Temperature # Name of the temperature column
        prop_col: Property # Name of the property column
```

Advanced format is required when you have a file with multiple columns.
Supported file formats: .txt (space/tab separated), .csv, and .xlsx.

### 3. Key-Value Pairs

For properties defined at specific temperature points:

```yaml
properties:
    density:
        key: [1605, 1735] # Temperature values in K
        val: [7262.34, 7037.47] # Corresponding property values
```

Three ways to specify temperature points:

#### 3.1 Explicit List

```yaml
key: [1735.00, 1730.00, 1720.00, 1715.00] # Explicit temperature list
val: [7037.470, 7060.150, 7110.460, 7127.680] # Corresponding values
```

#### 3.2 Reference to Defined Temperatures

```yaml
key: [solidus_temperature, liquidus_temperature] # References
val: [7262.34, 7037.47] # Corresponding values
```

#### 3.3 Tuple with Start and Increment

```yaml
key: (1735.00, -5) # Start at 1735.00K and decrement by 5K
val: [7037.470, 7060.150, 7088.800, 7110.460, 7127.680]
```

When using a tuple for key, the generated temperature points will be:
`[start, start+increment, start+2*increment, ...]` until matching the length of val.

### 4. Computed Properties

#### 4.1 Simple Format

```yaml
properties:
    density: compute
    thermal_diffusivity: compute
    energy_density: compute
```

Simple format uses the default model to compute the property.

#### 4.2 Advanced Format

```yaml
properties:
    energy_density:
        compute: enthalpy_based # Specific computation model
```

#### 4.3 Energy Density Temperature Array

When energy_density is computed, you must specify the temperature array:

```yaml
properties:
    energy_density: compute
    energy_density_temperature_array: (300, 3000, 541) # 541 points
```

The third parameter can be:
- An integer: Total number of points to generate
- A float: Temperature increment/decrement between points

## Computation Models and Required Properties

### Density (by thermal expansion)
- **Equation**: ρ(T) = ρ₀ / (1 + tec * (T - T₀))³
- **Required properties**: base_temperature, base_density, thermal_expansion_coefficient

### Thermal Diffusivity
- **Equation**: α(T) = k(T) / (ρ(T) * cp(T))
- **Required properties**: heat_conductivity, density, heat_capacity

### Energy Density (standard model)
- **Equation**: ρ(T) * (cp(T) * T + L)
- **Required properties**: density, heat_capacity, latent_heat_of_fusion

### Energy Density (enthalpy_based model)
- **Equation**: ρ(T) * (h(T) + L)
- **Required properties**: density, specific_enthalpy, latent_heat_of_fusion

### Energy Density (total_enthalpy model)
- **Equation**: ρ(T) * h(T)
- **Required properties**: density, specific_enthalpy

## Validation Rules

1. All required top-level fields must be present
2. Composition fractions must sum to approximately 1.0
3. Liquidus temperature must be greater than or equal to solidus temperature
4. Properties cannot be defined in multiple ways or multiple times
5. Required dependencies for computed properties must be present
6. Temperature arrays must be monotonic
7. Energy density arrays must be monotonic with respect to temperature
8. File paths must be valid and files must exist
9. For key-value pairs, key and val must have the same length
10. When using tuple notation for temperature arrays, the increment must be non-zero

## Important Notes

1. All numerical values must use period (.) as decimal separator, not comma
2. Interpolation between data points is performed automatically for file-based and key-val properties
3. Properties will be computed in the correct order regardless of their position in the file
4. To retrieve temperature from energy_density, use the default "interpolate" method from within the generated class
